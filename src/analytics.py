"""City, regional, temporal, episode and RDD analytics (Phase 4).

All inputs are MODELLED concentrations (CAMS via Open-Meteo), not official
monitoring-station measurements. Results describe the model data.

Spark notes:
  * groupBy().agg() is a wide transformation: Spark first computes partial
    aggregates inside each partition, then shuffles only those partial
    results by key and merges them (much less data than shuffling rows).
  * Window functions (rank, row_number, lag) compute a value per row from
    other rows in its partition (e.g. the previous year of the same city)
    without collapsing rows the way groupBy does.
  * percentile_approx uses a mergeable sketch, so percentiles are computed
    in parallel without sorting each group on one machine. With accuracy
    10000 the rank error is at most n/10000.
"""
from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from src.config import (AQI_BANDS, EPISODE_MIN_DAYS, INDEX_UNITS,
                        MIN_CITIES_PER_GROUP, MONTHLY_WEATHER, NAAQS_24H,
                        PERCENTILE_ACCURACY, POLLUTANT_UNITS,
                        RDD_CHECK_TOLERANCE, SEASON_ORDER, SUMMARY_POLLUTANTS,
                        TOP_DAYS_PER_CITY, WHO_2021_24H)

UNITS = {**POLLUTANT_UNITS, **INDEX_UNITS}
CATEGORY_ORDER = {label: i for i, (_, label) in enumerate(AQI_BANDS, start=1)}
SEASON_INDEX = {season: i for i, season in enumerate(SEASON_ORDER, start=1)}
LONG_ID_COLUMNS = ["city", "state", "region", "latitude", "longitude",
                   "date", "year", "season", "season_complete", "is_provisional"]
QUANTILES = {"p25": 0.25, "median": 0.5, "p75": 0.75, "p95": 0.95}


# ---------------------------------------------------------------- shared helpers

def literal_map(mapping: dict) -> Column:
    """A Spark map literal, e.g. literal_map(UNITS)[F.col('pollutant')]."""
    return F.create_map(*[F.lit(x) for pair in mapping.items() for x in pair])


def explode_structs(df: DataFrame, keep_cols: list[str], structs: list[Column]) -> DataFrame:
    """Turn several struct columns of one row into several rows (wide -> long)."""
    return (df.select(*keep_cols, F.explode(F.array(*structs)).alias("item"))
              .select(*keep_cols, "item.*"))


def to_long(daily: DataFrame, value_cols: list[str] = SUMMARY_POLLUTANTS) -> DataFrame:
    """Unpivot pollutant columns into (pollutant, value) rows.

    Adds the unit and, where one exists, the NAAQS and WHO 24-h limits
    (null for pollutants without a limit, so exceedance counts stay null).
    """
    numeric = daily.select(*LONG_ID_COLUMNS, *[F.col(c).cast("double") for c in value_cols])
    return (
        numeric.unpivot(LONG_ID_COLUMNS, value_cols, "pollutant", "value")
        .withColumn("unit", literal_map(UNITS)[F.col("pollutant")])
        .withColumn("naaqs_limit", literal_map(NAAQS_24H)[F.col("pollutant")])
        .withColumn("who_limit", literal_map(WHO_2021_24H)[F.col("pollutant")])
    )


def _above(limit_col: str) -> Column:
    # Null limit -> null comparison -> sum() is null: "not applicable".
    return F.sum((F.col("value") > F.col(limit_col)).cast("int"))


# ---------------------------------------------------------------- city / region / state

def city_pollutant_stats(long: DataFrame) -> DataFrame:
    """Descriptive statistics per city and pollutant, ranked by mean."""
    keys = ["city", "state", "region", "latitude", "longitude", "pollutant", "unit"]
    quantiles = F.percentile_approx("value", list(QUANTILES.values()), PERCENTILE_ACCURACY)
    stats = long.groupBy(*keys).agg(
        F.count("value").alias("n_days"),
        F.mean("value").alias("mean"),
        F.stddev("value").alias("std"),
        F.min("value").alias("min"),
        quantiles.alias("q"),
        F.max("value").alias("max"),
        _above("naaqs_limit").alias("days_above_naaqs"),
        _above("who_limit").alias("days_above_who"),
    )
    by_pollutant = Window.partitionBy("pollutant").orderBy(F.desc("mean"))
    return (
        stats.select("*", *[F.col("q")[i].alias(name) for i, name in enumerate(QUANTILES)])
        .drop("q")
        .withColumn("pct_days_above_naaqs", 100 * F.col("days_above_naaqs") / F.col("n_days"))
        .withColumn("pct_days_above_who", 100 * F.col("days_above_who") / F.col("n_days"))
        .withColumn("rank_highest_mean", F.rank().over(by_pollutant))
    )


def group_pollutant_stats(spark: SparkSession, long: DataFrame, group_cols: list[str]) -> DataFrame:
    """Region- or state-level statistics, written in Spark SQL.

    Every city has the same number of days, so the mean over all city-days
    of a group equals the average of its city means (equal city weights).
    Groups with fewer than MIN_CITIES_PER_GROUP cities are flagged.
    A state can span regions in this dataset (Maharashtra: West and
    Central), so non-region groups list their regions instead of grouping
    by them.
    """
    long.createOrReplaceTempView("pollutant_long")
    cols = ", ".join(group_cols)
    regions = ("" if "region" in group_cols
               else "array_join(array_sort(collect_set(region)), ', ') AS regions,")
    return spark.sql(f"""
        SELECT {cols}, pollutant, unit, {regions}
               COUNT(DISTINCT city)                                AS n_cities,
               COUNT(value)                                        AS n_city_days,
               AVG(value)                                          AS mean,
               percentile_approx(value, 0.5, {PERCENTILE_ACCURACY})  AS median,
               percentile_approx(value, 0.95, {PERCENTILE_ACCURACY}) AS p95,
               MAX(value)                                          AS max,
               100 * AVG(CAST(value > naaqs_limit AS DOUBLE))      AS pct_days_above_naaqs,
               100 * AVG(CAST(value > who_limit AS DOUBLE))        AS pct_days_above_who,
               COUNT(DISTINCT city) < {MIN_CITIES_PER_GROUP}       AS few_cities,
               RANK() OVER (PARTITION BY pollutant ORDER BY AVG(value) DESC) AS rank_highest_mean
        FROM pollutant_long
        GROUP BY {cols}, pollutant, unit
    """)


def aqi_category_distribution(daily: DataFrame) -> DataFrame:
    """Days and share of days per AQI category, for both Indian AQI variants."""
    category_cols = ["india_aqi_pm_category", "india_aqi_category"]
    counts = (
        daily.unpivot(["city", "region"], category_cols, "aqi_index", "category")
        .withColumn("aqi_index", F.regexp_replace("aqi_index", "_category$", ""))
        .groupBy("city", "region", "aqi_index", "category")
        .agg(F.count("*").alias("days"))
    )
    per_city_index = Window.partitionBy("city", "aqi_index")
    return (counts
            .withColumn("share", F.col("days") / F.sum("days").over(per_city_index))
            .withColumn("category_order", literal_map(CATEGORY_ORDER)[F.col("category")]))


# ---------------------------------------------------------------- temporal

def monthly_city(daily: DataFrame) -> DataFrame:
    """Monthly means of daily values per city (wide format, for time series)."""
    month_start = F.make_date("year", "month", F.lit(1))
    return (
        daily.groupBy("city", "region", "year", "month")
        .agg(F.count("*").alias("n_days"),
             F.sum(F.col("is_provisional").cast("int")).alias("provisional_days"),
             *[F.mean(c).alias(c) for c in SUMMARY_POLLUTANTS + MONTHLY_WEATHER])
        .withColumn("month_start", month_start)
        .withColumn("month_complete", F.col("n_days") == F.dayofmonth(F.last_day(month_start)))
    )


def complete_years(daily: DataFrame) -> list[int]:
    """Years in which every city has every calendar day (derived, not assumed)."""
    n_cities = daily.select("city").distinct().count()
    per_city = (
        daily.groupBy("city", "year").agg(F.countDistinct("date").alias("days"))
        .withColumn("expected", F.datediff(F.make_date(F.col("year") + 1, F.lit(1), F.lit(1)),
                                           F.make_date("year", F.lit(1), F.lit(1))))
    )
    years = (per_city.groupBy("year")
             .agg(F.count("*").alias("cities"),
                  F.min((F.col("days") == F.col("expected")).cast("int")).alias("all_complete"))
             .collect())   # one row per year: tiny
    return sorted(r.year for r in years if r.all_complete == 1 and r.cities == n_cities)


def annual_city(long: DataFrame, years: list[int]) -> DataFrame:
    """Annual means for complete years, with change from the previous year."""
    annual = (long.filter(F.col("year").isin(years))
              .groupBy("city", "region", "pollutant", "unit", "year")
              .agg(F.count("value").alias("n_days"), F.mean("value").alias("mean")))
    by_city = Window.partitionBy("city", "pollutant").orderBy("year")
    previous = F.lag("mean").over(by_city)
    consecutive = F.lag("year").over(by_city) == F.col("year") - 1
    return (annual
            .withColumn("previous_year_mean", F.when(consecutive, previous))
            .withColumn("change", F.col("mean") - F.col("previous_year_mean"))
            .withColumn("change_pct", F.when(F.col("previous_year_mean") != 0,
                                             100 * F.col("change") / F.col("previous_year_mean"))))


def seasonal_city(long: DataFrame) -> DataFrame:
    """Per city, season-year and pollutant; only complete seasons are kept."""
    return (long.filter(F.col("season_complete"))
            .groupBy("city", "region", "year", "season", "pollutant", "unit")
            .agg(F.count("value").alias("n_days"), F.mean("value").alias("mean"),
                 F.percentile_approx("value", 0.5, PERCENTILE_ACCURACY).alias("median"))
            .withColumn("season_order", literal_map(SEASON_INDEX)[F.col("season")]))


# ---------------------------------------------------------------- episodes and extremes

def pollution_episodes(daily: DataFrame, flag_col: str = "pm2_5_above_naaqs",
                       min_days: int = EPISODE_MIN_DAYS) -> DataFrame:
    """Runs of at least min_days consecutive calendar days with flag_col true.

    "Gaps and islands": number the flagged days of each city 1, 2, 3, ...
    in date order. For consecutive dates, date minus that number is
    constant, so it identifies the run. A missing or unflagged day breaks it.
    touches_data_boundary marks runs that may continue outside the data.
    """
    first_day, last_day = daily.agg(F.min("date"), F.max("date")).first()
    by_city = Window.partitionBy("city").orderBy("date")
    runs = (
        daily.filter(F.col(flag_col))
        .withColumn("run_id", F.date_sub("date", F.row_number().over(by_city)))
        .groupBy("city", "region", "run_id")
        .agg(F.min("date").alias("start_date"), F.max("date").alias("end_date"),
             F.count("*").alias("length_days"),
             F.max("pm2_5_mean").alias("peak_pm2_5"), F.mean("pm2_5_mean").alias("mean_pm2_5"),
             F.max(F.col("is_provisional").cast("int")).cast("boolean").alias("includes_provisional"))
        .filter(F.col("length_days") >= min_days)
    )
    return (runs
            .withColumn("touches_data_boundary",
                        (F.col("start_date") == F.lit(first_day)) | (F.col("end_date") == F.lit(last_day)))
            .withColumn("episode_id", F.row_number().over(Window.partitionBy("city").orderBy("start_date")))
            .drop("run_id"))


def top_pollution_days(daily: DataFrame, n: int = TOP_DAYS_PER_CITY) -> DataFrame:
    """The n highest-PM2.5 days of each city, with that day's weather."""
    by_city = Window.partitionBy("city").orderBy(F.desc("pm2_5_mean"), F.asc("date"))
    return (daily.withColumn("rank", F.row_number().over(by_city))
            .filter(F.col("rank") <= n)
            .select("city", "region", "rank", "date", "season", "pm2_5_mean", "pm10_mean",
                    "india_aqi_pm", "india_aqi_pm_category", "dust_mean",
                    "temperature_2m_mean", "relative_humidity_2m_mean",
                    "wind_speed_10m_max", "precipitation_sum", "is_provisional"))


# ---------------------------------------------------------------- RDD demonstration

def rdd_city_means(daily: DataFrame, value_col: str = "pm2_5_mean",
                   tolerance: float = RDD_CHECK_TOLERANCE) -> DataFrame:
    """City means computed with the RDD API, checked against the DataFrame API.

    map: each row -> (city, (value, 1)).  reduceByKey: add the pairs per key,
    first inside each partition, then across partitions after a shuffle.
    The lambdas run in Python worker processes, unlike DataFrame operations,
    which run inside the JVM and are optimised by Catalyst.
    """
    rows = daily.select("city", value_col).filter(F.col(value_col).isNotNull())
    sums = (rows.rdd
            .map(lambda row: (row[0], (row[1], 1)))
            .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])))
    rdd_means = (sums.map(lambda kv: (kv[0], kv[1][0] / kv[1][1], kv[1][1]))
                 .toDF(["city", "rdd_mean", "rdd_n"]))
    df_means = rows.groupBy("city").agg(F.mean(value_col).alias("dataframe_mean"),
                                        F.count(value_col).alias("dataframe_n"))
    # Summation order differs between the two APIs, so allow a tiny
    # floating-point difference relative to the size of the mean.
    allowed = tolerance * F.greatest(F.lit(1.0), F.abs("dataframe_mean"))
    return (rdd_means.join(df_means, "city", "full_outer")
            .withColumn("abs_diff", F.abs(F.col("rdd_mean") - F.col("dataframe_mean")))
            .withColumn("within_tolerance", F.coalesce(
                (F.col("abs_diff") <= allowed) & (F.col("rdd_n") == F.col("dataframe_n")),
                F.lit(False))))


# ---------------------------------------------------------------- summary

def dataset_summary(spark: SparkSession, daily: DataFrame, air_quality_rows: int,
                    weather: DataFrame, years: list[int]) -> DataFrame:
    """One-row table of dataset facts for the dashboard's overview page."""
    d = daily.agg(
        F.countDistinct("city").alias("cities"), F.countDistinct("state").alias("states"),
        F.countDistinct("region").alias("regions"), F.count("*").alias("joined_rows"),
        F.min("date").alias("first_date"), F.max("date").alias("last_date"),
        F.sum(F.col("is_provisional").cast("int")).alias("provisional_rows"),
        F.sum((~F.col("season_complete")).cast("int")).alias("rows_in_incomplete_seasons"),
    ).first().asDict()
    w = weather.agg(F.count("*").alias("weather_rows"), F.min("date").alias("weather_first_date"),
                    F.max("date").alias("weather_last_date")).first().asDict()
    row = {**d, **w, "air_quality_rows": air_quality_rows, "complete_years": years,
           "data_source": "Modelled (CAMS air quality, ERA5/IFS weather via Open-Meteo); "
                          "not monitoring-station measurements"}
    return spark.createDataFrame([row])
