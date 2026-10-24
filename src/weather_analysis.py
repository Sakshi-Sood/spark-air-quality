"""Weather-pollution associations: correlations and weather-bin comparisons.

These are observational associations in modelled data. They do not show that
a weather variable causes a pollution change. Season is a strong confounder
(the monsoon brings rain, humidity, wind and cleaner air at the same time),
so every result is also computed within each complete season.

Spark notes:
  * explode() turns one row into several, so a single groupBy can compute
    results for all scopes (India, each region, each city) and all season
    groups at once instead of looping over them in Python.
  * Spearman's rho is Pearson's r on ranks. Ranks come from window
    functions partitioned by group; tied values get their average rank.
"""
from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

from src.analytics import UNITS, explode_structs, literal_map
from src.config import (ALL_YEAR, BIN_POLLUTANTS, CORRELATION_WEATHER,
                        HUMIDITY_BAND_PCT, MIN_BIN_DAYS, MIN_CORRELATION_DAYS,
                        PERCENTILE_ACCURACY, RAIN_LABELS, TEMPERATURE_BAND_C,
                        WEATHER_ANALYSIS_POLLUTANTS, WIND_LABELS)

GROUP_COLS = ["scope_type", "scope", "season_group"]


def explode_scopes(daily: DataFrame) -> DataFrame:
    """Repeat each day once per scope and season group it belongs to.

    Scopes: all cities, the city's region, the city itself.
    Season groups: "All year", plus the day's season if that season is
    complete (incomplete seasons are excluded from within-season results).
    """
    scopes = F.array(
        F.struct(F.lit("all").alias("scope_type"), F.lit("All cities").alias("scope")),
        F.struct(F.lit("region").alias("scope_type"), F.col("region").alias("scope")),
        F.struct(F.lit("city").alias("scope_type"), F.col("city").alias("scope")),
    )
    seasons = F.array(F.lit(ALL_YEAR), F.when(F.col("season_complete"), F.col("season")))
    return (daily
            .withColumn("scope_struct", F.explode(scopes))
            .withColumn("season_group", F.explode(seasons))
            .filter(F.col("season_group").isNotNull())
            .select("*", "scope_struct.*").drop("scope_struct"))


# ---------------------------------------------------------------- correlations

def add_average_ranks(df: DataFrame, cols: list[str], partition_cols: list[str]) -> DataFrame:
    """Add <col>_rank: rank within the partition, ties get their average rank.

    rank() gives tied rows the lowest rank of the tie; adding (ties - 1) / 2
    turns that into the average rank. Ties are counted with a range frame of
    width 0 on the same ordering, so no extra shuffle is needed.
    """
    for c in cols:
        ordered = Window.partitionBy(*partition_cols).orderBy(c)
        ties = ordered.rangeBetween(0, 0)
        df = df.withColumn(f"{c}_rank", F.rank().over(ordered) + (F.count("*").over(ties) - 1) / 2)
    return df


def add_correlation_status(df: DataFrame, min_days: int) -> DataFrame:
    """Null out correlations that are undefined or based on too few days.

    Spark's corr() returns null for fewer than 2 rows and NaN when a variable
    is constant (zero variance); both become null with an explanatory status.
    """
    undefined = (F.col("pearson_r").isNull() | F.isnan("pearson_r")
                 | F.col("spearman_rho").isNull() | F.isnan("spearman_rho"))
    status = (F.when(F.col("n_days") < min_days, "insufficient_data")
              .when(undefined, "undefined_constant_values")
              .otherwise("ok"))
    return (df.withColumn("status", status)
              .withColumn("pearson_r", F.when(F.col("status") == "ok", F.col("pearson_r")))
              .withColumn("spearman_rho", F.when(F.col("status") == "ok", F.col("spearman_rho"))))


def weather_correlations(scoped: DataFrame, pollutants: list[str] = WEATHER_ANALYSIS_POLLUTANTS,
                         weather_vars: list[str] = CORRELATION_WEATHER,
                         min_days: int = MIN_CORRELATION_DAYS) -> DataFrame:
    """Pearson and Spearman correlation for every pollutant x weather pair, per group."""
    variables = pollutants + weather_vars
    ranked = add_average_ranks(scoped.dropna(subset=variables), variables, GROUP_COLS)
    pairs = [(p, w) for p in pollutants for w in weather_vars]
    aggs = [F.count("*").alias("n_days")]
    for p, w in pairs:
        aggs.append(F.corr(p, w).alias(f"{p}__{w}__pearson"))
        aggs.append(F.corr(f"{p}_rank", f"{w}_rank").alias(f"{p}__{w}__spearman"))
    wide = ranked.groupBy(*GROUP_COLS).agg(*aggs)
    structs = [F.struct(F.lit(p).alias("pollutant"), F.lit(w).alias("weather_variable"),
                        F.col(f"{p}__{w}__pearson").alias("pearson_r"),
                        F.col(f"{p}__{w}__spearman").alias("spearman_rho"))
               for p, w in pairs]
    return add_correlation_status(explode_structs(wide, GROUP_COLS + ["n_days"], structs), min_days)


# ---------------------------------------------------------------- weather bins

def _band_label(band: Column, width: int, unit: str) -> Column:
    # format_string follows Java's formatter, where a literal % is written %%.
    return F.format_string(f"%d–%d {unit.replace('%', '%%')}", band, band + width)


def weather_bins() -> Column:
    """One struct per binned weather variable: (variable, label, sort order)."""
    def entry(variable: str, label: Column, order: Column) -> Column:
        return F.struct(F.lit(variable).alias("weather_variable"),
                        label.alias("bin_label"), order.cast("int").alias("bin_order"))

    wind_order = literal_map({label: i for i, label in enumerate(WIND_LABELS)})
    rain_order = literal_map({label: i for i, label in enumerate(RAIN_LABELS)})
    return F.array(
        entry("wind_speed_10m_max", F.col("wind_bin"), wind_order[F.col("wind_bin")]),
        entry("precipitation_sum", F.col("rain_bin"), rain_order[F.col("rain_bin")]),
        entry("temperature_2m_mean",
              _band_label(F.col("temperature_band_c"), TEMPERATURE_BAND_C, "°C"),
              F.col("temperature_band_c")),
        entry("relative_humidity_2m_mean",
              _band_label(F.col("humidity_band_pct"), HUMIDITY_BAND_PCT, "%"),
              F.col("humidity_band_pct")),
    )


def weather_bin_stats(scoped: DataFrame, pollutants: list[str] = BIN_POLLUTANTS,
                      min_days: int = MIN_BIN_DAYS) -> DataFrame:
    """Mean and median pollution per weather bin, scope and season group."""
    keys = GROUP_COLS + ["weather_variable", "bin_label", "bin_order"]
    binned = (scoped.withColumn("bin", F.explode(weather_bins()))
              .select(*GROUP_COLS, *pollutants, "bin.*"))
    aggs = [F.count("*").alias("n_days")]
    for p in pollutants:
        aggs += [F.mean(p).alias(f"{p}__mean"),
                 F.percentile_approx(p, 0.5, PERCENTILE_ACCURACY).alias(f"{p}__median")]
    wide = binned.groupBy(*keys).agg(*aggs)
    structs = [F.struct(F.lit(p).alias("pollutant"), F.col(f"{p}__mean").alias("mean"),
                        F.col(f"{p}__median").alias("median")) for p in pollutants]
    return (explode_structs(wide, keys + ["n_days"], structs)
            .withColumn("unit", literal_map(UNITS)[F.col("pollutant")])
            .withColumn("low_sample", F.col("n_days") < min_days))
