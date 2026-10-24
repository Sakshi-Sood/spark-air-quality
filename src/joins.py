"""Join air quality, weather and city metadata into one city x day table.

Grain guarantee: one output row per (city, date) that exists in BOTH daily
datasets. Every join checks key uniqueness first and row counts afterwards,
so a many-to-many join cannot silently multiply rows.

Spark notes:
  * air_quality JOIN weather: Spark picks the strategy from size estimates.
    For large inputs it uses a sort-merge join (both sides shuffled by key,
    sorted, then merged). If one side is estimated below
    spark.sql.autoBroadcastJoinThreshold (10 MB by default), Spark broadcasts
    it instead. The chosen plan is saved to outputs/reports/plans/.
  * Joining cities: the 50-row table is explicitly broadcast, meaning a full
    copy is sent to every executor, so the large side is never shuffled.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import CITY_KEY, DAILY_KEY

# Columns derived from `date` exist in both processed daily tables; keep the
# air-quality copy so the joined table has a single version of each.
DATE_DERIVED = ["year", "month", "season"]


def assert_unique_key(df: DataFrame, key: list[str], name: str) -> int:
    """Raise if any key value appears more than once; return the row count."""
    rows = df.count()
    keys = df.select(*key).distinct().count()
    if rows != keys:
        raise ValueError(f"{name}: key {key} is not unique ({rows} rows, {keys} distinct keys)")
    return rows


def join_air_quality_weather(aq: DataFrame, weather: DataFrame) -> tuple[DataFrame, dict]:
    """Inner join on (city, date); returns the joined DataFrame and row counts."""
    aq_rows = assert_unique_key(aq, DAILY_KEY, "air_quality")
    weather_rows = assert_unique_key(weather, DAILY_KEY, "weather")

    weather_side = (weather.drop(*DATE_DERIVED)
                    .withColumnRenamed("is_provisional", "weather_is_provisional"))
    joined = (
        aq.join(weather_side, on=DAILY_KEY, how="inner")
        # A day is provisional if either source still treats it as provisional.
        .withColumn("is_provisional", F.col("is_provisional") | F.col("weather_is_provisional"))
        .drop("weather_is_provisional")
    )

    # left_anti keeps rows from the left side that have NO match on the right.
    stats = {
        "air_quality_rows": aq_rows,
        "weather_rows": weather_rows,
        "air_quality_without_weather": aq.join(weather, DAILY_KEY, "left_anti").count(),
        "weather_without_air_quality": weather.join(aq, DAILY_KEY, "left_anti").count(),
        "joined_rows": joined.count(),
    }
    expected = aq_rows - stats["air_quality_without_weather"]
    if stats["joined_rows"] != expected:
        raise ValueError(f"Join changed the grain: {stats['joined_rows']} rows, expected {expected}")
    return joined, stats


def attach_city_metadata(df: DataFrame, cities: DataFrame) -> DataFrame:
    """Broadcast-join state, region, coordinates and elevation onto each row."""
    assert_unique_key(cities, CITY_KEY, "cities")
    before = df.count()
    enriched = df.join(F.broadcast(cities), on=CITY_KEY, how="left")
    after = enriched.count()
    missing = enriched.filter(F.col("region").isNull()).count()
    if after != before or missing:
        raise ValueError(f"City join failed: {before} -> {after} rows, {missing} rows without metadata")
    return enriched


def attach_season_completeness(df: DataFrame, coverage: DataFrame) -> DataFrame:
    """Add the season_complete flag from the (city, year, season) coverage table."""
    key = ["city", "year", "season"]
    assert_unique_key(coverage, key, "season_coverage")
    before = df.count()
    flagged = df.join(F.broadcast(coverage.select(*key, "season_complete")), on=key, how="left")
    if flagged.count() != before:
        raise ValueError("Season completeness join changed the row count")
    return flagged
