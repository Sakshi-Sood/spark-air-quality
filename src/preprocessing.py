"""Cleaning and derived columns, built only from Spark built-in functions.

Spark notes:
  * Everything here is a *narrow transformation* (withColumn, filter, select):
    each output partition depends on one input partition, so no shuffle.
    The exceptions are dropDuplicates and season_coverage's groupBy, which
    are *wide*: rows with the same key must meet in the same partition, so
    Spark shuffles data across partitions.
  * Transformations are lazy: these functions only build a plan. Work runs
    when an action (count, write) is called.
"""
from datetime import timedelta

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from src.config import (DAILY_KEY, HUMIDITY_BAND_PCT, NAAQS_24H,
                        PROVISIONAL_DAYS, RAIN_EDGES_MM, RAIN_LABELS,
                        SEASON_MONTHS, TEMPERATURE_BAND_C, WHO_2021_24H,
                        WIND_EDGES_KMH, WIND_LABELS)
from src.schemas import CORRUPT_COLUMN
from src.utils import get_logger

log = get_logger(__name__)

SEASON_BY_MONTH = {m: season for season, (first, last) in SEASON_MONTHS.items()
                   for m in range(first, last + 1)}


# ---------------------------------------------------------------- generic steps

def clean_strings(df: DataFrame) -> DataFrame:
    """Trim whitespace in every string column and drop the corrupt-record column."""
    df = df.drop(CORRUPT_COLUMN)
    for field in df.schema.fields:
        if field.dataType.simpleString() == "string":
            df = df.withColumn(field.name, F.trim(F.col(field.name)))
    return df


def drop_duplicate_keys(df: DataFrame, key: list[str], name: str) -> tuple[DataFrame, int]:
    """Keep one row per key and report how many rows were removed.

    Validation has already confirmed that no key duplicate has conflicting
    values, so any removed row is an exact copy.
    """
    before = df.count()
    deduped = df.dropDuplicates(key)
    removed = before - deduped.count()
    log.info("%s: %d duplicate key rows removed", name, removed)
    return deduped, removed


def add_calendar_columns(df: DataFrame) -> DataFrame:
    """Add year, month and IMD season from the date column."""
    season_map = F.create_map(*[F.lit(x) for pair in SEASON_BY_MONTH.items() for x in pair])
    return (df.withColumn("year", F.year("date"))
              .withColumn("month", F.month("date"))
              .withColumn("season", season_map[F.col("month")]))


def latest_date(df: DataFrame):
    """The dataset's own maximum date (one small aggregation)."""
    return df.agg(F.max("date")).first()[0]


def add_provisional_flag(df: DataFrame, max_date) -> DataFrame:
    """Flag the last PROVISIONAL_DAYS calendar days up to max_date inclusive.

    Example with 10 days and max_date 2026-10-23: 2026-10-14 .. 2026-10-23.
    """
    cutoff = F.date_sub(F.lit(max_date), PROVISIONAL_DAYS - 1)
    return df.withColumn("is_provisional", F.col("date") >= cutoff)


# ---------------------------------------------------------------- bins and flags

def assign_bins(value: Column, edges: list[float], labels: list[str]) -> Column:
    """Return the label of the bin containing value; lower edges are inclusive.

    With edges [10, 15, 20]: <10 -> labels[0], [10,15) -> labels[1],
    [15,20) -> labels[2], >=20 -> labels[3]. Nulls stay null.
    """
    if len(labels) != len(edges) + 1:
        raise ValueError("Need exactly one more label than edges")
    expr = F.when(value.isNull(), F.lit(None).cast("string"))
    for edge, label in zip(edges, labels):
        expr = expr.when(value < edge, label)
    return expr.otherwise(labels[-1])


def band_lower_edge(value: Column, width: int, cap: int | None = None) -> Column:
    """Lower edge of a fixed-width band, e.g. width 5: 24.9 -> 20, -12.6 -> -15."""
    edge = (F.floor(value / width) * width).cast("int")
    return F.least(edge, F.lit(cap)) if cap is not None else edge


def add_weather_bins(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("wind_bin", assign_bins(F.col("wind_speed_10m_max"), WIND_EDGES_KMH, WIND_LABELS))
          .withColumn("rain_bin", assign_bins(F.col("precipitation_sum"), RAIN_EDGES_MM, RAIN_LABELS))
          .withColumn("temperature_band_c",
                      band_lower_edge(F.col("temperature_2m_mean"), TEMPERATURE_BAND_C))
          # 100 % humidity would form its own band, so fold it into 80-100.
          .withColumn("humidity_band_pct",
                      band_lower_edge(F.col("relative_humidity_2m_mean"), HUMIDITY_BAND_PCT,
                                      cap=100 - HUMIDITY_BAND_PCT))
    )


def add_exceedance_flags(df: DataFrame) -> DataFrame:
    """Flag days whose modelled 24-h mean is strictly above a reference limit.

    These compare model output with a limit value; they are not official
    compliance results (those rely on station monitoring).
    """
    for column, limit in NAAQS_24H.items():
        df = df.withColumn(f"{column.removesuffix('_mean')}_above_naaqs", F.col(column) > limit)
    for column, limit in WHO_2021_24H.items():
        df = df.withColumn(f"{column.removesuffix('_mean')}_above_who", F.col(column) > limit)
    return df


# ---------------------------------------------------------------- season completeness

def season_coverage(df: DataFrame) -> DataFrame:
    """One row per (city, year, season): observed vs expected days.

    expected_days is the number of calendar days in that season of that
    year (e.g. Winter 2024 = 31 + 29 = 60). A season is complete only when
    every expected day is present, so partial seasons at the start or end
    of the data (e.g. Monsoon 2022 starting 5 Aug) are excluded later.
    """
    first_month = F.create_map(*[F.lit(x) for s, (f, _) in SEASON_MONTHS.items() for x in (s, f)])
    last_month = F.create_map(*[F.lit(x) for s, (_, l) in SEASON_MONTHS.items() for x in (s, l)])
    season_start = F.make_date(F.col("year"), first_month[F.col("season")], F.lit(1))
    season_end = F.last_day(F.make_date(F.col("year"), last_month[F.col("season")], F.lit(1)))
    return (
        df.groupBy("city", "year", "season")
          .agg(F.countDistinct("date").alias("observed_days"))
          .withColumn("expected_days", F.datediff(season_end, season_start) + 1)
          .withColumn("season_complete", F.col("observed_days") == F.col("expected_days"))
    )


# ---------------------------------------------------------------- per-dataset pipelines

def preprocess_daily(df: DataFrame, name: str) -> tuple[DataFrame, dict]:
    """Shared steps for the two daily datasets."""
    df = clean_strings(df)
    df, removed = drop_duplicate_keys(df, DAILY_KEY, name)
    max_date = latest_date(df)
    df = add_provisional_flag(add_calendar_columns(df), max_date)
    stats = {"duplicate_key_rows_removed": removed, "max_date": max_date,
             "provisional_from": max_date - timedelta(days=PROVISIONAL_DAYS - 1)}
    return df, stats


def preprocess_air_quality(df: DataFrame) -> tuple[DataFrame, dict]:
    df, stats = preprocess_daily(df, "air_quality")
    return add_exceedance_flags(df), stats


def preprocess_weather(df: DataFrame) -> tuple[DataFrame, dict]:
    df, stats = preprocess_daily(df, "weather")
    return add_weather_bins(df), stats


def preprocess_cities(df: DataFrame) -> DataFrame:
    return clean_strings(df)
