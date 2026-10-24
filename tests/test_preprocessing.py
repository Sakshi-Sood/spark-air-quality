"""Season mapping, binning, provisional flag, exceedance flags, season coverage."""
from datetime import date

from pyspark.sql import functions as F

from src.config import RAIN_EDGES_MM, RAIN_LABELS, WIND_EDGES_KMH, WIND_LABELS
from src.preprocessing import (add_calendar_columns, add_exceedance_flags,
                               add_provisional_flag, add_weather_bins,
                               assign_bins, season_coverage)


def test_every_month_maps_to_imd_season(spark):
    df = spark.createDataFrame([(date(2024, m, 15),) for m in range(1, 13)], ["date"])
    seasons = [r.season for r in add_calendar_columns(df).orderBy("month").collect()]
    assert seasons == (["Winter"] * 2 + ["Pre-monsoon"] * 3
                       + ["Monsoon"] * 4 + ["Post-monsoon"] * 3)


def test_wind_bin_boundaries_are_lower_inclusive(spark):
    values = [0.0, 9.99, 10.0, 14.9, 15.0, 19.99, 20.0, 76.1, None]
    df = spark.createDataFrame([(v,) for v in values], "v double")
    got = [r.b for r in df.select(assign_bins(F.col("v"), WIND_EDGES_KMH, WIND_LABELS).alias("b")).collect()]
    assert got == [WIND_LABELS[0], WIND_LABELS[0], WIND_LABELS[1], WIND_LABELS[1],
                   WIND_LABELS[2], WIND_LABELS[2], WIND_LABELS[3], WIND_LABELS[3], None]


def test_rain_bins_follow_imd_categories(spark):
    values = [0.0, 0.1, 2.4, 2.5, 15.5, 15.6, 64.4, 64.5, 293.1]
    df = spark.createDataFrame([(v,) for v in values], "v double")
    got = [r.b for r in df.select(assign_bins(F.col("v"), RAIN_EDGES_MM, RAIN_LABELS).alias("b")).collect()]
    assert got == [RAIN_LABELS[i] for i in (0, 1, 1, 2, 2, 3, 3, 4, 4)]


def test_temperature_and_humidity_bands(spark):
    rows = [(-12.6, 0.0, 6), (24.9, 3.0, 59), (25.0, 3.0, 80), (40.3, 3.0, 100)]
    df = spark.createDataFrame(rows, "temperature_2m_mean double, precipitation_sum double, "
                                     "relative_humidity_2m_mean int")
    df = add_weather_bins(df.withColumn("wind_speed_10m_max", F.lit(5.0)))
    got = [(r.temperature_band_c, r.humidity_band_pct) for r in df.collect()]
    # 100 % humidity is folded into the 80-100 band.
    assert got == [(-15, 0), (20, 40), (25, 80), (40, 80)]


def test_provisional_flag_marks_exactly_last_ten_days(spark):
    df = spark.createDataFrame([(date(2026, 10, d),) for d in range(1, 24)], ["date"])
    flagged = add_provisional_flag(df, date(2026, 10, 23))
    days = sorted(r.date.day for r in flagged.filter("is_provisional").collect())
    assert days == list(range(14, 24))


def test_exceedance_flags_are_strictly_above_limit(spark):
    df = spark.createDataFrame([(60.0, 100.0), (60.01, 100.01), (14.0, 44.0)],
                               ["pm2_5_mean", "pm10_mean"])
    rows = add_exceedance_flags(df).collect()
    assert [(r.pm2_5_above_naaqs, r.pm10_above_naaqs) for r in rows] == [
        (False, False), (True, True), (False, False)]
    assert [(r.pm2_5_above_who, r.pm10_above_who) for r in rows] == [
        (True, True), (True, True), (False, False)]


def _days(spark, start: str, end: str):
    return spark.sql(f"SELECT 'X' AS city, explode(sequence(DATE'{start}', DATE'{end}')) AS date")


def test_season_completeness_uses_expected_calendar_days(spark):
    full_winter_2024 = _days(spark, "2024-01-01", "2024-02-29")      # leap year: 60 days
    partial_winter_2025 = _days(spark, "2025-01-01", "2025-02-27")   # 58 of 59 days
    df = add_calendar_columns(full_winter_2024.unionByName(partial_winter_2025))
    result = {r.year: (r.observed_days, r.expected_days, r.season_complete)
              for r in season_coverage(df).collect()}
    assert result == {2024: (60, 60, True), 2025: (58, 59, False)}
