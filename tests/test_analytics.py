"""Key analytical transformations and their edge cases."""
from datetime import date, timedelta

import pytest
from pyspark.sql import functions as F

from src.analytics import (aqi_category_distribution, annual_city,
                           city_pollutant_stats, complete_years,
                           group_pollutant_stats, pollution_episodes,
                           rdd_city_means, to_long)
from src.weather_analysis import (GROUP_COLS, add_average_ranks,
                                  weather_bin_stats, weather_correlations)

DAILY_SCHEMA = ("city string, state string, region string, latitude double, longitude double, "
                "date date, year int, season string, season_complete boolean, "
                "is_provisional boolean, pm2_5_mean double, no2_mean double")
CITY_META = {"A": ("S1", "North"), "B": ("S1", "West"), "C": ("S2", "West")}


def _daily(spark, rows):
    """rows: (city, date, pm2_5_mean[, no2_mean])."""
    data = []
    for city, day, pm, *rest in rows:
        state, region = CITY_META[city]
        data.append((city, state, region, 0.0, 0.0, day, day.year, "Winter", True, False,
                     pm, rest[0] if rest else 1.0))
    return spark.createDataFrame(data, DAILY_SCHEMA)


def _days(start: date, n: int):
    return [start + timedelta(days=i) for i in range(n)]


def test_city_stats_exceedance_and_ranking(spark):
    d = _days(date(2024, 1, 1), 3)
    daily = _daily(spark, [("A", d[0], 50.0), ("A", d[1], 70.0), ("A", d[2], 90.0),
                           ("B", d[0], 10.0), ("B", d[1], 12.0), ("B", d[2], 14.0)])
    stats = city_pollutant_stats(to_long(daily, ["pm2_5_mean", "no2_mean"]))
    a = stats.filter("city = 'A' AND pollutant = 'pm2_5_mean'").first()
    assert (a.n_days, a.mean, a.min, a.max, a.median) == (3, 70.0, 50.0, 90.0, 70.0)
    assert (a.days_above_naaqs, a.days_above_who, a.rank_highest_mean) == (2, 3, 1)
    assert a.unit == "µg/m³"
    no2 = stats.filter("city = 'A' AND pollutant = 'no2_mean'").first()
    assert no2.days_above_naaqs is None   # no limit defined -> not applicable


def test_state_stats_list_regions_and_flag_few_cities(spark):
    d = date(2024, 1, 1)
    daily = _daily(spark, [("A", d, 10.0), ("B", d, 30.0), ("C", d, 5.0)])
    states = group_pollutant_stats(spark, to_long(daily, ["pm2_5_mean"]), ["state"])
    s1 = states.filter("state = 'S1'").first()
    assert (s1.regions, s1.n_cities, s1.mean, s1.few_cities, s1.rank_highest_mean) == (
        "North, West", 2, 20.0, True, 1)


def test_complete_years_excludes_partial_years(spark):
    rows = [("A", d, 1.0) for d in _days(date(2024, 1, 1), 366 + 10)]   # 2024 leap + 10 days
    assert complete_years(_daily(spark, rows)) == [2024]


def test_annual_change_only_between_consecutive_years(spark):
    rows = [("A", date(y, 6, 1), pm) for y, pm in ((2021, 10.0), (2022, 15.0), (2024, 30.0))]
    annual = annual_city(to_long(_daily(spark, rows), ["pm2_5_mean"]), [2021, 2022, 2024])
    got = {r.year: (r.previous_year_mean, r.change, r.change_pct) for r in annual.collect()}
    assert got == {2021: (None, None, None), 2022: (10.0, 5.0, 50.0), 2024: (None, None, None)}


def test_episodes_need_consecutive_days(spark):
    d = _days(date(2024, 1, 1), 12)
    flags = [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0]     # runs of 3, 2 and 4 days
    rows = [("A", day, 80.0 if f else 20.0) for day, f in zip(d, flags)]
    daily = _daily(spark, rows).withColumn("pm2_5_above_naaqs", F.col("pm2_5_mean") > 60)
    episodes = pollution_episodes(daily, min_days=3).orderBy("start_date").collect()
    assert [(e.start_date, e.length_days) for e in episodes] == [(d[0], 3), (d[7], 4)]
    assert episodes[0].touches_data_boundary and not episodes[1].touches_data_boundary

    # A missing calendar day also breaks a run.
    gap = daily.filter(F.col("date") != d[8])
    assert [e.length_days for e in pollution_episodes(gap, min_days=3).collect()] == [3]


def test_average_ranks_handle_ties(spark):
    df = spark.createDataFrame([("g", v) for v in (10.0, 20.0, 20.0, 30.0)], ["g", "v"])
    ranks = sorted(r.v_rank for r in add_average_ranks(df, ["v"], ["g"]).collect())
    assert ranks == [1.0, 2.5, 2.5, 4.0]


def _scoped(spark, scope, xs, ys):
    return spark.createDataFrame([("city", scope, "All year", x, y) for x, y in zip(xs, ys)],
                                 GROUP_COLS + ["pm2_5_mean", "wind_speed_10m_max"])


def test_correlation_status_and_spearman(spark):
    xs = [float(i) for i in range(1, 41)]
    scoped = (_scoped(spark, "monotonic", xs, [x ** 3 for x in xs])
              .unionByName(_scoped(spark, "constant", xs, [5.0] * 40))
              .unionByName(_scoped(spark, "short", xs[:10], xs[:10])))
    result = {r.scope: r for r in weather_correlations(
        scoped, ["pm2_5_mean"], ["wind_speed_10m_max"], min_days=30).collect()}
    assert result["monotonic"].status == "ok"
    assert result["monotonic"].spearman_rho == pytest.approx(1.0)
    assert result["monotonic"].pearson_r < 0.95          # non-linear, so below 1
    assert result["constant"].status == "undefined_constant_values"
    assert result["constant"].pearson_r is None
    assert result["short"].status == "insufficient_data"
    assert result["short"].spearman_rho is None


def test_weather_bins_labels_and_low_sample(spark):
    row = ("all", "All cities", "All year", 40.0, 60.0, 80.0, "<10 km/h", "Dry (<0.1 mm)", 25, 80)
    scoped = spark.createDataFrame([row], GROUP_COLS + [
        "pm2_5_mean", "pm10_mean", "india_aqi_pm", "wind_bin", "rain_bin",
        "temperature_band_c", "humidity_band_pct"])
    bins = {r.weather_variable: r for r in weather_bin_stats(scoped, ["pm2_5_mean"], 30).collect()}
    assert bins["relative_humidity_2m_mean"].bin_label == "80–100 %"
    assert bins["temperature_2m_mean"].bin_label == "25–30 °C"
    assert bins["wind_speed_10m_max"].bin_order == 0
    assert all(r.low_sample and r.mean == 40.0 for r in bins.values())


def test_aqi_category_shares_sum_to_one(spark):
    rows = [("A", "North", "Good", "Moderate"), ("A", "North", "Good", "Moderate"),
            ("A", "North", "Poor", "Poor")]
    df = spark.createDataFrame(rows, ["city", "region", "india_aqi_pm_category", "india_aqi_category"])
    dist = aqi_category_distribution(df)
    totals = dist.groupBy("aqi_index").agg(F.sum("share").alias("s")).collect()
    assert all(r.s == pytest.approx(1.0) for r in totals)
    good = dist.filter("aqi_index = 'india_aqi_pm' AND category = 'Good'").first()
    assert (good.days, good.category_order) == (2, 1)


def test_rdd_means_match_dataframe_means(spark):
    d = _days(date(2024, 1, 1), 4)
    daily = _daily(spark, [("A", d[0], 0.1), ("A", d[1], 0.2), ("A", d[2], 0.3),
                           ("B", d[3], 7.0)])
    rows = {r.city: r for r in rdd_city_means(daily).collect()}
    assert all(r.within_tolerance for r in rows.values())
    assert rows["A"].rdd_mean == pytest.approx(0.2) and rows["A"].rdd_n == 3
