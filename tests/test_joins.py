"""Join row-count invariants: joins must never multiply or silently drop rows."""
from datetime import date

import pytest

from src.joins import (assert_unique_key, attach_city_metadata,
                       join_air_quality_weather)

D1, D2, D3 = date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)


def _aq(spark, rows):
    return spark.createDataFrame(
        [(c, d, 2024, 1, "Winter", False, pm) for c, d, pm in rows],
        "city string, date date, year int, month int, season string, "
        "is_provisional boolean, pm2_5_mean double")


def _weather(spark, rows):
    return spark.createDataFrame(
        [(c, d, 2024, 1, "Winter", prov, t) for c, d, t, prov in rows],
        "city string, date date, year int, month int, season string, "
        "is_provisional boolean, temperature_2m_mean double")


def test_inner_join_keeps_one_row_per_matching_key(spark):
    aq = _aq(spark, [("A", D1, 10.0), ("A", D2, 20.0), ("B", D1, 30.0)])
    weather = _weather(spark, [("A", D1, 15.0, False), ("A", D2, 16.0, True),
                               ("B", D1, 17.0, False), ("B", D3, 18.0, False)])
    joined, stats = join_air_quality_weather(aq, weather)
    assert stats["joined_rows"] == 3
    assert stats["air_quality_without_weather"] == 0
    assert stats["weather_without_air_quality"] == 1
    # Date-derived columns appear once; provisional = OR of both sources.
    assert joined.columns.count("season") == 1
    flags = {(r.city, r.date): r.is_provisional for r in joined.collect()}
    assert flags[("A", D2)] is True and flags[("A", D1)] is False


def test_join_rejects_duplicate_keys(spark):
    aq = _aq(spark, [("A", D1, 10.0)])
    weather = _weather(spark, [("A", D1, 15.0, False), ("A", D1, 15.5, False)])
    with pytest.raises(ValueError, match="not unique"):
        join_air_quality_weather(aq, weather)


def test_city_metadata_join_preserves_row_count(spark):
    daily = spark.createDataFrame([("A", D1), ("A", D2), ("B", D1)], "city string, date date")
    cities = spark.createDataFrame([("A", "North"), ("B", "South")], ["city", "region"])
    assert attach_city_metadata(daily, cities).count() == 3


def test_city_metadata_join_rejects_unknown_city(spark):
    daily = spark.createDataFrame([("A", D1), ("Z", D1)], "city string, date date")
    cities = spark.createDataFrame([("A", "North")], ["city", "region"])
    with pytest.raises(ValueError, match="without metadata"):
        attach_city_metadata(daily, cities)


def test_assert_unique_key_rejects_duplicate_city_metadata(spark):
    cities = spark.createDataFrame([("A", "North"), ("A", "South")], ["city", "region"])
    with pytest.raises(ValueError):
        assert_unique_key(cities, ["city"], "cities")
