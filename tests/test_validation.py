"""Schema and validation behaviour."""
import pytest

from src.config import RAW_FILES
from src.data_validation import (check_header, count_duplicates, count_nulls,
                                 count_violations, range_rules, read_csv_header)
from src.schemas import CORRUPT_COLUMN, SCHEMAS, read_raw_csv

CITIES_HEADER = "city,state,region,latitude,longitude,elevation_m"


@pytest.mark.parametrize("name", list(RAW_FILES))
def test_schemas_match_real_csv_headers(name):
    path = RAW_FILES[name]
    if not path.exists():
        pytest.skip(f"{path} not downloaded")
    assert read_csv_header(path) == SCHEMAS[name].fieldNames()


def test_header_check_reports_missing_and_extra_columns(tmp_path):
    path = tmp_path / "cities.csv"
    path.write_text("city,state,region,lat,longitude,elevation_m\nA,S,North,1,2,3\n")
    check = check_header("cities", path)
    assert not check["passed"]
    assert check["details"]["missing_columns"] == ["latitude"]
    assert check["details"]["unexpected_columns"] == ["lat"]


def test_malformed_value_goes_to_corrupt_column(spark, tmp_path):
    path = tmp_path / "cities.csv"
    path.write_text(f"{CITIES_HEADER}\nA,S,North,12.5,77.1,10\nB,S,South,not_a_number,77.2,10\n")
    df = read_raw_csv(spark, "cities", path).cache()
    bad = df.filter(df[CORRUPT_COLUMN].isNotNull()).collect()
    assert [r.city for r in bad] == ["B"]
    df.unpersist()


def test_duplicate_counting_separates_exact_and_conflicting(spark):
    rows = [("A", 1, 10.0), ("A", 1, 10.0),   # exact copy
            ("B", 1, 5.0), ("B", 1, 6.0),     # same key, different values
            ("C", 1, 1.0)]
    df = spark.createDataFrame(rows, ["city", "day", "value"])
    assert count_duplicates(df, ["city", "day"]) == {
        "rows": 5, "duplicate_rows": 1, "duplicate_key_rows": 2, "conflicting_key_rows": 1}


def test_null_counts_per_column(spark):
    df = spark.createDataFrame([("A", None), (None, 2.0), ("C", None)], "city string, v double")
    assert count_nulls(df, ["city", "v"]) == {"city": 1, "v": 2}


def test_weather_range_rules_flag_impossible_values(spark):
    row = dict(temperature_2m_max=30.0, temperature_2m_min=20.0, temperature_2m_mean=25.0,
               apparent_temperature_max=31.0, precipitation_sum=0.0, precipitation_hours=0.0,
               relative_humidity_2m_mean=50, wind_speed_10m_max=10.0,
               wind_direction_10m_dominant=90, shortwave_radiation_sum=18.0,
               et0_fao_evapotranspiration=4.0)
    bad = {**row, "relative_humidity_2m_mean": 120, "temperature_2m_min": 26.0}
    df = spark.createDataFrame([row, bad])
    result = count_violations(df, range_rules("weather"))
    assert result["humidity_out_of_range"] == 1
    assert result["temperature_order_violated"] == 1
    assert result["wind_direction_out_of_range"] == 0
