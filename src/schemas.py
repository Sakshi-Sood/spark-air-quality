"""Explicit Spark schemas for the three raw CSVs, and the raw CSV reader.

Why explicit schemas instead of inferSchema=True?
  * inferSchema makes Spark scan the whole file once just to guess types.
  * A guess can silently change (e.g. an integer column becoming double).
  * With a fixed schema, a malformed value is caught and counted instead.

Column names and types were verified against the actual files in Phase 1.
"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (DateType, DoubleType, IntegerType, StringType,
                               StructField, StructType)

from src.config import DATE_FORMAT, RAW_FILES

CORRUPT_COLUMN = "_corrupt_record"


def _fields(spec: list[tuple[str, object]]) -> StructType:
    return StructType([StructField(name, dtype, True) for name, dtype in spec])


AIR_QUALITY_SCHEMA = _fields([
    ("date", DateType()), ("city", StringType()),
    ("pm2_5_mean", DoubleType()), ("pm2_5_max", DoubleType()),
    ("pm10_mean", DoubleType()), ("pm10_max", DoubleType()),
    ("no2_mean", DoubleType()), ("so2_mean", DoubleType()),
    ("o3_8h_max", DoubleType()), ("co_8h_max", DoubleType()),
    ("dust_mean", DoubleType()), ("aod_mean", DoubleType()),
    ("us_aqi_max", DoubleType()), ("hours_available", IntegerType()),
    ("india_aqi", IntegerType()), ("india_aqi_category", StringType()),
    ("dominant_pollutant", StringType()),
    ("india_aqi_pm", IntegerType()), ("india_aqi_pm_category", StringType()),
])

WEATHER_SCHEMA = _fields([
    ("date", DateType()), ("city", StringType()),
    ("temperature_2m_max", DoubleType()), ("temperature_2m_min", DoubleType()),
    ("temperature_2m_mean", DoubleType()),
    ("apparent_temperature_max", DoubleType()),
    ("precipitation_sum", DoubleType()), ("precipitation_hours", DoubleType()),
    ("relative_humidity_2m_mean", IntegerType()),
    ("wind_speed_10m_max", DoubleType()),
    ("wind_direction_10m_dominant", IntegerType()),
    ("shortwave_radiation_sum", DoubleType()),
    ("et0_fao_evapotranspiration", DoubleType()),
    ("weather_code", IntegerType()),
])

CITIES_SCHEMA = _fields([
    ("city", StringType()), ("state", StringType()), ("region", StringType()),
    ("latitude", DoubleType()), ("longitude", DoubleType()),
    ("elevation_m", DoubleType()),
])

SCHEMAS = {
    "air_quality": AIR_QUALITY_SCHEMA,
    "weather": WEATHER_SCHEMA,
    "cities": CITIES_SCHEMA,
}


def read_raw_csv(spark: SparkSession, name: str, path=None) -> DataFrame:
    """Read one raw CSV with its explicit schema.

    mode=PERMISSIVE keeps malformed rows instead of failing; their original
    text lands in CORRUPT_COLUMN so validation can count them.
    enforceSchema=False makes Spark compare the CSV header with the schema's
    column names and raise an error on a mismatch (the default silently
    applies the schema by position).
    """
    schema = StructType(SCHEMAS[name].fields + [StructField(CORRUPT_COLUMN, StringType())])
    path = path or RAW_FILES[name]
    return (
        spark.read
        .option("header", True)
        .option("enforceSchema", False)
        .option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", CORRUPT_COLUMN)
        .option("dateFormat", DATE_FORMAT)
        .schema(schema)
        .csv(str(path))
    )
