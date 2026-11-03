"""Central project configuration: paths, dataset facts, and analysis constants.

Every value here was verified against the actual CSV files or the dataset card
(Hugging Face revision DATASET_REVISION) during Phase 1 inspection.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DIR = OUTPUT_DIR / "processed"
ANALYTICS_DIR = OUTPUT_DIR / "analytics"
REPORTS_DIR = OUTPUT_DIR / "reports"
PLANS_DIR = REPORTS_DIR / "plans"
SPARK_EVENTS_DIR = OUTPUT_DIR / "spark-events"
ML_DIR = OUTPUT_DIR / "ml"
ML_MODELS_DIR = ML_DIR / "models"
ML_METRICS_DIR = ML_DIR / "metrics"
ML_PREDICTIONS_DIR = ML_DIR / "predictions"
ML_REPORTS_DIR = ML_DIR / "reports"

RAW_FILES = {
    "air_quality": DATA_DIR / "air_quality.csv",
    "weather": DATA_DIR / "weather.csv",
    "cities": DATA_DIR / "cities.csv",
}

# ---------------------------------------------------------------- source
DATASET_REPO = "AkshayKarthick/india-weather-air-quality"
# Revision verified in Phase 1: the local CSVs are value-identical to it.
DATASET_REVISION = "339a4f37b76b9d2bae2f4b34cb189244019b4ed1"

# ---------------------------------------------------------------- keys
CITY_KEY = ["city"]
DAILY_KEY = ["city", "date"]          # grain of air_quality and weather
DATE_FORMAT = "yyyy-MM-dd"            # dates are local IST calendar days

# ---------------------------------------------------------------- Spark
SHUFFLE_PARTITIONS = 8                # default 200 is far too many for ~18 MB
DRIVER_MEMORY = "3g"                  # machine has 7.7 GB RAM
# Each processed table is small (a few MB), so one Parquet file per table
# avoids the "many tiny files" problem. See utils.write_parquet.
OUTPUT_FILES_PER_TABLE = 1

# ---------------------------------------------------------------- columns and units (dataset card)
POLLUTANT_UNITS = {
    "pm2_5_mean": "µg/m³", "pm2_5_max": "µg/m³",
    "pm10_mean": "µg/m³", "pm10_max": "µg/m³",
    "no2_mean": "µg/m³", "so2_mean": "µg/m³",
    "o3_8h_max": "µg/m³", "co_8h_max": "µg/m³",
    "dust_mean": "µg/m³", "aod_mean": "dimensionless",
}
WEATHER_UNITS = {
    "temperature_2m_max": "°C", "temperature_2m_min": "°C",
    "temperature_2m_mean": "°C", "apparent_temperature_max": "°C",
    "precipitation_sum": "mm", "precipitation_hours": "h",
    "relative_humidity_2m_mean": "%", "wind_speed_10m_max": "km/h",
    "wind_direction_10m_dominant": "° from north",
    "shortwave_radiation_sum": "MJ/m²", "et0_fao_evapotranspiration": "mm",
    "weather_code": "WMO code",
}

# ---------------------------------------------------------------- data quality
# The source marks the most recent ~10 days as provisional (may be revised).
# Rule: date >= max(date) - (PROVISIONAL_DAYS - 1), i.e. the last 10 calendar
# days, where max(date) is taken from each dataset itself.
PROVISIONAL_DAYS = 10
MIN_HOURS_AVAILABLE = 18              # source drops days with fewer hours

# CPCB National AQI bands (upper bound inclusive) -> category name.
AQI_BANDS = [
    (50, "Good"), (100, "Satisfactory"), (200, "Moderate"),
    (300, "Poor"), (400, "Very Poor"), (500, "Severe"),
]
DOMINANT_POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "O3", "CO"]

# ---------------------------------------------------------------- thresholds
# India NAAQS (CPCB, Nov 2009), 24-hour limits. We flag days whose *modelled*
# 24-h mean is strictly above the limit. This is NOT a compliance assessment:
# official compliance uses station monitoring and annual statistics.
NAAQS_24H = {"pm2_5_mean": 60.0, "pm10_mean": 100.0}
# WHO 2021 Air Quality Guidelines, 24-hour levels. Reference only.
WHO_2021_24H = {"pm2_5_mean": 15.0, "pm10_mean": 45.0}

# ---------------------------------------------------------------- seasons (IMD)
# India Meteorological Department seasons; none crosses a calendar year, so a
# season's year is simply the calendar year.
SEASON_MONTHS = {
    "Winter": (1, 2),
    "Pre-monsoon": (3, 5),
    "Monsoon": (6, 9),
    "Post-monsoon": (10, 12),
}
SEASON_ORDER = list(SEASON_MONTHS)

# ---------------------------------------------------------------- weather bins
# (lower edges of every bin after the first, labels). Lower edge inclusive.
WIND_EDGES_KMH = [10.0, 15.0, 20.0]
WIND_LABELS = ["<10 km/h", "10–15 km/h", "15–20 km/h", "≥20 km/h"]

# IMD daily rainfall categories (mm), merged above "heavy".
RAIN_EDGES_MM = [0.1, 2.5, 15.6, 64.5]
RAIN_LABELS = [
    "Dry (<0.1 mm)", "Very light (0.1–2.4 mm)", "Light (2.5–15.5 mm)",
    "Moderate (15.6–64.4 mm)", "Heavy (≥64.5 mm)",
]

TEMPERATURE_BAND_C = 5        # band = lower edge, e.g. 20 means 20–25 °C
HUMIDITY_BAND_PCT = 20        # band = lower edge, 100 % is folded into 80

# ---------------------------------------------------------------- analytics (Phase 4)
# Primary indicators: pm2_5_mean and india_aqi_pm. india_aqi is included only
# with its ozone caveat (CAMS overestimates O3 over India, so O3 dominates it).
INDEX_UNITS = {"india_aqi_pm": "index (PM only)", "india_aqi": "index (all pollutants)"}
SUMMARY_POLLUTANTS = [
    "pm2_5_mean", "india_aqi_pm", "pm10_mean", "india_aqi",
    "no2_mean", "so2_mean", "o3_8h_max", "co_8h_max", "dust_mean", "aod_mean",
]
WEATHER_ANALYSIS_POLLUTANTS = ["pm2_5_mean", "india_aqi_pm", "pm10_mean", "no2_mean", "o3_8h_max"]
BIN_POLLUTANTS = ["pm2_5_mean", "india_aqi_pm", "pm10_mean"]
CORRELATION_WEATHER = [
    "temperature_2m_mean", "relative_humidity_2m_mean",
    "wind_speed_10m_max", "precipitation_sum",
]
MONTHLY_WEATHER = CORRELATION_WEATHER

ALL_YEAR = "All year"                 # label for "no season filter"
MIN_CORRELATION_DAYS = 30             # fewer paired days -> correlation not reported
MIN_BIN_DAYS = 30                     # fewer days -> bin marked low_sample
MIN_CITIES_PER_GROUP = 3              # regions/states below this are flagged
EPISODE_MIN_DAYS = 3                  # consecutive days above the PM2.5 NAAQS limit
TOP_DAYS_PER_CITY = 20
PERCENTILE_ACCURACY = 10000           # percentile_approx: rank error <= 1/10000 of n
RDD_CHECK_TOLERANCE = 1e-9            # relative tolerance, RDD vs DataFrame means
