"""Display names, units, caveat texts and colours used across the dashboard.

Units and column lists come from src.config, so the dashboard and the Spark
pipeline always describe the same columns the same way.
"""
from src.config import (INDEX_UNITS, NAAQS_24H, POLLUTANT_UNITS, WEATHER_UNITS,
                        WHO_2021_24H)

POLLUTANT_NAMES = {
    "pm2_5_mean": "PM2.5 (24-h mean)",
    "india_aqi_pm": "Indian AQI, PM only",
    "pm10_mean": "PM10 (24-h mean)",
    "india_aqi": "Indian AQI, all pollutants ⚠ ozone-inflated",
    "no2_mean": "NO₂ (24-h mean)",
    "so2_mean": "SO₂ (24-h mean)",
    "o3_8h_max": "O₃ (max 8-h mean)",
    "co_8h_max": "CO (max 8-h mean)",
    "dust_mean": "Dust (24-h mean)",
    "aod_mean": "Aerosol optical depth",
}
# Short names for tight spaces such as KPI card labels.
SHORT_NAMES = {
    "pm2_5_mean": "PM2.5", "india_aqi_pm": "AQI (PM only)", "pm10_mean": "PM10",
    "india_aqi": "AQI (all)", "no2_mean": "NO₂", "so2_mean": "SO₂", "o3_8h_max": "O₃",
    "co_8h_max": "CO", "dust_mean": "Dust", "aod_mean": "AOD",
}
WEATHER_NAMES = {
    "temperature_2m_mean": "Mean temperature",
    "relative_humidity_2m_mean": "Relative humidity",
    "wind_speed_10m_max": "Max wind speed",
    "precipitation_sum": "Daily rainfall",
}
UNITS = {**POLLUTANT_UNITS, **INDEX_UNITS, **WEATHER_UNITS}


def label(column: str) -> str:
    """Display name with unit, e.g. 'PM2.5 (24-h mean), µg/m³'."""
    name = POLLUTANT_NAMES.get(column) or WEATHER_NAMES.get(column) or column
    unit = UNITS.get(column)
    return f"{name}, {unit}" if unit else name


# ---------------------------------------------------------------- caveats (shown in the UI)
MODEL_DATA_NOTE = (
    "All values are **modelled** (CAMS global air-quality model and ERA5/IFS weather "
    "reanalysis, via Open-Meteo), not readings from ground monitoring stations. "
    "They are smooth over 9–45 km grid cells and can miss very local peaks."
)
OZONE_NOTE = (
    "**Indian AQI (all pollutants)** is dominated by ozone on about 90 % of days because "
    "the CAMS model overestimates surface ozone over India. It runs higher than station "
    "AQI. Use PM2.5 or the **PM-only Indian AQI** for comparisons."
)
EXCEEDANCE_NOTE = (
    "\"Days above limit\" counts days whose *modelled* 24-h mean is above the national "
    f"NAAQS value (PM2.5 {NAAQS_24H['pm2_5_mean']:.0f}, PM10 {NAAQS_24H['pm10_mean']:.0f} µg/m³) "
    f"or the WHO 2021 guideline (PM2.5 {WHO_2021_24H['pm2_5_mean']:.0f}, "
    f"PM10 {WHO_2021_24H['pm10_mean']:.0f} µg/m³). It is **not** a regulatory compliance result."
)
CORRELATION_NOTE = (
    "Correlations show how pollution and weather **move together** in observational data. "
    "They do not show that weather causes the change. Season is a strong confounder: "
    "compare the all-year value with the within-season values."
)
PROVISIONAL_NOTE = "The last 10 days of data are provisional and may be revised by the source."
TREND_NOTE = (
    "Only three complete years (2023–2025) exist. Year-to-year differences are descriptive "
    "and are too few to establish a long-term trend."
)
SHORT_MODEL_NOTE = "Modelled data (CAMS air quality, ERA5 weather), not station measurements."
SHORT_OZONE_NOTE = "This index is ozone-inflated by the model; prefer PM2.5 or the PM-only AQI."

# ---------------------------------------------------------------- colours
# Validated with the dataviz palette checker on a white surface: the categorical
# set passes lightness, chroma, colour-blind separation and 3:1 contrast; the
# year ramp passes the ordinal checks (one hue, monotone lightness).
ACCENT = "#14614f"                                       # deep teal: theme and emphasis
SERIES = ["#008a70", "#d97a26", "#3b6fb6", "#bf4d80", "#7f9a2c", "#6b5bd2"]
PRIMARY = "#2f8f78"                                      # single-series marks
MUTED = "#a3aca8"
YEAR_RAMP = ["#76b5a4", "#2f8f78", "#14614f"]           # ordinal, light -> dark
SEQUENTIAL = ["#e3f1ec", "#a9d5c8", "#5fae9b", "#2f8f78", "#14614f", "#0b3f33"]
# Negative = teal, positive = terracotta, neutral grey at zero.
DIVERGING = [[0.0, "#14614f"], [0.25, "#76b5a4"], [0.5, "#f0efec"],
             [0.75, "#e3a07f"], [1.0, "#a8432a"]]
# CPCB's AQI colour convention (green -> yellow -> red -> maroon), kept because
# viewers know it, but muted so it does not overpower the rest of the page.
AQI_COLOURS = {"Good": "#3f9a5f", "Satisfactory": "#a2c96f", "Moderate": "#e2c254",
               "Poor": "#e7954f", "Very Poor": "#cf5641", "Severe": "#8c2b2b"}
MAX_COMPARED_CITIES = len(SERIES)

