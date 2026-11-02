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
# The dashboard uses a dark theme (.streamlit/config.toml). SURFACE is the card
# background that charts sit on; every chart colour below was validated with
# the dataviz palette checker against it: the categorical set passes lightness,
# chroma, colour-blind separation and 3:1 contrast; the year ramp passes the
# ordinal checks (one hue, monotone lightness).
PAGE = "#0d1311"
SURFACE = "#151d1b"
ACCENT = "#22a986"                                       # teal: theme and emphasis
SERIES = ["#22a986", "#cc7a2a", "#5b8fd9", "#d0679a", "#86a032", "#8f7ee6"]
PRIMARY = "#22a986"                                      # single-series marks
MUTED = "#5f6b67"
YEAR_RAMP = ["#2f7d6a", "#3fa98c", "#7fd3b9"]           # ordinal, oldest dim -> newest bright
SEQUENTIAL = ["#173a32", "#1f5a4c", "#2a7a66", "#3fa98c", "#7fd3b9", "#c8f0e2"]   # low dim -> high bright
# Negative = teal, positive = terracotta, neutral dark grey at zero.
DIVERGING = [[0.0, "#2fb592"], [0.25, "#1f6b59"], [0.5, "#2c3431"],
             [0.75, "#9a4f33"], [1.0, "#e0855f"]]
# CPCB's AQI colour convention (green -> yellow -> red -> maroon), kept because
# viewers know it; Severe is lifted so it stays visible on the dark surface.
AQI_COLOURS = {"Good": "#3f9a5f", "Satisfactory": "#7fb956", "Moderate": "#e6c14a",
               "Poor": "#e7954f", "Very Poor": "#cf5641", "Severe": "#a8406e"}
MAX_COMPARED_CITIES = len(SERIES)

