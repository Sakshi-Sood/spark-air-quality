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

# ---------------------------------------------------------------- colours (validated palette)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
PRIMARY = SERIES[0]
MUTED = "#898781"
YEAR_RAMP = ["#86b6ef", "#3987e5", "#1c5cab"]           # ordinal, light -> dark
SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
DIVERGING = [[0.0, "#1c5cab"], [0.25, "#86b6ef"], [0.5, "#f0efec"],
             [0.75, "#f0a09f"], [1.0, "#b8302f"]]
# CPCB's published AQI colour convention, kept because viewers know it.
AQI_COLOURS = {"Good": "#00b050", "Satisfactory": "#92d050", "Moderate": "#e6c200",
               "Poor": "#ff9900", "Very Poor": "#e0281e", "Severe": "#a0001e"}
MAX_COMPARED_CITIES = len(SERIES)

