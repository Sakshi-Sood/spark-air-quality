"""Page 1 - Overview: KPIs, coverage, regional comparison, key findings."""
import streamlit as st

from dashboard.charts import ranking_bar
from dashboard.data_loader import MissingOutputError, load_table, stop_with_missing
from dashboard.filters import get_filters
from dashboard.labels import (EXCEEDANCE_NOTE, MODEL_DATA_NOTE, OZONE_NOTE,
                              PROVISIONAL_NOTE, UNITS, label)
from dashboard.logic import filter_places, key_findings

filters = get_filters()
try:
    summary = load_table("dataset_summary").iloc[0]
    city_stats = load_table("city_pollutant_stats")
    region_stats = load_table("region_pollutant_stats")
    seasonal = load_table("seasonal_city", ("year", "season", "pollutant", "mean"))
    bins = load_table("weather_bin_stats")
except MissingOutputError as error:
    stop_with_missing(error)

st.title("Air Quality and Weather Across 50 Indian Cities")
st.caption("Processed with Apache Spark (PySpark, local mode on one laptop). "
           "This page reads precomputed results only.")
st.info(MODEL_DATA_NOTE)

# ---------------------------------------------------------------- KPIs
pollutant, unit = filters.pollutant, UNITS[filters.pollutant]
selected = filter_places(city_stats[city_stats["pollutant"] == pollutant],
                         filters.regions, filters.cities)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Cities analysed", f"{summary['cities']}", f"{summary['regions']} regions, "
          f"{summary['states']} states", delta_color="off")
c2.metric("Air-quality observations", f"{summary['air_quality_rows']:,}", "city-days",
          delta_color="off")
c3.metric("Weather observations", f"{summary['weather_rows']:,}",
          f"from {summary['weather_first_date']}", delta_color="off")
c4.metric("Joined analytical rows", f"{summary['joined_rows']:,}", "air quality ⋈ weather",
          delta_color="off")

c5, c6, c7 = st.columns([1, 1, 2])
if selected.empty:
    st.warning("No cities match the current filters.")
else:
    top = selected.loc[selected["max"].idxmax()]
    scope = "selected cities" if (filters.regions or filters.cities) else "all cities"
    # Every city has the same number of days, so the mean of city means is the
    # mean over all their city-days.
    c5.metric(f"Mean {label(pollutant)}", f"{selected['mean'].mean():.1f}", scope,
              delta_color="off")
    c6.metric("Highest daily value", f"{top['max']:.1f}", f"{top['city']} (single day)",
              delta_color="off")
c7.metric("Date coverage (air quality)", f"{summary['first_date']} → {summary['last_date']}",
          f"complete years: {', '.join(map(str, summary['complete_years']))}", delta_color="off")
if pollutant == "india_aqi":
    st.warning(OZONE_NOTE)

# ---------------------------------------------------------------- regional comparison and findings
left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Regional comparison")
    regions = region_stats[region_stats["pollutant"] == pollutant].copy()
    regions["label"] = regions["region"] + " (" + regions["n_cities"].astype(str) + " cities)"
    st.plotly_chart(ranking_bar(regions, "label", f"Mean of daily values, {unit}", unit),
                    width="stretch")
    st.caption("Regions differ in city count; a region mean can hide large differences "
               "between its cities (e.g. clean hill cities alongside Delhi in the North).")
with right:
    st.subheader("Key findings (PM2.5)")
    for finding in key_findings(city_stats, region_stats, seasonal, bins):
        st.markdown(f"- {finding}")
    st.caption(EXCEEDANCE_NOTE)
    st.caption(PROVISIONAL_NOTE + f" ({summary['provisional_rows']} city-days.)")
