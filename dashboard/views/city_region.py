"""Page 2 - City and regional analysis: rankings, statistics, map, regions/states, AQI mix."""
import streamlit as st

from dashboard.charts import aqi_stack, city_map, range_box, ranking_bar
from dashboard.data_loader import MissingOutputError, load_table, stop_with_missing
from dashboard.filters import get_filters
from dashboard.labels import EXCEEDANCE_NOTE, OZONE_NOTE, UNITS, label
from dashboard.logic import filter_places

filters = get_filters()
try:
    city_stats = load_table("city_pollutant_stats")
    region_stats = load_table("region_pollutant_stats")
    state_stats = load_table("state_pollutant_stats")
    aqi_mix = load_table("aqi_category_distribution")
except MissingOutputError as error:
    stop_with_missing(error)

pollutant, unit = filters.pollutant, UNITS[filters.pollutant]
st.title("City and Regional Analysis")
st.caption(f"Indicator: **{label(pollutant)}**. Change it, or narrow regions and cities, "
           "in the sidebar.")
if pollutant == "india_aqi":
    st.warning(OZONE_NOTE)

cities = filter_places(city_stats[city_stats["pollutant"] == pollutant],
                       filters.regions, filters.cities)
if cities.empty:
    st.warning("No cities match the current filters.")
    st.stop()

tab_rank, tab_map, tab_groups, tab_aqi = st.tabs(
    ["Rankings & statistics", "Map", "Regions & states", "AQI categories"])

with tab_rank:
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.subheader("Ranking by mean")
        st.plotly_chart(ranking_bar(cities, "city", f"Mean of daily values, {unit}", unit),
                        width="stretch")
    with right:
        st.subheader("Distribution of daily values")
        st.plotly_chart(range_box(cities, unit), width="stretch")
        st.caption("Box: 25th–75th percentile with median. Whiskers: minimum to 95th percentile "
                   "(precomputed in Spark with percentile_approx).")
    st.subheader("Descriptive statistics")
    columns = ["rank_highest_mean", "city", "state", "region", "n_days", "mean", "median",
               "std", "min", "p25", "p75", "p95", "max"]
    if cities["pct_days_above_naaqs"].notna().any():
        columns += ["days_above_naaqs", "pct_days_above_naaqs", "pct_days_above_who"]
        st.caption(EXCEEDANCE_NOTE)
    st.dataframe(cities.sort_values("rank_highest_mean")[columns].round(2),
                 hide_index=True, width="stretch")
    st.caption("Rank 1 = highest mean among all 50 cities, for this indicator.")

with tab_map:
    st.subheader(f"Mean {label(pollutant)} by city")
    st.plotly_chart(city_map(cities, unit), width="stretch")
    st.caption("Points are the model grid cells at city centres. The base map is downloaded "
               "by Plotly, so it needs an internet connection; the points do not.")

with tab_groups:
    regions = region_stats[region_stats["pollutant"] == pollutant]
    if filters.regions:
        regions = regions[regions["region"].isin(filters.regions)]
    regions = regions.assign(label=regions["region"] + " (" + regions["n_cities"].astype(str) + ")")
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Regions")
        st.plotly_chart(ranking_bar(regions, "label", f"Mean of daily values, {unit}", unit),
                        width="stretch")
    with right:
        st.subheader("States")
        states = state_stats[state_stats["pollutant"] == pollutant]
        if filters.regions:
            states = states[states["regions"].str.contains("|".join(filters.regions))]
        st.dataframe(states.sort_values("mean", ascending=False)[
            ["rank_highest_mean", "state", "regions", "n_cities", "mean", "median", "p95",
             "few_cities"]].round(2), hide_index=True, width="stretch", height=420)
        st.caption("`few_cities` = fewer than 3 cities: the state figure may reflect a single "
                   "city and is not representative of the whole state.")

with tab_aqi:
    index = st.radio("AQI variant", ["india_aqi_pm", "india_aqi"], horizontal=True,
                     format_func=label, key="aqi_variant")
    if index == "india_aqi":
        st.warning(OZONE_NOTE)
    mix = filter_places(aqi_mix[aqi_mix["aqi_index"] == index], filters.regions, filters.cities)
    # Order cities by their share of Good + Satisfactory days, cleanest at the top.
    clean = (mix[mix["category_order"] <= 2].groupby("city")["share"].sum()
             .reindex(mix["city"].unique(), fill_value=0).sort_values())
    st.plotly_chart(aqi_stack(mix.set_index("city").loc[clean.index].reset_index()),
                    width="stretch")
    with st.expander("Data table"):
        st.dataframe(mix.pivot_table(index="city", columns="category", values="days", fill_value=0),
                     width="stretch")
