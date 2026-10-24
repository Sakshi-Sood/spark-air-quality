"""Page 4 - Temporal analysis: daily and monthly series, years, seasons, episodes, top days."""
from datetime import timedelta

import streamlit as st

from dashboard.charts import annual_bars, city_lines, episode_timeline, season_bars
from dashboard.data_loader import (MissingOutputError, load_cities, load_daily,
                                   load_table, stop_with_missing)
from dashboard.filters import get_filters
from dashboard.labels import (EXCEEDANCE_NOTE, MAX_COMPARED_CITIES, OZONE_NOTE,
                              PROVISIONAL_NOTE, SERIES, TREND_NOTE, UNITS, label)
from dashboard.logic import assign_stable_colours, filter_places
from src.config import NAAQS_24H, SEASON_ORDER

DEFAULT_CITIES = ["Delhi", "Kolkata", "Mumbai", "Bengaluru"]

filters = get_filters()
try:
    summary = load_table("dataset_summary").iloc[0]
    monthly = load_table("monthly_city")
    annual = load_table("annual_city")
    seasonal = load_table("seasonal_city")
    episodes = load_table("pollution_episodes")
    top_days = load_table("top_pollution_days")
    all_cities = sorted(load_cities()["city"])
except MissingOutputError as error:
    stop_with_missing(error)

pollutant, unit = filters.pollutant, UNITS[filters.pollutant]
st.title("Temporal Analysis")
if pollutant == "india_aqi":
    st.warning(OZONE_NOTE)

# ---------------------------------------------------------------- which cities to compare
candidates = filters.cities or [c for c in DEFAULT_CITIES if c in all_cities]
if not filters.cities and filters.regions:
    candidates = sorted(monthly.loc[monthly["region"].isin(filters.regions), "city"].unique())
if len(candidates) > MAX_COMPARED_CITIES:
    st.info(f"Comparing the first {MAX_COMPARED_CITIES} of {len(candidates)} cities so lines "
            "stay readable. Pick specific cities in the sidebar to change them.")
compare = candidates[:MAX_COMPARED_CITIES]
if not compare:
    st.warning("No cities match the current filters.")
    st.stop()
# Each city keeps its colour while others are added or removed.
colours = assign_stable_colours(compare, st.session_state.get("city_colours", {}), SERIES)
st.session_state["city_colours"] = colours
st.caption(f"Indicator: **{label(pollutant)}** · Cities: {', '.join(compare)}")

tab_daily, tab_month, tab_year, tab_season, tab_episode, tab_top = st.tabs(
    ["Daily", "Monthly", "Annual", "Seasonal", "Episodes", "Top days"])

with tab_daily:
    first_day, last_day = summary["first_date"], summary["last_date"]
    start, end = st.slider("Date range", min_value=first_day, max_value=last_day,
                           value=(last_day - timedelta(days=365), last_day),
                           format="YYYY-MM-DD", key="t_dates")
    daily = load_daily(("city", "date", pollutant, "is_provisional"), tuple(compare), start, end)
    if daily.empty:
        st.info("No daily rows in this date range.")
    else:
        fig = city_lines(daily.sort_values("date"), "date", pollutant, colours, label(pollutant))
        if pollutant in NAAQS_24H:
            fig.add_hline(y=NAAQS_24H[pollutant], line_dash="dot", line_color="#52514e",
                          annotation_text="NAAQS 24-h value (reference)",
                          annotation_position="top left")
            st.caption(EXCEEDANCE_NOTE)
        st.plotly_chart(fig, width="stretch")
        if daily["is_provisional"].any():
            st.caption(PROVISIONAL_NOTE)

with tab_month:
    data = monthly[monthly["city"].isin(compare)].sort_values("month_start")
    st.plotly_chart(city_lines(data, "month_start", pollutant, colours,
                               f"Monthly mean of daily {label(pollutant)}"),
                    width="stretch")
    partial = data.loc[~data["month_complete"], "month_start"].drop_duplicates()
    st.caption("Partial months (first and last month of the data): "
               + ", ".join(d.strftime("%b %Y") for d in partial) + ". "
               "Monthly means are not compared with the 24-h limit.")

with tab_year:
    data = annual[(annual["city"].isin(compare)) & (annual["pollutant"] == pollutant)]
    st.plotly_chart(annual_bars(data, f"Annual mean, {unit}"), width="stretch")
    st.info(TREND_NOTE)
    st.dataframe(data.pivot_table(index="city", columns="year", values="mean").round(1),
                 width="stretch")

with tab_season:
    data = seasonal[(seasonal["city"].isin(compare)) & (seasonal["pollutant"] == pollutant)]
    by_season = (data.groupby(["city", "season"], as_index=False)
                 .agg(mean=("mean", "mean"), season_years=("year", "nunique")))
    st.plotly_chart(season_bars(by_season, colours, f"Mean over complete seasons, {unit}"),
                    width="stretch")
    st.caption("IMD seasons: Winter Jan–Feb, Pre-monsoon Mar–May, Monsoon Jun–Sep, "
               "Post-monsoon Oct–Dec. Only complete season-years are included "
               "(Monsoon 2022 and Post-monsoon 2026 are partial and excluded).")
    with st.expander("Season-by-year table"):
        st.dataframe(data.pivot_table(index=["city", "year"], columns="season", values="mean")
                     .reindex(columns=SEASON_ORDER).round(1), width="stretch")

with tab_episode:
    st.markdown("Episodes: **3 or more consecutive days** with modelled PM2.5 24-h mean "
                "above the NAAQS value of 60 µg/m³.")
    data = filter_places(episodes, filters.regions, filters.cities)
    min_len = st.slider("Minimum episode length (days)", 3, 30, 7, key="t_min_len")
    data = data[data["length_days"] >= min_len]
    if data.empty:
        st.info("No episodes match these filters.")
    else:
        st.plotly_chart(episode_timeline(data), width="stretch")
        st.caption("Orange episodes touch the first or last date of the data, so they may have "
                   "started earlier or still be ongoing (potentially truncated).")
        st.dataframe(data.sort_values("length_days", ascending=False)[
            ["city", "region", "start_date", "end_date", "length_days", "peak_pm2_5",
             "mean_pm2_5", "touches_data_boundary", "includes_provisional"]].round(1),
            hide_index=True, width="stretch")

with tab_top:
    city = st.selectbox("City", compare, key="t_top_city")
    st.dataframe(top_days[top_days["city"] == city].drop(columns=["region"]).round(1),
                 hide_index=True, width="stretch")
    st.caption("Ranked by modelled PM2.5 24-h mean, with that day's weather.")
