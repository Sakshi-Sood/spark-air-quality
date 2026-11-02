"""Page 4 - Temporal analysis: daily and monthly series, years, seasons, episodes, top days."""
from datetime import timedelta

import streamlit as st

from dashboard.charts import INK_2, annual_bars, city_lines, episode_timeline, season_bars
from dashboard.data_loader import (MissingOutputError, load_cities, load_daily,
                                   load_table, stop_with_missing)
from dashboard.filters import get_filters
from dashboard.labels import (EXCEEDANCE_NOTE, MAX_COMPARED_CITIES, OZONE_NOTE,
                              PROVISIONAL_NOTE, SERIES, TREND_NOTE, UNITS, label)
from dashboard.logic import assign_stable_colours, filter_places
from dashboard.ui import (card, number_columns, page_header, section, show_chart,
                          table_height, text_columns)
from src.config import NAAQS_24H, SEASON_ORDER

DEFAULT_CITIES = ["Delhi", "Kolkata", "Mumbai", "Bengaluru"]
DATE_COLUMN = {"format": "D MMM YYYY"}

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

# ---------------------------------------------------------------- which cities to compare
candidates = filters.cities or [c for c in DEFAULT_CITIES if c in all_cities]
if not filters.cities and filters.regions:
    candidates = sorted(monthly.loc[monthly["region"].isin(filters.regions), "city"].unique())
compare = candidates[:MAX_COMPARED_CITIES]
page_header("Trends over time",
            f"Indicator: **{label(pollutant)}** · comparing {', '.join(compare) or 'no cities'}")
if pollutant == "india_aqi":
    st.warning(OZONE_NOTE, icon=":material/warning:")
if len(candidates) > MAX_COMPARED_CITIES:
    st.caption(f":material/info: Showing the first {MAX_COMPARED_CITIES} of {len(candidates)} cities "
               "so lines stay readable; pick cities in the sidebar to change them.")
if not compare:
    st.warning("No cities match the current filters.")
    st.stop()
# Each city keeps its colour while others are added or removed.
colours = assign_stable_colours(compare, st.session_state.get("city_colours", {}), SERIES)
st.session_state["city_colours"] = colours

tab_daily, tab_month, tab_year, tab_season, tab_episode, tab_top = st.tabs(
    ["Daily", "Monthly", "Annual", "Seasonal", "Episodes", "Top days"])

with tab_daily, card("t-daily"):
    first_day, last_day = summary["first_date"], summary["last_date"]
    head, control = st.columns([1, 1], vertical_alignment="bottom")
    with head:
        section("Daily values", f"{unit} · 24-h means for the selected period")
    start, end = control.slider("Date range", min_value=first_day, max_value=last_day,
                                value=(last_day - timedelta(days=365), last_day),
                                format="D MMM YYYY", key="t_dates")
    daily = load_daily(("city", "date", pollutant, "is_provisional"), tuple(compare), start, end)
    if daily.empty:
        st.info("No daily rows in this date range.")
    else:
        fig = city_lines(daily.sort_values("date"), "date", pollutant, colours, label(pollutant))
        if pollutant in NAAQS_24H:
            fig.add_hline(y=NAAQS_24H[pollutant], line_dash="dot", line_color=INK_2, line_width=1,
                          annotation_text=f"NAAQS 24-h value, {NAAQS_24H[pollutant]:.0f}",
                          annotation_font=dict(size=11, color=INK_2),
                          annotation_bgcolor="rgba(21,29,27,0.85)",
                          annotation_position="top right")
        show_chart(fig)
        notes = [EXCEEDANCE_NOTE] if pollutant in NAAQS_24H else []
        if daily["is_provisional"].any():
            notes.append(PROVISIONAL_NOTE)
        for note in notes:
            st.caption(note)

with tab_month, card("t-monthly"):
    section("Monthly means", f"Mean of daily values, {unit}")
    data = monthly[monthly["city"].isin(compare)].sort_values("month_start")
    show_chart(city_lines(data, "month_start", pollutant, colours,
                          f"Monthly mean, {unit}", monthly=True))
    partial = data.loc[~data["month_complete"], "month_start"].drop_duplicates()
    st.caption("Partial months: " + ", ".join(d.strftime("%b %Y") for d in partial) + ". "
               "Monthly means are not compared with the 24-h limit.")

with tab_year, card("t-annual"):
    section("Annual means, complete years only", f"{unit} · 2023–2025")
    data = annual[(annual["city"].isin(compare)) & (annual["pollutant"] == pollutant)]
    show_chart(annual_bars(data, f"Annual mean, {unit}"))
    st.caption(f":material/info: {TREND_NOTE}")
    table = data.pivot_table(index="city", columns="year", values="mean").reset_index()
    table.columns = [str(c) for c in table.columns]
    st.dataframe(table, hide_index=True, width="stretch", height=table_height(len(table)),
                 column_config={**text_columns(["city"]),
                                **number_columns([c for c in table.columns if c != "city"])})

with tab_season, card("t-seasonal"):
    section("Seasonal means, complete seasons only", f"{unit} · averaged over complete season-years")
    data = seasonal[(seasonal["city"].isin(compare)) & (seasonal["pollutant"] == pollutant)]
    by_season = (data.groupby(["city", "season"], as_index=False)
                 .agg(mean=("mean", "mean"), season_years=("year", "nunique")))
    show_chart(season_bars(by_season, colours, f"Mean, {unit}"))
    st.caption("IMD seasons: Winter Jan–Feb · Pre-monsoon Mar–May · Monsoon Jun–Sep · "
               "Post-monsoon Oct–Dec. Monsoon 2022 and Post-monsoon 2026 are partial and excluded.")
    with st.expander("Season-by-year table"):
        st.dataframe(data.pivot_table(index=["city", "year"], columns="season", values="mean")
                     .reindex(columns=SEASON_ORDER).round(1), width="stretch")

with tab_episode, card("t-episodes"):
    head, control = st.columns([1, 1], vertical_alignment="bottom")
    with head:
        section("PM2.5 pollution episodes",
                "Runs of 3 or more consecutive days with modelled PM2.5 above 60 µg/m³ (NAAQS 24-h)")
    min_len = control.slider("Minimum episode length (days)", 3, 30, 7, key="t_min_len")
    data = filter_places(episodes, filters.regions, filters.cities)
    data = data[data["length_days"] >= min_len]
    if data.empty:
        st.info("No episodes match these filters.")
    else:
        show_chart(episode_timeline(data))
        st.caption("Amber episodes touch the first or last date of the data, so they may have "
                   "started earlier or still be ongoing (potentially truncated).")
        table = data.sort_values("length_days", ascending=False)[
            ["city", "region", "start_date", "end_date", "length_days", "peak_pm2_5",
             "mean_pm2_5", "touches_data_boundary", "includes_provisional"]]
        st.dataframe(table, hide_index=True, width="stretch", height=table_height(len(table), 10),
                     column_config={
                         **text_columns(["city", "region"]),
                         "start_date": st.column_config.DateColumn("Start", **DATE_COLUMN),
                         "end_date": st.column_config.DateColumn("End", **DATE_COLUMN),
                         "length_days": st.column_config.NumberColumn("Days", format="%d"),
                         **number_columns(["peak_pm2_5", "mean_pm2_5"], labels={
                             "peak_pm2_5": "Peak, µg/m³", "mean_pm2_5": "Mean, µg/m³"}),
                         "touches_data_boundary": st.column_config.CheckboxColumn("Possibly truncated"),
                         "includes_provisional": st.column_config.CheckboxColumn("Provisional days"),
                     })

with tab_top, card("t-top"):
    head, control = st.columns([1, 1], vertical_alignment="bottom")
    with head:
        section("Highest PM2.5 days", "Ranked by modelled 24-h mean, with that day's weather · "
                "PM2.5, PM10, dust in µg/m³ · temperature °C · humidity % · wind km/h · rain mm")
    city = control.selectbox("City", compare, key="t_top_city")
    table = top_days[top_days["city"] == city].drop(columns=["region", "city"])
    st.dataframe(table, hide_index=True, width="stretch", height=table_height(len(table), 20),
                 column_config={
                     "rank": st.column_config.NumberColumn("Rank", format="%d"),
                     "date": st.column_config.DateColumn("Date", **DATE_COLUMN),
                     **text_columns(["season"]),
                     "india_aqi_pm_category": st.column_config.TextColumn("Category"),
                     "india_aqi_pm": st.column_config.NumberColumn("AQI (PM)", format="%d"),
                     **number_columns(["pm2_5_mean", "pm10_mean", "dust_mean"], labels={
                         "pm2_5_mean": "PM2.5", "pm10_mean": "PM10", "dust_mean": "Dust"}),
                     **number_columns(["temperature_2m_mean", "relative_humidity_2m_mean",
                                       "wind_speed_10m_max", "precipitation_sum"], labels={
                         "temperature_2m_mean": "Temp", "relative_humidity_2m_mean": "Humidity",
                         "wind_speed_10m_max": "Wind", "precipitation_sum": "Rain"}),
                     "is_provisional": st.column_config.CheckboxColumn("Provisional"),
                 })
