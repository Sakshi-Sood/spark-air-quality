"""Page 2 - City and regional analysis: rankings, statistics, map, regions/states, AQI mix."""
import streamlit as st

from dashboard.charts import aqi_stack, city_map, ranking_bar, ranking_with_distribution
from dashboard.data_loader import MissingOutputError, load_table, stop_with_missing
from dashboard.filters import get_filters
from dashboard.labels import EXCEEDANCE_NOTE, OZONE_NOTE, POLLUTANT_NAMES, UNITS, label
from dashboard.logic import filter_places
from dashboard.ui import (card, number_columns, page_header, section, show_chart,
                          table_height, text_columns)

TOP_N = 15

filters = get_filters()
try:
    city_stats = load_table("city_pollutant_stats")
    region_stats = load_table("region_pollutant_stats")
    state_stats = load_table("state_pollutant_stats")
    aqi_mix = load_table("aqi_category_distribution")
except MissingOutputError as error:
    stop_with_missing(error)

pollutant, unit = filters.pollutant, UNITS[filters.pollutant]
page_header("City and Regional Air Quality Rankings",
            f"Indicator: **{label(pollutant)}** · change it in the sidebar")
if pollutant == "india_aqi":
    st.warning(OZONE_NOTE, icon=":material/warning:")

cities = filter_places(city_stats[city_stats["pollutant"] == pollutant],
                       filters.regions, filters.cities)
if cities.empty:
    st.warning("No cities match the current filters.")
    st.stop()

tab_rank, tab_map, tab_groups, tab_aqi = st.tabs(
    ["Rankings & statistics", "Map", "Regions & states", "AQI categories"])

with tab_rank:
    with card("city-ranking"):
        section(f"{POLLUTANT_NAMES[pollutant]} by city",
                "Left: mean of daily values. Right: box = 25th–75th percentile with median, "
                "whisker = minimum to 95th percentile, dot = maximum.")
        shown = st.segmented_control(
            "Cities shown", ["All", f"Highest {TOP_N}", f"Lowest {TOP_N}"], default="All",
            key="c_shown", label_visibility="collapsed", disabled=len(cities) <= TOP_N)
        ranked = cities.sort_values("mean", ascending=False)
        if shown == f"Highest {TOP_N}":
            ranked = ranked.head(TOP_N)
        elif shown == f"Lowest {TOP_N}":
            ranked = ranked.tail(TOP_N)
        show_chart(ranking_with_distribution(ranked, unit))

    with card("city-stats"):
        days = int(cities["n_days"].max())
        section("Descriptive statistics",
                f"{unit} · {days:,} daily values per city · rank 1 = highest mean among all 50 cities")
        columns = ["rank_highest_mean", "city", "state", "region", "mean", "median",
                   "std", "min", "p25", "p75", "p95", "max"]
        config = {"rank_highest_mean": st.column_config.NumberColumn("Rank", format="%d", width="small"),
                  **{c: st.column_config.TextColumn(c.capitalize(), width="medium")
                     for c in ("city", "state", "region")},
                  **number_columns(["mean", "median", "std", "min", "p25", "p75", "p95", "max"])}
        if cities["pct_days_above_naaqs"].notna().any():
            columns += ["pct_days_above_naaqs", "pct_days_above_who"]
            config.update(number_columns(["pct_days_above_naaqs", "pct_days_above_who"], "%.1f %%",
                                         {"pct_days_above_naaqs": "Days > NAAQS",
                                          "pct_days_above_who": "Days > WHO"}))
        table = cities.sort_values("rank_highest_mean")[columns]
        st.dataframe(table, hide_index=True, width="stretch", column_config=config,
                     height=table_height(len(table)))
        if "pct_days_above_naaqs" in columns:
            st.caption(EXCEEDANCE_NOTE)

with tab_map:
    with card("city-map"):
        section(f"Mean {POLLUTANT_NAMES[pollutant]} by city", f"{unit} · brighter = higher")
        show_chart(city_map(cities, unit))
        st.caption("Points are the model grid cells at city centres. The base map is drawn "
                   "from Plotly's online map data, so it needs an internet connection.")

with tab_groups:
    regions = region_stats[region_stats["pollutant"] == pollutant]
    if filters.regions:
        regions = regions[regions["region"].isin(filters.regions)]
    regions = regions.assign(label=regions["region"] + " (" + regions["n_cities"].astype(str) + ")")
    left, right = st.columns([2, 3], gap="medium")
    with left, card("regions", height="stretch"):
        section("Regions", f"Mean, {unit} · cities in brackets")
        show_chart(ranking_bar(regions, "label", f"Mean of daily values, {unit}", unit))
    with right, card("states", height="stretch"):
        section("States", "Flagged states have fewer than 3 cities and may reflect a single city.")
        states = state_stats[state_stats["pollutant"] == pollutant]
        if filters.regions:
            states = states[states["regions"].str.contains("|".join(filters.regions))]
        states = states.sort_values("mean", ascending=False)
        st.dataframe(
            states[["rank_highest_mean", "state", "regions", "n_cities", "mean", "median", "p95",
                    "few_cities"]],
            hide_index=True, width="stretch", height=table_height(len(states), 12),
            column_config={"rank_highest_mean": st.column_config.NumberColumn("Rank", format="%d"),
                           "n_cities": st.column_config.NumberColumn("Cities", format="%d"),
                           "few_cities": st.column_config.CheckboxColumn("< 3 cities"),
                           **text_columns(["state", "regions"]),
                           **number_columns(["mean", "median", "p95"])})

with tab_aqi:
    with card("aqi"):
        head, control = st.columns([3, 2], vertical_alignment="bottom")
        with head:
            section("Share of days in each AQI category",
                    "CPCB category colours · cleanest cities at the top")
        index = control.segmented_control("AQI variant", ["india_aqi_pm", "india_aqi"],
                                          default="india_aqi_pm", key="aqi_variant",
                                          format_func={"india_aqi_pm": "PM-only AQI",
                                                       "india_aqi": "All-pollutant AQI"}.get,
                                          label_visibility="collapsed") or "india_aqi_pm"
        if index == "india_aqi":
            st.warning(OZONE_NOTE, icon=":material/warning:")
        mix = filter_places(aqi_mix[aqi_mix["aqi_index"] == index], filters.regions, filters.cities)
        # Cleanest at the top: order by share of Good + Satisfactory days.
        clean = (mix[mix["category_order"] <= 2].groupby("city")["share"].sum()
                 .reindex(mix["city"].unique(), fill_value=0).sort_values(ascending=False))
        show_chart(aqi_stack(mix.set_index("city").loc[clean.index].reset_index()))
        with st.expander("Data table (days per category)"):
            st.dataframe(mix.pivot_table(index="city", columns="category", values="days", fill_value=0),
                         width="stretch")
