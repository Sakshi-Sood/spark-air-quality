"""Page 1 - Overview: KPIs, coverage, regional comparison, key findings."""
import streamlit as st

from dashboard.charts import ranking_bar
from dashboard.data_loader import MissingOutputError, load_table, stop_with_missing
from dashboard.filters import get_filters
from dashboard.labels import (EXCEEDANCE_NOTE, OZONE_NOTE, POLLUTANT_NAMES,
                              PROVISIONAL_NOTE, SHORT_NAMES, UNITS)
from dashboard.logic import filter_places, key_findings
from dashboard.ui import card, fact_row, fmt_date, kpi_row, page_header, section, show_chart

filters = get_filters()
try:
    summary = load_table("dataset_summary").iloc[0]
    city_stats = load_table("city_pollutant_stats")
    region_stats = load_table("region_pollutant_stats")
    seasonal = load_table("seasonal_city", ("city", "region", "year", "season", "pollutant", "mean"))
    bins = load_table("weather_bin_stats")
except MissingOutputError as error:
    stop_with_missing(error)

pollutant, unit = filters.pollutant, UNITS[filters.pollutant]
name = POLLUTANT_NAMES[pollutant]
page_header("Air quality and weather across 50 Indian cities",
            f"Modelled data · processed with Apache Spark (local mode) · "
            f"{fmt_date(summary['first_date'])} – {fmt_date(summary['last_date'])}")
if pollutant == "india_aqi":
    st.warning(OZONE_NOTE, icon=":material/warning:")

# ---------------------------------------------------------------- headline metrics
selected = filter_places(city_stats[city_stats["pollutant"] == pollutant],
                         filters.regions, filters.cities)
if selected.empty:
    st.warning("No cities match the current filters.")
    st.stop()

filtered = bool(filters.regions or filters.cities)
scope = f"{len(selected)} selected cities" if filtered else f"all {summary['cities']} cities"
top = selected.loc[selected["max"].idxmax()]
# Every city has the same number of days, so the mean of the city means equals
# the mean over all of their city-days.
short = SHORT_NAMES[pollutant]
items = [
    (f"Mean {short} · {unit}", f"{selected['mean'].mean():.1f}", scope,
     f"Mean of daily {name} values over {scope}."),
    (f"Highest day · {unit}", f"{top['max']:.1f}", f"{top['city']}",
     "Highest single modelled daily value among the cities in view."),
]
if selected["days_above_naaqs"].notna().any():
    share = 100 * selected["days_above_naaqs"].sum() / selected["n_days"].sum()
    items.append(("Days above NAAQS", f"{share:.1f} %", "of city-days", EXCEEDANCE_NOTE))
else:
    items.append(("Days above NAAQS", "—", "no 24-h limit",
                  "India's NAAQS 24-h limits used here exist only for PM2.5 and PM10."))
items.append(("Cities in view", f"{len(selected)} of {summary['cities']}",
              f"{selected['region'].nunique()} regions",
              f"{selected['region'].nunique()} regions, {selected['state'].nunique()} states."))
kpi_row(items, key="overview-kpi")

years = summary["complete_years"]
fact_row([
    ("Air-quality observations", f"{summary['air_quality_rows']:,} city-days"),
    ("Weather observations", f"{summary['weather_rows']:,} city-days (from {fmt_date(summary['weather_first_date'])})"),
    ("Joined analytical rows", f"{summary['joined_rows']:,}"),
    ("Complete calendar years", f"{min(years)} – {max(years)}"),
])

# ---------------------------------------------------------------- regions and findings
left, right = st.columns([3, 2], gap="medium")
with left, card("overview-regions", height="stretch"):
    section(f"Mean {name} by region", f"{unit} · number of cities in brackets")
    regions = region_stats[region_stats["pollutant"] == pollutant].copy()
    regions["label"] = regions["region"] + " (" + regions["n_cities"].astype(str) + ")"
    show_chart(ranking_bar(regions, "label", f"Mean of daily values, {unit}", unit, height=320))
    st.caption("A region mean can hide large differences between its cities, "
               "e.g. clean hill cities alongside Delhi in the North.")
with right, card("overview-findings", height="stretch"):
    section("Key findings", f"{name} · {scope}")
    findings = key_findings(city_stats, region_stats, seasonal, bins, pollutant, name, unit,
                            filters.regions, filters.cities)
    st.markdown("\n".join(f"- {f}" for f in findings) or "No findings for this selection.")
    with st.expander("How to read these figures"):
        st.caption(EXCEEDANCE_NOTE)
        st.caption(PROVISIONAL_NOTE + f" ({summary['provisional_rows']} city-days.)")
