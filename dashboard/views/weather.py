"""Page 3 - Weather and pollution: correlations, within-season view, scatter, weather bins."""
import pandas as pd
import streamlit as st

from dashboard.charts import (correlation_heatmap, scatter_with_fit,
                              season_comparison, weather_bins_bar)
from dashboard.data_loader import (MissingOutputError, load_cities, load_daily,
                                   load_table, stop_with_missing)
from dashboard.filters import get_filters, page_pollutant
from dashboard.labels import CORRELATION_NOTE, POLLUTANT_NAMES, WEATHER_NAMES, label
from dashboard.logic import correlation_strength
from dashboard.ui import card, number_columns, page_header, section, show_chart, table_height
from src.config import (ALL_YEAR, BIN_POLLUTANTS, CORRELATION_WEATHER,
                        SEASON_ORDER, WEATHER_ANALYSIS_POLLUTANTS)

filters = get_filters()
try:
    correlations = load_table("weather_correlations")
    bins = load_table("weather_bin_stats")
    cities = load_cities()
except MissingOutputError as error:
    stop_with_missing(error)

page_header("Weather and Pollution Associations",
            "How modelled pollution moves with weather · observational associations, not evidence of cause")
pollutant = page_pollutant(filters, WEATHER_ANALYSIS_POLLUTANTS)

# ---------------------------------------------------------------- page filters
city_options = filters.cities or sorted(cities["city"])
region_options = filters.regions or sorted(cities["region"].unique())
with card("weather-filters"):
    f1, f2, f3, f4 = st.columns(4)
    scope_type = f1.selectbox("Scope", ["city", "region", "all"], key="w_scope_type",
                              format_func={"city": "One city", "region": "One region",
                                           "all": "All 50 cities pooled"}.get)
    if scope_type == "city":
        scope = f2.selectbox("City", city_options, key="w_city")
    elif scope_type == "region":
        scope = f2.selectbox("Region", region_options, key="w_region")
    else:
        scope = "All cities"
        f2.caption("Pooling mixes differences *between* cities with day-to-day variation; "
                   "a city or region view is usually more meaningful.")
    weather_var = f3.selectbox("Weather variable", CORRELATION_WEATHER, format_func=label, key="w_var")
    season = f4.selectbox("Season", [ALL_YEAR] + SEASON_ORDER, key="w_season")
    metric = st.radio("Correlation measure", ["spearman_rho", "pearson_r"], horizontal=True,
                      key="w_metric",
                      format_func={"spearman_rho": "Spearman ρ (rank-based, robust to outliers)",
                                   "pearson_r": "Pearson r (linear)"}.get)
metric_name = "Spearman ρ" if metric == "spearman_rho" else "Pearson r"

scoped = correlations[(correlations["scope_type"] == scope_type) & (correlations["scope"] == scope)]
pair = scoped[(scoped["pollutant"] == pollutant) & (scoped["weather_variable"] == weather_var)]

# ---------------------------------------------------------------- correlations
left, right = st.columns([3, 2], gap="medium")
with left, card("weather-matrix", height="stretch"):
    cell = scoped[scoped["season_group"] == season]
    section(f"Correlation matrix · {scope} · {season}",
            "Teal = move in opposite directions · terracotta = move together")
    matrix = (cell.pivot(index="pollutant", columns="weather_variable", values=metric)
              .reindex(index=WEATHER_ANALYSIS_POLLUTANTS, columns=CORRELATION_WEATHER)
              .rename(index=POLLUTANT_NAMES, columns=WEATHER_NAMES))
    if matrix.isna().all().all():
        st.info("No correlations available for this selection.")
    else:
        show_chart(correlation_heatmap(matrix, metric_name))
        st.caption(f"{metric_name} over {int(cell['n_days'].iloc[0]):,} days. {CORRELATION_NOTE}")
with right, card("weather-seasons", height="stretch"):
    section("All year vs within season",
            f"{POLLUTANT_NAMES[pollutant]} vs {WEATHER_NAMES[weather_var].lower()}")
    if pair.empty:
        st.info("No data for this selection.")
    else:
        show_chart(season_comparison(pair, metric, metric_name))
        rows = pair[pair["season_group"] == season]
        row = rows.iloc[0] if not rows.empty else None
        if row is None:
            st.info(f"No result for {season}.")
        elif row["status"] != "ok":
            st.info(f"Not reported for {season}: {row['status'].replace('_', ' ')}.")
        else:
            st.markdown(f"**{season}:** {metric_name} = {row[metric]:.2f} "
                        f"({correlation_strength(row[metric])}, {row['n_days']:,} days)")
        st.caption("When season bars differ from the grey all-year bar, part of the all-year "
                   "association reflects seasonal differences rather than the weather variable.")

# ---------------------------------------------------------------- daily scatter and weather bins
left, right = st.columns(2, gap="medium")
with left, card("weather-scatter", height="stretch"):
    section("Daily values", "One city at a time")
    scatter_city = scope if scope_type == "city" else st.selectbox(
        "City for the scatter plot", city_options, key="w_scatter_city")
    daily = load_daily(("city", "date", "season", "season_complete", pollutant, weather_var),
                       (scatter_city,))
    if season != ALL_YEAR:
        daily = daily[(daily["season"] == season) & daily["season_complete"]]
    if daily.empty:
        st.info("No daily rows for this selection.")
    else:
        show_chart(scatter_with_fit(daily, weather_var, pollutant, label(weather_var), label(pollutant)))
        st.caption(f"{scatter_city} · {season} · {len(daily):,} days. "
                   "The line is a least-squares fit, shown only as a visual guide.")

with right, card("weather-bins", height="stretch"):
    bin_pollutant = pollutant if pollutant in BIN_POLLUTANTS else BIN_POLLUTANTS[0]
    section("Pollution by weather category", f"Mean {label(bin_pollutant)} · {scope} · {season}")
    selected_bins = bins[(bins["scope_type"] == scope_type) & (bins["scope"] == scope)
                         & (bins["season_group"] == season) & (bins["weather_variable"] == weather_var)
                         & (bins["pollutant"] == bin_pollutant)]
    if selected_bins.empty:
        st.info("No weather-bin results for this selection.")
    else:
        show_chart(weather_bins_bar(selected_bins, label(weather_var), f"Mean, {label(bin_pollutant).split(', ')[-1]}"))
        if selected_bins["low_sample"].any():
            st.caption("Hatched grey bars have fewer than 30 days and should not be compared.")
        with st.expander("Data table"):
            table = selected_bins.sort_values("bin_order")[["bin_label", "n_days", "mean", "median", "low_sample"]]
            st.dataframe(table, hide_index=True, width="stretch", height=table_height(len(table)),
                         column_config={"bin_label": st.column_config.TextColumn("Category"),
                                        "n_days": st.column_config.NumberColumn("Days", format="%d"),
                                        "low_sample": st.column_config.CheckboxColumn("< 30 days"),
                                        **number_columns(["mean", "median"])})

# ---------------------------------------------------------------- spread across cities
with st.expander(f"How {metric_name} for this pair varies across the 50 cities ({season})"):
    per_city = correlations[(correlations["scope_type"] == "city") & (correlations["season_group"] == season)
                            & (correlations["pollutant"] == pollutant)
                            & (correlations["weather_variable"] == weather_var)]
    table = pd.DataFrame({"city": per_city["scope"], "value": per_city[metric],
                          "strength": per_city[metric].map(correlation_strength),
                          "days": per_city["n_days"]}).sort_values("value")
    st.dataframe(table, hide_index=True, width="stretch", height=table_height(len(table), 12),
                 column_config={"city": st.column_config.TextColumn("City"),
                                "value": st.column_config.NumberColumn(metric_name, format="%.2f"),
                                "strength": st.column_config.TextColumn("Strength"),
                                "days": st.column_config.NumberColumn("Days", format="%d")})
