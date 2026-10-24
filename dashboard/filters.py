"""Global sidebar filters, shared by every page.

The sidebar is drawn by app.py before the selected page runs, and every
widget has a fixed key, so the same selections apply on every page.
"""
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from dashboard.labels import MODEL_DATA_NOTE, OZONE_NOTE, label
from src.config import SUMMARY_POLLUTANTS


@dataclass(frozen=True)
class Filters:
    pollutant: str
    regions: list[str]
    cities: list[str]          # empty = all cities in the selected regions


def render_sidebar(cities: pd.DataFrame) -> Filters:
    st.sidebar.header("Filters")
    pollutant = st.sidebar.selectbox("Pollutant / index", SUMMARY_POLLUTANTS,
                                     format_func=label, key="pollutant")
    regions = st.sidebar.multiselect("Regions", sorted(cities["region"].unique()),
                                     key="regions", placeholder="All regions")
    in_regions = cities[cities["region"].isin(regions)] if regions else cities
    options = sorted(in_regions["city"])
    # Drop cities that the region filter just removed (Streamlit rejects
    # default values that are not among the options).
    if "cities" in st.session_state:
        st.session_state["cities"] = [c for c in st.session_state["cities"] if c in options]
    selected = st.sidebar.multiselect("Cities", options, key="cities", placeholder="All cities")
    if pollutant == "india_aqi":
        st.sidebar.warning(OZONE_NOTE)
    st.sidebar.caption(MODEL_DATA_NOTE)
    # No cities picked: an empty list means "all", unless regions narrow it.
    effective_cities = selected if selected else (options if regions else [])
    filters = Filters(pollutant, regions, effective_cities)
    st.session_state["filters"] = filters
    return filters


def get_filters() -> Filters:
    return st.session_state["filters"]


def page_pollutant(filters: Filters, allowed: list[str]) -> str:
    """The global pollutant if this page has results for it, else PM2.5 (with a note)."""
    if filters.pollutant in allowed:
        return filters.pollutant
    st.info(f"{label(filters.pollutant)} is not analysed on this page; showing {label(allowed[0])}.")
    return allowed[0]
