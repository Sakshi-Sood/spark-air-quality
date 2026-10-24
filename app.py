"""Streamlit dashboard entry point.

Run from the project folder:
    .\\venv\\Scripts\\python.exe -m streamlit run app.py

The dashboard only reads precomputed outputs (Parquet/JSON written by
main.py). It never starts Spark, so filters respond instantly.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dashboard.data_loader import MissingOutputError, load_cities, stop_with_missing  # noqa: E402
from dashboard.filters import render_sidebar  # noqa: E402

st.set_page_config(page_title="India Air Quality & Weather | Spark", page_icon="🌫️",
                   layout="wide")

VIEWS = Path("dashboard") / "views"
pages = st.navigation([
    st.Page(VIEWS / "overview.py", title="Overview", icon="🏠", default=True),
    st.Page(VIEWS / "city_region.py", title="City & Regional Analysis", icon="🏙️"),
    st.Page(VIEWS / "weather.py", title="Weather & Pollution", icon="🌦️"),
    st.Page(VIEWS / "temporal.py", title="Temporal Analysis", icon="📈"),
    st.Page(VIEWS / "performance.py", title="Spark Performance", icon="⚙️"),
])

try:
    render_sidebar(load_cities())
except MissingOutputError as error:
    stop_with_missing(error)

pages.run()
