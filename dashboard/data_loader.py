"""Cached readers for the pipeline's outputs. The dashboard never starts Spark.

st.cache_data stores each result keyed by its arguments, so a table is read
from disk once per session (until the files change and the app restarts),
not on every filter change. All tables here are small aggregated outputs;
daily data is read only for the requested columns, cities and dates.
"""
import json
from datetime import date

import pandas as pd
import streamlit as st

from src.config import (
    ANALYTICS_DIR,
    ML_METRICS_DIR,
    ML_PREDICTIONS_DIR,
    ML_REPORTS_DIR,
    OUTPUT_DIR,
    PLANS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
)

RERUN_HINT = "Generate the outputs with: `.\\venv\\Scripts\\python.exe main.py --stage all`"


class MissingOutputError(FileNotFoundError):
    """A pipeline output the dashboard needs has not been generated yet."""


def _require(path):
    if not path.exists():
        shown = path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
        raise MissingOutputError(f"`{shown.as_posix()}` was not found.")
    return path


@st.cache_data(show_spinner="Loading analytical table…")
def load_table(name: str, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Read one table from outputs/analytics/ (optionally only some columns)."""
    path = _require(ANALYTICS_DIR / name)
    return pd.read_parquet(path, columns=list(columns) if columns else None)


@st.cache_data(show_spinner=False)
def load_cities() -> pd.DataFrame:
    return pd.read_parquet(_require(PROCESSED_DIR / "cities"))


@st.cache_data(show_spinner="Loading daily data…")
def load_daily(columns: tuple[str, ...], cities: tuple[str, ...],
               start: date | None = None, end: date | None = None) -> pd.DataFrame:
    """Read selected columns of the city x day table for the given cities and dates.

    The filters are passed to pyarrow, so rows outside the selection are
    dropped while reading instead of after loading the whole table.
    """
    filters = [("city", "in", list(cities))]
    if start:
        filters.append(("date", ">=", start))
    if end:
        filters.append(("date", "<=", end))
    path = _require(PROCESSED_DIR / "daily_joined")
    return pd.read_parquet(path, columns=list(columns), filters=filters)


@st.cache_data(show_spinner=False)
def load_report(name: str) -> dict | None:
    """A JSON report from outputs/reports/, or None if it does not exist."""
    path = REPORTS_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_text(name: str) -> str | None:
    path = PLANS_DIR / name
    return path.read_text(encoding="utf-8") if path.exists() else None


@st.cache_data(show_spinner=False)
def load_performance_results() -> pd.DataFrame | None:
    """Phase 6 results, if the experiments have been run."""
    path = OUTPUT_DIR / "performance" / "results.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_ml_comparison() -> dict | None:
    """Model comparison report from outputs/ml/metrics/model_comparison.json."""
    path = ML_METRICS_DIR / "model_comparison.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_ml_evaluation() -> dict | None:
    """Detailed evaluation metrics from outputs/ml/metrics/evaluation_metrics.json."""
    path = ML_METRICS_DIR / "evaluation_metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_ml_report() -> dict | None:
    """Training report from outputs/ml/reports/ml_training_report.json."""
    path = ML_REPORTS_DIR / "ml_training_report.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data(show_spinner="Loading batch forecasts…")
def load_ml_forecasts() -> pd.DataFrame:
    """Next-day PM2.5 forecasts for all cities."""
    path = _require(ML_PREDICTIONS_DIR / "forecast_results")
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading test predictions…")
def load_ml_test_predictions(cities: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Test set predictions and residuals."""
    filters = [("city", "in", list(cities))] if cities else None
    path = _require(ML_PREDICTIONS_DIR / "test_predictions")
    return pd.read_parquet(path, filters=filters)


def stop_with_missing(error: MissingOutputError) -> None:
    """Show an understandable message instead of a stack trace, then stop the page."""
    st.error(f"Required pipeline output is missing: {error}")
    st.info(RERUN_HINT)
    st.stop()
