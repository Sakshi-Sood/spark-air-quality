"""Shared page-layout helpers so every page looks the same.

Most styling comes from the dark theme in .streamlit/config.toml. The only
custom CSS gives cards a raised surface colour (the one charts are validated
against), slightly lighter than the page. It targets the `st-key-<key>` class that
Streamlit documents for containers created with a key, rather than
Streamlit's internal DOM structure.
"""
from datetime import date

import pandas as pd
import streamlit as st

from dashboard.labels import SURFACE

CARD_CSS = f"""
<style>
div[class*="st-key-card-"] {{ background: {SURFACE}; }}
/* Less empty space above the page title (falls back to the default if absent). */
[data-testid="stMainBlockContainer"] {{ padding-top: 2.5rem; }}
</style>
"""
PLOTLY_CONFIG = {"displaylogo": False,
                 "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


def apply_page_style() -> None:
    st.html(CARD_CSS)


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def card(key: str, **kwargs):
    """A bordered card container; use as `with card("name"):`."""
    return st.container(border=True, key=f"card-{key}", **kwargs)


def kpi(label: str, value: str, note: str | None = None, help: str | None = None) -> None:
    """A headline metric. The note is neutral supporting text (no arrow, no colour);
    longer explanations go in the help tooltip so the card stays compact."""
    st.metric(label, value, note, delta_color="off", delta_arrow="off", help=help)


def kpi_row(items: list[tuple], key: str) -> None:
    """Equal-width KPI cards in one row. Items: (label, value, note[, help])."""
    for i, (column, item) in enumerate(zip(st.columns(len(items)), items)):
        with column, card(f"{key}-{i}", height="stretch"):
            kpi(*item)


def fact_row(items: list[tuple[str, str]]) -> None:
    """Smaller supporting facts (label above, value below), no cards."""
    for column, (label, value) in zip(st.columns(len(items)), items):
        column.markdown(f":gray[{label}]  \n**{value}**")


def show_chart(fig, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=key)


def section(title: str, caption: str | None = None) -> None:
    st.subheader(title)
    if caption:
        st.caption(caption)


def fmt_date(value) -> str:
    """e.g. 5 Aug 2022."""
    d = pd.Timestamp(value).date() if not isinstance(value, date) else value
    return f"{d.day} {d:%b %Y}"


COLUMN_LABELS = {
    "city": "City", "state": "State", "region": "Region", "regions": "Regions",
    "mean": "Mean", "median": "Median", "std": "Std dev", "min": "Min", "max": "Max",
    "p25": "25th pct", "p75": "75th pct", "p95": "95th pct", "n_days": "Days",
    "n_cities": "Cities", "rank_highest_mean": "Rank", "date": "Date", "season": "Season",
}


def number_columns(columns: list[str], fmt: str = "%.1f", labels: dict | None = None) -> dict:
    """st.dataframe column_config entries with one number format and readable headers."""
    labels = {**COLUMN_LABELS, **(labels or {})}
    return {c: st.column_config.NumberColumn(labels.get(c, c), format=fmt) for c in columns}


def text_columns(columns: list[str]) -> dict:
    """Readable headers for text columns."""
    return {c: st.column_config.TextColumn(COLUMN_LABELS.get(c, c)) for c in columns}


def table_height(rows: int, max_rows: int = 15) -> int:
    """Height that fits up to max_rows rows (about 33 px each at this theme's font size)."""
    return 6 + 33 * (min(rows, max_rows) + 1)
