"""Plotly chart builders shared by the dashboard pages.

Every figure goes through style(), so fonts, gridlines, margins, legends and
tooltips match on all pages. Conventions: one y-axis per chart, one colour
for single-series charts, city colours that follow the city (not its rank),
units in axis titles and tooltips.
"""
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.labels import (AQI_COLOURS, DIVERGING, MUTED, PRIMARY, SEQUENTIAL,
                              SERIES, YEAR_RAMP)
from src.config import ALL_YEAR, AQI_BANDS, SEASON_ORDER

INK, INK_2, GRID, AXIS = "#1d2a26", "#5b6670", "#e8ebe8", "#c9cfcb"
FONT = "Source Sans Pro, system-ui, -apple-system, Segoe UI, sans-serif"
ROW_PX = 22          # height per category row in horizontal charts


def style(fig: go.Figure, x_title: str = "", y_title: str = "",
          height: int = 360, legend: bool = True) -> go.Figure:
    """Apply the shared look; axis titles carry the units."""
    fig.update_layout(
        height=height, margin=dict(l=8, r=16, t=48 if legend else 12, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=12),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, title_text="",
                    font=dict(size=12, color=INK_2)),
        hoverlabel=dict(bgcolor="white", bordercolor=AXIS, font=dict(family=FONT, color=INK, size=12)),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=INK_2),
                     title_font=dict(color=INK_2, size=12), automargin=True)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=INK_2),
                     title_font=dict(color=INK_2, size=12), automargin=True)
    fig.update_xaxes(title_text=x_title)      # always set: clears Plotly Express defaults
    fig.update_yaxes(title_text=y_title)
    return fig


def _rows_height(n: int, extra: int = 70, minimum: int = 240) -> int:
    return max(minimum, ROW_PX * n + extra)


# ---------------------------------------------------------------- rankings and distributions

def ranking_bar(df: pd.DataFrame, category: str, value_title: str, unit: str,
                height: int | None = None) -> go.Figure:
    """Horizontal bars, highest at the top. df needs `category`, mean, median, p95."""
    data = df.sort_values("mean")
    fig = go.Figure(go.Bar(
        x=data["mean"], y=data[category], orientation="h", marker_color=PRIMARY,
        customdata=data[["median", "p95"]],
        hovertemplate=(f"<b>%{{y}}</b><br>Mean %{{x:.1f}} {unit}<br>Median %{{customdata[0]:.1f}}"
                       f"<br>95th percentile %{{customdata[1]:.1f}}<extra></extra>"),
    ))
    fig.update_layout(bargap=0.35)
    return style(fig, value_title, "", height=height or _rows_height(len(data)), legend=False)


def ranking_with_distribution(df: pd.DataFrame, unit: str) -> go.Figure:
    """City ranking (left) and daily-value distribution (right) sharing one city axis.

    Distribution from precomputed quantiles: box = 25th-75th percentile with
    the median, whisker = minimum to 95th percentile, dot = maximum, so high
    values stay visible instead of being cut off.
    """
    data = df.sort_values("mean")
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        column_widths=[0.38, 0.62],
                        subplot_titles=(f"Mean, {unit}", f"Daily values, {unit}"))
    fig.add_trace(go.Bar(
        x=data["mean"], y=data["city"], orientation="h", marker_color=PRIMARY, name="mean",
        hovertemplate=f"<b>%{{y}}</b><br>Mean %{{x:.1f}} {unit}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Box(
        y=data["city"], q1=data["p25"], median=data["median"], q3=data["p75"],
        lowerfence=data["min"], upperfence=data["p95"], orientation="h", name="distribution",
        marker_color=PRIMARY, line_width=1.2, fillcolor="rgba(47,143,120,0.18)",
        hoverinfo="x+y"), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=data["max"], y=data["city"], mode="markers", name="maximum",
        marker=dict(size=7, color="white", line=dict(color=INK_2, width=1.2)),
        hovertemplate=f"<b>%{{y}}</b><br>Maximum %{{x:.1f}} {unit}<extra></extra>"), row=1, col=2)
    fig.update_layout(bargap=0.35)
    for annotation in fig.layout.annotations:          # subplot titles
        annotation.update(font=dict(size=12, color=INK_2), x=annotation.x, xanchor="center")
    fig = style(fig, height=_rows_height(len(data), extra=60), legend=False)
    fig.update_layout(margin=dict(t=30))
    return fig


def city_map(df: pd.DataFrame, unit: str) -> go.Figure:
    fig = px.scatter_geo(df, lat="latitude", lon="longitude", color="mean", hover_name="city",
                         hover_data={"region": True, "mean": ":.1f", "latitude": False, "longitude": False},
                         color_continuous_scale=SEQUENTIAL[1:])
    fig.update_traces(marker=dict(size=11, line=dict(width=1, color="white")))
    fig.update_geos(fitbounds="locations", showcountries=True, countrycolor=AXIS,
                    showland=True, landcolor="#eef1ee", showocean=True, oceancolor="#f7f9f8",
                    showframe=False, lakecolor="#f7f9f8", projection_type="mercator")
    fig.update_layout(coloraxis_colorbar=dict(title=dict(text=unit, font=dict(size=12)),
                                              thickness=12, len=0.7))
    fig = style(fig, height=540, legend=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig


def aqi_stack(df: pd.DataFrame) -> go.Figure:
    """100 % stacked bars of AQI categories per city."""
    order = [label for _, label in AQI_BANDS]
    data = df.assign(percent=100 * df["share"])
    fig = px.bar(data, x="percent", y="city", color="category", orientation="h",
                 category_orders={"category": order, "city": list(dict.fromkeys(data["city"]))},
                 color_discrete_map=AQI_COLOURS, custom_data=["days"])
    fig.update_traces(marker_line_width=1, marker_line_color="white",
                      hovertemplate="<b>%{y}</b><br>%{fullData.name}: %{x:.1f} % of days"
                                    " (%{customdata[0]} days)<extra></extra>")
    fig.update_layout(barmode="stack", bargap=0.3)
    fig.update_xaxes(range=[0, 100], ticksuffix=" %")
    fig.update_yaxes(autorange="reversed")            # first city in the given order at the top
    return style(fig, "Share of days", "", height=_rows_height(data["city"].nunique(), extra=90))


# ---------------------------------------------------------------- weather

def correlation_heatmap(matrix: pd.DataFrame, metric: str) -> go.Figure:
    columns = [c.replace(" ", "<br>", 1) for c in matrix.columns]      # wrap, keep labels level
    fig = go.Figure(go.Heatmap(
        z=matrix.values, x=columns, y=list(matrix.index), zmin=-1, zmax=1,
        colorscale=DIVERGING, texttemplate="%{z:.2f}", textfont=dict(size=13), xgap=3, ygap=3,
        colorbar=dict(thickness=12, len=0.9, tickvals=[-1, -0.5, 0, 0.5, 1]),
        hovertemplate="%{y} vs %{x}<br>" + metric + " = %{z:.2f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="bottom", tickangle=0)
    return style(fig, height=330, legend=False)


def season_comparison(df: pd.DataFrame, metric_col: str, metric: str) -> go.Figure:
    """Correlation for 'All year' (grey) next to each complete season (teal)."""
    order = [ALL_YEAR] + SEASON_ORDER
    data = df.set_index("season_group").reindex(order).dropna(subset=[metric_col]).reset_index()
    colours = [MUTED if s == ALL_YEAR else PRIMARY for s in data["season_group"]]
    ticks = data["season_group"].str.replace("-", "-<br>", regex=False).str.replace("All year", "All<br>year")
    fig = go.Figure(go.Bar(x=ticks, y=data[metric_col], marker_color=colours,
                           customdata=data["n_days"], text=data[metric_col].round(2),
                           textposition="outside", cliponaxis=False, textfont=dict(color=INK_2),
                           hovertemplate="%{x}<br>" + metric + " %{y:.2f}<br>%{customdata:,} days<extra></extra>"))
    fig.add_hline(y=0, line_color=AXIS, line_width=1)
    fig.update_yaxes(range=[-1, 1], tickvals=[-1, -0.5, 0, 0.5, 1])
    fig.update_xaxes(tickangle=0)
    fig.update_layout(bargap=0.45)
    return style(fig, "", metric, height=320, legend=False)


def weather_bins_bar(df: pd.DataFrame, weather_title: str, pollutant_title: str) -> go.Figure:
    """Mean pollution per weather bin; low-sample bins are hatched and grey."""
    data = df.sort_values("bin_order")
    low = data["low_sample"].to_numpy()
    fig = go.Figure(go.Bar(
        x=data["bin_label"], y=data["mean"],
        marker=dict(color=[MUTED if flag else PRIMARY for flag in low],
                    pattern_shape=["/" if flag else "" for flag in low]),
        customdata=np.stack([data["n_days"], data["median"]], axis=-1),
        hovertemplate="%{x}<br>Mean %{y:.1f}<br>Median %{customdata[1]:.1f}"
                      "<br>%{customdata[0]:,} days<extra></extra>",
    ))
    fig.update_layout(bargap=0.4)
    return style(fig, weather_title, pollutant_title, height=340, legend=False)


def scatter_with_fit(df: pd.DataFrame, x: str, y: str, x_title: str, y_title: str) -> go.Figure:
    """Daily points with a least-squares line, shown only as a visual guide."""
    fig = go.Figure(go.Scattergl(
        x=df[x], y=df[y], mode="markers", name="days",
        marker=dict(size=7, color=PRIMARY, opacity=0.4, line=dict(width=0)),
        customdata=df["date"].astype(str),
        hovertemplate="%{customdata}<br>" + x_title + ": %{x:.1f}<br>" + y_title + ": %{y:.1f}<extra></extra>",
    ))
    if len(df) >= 3 and df[x].nunique() > 1:
        slope, intercept = np.polyfit(df[x], df[y], 1)
        xs = np.array([df[x].min(), df[x].max()])
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="linear fit",
                                 line=dict(color=INK, width=2), hoverinfo="skip"))
    return style(fig, x_title, y_title, height=420, legend=False)


# ---------------------------------------------------------------- time

def city_lines(df: pd.DataFrame, x: str, y: str, colours: dict[str, str],
               y_title: str, monthly: bool = False) -> go.Figure:
    fig = go.Figure()
    for city, colour in colours.items():
        part = df[df["city"] == city]
        fig.add_trace(go.Scatter(x=part[x], y=part[y], mode="lines", name=city,
                                 line=dict(color=colour, width=2 if monthly else 1.6),
                                 hovertemplate=f"{city}: %{{y:.1f}}<extra></extra>"))
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(tickformat="%b %Y", hoverformat="%b %Y" if monthly else "%d %b %Y")
    return style(fig, "", y_title, height=400)


def annual_bars(df: pd.DataFrame, y_title: str) -> go.Figure:
    years = sorted(df["year"].unique())
    colour_map = {str(y): YEAR_RAMP[i % len(YEAR_RAMP)] for i, y in enumerate(years)}
    data = df.assign(year=df["year"].astype(str))
    fig = px.bar(data, x="city", y="mean", color="year", barmode="group",
                 color_discrete_map=colour_map, category_orders={"year": list(colour_map)})
    fig.update_traces(marker_line_width=1, marker_line_color="white",
                      hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:.1f}<extra></extra>")
    fig.update_layout(bargap=0.3, bargroupgap=0.05)
    return style(fig, "", y_title, height=360)


def season_bars(df: pd.DataFrame, colours: dict[str, str], y_title: str) -> go.Figure:
    fig = px.bar(df, x="season", y="mean", color="city", barmode="group",
                 color_discrete_map=colours, category_orders={"season": SEASON_ORDER, "city": list(colours)},
                 custom_data=["season_years"])
    fig.update_traces(marker_line_width=1, marker_line_color="white",
                      hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y:.1f}"
                                    "<br>%{customdata[0]} complete season-years<extra></extra>")
    fig.update_layout(bargap=0.3, bargroupgap=0.05)
    return style(fig, "", y_title, height=360)


def episode_timeline(df: pd.DataFrame) -> go.Figure:
    """Episodes as horizontal spans; possibly truncated ones in a second colour."""
    complete, truncated = "Complete episode", "Potentially truncated (touches data edge)"
    data = df.assign(
        end_display=pd.to_datetime(df["end_date"]) + timedelta(days=1),   # inclusive end
        kind=np.where(df["touches_data_boundary"], truncated, complete))
    order = list(data.groupby("city")["length_days"].sum().sort_values(ascending=False).index)
    fig = px.timeline(data, x_start="start_date", x_end="end_display", y="city", color="kind",
                      color_discrete_map={complete: PRIMARY, truncated: SERIES[1]},
                      category_orders={"city": order, "kind": [complete, truncated]},
                      custom_data=["length_days", "peak_pm2_5", "start_date", "end_date"])
    fig.update_traces(hovertemplate="<b>%{y}</b><br>%{customdata[2]} to %{customdata[3]}"
                                    "<br>%{customdata[0]} days, peak %{customdata[1]:.0f} µg/m³<extra></extra>")
    fig.update_xaxes(tickformat="%b %Y")
    fig.update_yaxes(autorange="reversed")            # most episode-days at the top
    return style(fig, "", "", height=_rows_height(len(order), extra=90))


# ---------------------------------------------------------------- performance

def experiment_comparison(results: pd.DataFrame, experiment: str) -> go.Figure:
    """Median run time per variant with min-max range, for one measured experiment.

    Expects the Phase 6 results schema: experiment, variant, seconds (one row per run).
    """
    runs = results[results["experiment"] == experiment]
    stats = runs.groupby("variant", sort=False)["seconds"].agg(["median", "min", "max", "count"]).reset_index()
    fig = go.Figure(go.Bar(
        x=stats["variant"].astype(str), y=stats["median"], marker_color=PRIMARY,
        error_y=dict(type="data", symmetric=False, array=stats["max"] - stats["median"],
                     arrayminus=stats["median"] - stats["min"], color=INK_2, thickness=1.2),
        customdata=stats[["min", "max", "count"]],
        hovertemplate="%{x}<br>Median %{y:.2f} s<br>Range %{customdata[0]:.2f}–%{customdata[1]:.2f} s"
                      "<br>%{customdata[2]} runs<extra></extra>"))
    fig.update_layout(bargap=0.45)
    return style(fig, "", "Seconds (median, min–max)", height=320, legend=False)
