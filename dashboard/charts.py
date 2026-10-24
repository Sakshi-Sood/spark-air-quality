"""Plotly chart builders shared by the dashboard pages.

Conventions (validated palette, see dashboard/labels.py): one y-axis per
chart, single-series charts in one colour, colours follow the entity (city)
rather than its rank, thin marks, hairline grid, units in every axis title
and tooltip.
"""
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.labels import (AQI_COLOURS, DIVERGING, MUTED, PRIMARY,
                              SEQUENTIAL, YEAR_RAMP)
from src.config import ALL_YEAR, AQI_BANDS, SEASON_ORDER

INK, INK_2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def style(fig: go.Figure, x_title: str = "", y_title: str = "",
          height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK, size=13),
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(bgcolor="white", font_color=INK),
    )
    for axis in (fig.layout.xaxis, fig.layout.yaxis):
        axis.update(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS,
                    tickfont=dict(color=INK_2), title_font=dict(color=INK_2))
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text=y_title)
    return fig


def ranking_bar(df: pd.DataFrame, category: str, value_title: str, unit: str) -> go.Figure:
    """Horizontal bars, highest at the top. df needs `category`, mean, median, p95."""
    data = df.sort_values("mean")
    fig = go.Figure(go.Bar(
        x=data["mean"], y=data[category], orientation="h", marker_color=PRIMARY,
        customdata=data[["median", "p95"]],
        hovertemplate=(f"<b>%{{y}}</b><br>mean %{{x:.1f}} {unit}<br>median %{{customdata[0]:.1f}}"
                       f"<br>95th pct %{{customdata[1]:.1f}}<extra></extra>"),
    ))
    fig.update_layout(bargap=0.25)
    return style(fig, value_title, "", height=max(260, 22 * len(data) + 60), legend=False)


def range_box(df: pd.DataFrame, unit: str) -> go.Figure:
    """Distribution per city from precomputed quantiles (box: p25-p75, whiskers: min to p95)."""
    data = df.sort_values("median", ascending=False)
    fig = go.Figure(go.Box(
        x=data["city"], q1=data["p25"], median=data["median"], q3=data["p75"],
        lowerfence=data["min"], upperfence=data["p95"], marker_color=PRIMARY,
        line_width=1.5, fillcolor="rgba(42,120,214,0.15)", name="",
    ))
    return style(fig, "", f"Daily value, {unit}", height=420, legend=False)


def city_map(df: pd.DataFrame, unit: str) -> go.Figure:
    fig = px.scatter_geo(df, lat="latitude", lon="longitude", color="mean", hover_name="city",
                         hover_data={"region": True, "mean": ":.1f", "latitude": False, "longitude": False},
                         color_continuous_scale=SEQUENTIAL)
    fig.update_traces(marker=dict(size=11, line=dict(width=1, color="white")))
    fig.update_geos(fitbounds="locations", showcountries=True, countrycolor=AXIS,
                    showland=True, landcolor="#f4f3ef", showocean=False, lakecolor=SURFACE)
    fig.update_layout(coloraxis_colorbar=dict(title=unit))
    return style(fig, height=520)


def aqi_stack(df: pd.DataFrame) -> go.Figure:
    """100 % stacked bars of AQI categories per city."""
    order = [label for _, label in AQI_BANDS]
    data = df.assign(percent=100 * df["share"])
    fig = px.bar(data, x="percent", y="city", color="category", orientation="h",
                 category_orders={"category": order}, color_discrete_map=AQI_COLOURS,
                 custom_data=["days"])
    fig.update_traces(marker_line_width=1, marker_line_color=SURFACE,
                      hovertemplate="<b>%{y}</b><br>%{fullData.name}: %{x:.1f} % of days"
                                    " (%{customdata[0]} days)<extra></extra>")
    fig.update_layout(barmode="stack", bargap=0.25, legend_title_text="")
    return style(fig, "Share of days, %", "", height=max(300, 22 * data["city"].nunique() + 90))


def correlation_heatmap(matrix: pd.DataFrame, metric: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=matrix.values, x=list(matrix.columns), y=list(matrix.index), zmin=-1, zmax=1,
        colorscale=DIVERGING, texttemplate="%{z:.2f}", xgap=2, ygap=2,
        colorbar=dict(title=metric),
        hovertemplate="%{y}<br>%{x}<br>" + metric + " = %{z:.2f}<extra></extra>",
    ))
    return style(fig, height=340, legend=False)


def season_comparison(df: pd.DataFrame, metric_col: str, metric: str) -> go.Figure:
    """Correlation for 'All year' (grey) next to each complete season (blue)."""
    order = [ALL_YEAR] + SEASON_ORDER
    data = df.set_index("season_group").reindex(order).dropna(subset=[metric_col]).reset_index()
    colours = [MUTED if s == ALL_YEAR else PRIMARY for s in data["season_group"]]
    fig = go.Figure(go.Bar(x=data["season_group"], y=data[metric_col], marker_color=colours,
                           customdata=data["n_days"],
                           hovertemplate="%{x}<br>" + metric + " %{y:.2f}<br>%{customdata} days<extra></extra>"))
    fig.add_hline(y=0, line_color=AXIS, line_width=1)
    fig.update_yaxes(range=[-1, 1])
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
        hovertemplate="%{x}<br>mean %{y:.1f}<br>median %{customdata[1]:.1f}"
                      "<br>%{customdata[0]} days<extra></extra>",
    ))
    fig.update_layout(bargap=0.3)
    return style(fig, weather_title, pollutant_title, height=340, legend=False)


def scatter_with_fit(df: pd.DataFrame, x: str, y: str, x_title: str, y_title: str) -> go.Figure:
    """Daily points with a least-squares line, shown only as a visual guide."""
    fig = go.Figure(go.Scattergl(
        x=df[x], y=df[y], mode="markers", name="days",
        marker=dict(size=8, color=PRIMARY, opacity=0.35, line=dict(width=0)),
        customdata=df["date"].astype(str),
        hovertemplate="%{customdata}<br>x %{x:.1f}<br>y %{y:.1f}<extra></extra>",
    ))
    if len(df) >= 3 and df[x].nunique() > 1:
        slope, intercept = np.polyfit(df[x], df[y], 1)
        xs = np.array([df[x].min(), df[x].max()])
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="linear fit",
                                 line=dict(color=INK, width=2), hoverinfo="skip"))
    return style(fig, x_title, y_title, height=420, legend=False)


def city_lines(df: pd.DataFrame, x: str, y: str, colours: dict[str, str],
               y_title: str, hover_extra: str = "") -> go.Figure:
    fig = go.Figure()
    for city, colour in colours.items():
        part = df[df["city"] == city]
        fig.add_trace(go.Scatter(x=part[x], y=part[y], mode="lines", name=city,
                                 line=dict(color=colour, width=2),
                                 hovertemplate=f"<b>{city}</b><br>%{{x}}<br>%{{y:.1f}}{hover_extra}<extra></extra>"))
    fig.update_layout(hovermode="x unified")
    return style(fig, "", y_title, height=400, legend=True)


def annual_bars(df: pd.DataFrame, y_title: str) -> go.Figure:
    years = sorted(df["year"].unique())
    colour_map = {str(y): YEAR_RAMP[i % len(YEAR_RAMP)] for i, y in enumerate(years)}
    data = df.assign(year=df["year"].astype(str))
    fig = px.bar(data, x="city", y="mean", color="year", barmode="group",
                 color_discrete_map=colour_map, category_orders={"year": list(colour_map)})
    fig.update_traces(marker_line_width=1, marker_line_color=SURFACE,
                      hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:.1f}<extra></extra>")
    fig.update_layout(legend_title_text="")
    return style(fig, "", y_title, height=380)


def season_bars(df: pd.DataFrame, colours: dict[str, str], y_title: str) -> go.Figure:
    fig = px.bar(df, x="season", y="mean", color="city", barmode="group",
                 color_discrete_map=colours, category_orders={"season": SEASON_ORDER},
                 custom_data=["season_years"])
    fig.update_traces(marker_line_width=1, marker_line_color=SURFACE,
                      hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y:.1f}"
                                    "<br>%{customdata[0]} complete season-years<extra></extra>")
    fig.update_layout(legend_title_text="")
    return style(fig, "", y_title, height=380)


def episode_timeline(df: pd.DataFrame) -> go.Figure:
    """Episodes as horizontal spans; possibly truncated ones in a second colour."""
    data = df.assign(
        end_display=pd.to_datetime(df["end_date"]) + timedelta(days=1),   # inclusive end
        kind=np.where(df["touches_data_boundary"], "Potentially truncated (touches data edge)",
                      "Complete episode"))
    fig = px.timeline(data, x_start="start_date", x_end="end_display", y="city", color="kind",
                      color_discrete_map={"Complete episode": PRIMARY,
                                          "Potentially truncated (touches data edge)": "#eb6834"},
                      custom_data=["length_days", "peak_pm2_5", "start_date", "end_date"])
    fig.update_traces(hovertemplate="<b>%{y}</b><br>%{customdata[2]} to %{customdata[3]}"
                                    "<br>%{customdata[0]} days, peak %{customdata[1]:.0f} µg/m³<extra></extra>")
    fig.update_layout(legend_title_text="")
    return style(fig, "", "", height=max(280, 28 * data["city"].nunique() + 100))
