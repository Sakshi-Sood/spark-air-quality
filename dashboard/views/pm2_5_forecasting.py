"""Page: PM2.5 Forecasting · Next-Day Predictions & MLlib Model Evaluation.

Integrates Apache Spark MLlib regression models (Linear Regression with L2 regularization,
Decision Tree Regressor, Random Forest Regressor) compared against the Naïve Persistence
Baseline for next-day city-level PM2.5 forecasting.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.charts import (
    AXIS,
    GRID,
    INK,
    INK_2,
    PRIMARY,
    SERIES,
    style,
)
from dashboard.data_loader import (
    MissingOutputError,
    load_cities,
    load_daily,
    load_ml_comparison,
    load_ml_evaluation,
    load_ml_forecasts,
    load_ml_report,
    load_ml_test_predictions,
    stop_with_missing,
)
from dashboard.filters import get_filters
from dashboard.labels import NAAQS_24H
from dashboard.ui import (
    card,
    fact_row,
    fmt_date,
    kpi_row,
    number_columns,
    page_header,
    section,
    show_chart,
)

filters = get_filters()

try:
    comparison = load_ml_comparison()
    report = load_ml_report()
    forecasts = load_ml_forecasts()
    all_cities_df = load_cities()
except MissingOutputError as error:
    stop_with_missing(error)

if not comparison or not report:
    st.error("ML model outputs not found. Run: `python main.py --stage ml`")
    st.stop()

page_header(
    "Next-day PM2.5 forecasting with Apache Spark MLlib",
    "Chronological time-series regression across 50 Indian cities · Linear Regression, Decision Tree & Random Forest",
)

# ---------------------------------------------------------------- 1. Headline KPIs
best_key = comparison.get("best_model", "linear_regression")
best_display = report.get("selected_model_display", best_key.replace("_", " ").title())
split_meta = report.get("dataset_split", {}).get("splits", {})
train_meta = split_meta.get("train", {})
test_meta = split_meta.get("test", {})

val_metrics = report.get("model_comparison", [{}])
best_val_row = next((m for m in val_metrics if m.get("model_key") == best_key), {})
pers_val_row = next((m for m in val_metrics if m.get("model_key") == "persistence"), {})

val_rmse = best_val_row.get("val_rmse", 0.0)
val_mae = best_val_row.get("val_mae", 0.0)
val_r2 = best_val_row.get("val_r2", 0.0)
pers_rmse = pers_val_row.get("val_rmse", 1.0)
improvement = round(((pers_rmse - val_rmse) / pers_rmse) * 100, 1)

kpis = [
    ("Selected best model", best_display, "Validation RMSE criterion",
     f"Selected automatically from validation set performance: {comparison.get('justification', '')}"),
    ("Validation RMSE", f"{val_rmse:.2f} µg/m³", f"{improvement}% vs persistence",
     "Root Mean Squared Error on chronological validation period."),
    ("Validation MAE", f"{val_mae:.2f} µg/m³", f"R² = {val_r2:.4f}",
     "Mean Absolute Error and coefficient of determination."),
    ("Training window", f"{train_meta.get('num_dates', 0)} days",
     f"{fmt_date(train_meta.get('start_date', ''))} – {fmt_date(train_meta.get('end_date', ''))}",
     "Earliest 70% of chronological dates across all 50 cities (no lookahead leakage)."),
]
kpi_row(kpis, key="ml-kpi")

st.markdown(f"> **Model Selection Justification:** {comparison.get('justification', '')}")

# ---------------------------------------------------------------- 2. Forecast Interface
section("Next-day city forecast interface",
        "Generate and inspect next-day PM2.5 forecast for any city using the latest historical records.")

city_list = sorted(forecasts["city"].unique().tolist())
# Default to filter city if selected, otherwise Delhi
default_city = "Delhi" if "Delhi" in city_list else city_list[0]
if filters.cities and filters.cities[0] in city_list:
    default_city = filters.cities[0]

sel_city = st.selectbox("Select city to inspect forecast:", city_list, index=city_list.index(default_city))

city_fc = forecasts[forecasts["city"] == sel_city].iloc[0]
obs_date = city_fc["observation_date"]
fc_date = city_fc["forecast_date"]
latest_actual = city_fc["latest_actual_pm2_5"]
pred_pm2_5 = city_fc["predicted_pm2_5"]
expected_delta = city_fc["expected_change"]

with card("forecast-box"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Latest actual PM2.5", f"{latest_actual:.1f} µg/m³", f"Observed on {fmt_date(obs_date)}")
    with col2:
        st.metric("Forecasted PM2.5 (T+1)", f"{pred_pm2_5:.1f} µg/m³",
                  f"{expected_delta:+.1f} µg/m³ expected change",
                  delta_color="inverse")
    with col3:
        st.metric("Forecast date", fmt_date(fc_date), f"Model: {best_display}")
    with col4:
        naaqs_limit = NAAQS_24H["pm2_5_mean"]
        status = "Above NAAQS (60 µg/m³)" if pred_pm2_5 > naaqs_limit else "Within NAAQS limit"
        badge_color = ":red" if pred_pm2_5 > naaqs_limit else ":green"
        st.markdown(f"**NAAQS 24-h standard**  \n{badge_color}[{status}]")

# Load recent daily history for this city
recent_daily = load_daily(
    columns=("city", "date", "pm2_5_mean"),
    cities=(sel_city,),
    start=pd.Timestamp(obs_date).date() - pd.Timedelta(days=21),
    end=pd.Timestamp(obs_date).date(),
)

if not recent_daily.empty:
    recent_daily = recent_daily.sort_values("date")
    fig_trend = go.Figure()

    # Historical line
    fig_trend.add_trace(go.Scatter(
        x=recent_daily["date"],
        y=recent_daily["pm2_5_mean"],
        mode="lines+markers",
        name="Observed PM2.5",
        line=dict(color=PRIMARY, width=2.5),
        marker=dict(size=6, color=PRIMARY),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Observed: %{y:.1f} µg/m³<extra></extra>",
    ))

    # Forecast point
    fig_trend.add_trace(go.Scatter(
        x=[fc_date],
        y=[pred_pm2_5],
        mode="markers",
        name=f"Next-day forecast ({best_display})",
        marker=dict(size=14, color="#f59e0b", symbol="star", line=dict(color="#ffffff", width=1.5)),
        hovertemplate=f"<b>Forecast: %{{x|%d %b %Y}}</b><br>Predicted: %{{y:.1f}} µg/m³<extra></extra>",
    ))

    # NAAQS Reference Line
    fig_trend.add_hline(
        y=60.0, line_dash="dash", line_color="#ef4444",
        annotation_text="NAAQS 24-h limit (60 µg/m³)",
        annotation_position="bottom right",
    )

    style(fig_trend, x_title="Date", y_title="PM2.5 (µg/m³)", height=320)
    st.plotly_chart(fig_trend, use_container_width=True)

# ---------------------------------------------------------------- 3. Model Performance Comparison
section("Model performance & baseline comparison",
        "Rigorous comparison of the three Spark MLlib models against the persistence baseline on validation and untouched test sets.")

comp_table = pd.DataFrame(comparison["comparison_table"])
display_comp = comp_table[[
    "model_name", "val_mae", "val_rmse", "val_r2", "test_mae", "test_rmse", "test_r2",
]].rename(columns={
    "model_name": "Model",
    "val_mae": "Val MAE",
    "val_rmse": "Val RMSE",
    "val_r2": "Val R²",
    "test_mae": "Test MAE",
    "test_rmse": "Test RMSE",
    "test_r2": "Test R²",
})

col_tbl, col_bar = st.columns([1, 1])

with col_tbl:
    st.markdown("##### Performance Summary")
    st.dataframe(
        display_comp,
        column_config={
            "Val MAE": st.column_config.NumberColumn(format="%.2f"),
            "Val RMSE": st.column_config.NumberColumn(format="%.2f"),
            "Val R²": st.column_config.NumberColumn(format="%.4f"),
            "Test MAE": st.column_config.NumberColumn(format="%.2f"),
            "Test RMSE": st.column_config.NumberColumn(format="%.2f"),
            "Test R²": st.column_config.NumberColumn(format="%.4f"),
        },
        hide_index=True,
        use_container_width=True,
    )
    st.caption("Lower MAE & RMSE indicate higher accuracy. R² measures explained variance relative to mean.")

with col_bar:
    st.markdown("##### RMSE Comparison by Dataset")
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=comp_table["model_name"],
        y=comp_table["val_rmse"],
        name="Validation RMSE",
        marker_color=SERIES[0],
        text=[f"{v:.2f}" for v in comp_table["val_rmse"]],
        textposition="auto",
    ))
    fig_comp.add_trace(go.Bar(
        x=comp_table["model_name"],
        y=comp_table["test_rmse"],
        name="Test RMSE",
        marker_color=SERIES[1],
        text=[f"{v:.2f}" for v in comp_table["test_rmse"]],
        textposition="auto",
    ))
    style(fig_comp, x_title="Model", y_title="RMSE (µg/m³)", height=300)
    st.plotly_chart(fig_comp, use_container_width=True)

# ---------------------------------------------------------------- 4. Model Evaluation Visualizations
section("Model evaluation visualizations (untouched test period)",
        f"Evaluation of {best_display} predictions on the final 15% chronological test period ({fmt_date(test_meta.get('start_date', ''))} – {fmt_date(test_meta.get('end_date', ''))}).")

test_preds = load_ml_test_predictions()

col_scatter, col_resid = st.columns([1, 1])

with col_scatter:
    st.markdown("##### Actual vs Predicted PM2.5 (Scatter Plot)")
    # Sample 1500 points for crisp interactive rendering
    sample_df = test_preds.sample(n=min(1500, len(test_preds)), random_state=42)
    max_val = max(sample_df["actual_target_pm2_5"].max(), sample_df["predicted_pm2_5"].max())

    fig_scat = go.Figure()
    fig_scat.add_trace(go.Scatter(
        x=sample_df["actual_target_pm2_5"],
        y=sample_df["predicted_pm2_5"],
        mode="markers",
        marker=dict(size=5, color=PRIMARY, opacity=0.6),
        name="Test Observations",
        hovertemplate="<b>%{text}</b><br>Actual: %{x:.1f}<br>Predicted: %{y:.1f}<extra></extra>",
        text=sample_df["city"],
    ))
    # 1:1 Reference Line
    fig_scat.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines", name="Ideal 1:1 fit",
        line=dict(color="#ef4444", dash="dash", width=1.5),
    ))
    style(fig_scat, x_title="Actual PM2.5 (µg/m³)", y_title="Predicted PM2.5 (µg/m³)", height=340)
    st.plotly_chart(fig_scat, use_container_width=True)

with col_resid:
    st.markdown("##### Residual Distribution (Error = Prediction − Actual)")
    residuals = test_preds["residual"].dropna()
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=residuals,
        nbinsx=50,
        marker_color=SERIES[2],
        name="Residuals",
        hovertemplate="Residual: %{x:.1f} µg/m³<br>Count: %{y}<extra></extra>",
    ))
    fig_hist.add_vline(x=0.0, line_dash="dash", line_color="#ffffff", annotation_text="Zero error")
    style(fig_hist, x_title="Residual Error (µg/m³)", y_title="Count", height=340, legend=False)
    st.plotly_chart(fig_hist, use_container_width=True)

# Actual vs Predicted Time-Series Plot for Selected City
st.markdown(f"##### Test Period Time-Series: Actual vs Predicted PM2.5 for {sel_city}")
city_test_preds = test_preds[test_preds["city"] == sel_city].sort_values("forecast_date")

if not city_test_preds.empty:
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=city_test_preds["forecast_date"],
        y=city_test_preds["actual_target_pm2_5"],
        mode="lines",
        name="Actual PM2.5",
        line=dict(color=PRIMARY, width=2),
    ))
    fig_ts.add_trace(go.Scatter(
        x=city_test_preds["forecast_date"],
        y=city_test_preds["predicted_pm2_5"],
        mode="lines",
        name=f"Predicted ({best_display})",
        line=dict(color="#f59e0b", width=2, dash="dot"),
    ))
    fig_ts.add_trace(go.Scatter(
        x=city_test_preds["forecast_date"],
        y=city_test_preds["persistence_pred"],
        mode="lines",
        name="Persistence Baseline",
        line=dict(color=INK_2, width=1, dash="dash"),
    ))
    style(fig_ts, x_title="Forecast Date", y_title="PM2.5 (µg/m³)", height=320)
    st.plotly_chart(fig_ts, use_container_width=True)

# ---------------------------------------------------------------- 5. Feature Importance & Coefficients
section("Model interpretability & feature weights",
        "Learned feature weights from the trained Spark MLlib pipeline.")

feature_weights = report.get("feature_importances", [])
if feature_weights:
    feat_df = pd.DataFrame(feature_weights).head(12)
    # Determine if tree importance or linear coefficient
    val_col = "coefficient" if "coefficient" in feat_df.columns else "importance"
    feat_df = feat_df.sort_values(by=val_col, ascending=True)

    fig_feat = go.Figure()
    fig_feat.add_trace(go.Bar(
        x=feat_df[val_col],
        y=feat_df["feature"],
        orientation="h",
        marker_color=PRIMARY,
        hovertemplate="<b>%{y}</b><br>Weight: %{x:.4f}<extra></extra>",
    ))
    style(fig_feat, x_title="Learned Weight / Coefficient", y_title="Feature", height=350, legend=False)
    st.plotly_chart(fig_feat, use_container_width=True)

# ---------------------------------------------------------------- 6. Model Metadata and Training Audit
section("Training audit & reproducibility details",
        "Model specifications, date split boundaries, and data leakage safeguards.")

with card("audit-card"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**Training Set (70%)**  \n"
                    f"• {fmt_date(train_meta.get('start_date', ''))} to {fmt_date(train_meta.get('end_date', ''))}  \n"
                    f"• {train_meta.get('num_dates', 0)} dates ({train_meta.get('num_dates', 0) * 50:,} city-days)")
    with c2:
        val_meta = split_meta.get("validation", {})
        st.markdown(f"**Validation Set (15%)**  \n"
                    f"• {fmt_date(val_meta.get('start_date', ''))} to {fmt_date(val_meta.get('end_date', ''))}  \n"
                    f"• {val_meta.get('num_dates', 0)} dates ({val_meta.get('num_dates', 0) * 50:,} city-days)")
    with c3:
        st.markdown(f"**Test Set (15%)**  \n"
                    f"• {fmt_date(test_meta.get('start_date', ''))} to {fmt_date(test_meta.get('end_date', ''))}  \n"
                    f"• {test_meta.get('num_dates', 0)} dates ({test_meta.get('num_dates', 0) * 50:,} city-days)")

    st.markdown("---")
    st.markdown(
        f"**Leakage Safeguards:**  \n"
        f"• **Target Isolation:** Target $PM2.5(T+1)$ is strictly excluded from feature computation.  \n"
        f"• **Rolling Windows:** 3-day and 7-day averages terminate at day $T$, never referencing future rows.  \n"
        f"• **City Partitioning:** Spark window functions are strictly partitioned by city, preventing cross-city data bleed.  \n"
        f"• **Preprocessing Isolation:** StringIndexer and OneHotEncoder fit solely on the training partition.  \n"
        f"• **Untouched Test Set:** Test partition evaluated only once after final model selection."
    )
