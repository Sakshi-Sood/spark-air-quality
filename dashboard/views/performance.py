"""Page 5 - Spark performance: environment, data sizes, measured pipeline timings, experiments."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.charts import style
from dashboard.data_loader import (RERUN_HINT, load_performance_results,
                                   load_report, load_text)
from dashboard.labels import PRIMARY
from dashboard.logic import experiment_status

st.title("Spark Performance")
st.info("All processing ran in **Spark local mode on one laptop** (driver and executor threads "
        "in one JVM). This is parallel execution on one machine, not a multi-machine cluster.")

validation = load_report("validation_report")
processing = load_report("processing_report")
analytics = load_report("analytics_report")
if not (validation and processing and analytics):
    st.error("Pipeline reports are missing.")
    st.info(RERUN_HINT)
    st.stop()

# ---------------------------------------------------------------- environment
st.subheader("Execution environment (recorded by the pipeline)")
env = analytics["environment"]
st.dataframe(pd.DataFrame({"setting": list(env), "value": [str(v) for v in env.values()]}),
             hide_index=True, width="stretch")

# ---------------------------------------------------------------- dataset sizes
st.subheader("Dataset sizes")
left, right = st.columns(2, gap="large")
with left:
    raw = pd.DataFrame([{"file": d["file"]["path"], "rows": d.get("rows"),
                         "size_MB": d["file"]["bytes"] / 1e6, "spark_partitions": d.get("spark_input_partitions")}
                        for d in validation["datasets"].values()])
    st.markdown("**Raw CSV input**")
    st.dataframe(raw.round(2), hide_index=True, width="stretch")
with right:
    out = pd.DataFrame([{"table": name, "rows": t["rows"], "files": t["files"], "size_MB": t["bytes"] / 1e6}
                        for name, t in processing["outputs"].items()])
    st.markdown("**Processed Parquet (snappy-compressed)**")
    st.dataframe(out.round(2), hide_index=True, width="stretch")
st.caption("Parquet is columnar and compressed, so the same rows take much less space than CSV.")

# ---------------------------------------------------------------- measured pipeline timings
st.subheader("Measured pipeline timings (single runs)")
stages = {**processing.get("timings_seconds", {}), **analytics.get("timings_seconds", {})}
st.dataframe(pd.DataFrame({"stage": list(stages), "seconds": list(stages.values())}),
             hide_index=True, width="stretch")
st.caption("From the most recent `process` and `analytics` runs (separate Spark sessions, so each "
           "has its own startup time). These are not controlled experiments: repeated runs vary "
           "by roughly ±10 % or more, and Spark startup alone ranged from about 5 to 24 s.")
writes = analytics.get("table_write_seconds")
if writes:
    data = pd.Series(writes).sort_values()
    fig = go.Figure(go.Bar(x=data.values, y=data.index, orientation="h", marker_color=PRIMARY,
                           hovertemplate="%{y}: %{x:.1f} s<extra></extra>"))
    st.plotly_chart(style(fig, "Seconds (single run)", "", height=420, legend=False),
                    width="stretch")
    st.caption("Time of each analytical table's write action, which runs its whole plan. "
               "weather_correlations is slowest: it ranks 9 variables with window functions "
               "over the scope- and season-expanded data (about 0.45 million rows).")

# ---------------------------------------------------------------- experiments
st.subheader("Performance experiments (Phase 6)")
results = load_performance_results()
st.dataframe(experiment_status(results), hide_index=True, width="stretch")
if results is None:
    st.warning("No experiment has been run yet, so no experiment results are shown. "
               "Results will appear here after Phase 6.")

# ---------------------------------------------------------------- execution plan
st.subheader("Execution plan of the joined table")
plan = load_text("daily_joined_plan.txt")
if plan:
    st.caption("Saved with explain('formatted') during processing. Look for BroadcastHashJoin "
               "(small side copied to every task, no shuffle) and Exchange (a shuffle).")
    with st.expander("Show physical plan"):
        st.code(plan, language="text")
else:
    st.info("Plan file not found. " + RERUN_HINT)
