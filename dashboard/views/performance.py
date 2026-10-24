"""Page 5 - Spark performance: environment, data sizes, measured baseline timings, experiments."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.charts import experiment_comparison, style
from dashboard.data_loader import (RERUN_HINT, load_performance_results,
                                   load_report, load_text)
from dashboard.labels import PRIMARY
from dashboard.logic import EXPERIMENTS, experiment_status
from dashboard.ui import (card, kpi_row, number_columns, page_header, section,
                          show_chart, table_height)

page_header("Spark performance",
            "How the pipeline ran · measured values only; experiments not yet run are marked Pending")
st.caption(":material/info: All processing ran in **Spark local mode on one laptop** (driver and "
           "executor threads in one JVM): parallel execution on one machine, not a multi-machine cluster.")

validation = load_report("validation_report")
processing = load_report("processing_report")
analytics = load_report("analytics_report")
if not (validation and processing and analytics):
    st.error("Pipeline reports are missing.")
    st.info(RERUN_HINT)
    st.stop()

# ---------------------------------------------------------------- environment
env = analytics["environment"]
section("Execution environment", "Recorded by the pipeline at run time")
kpi_row([
    ("Apache Spark", env["spark_version"], f"Scala {env['scala_version']}"),
    ("Java", env["java_version"], f"Python {env['python_version']}"),
    ("Master", env["master"], f"{env['default_parallelism']} parallel tasks"),
    ("Shuffle partitions", str(env["shuffle_partitions"]),
     f"AQE {'on' if str(env['adaptive_query_execution']).lower() == 'true' else 'off'}"),
], key="perf-env")
with st.expander("All recorded settings"):
    st.dataframe(pd.DataFrame({"setting": list(env), "value": [str(v) for v in env.values()]}),
                 hide_index=True, width="stretch", height=table_height(len(env)))

# ---------------------------------------------------------------- dataset sizes
left, right = st.columns(2, gap="medium")
with left, card("perf-raw", height="stretch"):
    section("Raw CSV input", "Rows, size and the partitions Spark created when reading")
    raw = pd.DataFrame([{"file": d["file"]["path"].replace("\\", "/"), "rows": d.get("rows"),
                         "size_mb": d["file"]["bytes"] / 1e6, "partitions": d.get("spark_input_partitions")}
                        for d in validation["datasets"].values()])
    st.dataframe(raw, hide_index=True, width="stretch",
                 column_config={"file": st.column_config.TextColumn("File"),
                                "rows": st.column_config.NumberColumn("Rows", format="localized"),
                                "partitions": st.column_config.NumberColumn("Partitions", format="%d"),
                                **number_columns(["size_mb"], "%.2f", {"size_mb": "Size, MB"})})
with right, card("perf-parquet", height="stretch"):
    section("Processed Parquet", "Snappy-compressed, one file per table")
    out = pd.DataFrame([{"table": name, "rows": t["rows"], "files": t["files"], "size_mb": t["bytes"] / 1e6}
                        for name, t in processing["outputs"].items()])
    st.dataframe(out, hide_index=True, width="stretch",
                 column_config={"table": st.column_config.TextColumn("Table"),
                                "rows": st.column_config.NumberColumn("Rows", format="localized"),
                                "files": st.column_config.NumberColumn("Files", format="%d"),
                                **number_columns(["size_mb"], "%.2f", {"size_mb": "Size, MB"})})

# ---------------------------------------------------------------- measured baseline timings
with card("perf-baseline"):
    section("Baseline pipeline timings",
            "Measured, single runs · not controlled experiments (repeated runs vary by about ±10 %, "
            "Spark startup alone ranged from about 5 to 24 s)")
    stages = {**processing.get("timings_seconds", {}), **analytics.get("timings_seconds", {})}
    names = {"spark_startup_s": "Spark startup", "validation_s": "Validation",
             "processing_s": "Cleaning + joins", "analytics_s": "Analytics"}
    left, right = st.columns([1, 2], gap="medium")
    with left:
        st.dataframe(pd.DataFrame({"stage": [names.get(k, k) for k in stages], "seconds": list(stages.values())}),
                     hide_index=True, width="stretch",
                     column_config={"stage": st.column_config.TextColumn("Stage"),
                                    **number_columns(["seconds"], "%.1f", {"seconds": "Seconds"})})
        st.caption("From the latest `process` and `analytics` runs, each in its own Spark session.")
    writes = analytics.get("table_write_seconds")
    with right:
        if writes:
            data = pd.Series(writes).sort_values()
            fig = go.Figure(go.Bar(x=data.values, y=data.index, orientation="h", marker_color=PRIMARY,
                                   hovertemplate="%{y}: %{x:.1f} s<extra></extra>"))
            fig.update_layout(bargap=0.35)
            show_chart(style(fig, "Seconds to write each analytical table (single run)", "",
                             height=380, legend=False))
            st.caption("Each write is the action that runs that table's whole plan. "
                       "weather_correlations is slowest: it ranks 9 variables with window functions "
                       "over about 0.45 million scope- and season-expanded rows.")

# ---------------------------------------------------------------- experiments
with card("perf-experiments"):
    results = load_performance_results()
    status = experiment_status(results)
    measured = int((status["status"] == "Measured").sum())
    section("Performance experiments (Phase 6)",
            f"{measured} of {len(status)} measured · charts appear only for measured experiments")
    st.dataframe(status, hide_index=True, width="stretch", height=table_height(len(status)),
                 column_config={"experiment": st.column_config.TextColumn("ID", width="small"),
                                "description": st.column_config.TextColumn("Experiment", width="large"),
                                "status": st.column_config.TextColumn("Status", width="small")})
    if results is None:
        st.caption(":material/schedule: No experiment has been run yet, so no results are shown.")
    else:
        for experiment in [e for e in EXPERIMENTS if e in set(results["experiment"])]:
            section(f"{experiment} · {EXPERIMENTS[experiment]}")
            show_chart(experiment_comparison(results, experiment), key=f"exp-{experiment}")

# ---------------------------------------------------------------- execution plan
with st.expander("Physical plan of the joined table (explain, formatted)"):
    plan = load_text("daily_joined_plan.txt")
    if plan:
        st.caption("Look for BroadcastHashJoin (small table copied to every task, no shuffle) and "
                   "Exchange (a shuffle between stages).")
        st.code(plan, language="text")
    else:
        st.info("Plan file not found. " + RERUN_HINT)
