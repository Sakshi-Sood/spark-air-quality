"""Pipeline entry point.

Usage (from the project folder, with the venv's Python):
    python main.py --stage validate    # validate raw CSVs -> validation_report.json
    python main.py --stage process     # validate, then clean + join -> outputs/processed/
    python main.py --stage analytics   # analytical tables from processed data -> outputs/analytics/
    python main.py --stage ml          # train, evaluate, and save ML models & forecasts -> outputs/ml/
    python main.py --stage all         # validate + process + analytics + ml
"""
import argparse
import sys
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.analytics import (aqi_category_distribution, annual_city,
                           city_pollutant_stats, complete_years,
                           dataset_summary, group_pollutant_stats,
                           monthly_city, pollution_episodes, rdd_city_means,
                           seasonal_city, to_long, top_pollution_days)
from src.config import (ANALYTICS_DIR, DAILY_KEY, PLANS_DIR, PROCESSED_DIR,
                        REPORTS_DIR)
from src.data_validation import count_nulls, validate_all
from src.joins import (attach_city_metadata, attach_season_completeness,
                       join_air_quality_weather)
from src.ml.feature_engineering import (
    create_forecasting_features,
    split_time_series_data,
    validate_and_filter_ml_data,
)
from src.ml.forecasting import generate_batch_forecasts
from src.ml.ml_config import (
    FORECAST_DATE_COL,
    ML_PATHS,
    PERSISTENCE_COL,
    TARGET_COL,
)
from src.ml.model_evaluation import evaluate_models_and_baseline
from src.ml.model_selection import save_ml_artifacts, select_best_model
from src.ml.model_training import train_all_models
from src.preprocessing import (preprocess_air_quality, preprocess_cities,
                               preprocess_weather, season_coverage)
from src.schemas import read_raw_csv
from src.spark_session import create_spark_session, describe_environment
from src.utils import (explain_to_text, get_logger, parquet_file_stats,
                       write_json, write_parquet)
from src.weather_analysis import (explode_scopes, weather_bin_stats,
                                  weather_correlations)

log = get_logger("main")


def run_validation(spark: SparkSession) -> dict:
    """Validate the raw CSVs, write the report, and stop on critical failures."""
    report = validate_all(spark)
    path = REPORTS_DIR / "validation_report.json"
    write_json(report, path)
    summary = report["summary"]
    log.info("Validation: %d/%d checks passed -> %s",
             summary["passed"], summary["checks_run"], path)
    for level in ("warnings", "info"):
        for item in summary[level]:
            log.warning("%s: %s", level, item)
    if summary["critical_failures"]:
        for item in summary["critical_failures"]:
            log.error("critical: %s", item)
        sys.exit("Validation found critical problems; see the report. Processing stopped.")
    return report


def run_processing(spark: SparkSession) -> dict:
    """Clean each dataset, save it, then build and save the joined daily table."""
    raw = {name: read_raw_csv(spark, name) for name in ("air_quality", "weather", "cities")}
    aq, aq_stats = preprocess_air_quality(raw["air_quality"])
    weather, weather_stats = preprocess_weather(raw["weather"])
    cities = preprocess_cities(raw["cities"])

    write_parquet(aq, PROCESSED_DIR / "air_quality", sort_cols=DAILY_KEY)
    write_parquet(weather, PROCESSED_DIR / "weather", sort_cols=DAILY_KEY)
    write_parquet(cities, PROCESSED_DIR / "cities", sort_cols=["city"])

    # Continue from the Parquet copies: the plan no longer re-parses CSV text,
    # and Spark reads only the columns each step needs (column pruning).
    aq = spark.read.parquet((PROCESSED_DIR / "air_quality").as_posix())
    weather = spark.read.parquet((PROCESSED_DIR / "weather").as_posix())
    cities = spark.read.parquet((PROCESSED_DIR / "cities").as_posix())

    joined, join_stats = join_air_quality_weather(aq, weather)
    # cache(): the enriched table feeds several actions below (counts, the
    # coverage aggregation, the write); without it each action would redo
    # the join from the Parquet files.
    daily = attach_city_metadata(joined, cities).cache()
    coverage = season_coverage(daily)
    daily = attach_season_completeness(daily, coverage)
    leading = [*DAILY_KEY, "year", "month", "season"]
    daily = daily.select(*leading, *[c for c in daily.columns if c not in leading])

    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    (PLANS_DIR / "daily_joined_plan.txt").write_text(explain_to_text(daily), encoding="utf-8")

    write_parquet(coverage, PROCESSED_DIR / "season_coverage", sort_cols=["city", "year", "season"])
    write_parquet(daily, PROCESSED_DIR / "daily_joined", sort_cols=DAILY_KEY)
    daily.unpersist()

    return build_processing_report(spark, aq_stats, weather_stats, join_stats)


def build_processing_report(spark, aq_stats, weather_stats, join_stats) -> dict:
    """Re-read every output from disk and record counts, files and schema."""
    outputs = {}
    for table in ("air_quality", "weather", "cities", "season_coverage", "daily_joined"):
        path = PROCESSED_DIR / table
        df = spark.read.parquet(path.as_posix())
        outputs[table] = {"rows": df.count(), "columns": len(df.columns),
                          **parquet_file_stats(path)}

    daily = spark.read.parquet((PROCESSED_DIR / "daily_joined").as_posix())
    coverage = spark.read.parquet((PROCESSED_DIR / "season_coverage").as_posix())
    # At most a few dozen rows: safe to collect.
    partial = (coverage.filter(~coverage.season_complete)
               .groupBy("year", "season", "observed_days", "expected_days")
               .count().withColumnRenamed("count", "cities")
               .orderBy("year", "season").collect())
    return {
        "preprocessing": {"air_quality": aq_stats, "weather": weather_stats},
        "joins": join_stats,
        "outputs": outputs,
        "daily_joined": {
            "schema": [{"name": f.name, "type": f.dataType.simpleString()}
                       for f in daily.schema.fields],
            "provisional_rows": daily.filter("is_provisional").count(),
            "rows_in_incomplete_seasons": daily.filter("NOT season_complete").count(),
            "null_region_rows": daily.filter("region IS NULL").count(),
        },
        "incomplete_seasons": [r.asDict() for r in partial],
    }


def run_analytics(spark: SparkSession) -> dict:
    """Build every analytical table from the processed Parquet outputs."""
    daily_path = PROCESSED_DIR / "daily_joined"
    if not daily_path.exists():
        sys.exit(f"{daily_path} not found. Run: python main.py --stage process")

    # cache(): these three DataFrames each feed several output tables.
    daily = spark.read.parquet(daily_path.as_posix()).cache()
    long = to_long(daily).cache()
    scoped = explode_scopes(daily).cache()
    weather = spark.read.parquet((PROCESSED_DIR / "weather").as_posix())
    air_quality_rows = spark.read.parquet((PROCESSED_DIR / "air_quality").as_posix()).count()
    years = complete_years(daily)
    log.info("Complete years (every city, every day): %s", years)

    tables = {
        "dataset_summary": dataset_summary(spark, daily, air_quality_rows, weather, years),
        "city_pollutant_stats": city_pollutant_stats(long),
        "region_pollutant_stats": group_pollutant_stats(spark, long, ["region"]),
        "state_pollutant_stats": group_pollutant_stats(spark, long, ["state"]),
        "aqi_category_distribution": aqi_category_distribution(daily),
        "monthly_city": monthly_city(daily),
        "annual_city": annual_city(long, years),
        "seasonal_city": seasonal_city(long),
        "weather_correlations": weather_correlations(scoped),
        "weather_bin_stats": weather_bin_stats(scoped),
        "pollution_episodes": pollution_episodes(daily),
        "top_pollution_days": top_pollution_days(daily),
        "rdd_city_means": rdd_city_means(daily),
    }
    # Each write is the action that runs that table's plan, so its time also
    # includes filling a cache the first time a cached input is used.
    write_seconds = {}
    for name, df in tables.items():
        t0 = time.perf_counter()
        write_parquet(df, ANALYTICS_DIR / name)
        write_seconds[name] = round(time.perf_counter() - t0, 2)
        log.info("wrote %-26s in %5.1f s", name, write_seconds[name])
    for df in (daily, long, scoped):
        df.unpersist()
    report = build_analytics_report(spark, list(tables), years)
    report["table_write_seconds"] = write_seconds
    return report


def build_analytics_report(spark: SparkSession, names: list[str], years: list[int]) -> dict:
    """Re-read each analytical table and record rows, schema, nulls and files."""
    tables = {}
    for name in names:
        path = ANALYTICS_DIR / name
        df = spark.read.parquet(path.as_posix())
        nulls = count_nulls(df, df.columns)
        tables[name] = {
            "rows": df.count(),
            "schema": [f"{f.name}:{f.dataType.simpleString()}" for f in df.schema.fields],
            "columns_with_nulls": {c: n for c, n in nulls.items() if n},
            **parquet_file_stats(path),
        }
    rdd = spark.read.parquet((ANALYTICS_DIR / "rdd_city_means").as_posix())
    check = rdd.agg(F.count("*").alias("cities"), F.max("abs_diff").alias("max_abs_diff"),
                    F.sum((~F.col("within_tolerance")).cast("int")).alias("failures")).first()
    if check.failures:
        sys.exit(f"RDD and DataFrame city means disagree for {check.failures} cities")
    return {"complete_years": years, "tables": tables, "rdd_check": check.asDict()}


def run_ml(spark: SparkSession) -> dict:
    """Train MLlib regression models, evaluate against persistence baseline, and save artifacts."""
    daily_path = PROCESSED_DIR / "daily_joined"
    if not daily_path.exists():
        sys.exit(f"{daily_path} not found. Run: python main.py --stage process")

    log.info("Loading daily processed data for ML from %s...", daily_path)
    daily = spark.read.parquet(daily_path.as_posix())

    # 1. Feature Engineering
    log.info("Engineering lag, rolling, and temporal features...")
    features_df = create_forecasting_features(daily)
    valid_ml_df = validate_and_filter_ml_data(features_df, require_target=True).cache()

    # 2. Chronological Splitting
    log.info("Performing chronological time-series splitting...")
    train_df, val_df, test_df, split_metadata = split_time_series_data(valid_ml_df)
    train_df = train_df.cache()
    val_df = val_df.cache()
    test_df = test_df.cache()

    # 3. Model Training
    log.info("Training Linear Regression, Decision Tree, and Random Forest...")
    models = train_all_models(train_df)

    # 4. Validation Evaluation
    log.info("Evaluating models and persistence baseline on validation set...")
    val_results = evaluate_models_and_baseline(models, val_df, dataset_name="validation")

    # 5. Model Selection
    best_key, justification = select_best_model(val_results)

    # 6. Final Evaluation on Untouched Test Period
    log.info("Evaluating all models on untouched chronological test set...")
    test_results = evaluate_models_and_baseline(models, test_df, dataset_name="test")

    # 7. Generate Test Predictions for Best Model
    best_test_preds = models[best_key].transform(test_df)
    test_predictions_df = (
        best_test_preds
        .select(
            "city",
            "date",
            F.col(FORECAST_DATE_COL).alias("forecast_date"),
            F.round(F.col(TARGET_COL), 2).alias("actual_target_pm2_5"),
            F.round(F.col(PERSISTENCE_COL), 2).alias("persistence_pred"),
            F.round(F.col("prediction"), 2).alias("predicted_pm2_5"),
            F.round(F.col("prediction") - F.col(TARGET_COL), 2).alias("residual"),
            "region",
            "state",
        )
    )

    # 8. Persist Models, Metrics, Predictions, and Report
    log.info("Persisting ML models and reports to outputs/ml/...")
    report = save_ml_artifacts(
        models=models,
        best_key=best_key,
        val_results=val_results,
        test_results=test_results,
        test_predictions_df=test_predictions_df,
        split_metadata=split_metadata,
        justification=justification,
    )

    # 9. Batch Forecasting for All Cities (for Tomorrow)
    log.info("Generating next-day forecasts for all cities...")
    generate_batch_forecasts(
        model=models[best_key],
        daily_df=daily,
        model_name=val_results[best_key]["model_name"],
    )

    # Cleanup caches
    for df in (valid_ml_df, train_df, val_df, test_df):
        df.unpersist()

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Air quality and weather Spark pipeline")
    parser.add_argument("--stage", choices=["validate", "process", "analytics", "ml", "all"],
                        default="all", help="'process' and 'all' always run validation first")
    args = parser.parse_args()

    start = time.perf_counter()
    spark = create_spark_session()
    startup_s = time.perf_counter() - start
    log.info("Spark %s started in %.1f s (UI: %s)", spark.version, startup_s,
             spark.sparkContext.uiWebUrl)
    timings = {"spark_startup_s": startup_s}
    try:
        if args.stage in ("validate", "process", "all"):
            t0 = time.perf_counter()
            run_validation(spark)
            timings["validation_s"] = time.perf_counter() - t0

        if args.stage in ("process", "all"):
            t0 = time.perf_counter()
            report = run_processing(spark)
            timings["processing_s"] = time.perf_counter() - t0
            report["timings_seconds"] = {k: round(v, 2) for k, v in timings.items()}
            report["environment"] = describe_environment(spark)
            write_json(report, REPORTS_DIR / "processing_report.json")
            for table, info in report["outputs"].items():
                log.info("%-16s %7d rows  %d file(s)  %6.2f MB", table, info["rows"],
                         info["files"], info["bytes"] / 1e6)
            log.info("Joins: %s", report["joins"])

        if args.stage in ("analytics", "all"):
            t0 = time.perf_counter()
            report = run_analytics(spark)
            timings["analytics_s"] = time.perf_counter() - t0
            report["timings_seconds"] = {k: round(v, 2) for k, v in timings.items()}
            report["environment"] = describe_environment(spark)
            write_json(report, REPORTS_DIR / "analytics_report.json")
            for table, info in report["tables"].items():
                log.info("%-26s %7d rows  %d file(s)", table, info["rows"], info["files"])
            log.info("RDD vs DataFrame check: %s", report["rdd_check"])

        if args.stage in ("ml", "all"):
            t0 = time.perf_counter()
            report = run_ml(spark)
            timings["ml_s"] = time.perf_counter() - t0
            log.info("ML stage complete. Best model: %s", report["selected_model"])

        log.info("Timings: %s", {k: round(v, 1) for k, v in timings.items()})
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
