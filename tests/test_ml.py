"""Unit tests for MLlib PM2.5 forecasting module.

Covers:
- Target and lag feature generation
- Rolling window calculations and future leakage prevention
- City-wise partition isolation
- Chronological splitting with disjoint date ranges
- Spark MLlib pipeline training, evaluation metrics, and model persistence
- Reusable forecasting function and input validation
- Dashboard loader integration
"""
from datetime import date, timedelta
import math
import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from src.ml.feature_engineering import (
    create_forecasting_features,
    split_time_series_data,
    validate_and_filter_ml_data,
)
from src.ml.forecasting import forecast_city_next_day
from src.ml.ml_config import TARGET_COL
from src.ml.model_evaluation import evaluate_predictions
from src.ml.model_training import train_linear_regression
from dashboard.data_loader import (
    load_ml_comparison,
    load_ml_forecasts,
    load_ml_report,
    load_ml_test_predictions,
)

SYNTHETIC_SCHEMA = StructType([
    StructField("city", StringType(), False),
    StructField("date", DateType(), False),
    StructField("year", IntegerType(), False),
    StructField("month", IntegerType(), False),
    StructField("season", StringType(), False),
    StructField("pm2_5_mean", DoubleType(), False),
    StructField("pm10_mean", DoubleType(), False),
    StructField("no2_mean", DoubleType(), False),
    StructField("so2_mean", DoubleType(), False),
    StructField("co_8h_max", DoubleType(), False),
    StructField("temperature_2m_mean", DoubleType(), False),
    StructField("relative_humidity_2m_mean", IntegerType(), False),
    StructField("wind_speed_10m_max", DoubleType(), False),
    StructField("precipitation_sum", DoubleType(), False),
    StructField("state", StringType(), False),
    StructField("region", StringType(), False),
])


def _create_synthetic_daily(spark, n_days=15, cities=("Delhi", "Mumbai")):
    """Generate deterministic synthetic daily air quality and weather data."""
    start_date = date(2024, 1, 1)
    rows = []
    for city_idx, city in enumerate(cities):
        for i in range(n_days):
            cur_date = start_date + timedelta(days=i)
            # Deterministic linear values for easy verification
            base_pm = 50.0 + (city_idx * 20.0) + float(i * 2)
            rows.append((
                city,
                cur_date,
                cur_date.year,
                cur_date.month,
                "Winter",
                base_pm,
                base_pm * 1.5,
                25.0 + float(i),
                10.0,
                1.2,
                18.0 + float(i * 0.5),
                60,
                12.0,
                0.0,
                "StateX",
                "North" if city_idx == 0 else "West",
            ))
    return spark.createDataFrame(rows, SYNTHETIC_SCHEMA)


def test_target_generation_and_city_isolation(spark):
    """Verify target_pm2_5 is day T+1's pm2_5_mean and cities never bleed into each other."""
    df = _create_synthetic_daily(spark, n_days=10, cities=("CityA", "CityB"))
    features_df = create_forecasting_features(df)

    rows_a = features_df.filter("city = 'CityA'").orderBy("date").collect()
    rows_b = features_df.filter("city = 'CityB'").orderBy("date").collect()

    # Day 0 target should be Day 1's actual PM2.5
    assert rows_a[0]["target_pm2_5"] == rows_a[1]["pm2_5_lag1"]
    assert rows_b[0]["target_pm2_5"] == rows_b[1]["pm2_5_lag1"]

    # Day 0 lag 1 should be Day 0 PM2.5
    assert rows_a[0]["pm2_5_lag1"] == rows_a[0]["pm2_5_mean"]
    assert rows_b[0]["pm2_5_lag1"] == rows_b[0]["pm2_5_mean"]

    # City B values should not bleed into City A
    assert rows_a[0]["pm2_5_lag1"] != rows_b[0]["pm2_5_lag1"]

    # The last row has no tomorrow, so target must be None
    assert rows_a[-1]["target_pm2_5"] is None
    assert rows_b[-1]["target_pm2_5"] is None


def test_rolling_averages_and_lags(spark):
    """Verify lag 1, lag 2, lag 7, rolling 3d, and rolling 7d calculations."""
    df = _create_synthetic_daily(spark, n_days=12, cities=("CityA",))
    features_df = create_forecasting_features(df).orderBy("date").collect()

    # Row 7 (index 7, day 8):
    # Has full 7 lags:
    # pm2_5_mean values: 50, 52, 54, 56, 58, 60, 62, 64
    r7 = features_df[7]
    assert r7["pm2_5_lag1"] == 64.0  # Day T (current observation)
    assert r7["pm2_5_lag2"] == 62.0  # Day T-1
    assert r7["pm2_5_lag7"] == 52.0  # Day T-6

    # Rolling 3-day: average of (60, 62, 64) = 62.0
    assert abs(r7["pm2_5_rolling_3d"] - 62.0) < 1e-4

    # Rolling 7-day: average of (52, 54, 56, 58, 60, 62, 64) = 58.0
    assert abs(r7["pm2_5_rolling_7d"] - 58.0) < 1e-4


def test_no_future_data_leakage(spark):
    """Verify that altering day T+1 values does NOT affect day T engineered features."""
    df1 = _create_synthetic_daily(spark, n_days=8, cities=("CityA",))
    f1 = create_forecasting_features(df1).filter("date = '2024-01-05'").collect()[0]

    # Modify day 2024-01-06 (day T+1) significantly
    modified_rows = []
    for r in df1.collect():
        row_dict = r.asDict()
        if str(r.date) == "2024-01-06":
            row_dict["pm2_5_mean"] = 999.0
            row_dict["temperature_2m_mean"] = 99.0
        modified_rows.append(tuple(row_dict[f.name] for f in SYNTHETIC_SCHEMA.fields))

    df2 = spark.createDataFrame(modified_rows, SYNTHETIC_SCHEMA)
    f2 = create_forecasting_features(df2).filter("date = '2024-01-05'").collect()[0]

    # Input features at day T MUST remain identical
    assert f1["pm2_5_lag1"] == f2["pm2_5_lag1"]
    assert f1["pm2_5_rolling_3d"] == f2["pm2_5_rolling_3d"]
    assert f1["pm2_5_rolling_7d"] == f2["pm2_5_rolling_7d"]
    assert f1["temperature_lag1"] == f2["temperature_lag1"]

    # Only target_pm2_5 (which is day T+1 ground truth) should reflect the change
    assert f1["target_pm2_5"] != f2["target_pm2_5"]
    assert f2["target_pm2_5"] == 999.0


def test_chronological_splitting_no_overlap(spark):
    """Verify train, val, and test partitions have zero date overlap and correct order."""
    df = _create_synthetic_daily(spark, n_days=30, cities=("CityA", "CityB"))
    features = create_forecasting_features(df)
    valid = validate_and_filter_ml_data(features, require_target=True)

    train_df, val_df, test_df, meta = split_time_series_data(
        valid, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20
    )

    train_max = train_df.agg(F.max("date")).collect()[0][0]
    val_min = val_df.agg(F.min("date")).collect()[0][0]
    val_max = val_df.agg(F.max("date")).collect()[0][0]
    test_min = test_df.agg(F.min("date")).collect()[0][0]

    assert train_max < val_min, f"Train overlap with Val: {train_max} >= {val_min}"
    assert val_max < test_min, f"Val overlap with Test: {val_max} >= {test_min}"

    # Confirm across both cities dates are synchronously assigned
    train_cities = [r[0] for r in train_df.select("city").distinct().collect()]
    assert set(train_cities) == {"CityA", "CityB"}


def test_missing_feature_and_warmup_filtering(spark):
    """Verify warmup period without 7-day lags is cleanly filtered for training."""
    df = _create_synthetic_daily(spark, n_days=10, cities=("CityA",))
    features = create_forecasting_features(df)

    # For 10 days:
    # Days 0..5 (first 6 days) have lag7 = None
    # Day 9 has target = None
    # Usable days with complete features & target = 10 - 6 - 1 = 3 days
    valid_train = validate_and_filter_ml_data(features, require_target=True)
    assert valid_train.count() == 3

    # For live forecasting (require_target=False), day 9 is retained because lags are present
    valid_forecast = validate_and_filter_ml_data(features, require_target=False)
    assert valid_forecast.count() == 4


def test_model_training_and_evaluation_metrics(spark):
    """Verify PySpark pipeline training and MAE, RMSE, R² calculations."""
    df = _create_synthetic_daily(spark, n_days=25, cities=("CityA", "CityB"))
    features = create_forecasting_features(df)
    valid = validate_and_filter_ml_data(features, require_target=True)

    train_df, val_df, _, _ = split_time_series_data(valid, 0.7, 0.15, 0.15)

    model = train_linear_regression(train_df)
    preds = model.transform(val_df)

    metrics = evaluate_predictions(preds, prediction_col="prediction")

    assert metrics["mae"] >= 0.0
    assert metrics["rmse"] >= metrics["mae"]  # Mathematical property: RMSE >= MAE
    assert -1.0 <= metrics["r2"] <= 1.0


def test_model_persistence_and_reloading(spark, tmp_path):
    """Verify fitted pipeline can be saved to disk and loaded without loss of predictions."""
    df = _create_synthetic_daily(spark, n_days=20, cities=("CityA",))
    features = create_forecasting_features(df)
    valid = validate_and_filter_ml_data(features, require_target=True)

    train_df, val_df, _, _ = split_time_series_data(valid, 0.7, 0.15, 0.15)
    model = train_linear_regression(train_df)

    preds_before = [r["prediction"] for r in model.transform(val_df).select("prediction").collect()]

    save_path = tmp_path / "test_pipeline_model"
    model.write().overwrite().save(save_path.as_posix())

    from pyspark.ml import PipelineModel
    loaded_model = PipelineModel.load(save_path.as_posix())
    preds_after = [r["prediction"] for r in loaded_model.transform(val_df).select("prediction").collect()]

    for b, a in zip(preds_before, preds_after):
        assert abs(b - a) < 1e-5


def test_forecast_city_function(spark):
    """Verify forecast_city_next_day behaves correctly with sufficient and insufficient data."""
    # 1. Sufficient data (>= 7 days)
    df_sufficient = _create_synthetic_daily(spark, n_days=10, cities=("Delhi",))
    features = create_forecasting_features(df_sufficient)
    valid = validate_and_filter_ml_data(features, require_target=True)
    model = train_linear_regression(valid)

    res = forecast_city_next_day(model, df_sufficient, "Delhi", model_name="TestLR")
    assert res["status"] == "success"
    assert res["city"] == "Delhi"
    assert res["predicted_pm2_5"] > 0.0
    assert res["forecast_date"] == "2024-01-11"

    # 2. Insufficient data (< 7 days)
    df_short = _create_synthetic_daily(spark, n_days=5, cities=("Delhi",))
    res_short = forecast_city_next_day(model, df_short, "Delhi")
    assert res_short["status"] == "error"
    assert "Insufficient historical observations" in res_short["message"]


def test_dashboard_ml_data_loaders():
    """Verify that dashboard cached loaders read generated ML artifacts successfully."""
    comparison = load_ml_comparison()
    report = load_ml_report()
    forecasts = load_ml_forecasts()
    test_preds = load_ml_test_predictions()

    assert comparison is not None
    assert "best_model" in comparison
    assert len(comparison["comparison_table"]) == 4

    assert report is not None
    assert "dataset_split" in report

    assert len(forecasts) == 50
    assert "predicted_pm2_5" in forecasts.columns

    assert len(test_preds) > 0
    assert "actual_target_pm2_5" in test_preds.columns
    assert "residual" in test_preds.columns
