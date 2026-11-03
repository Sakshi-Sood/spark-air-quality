"""Forecasting module for next-day PM2.5 predictions.

Provides reusable inference functions to generate next-day PM2.5 forecasts
given historical air quality and weather observations for any city.
"""
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.ml.feature_engineering import (
    create_forecasting_features,
    validate_and_filter_ml_data,
)
from src.ml.ml_config import (
    FORECAST_DATE_COL,
    ML_PATHS,
    PERSISTENCE_COL,
)
from src.utils import get_logger, write_parquet

log = get_logger("ml.forecasting")


def load_best_model(model_path: Optional[str] = None) -> PipelineModel:
    """Load the persisted best Spark MLlib pipeline model.

    Parameters
    ----------
    model_path : Optional[str]
        Path to model directory. Defaults to outputs/ml/models/best_model.

    Returns
    -------
    PipelineModel
        Fitted Spark MLlib pipeline model ready for inference.
    """
    path = model_path or (ML_PATHS["models"] / "best_model").as_posix()
    log.info("Loading MLlib pipeline model from %s", path)
    return PipelineModel.load(path)


def forecast_city_next_day(
    model: PipelineModel,
    daily_df: DataFrame,
    city_name: str,
    model_name: str = "Best Model",
) -> Dict[str, Any]:
    """Generate next-day PM2.5 forecast for a specific city.

    Validates that at least 7 consecutive historical daily observations exist
    up to the latest observation date. Returns an informative error message if
    data is insufficient.

    Parameters
    ----------
    model : PipelineModel
        Fitted ML pipeline model.
    daily_df : DataFrame
        Daily joined dataset.
    city_name : str
        Target city name (e.g., 'Delhi').
    model_name : str
        Display name of the model.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing forecast results or validation failure message.
    """
    # Filter for target city
    city_df = daily_df.filter(F.col("city") == city_name)
    n_records = city_df.count()

    if n_records < 7:
        return {
            "status": "error",
            "city": city_name,
            "message": f"Insufficient historical observations for {city_name}. "
                       f"At least 7 consecutive daily records are required, but found {n_records}.",
        }

    # Generate features across the city's timeline
    features_df = create_forecasting_features(city_df)

    # Filter with require_target=False because the latest date's target (tomorrow) is not yet observed
    valid_features = validate_and_filter_ml_data(features_df, require_target=False)

    if valid_features.count() == 0:
        return {
            "status": "error",
            "city": city_name,
            "message": f"Historical data for {city_name} is missing required lag or weather features.",
        }

    # Get the latest observation row
    latest_row_df = valid_features.orderBy(F.col("date").desc()).limit(1)
    predictions_df = model.transform(latest_row_df)

    pred_row = predictions_df.collect()[0]

    obs_date = str(pred_row["date"])
    forecast_date = str(pred_row[FORECAST_DATE_COL])
    predicted_val = round(float(pred_row["prediction"]), 2)
    latest_actual = round(float(pred_row[PERSISTENCE_COL]), 2)

    # Fetch last 14 days of actual history for trend visualization
    recent_history = (
        city_df.orderBy(F.col("date").desc())
        .limit(14)
        .select("date", "pm2_5_mean")
        .orderBy("date")
        .collect()
    )
    history_records = [
        {"date": str(r["date"]), "pm2_5": round(float(r["pm2_5_mean"]), 2)}
        for r in recent_history
    ]

    return {
        "status": "success",
        "city": city_name,
        "observation_date": obs_date,
        "forecast_date": forecast_date,
        "latest_actual_pm2_5": latest_actual,
        "predicted_pm2_5": predicted_val,
        "model_used": model_name,
        "recent_history": history_records,
    }


def generate_batch_forecasts(
    model: PipelineModel,
    daily_df: DataFrame,
    model_name: str = "Best Model",
) -> DataFrame:
    """Generate next-day PM2.5 forecasts for all cities simultaneously using Spark.

    Parameters
    ----------
    model : PipelineModel
        Trained model pipeline.
    daily_df : DataFrame
        Complete daily joined dataset.
    model_name : str
        Name of the model.

    Returns
    -------
    DataFrame
        Forecast results DataFrame saved to outputs/ml/predictions/forecast_results.
    """
    # 1. Feature engineering
    features_df = create_forecasting_features(daily_df)
    valid_features = validate_and_filter_ml_data(features_df, require_target=False)

    # 2. Extract the latest date for each city
    max_dates = valid_features.groupBy("city").agg(F.max("date").alias("max_date"))
    latest_by_city = (
        valid_features
        .join(max_dates, (valid_features.city == max_dates.city) & (valid_features.date == max_dates.max_date))
        .drop(max_dates.city)
        .drop(max_dates.max_date)
    )

    # 3. Predict next-day PM2.5
    preds = model.transform(latest_by_city)

    forecast_results = (
        preds
        .select(
            "city",
            F.col("date").alias("observation_date"),
            F.col(FORECAST_DATE_COL).alias("forecast_date"),
            F.round(F.col(PERSISTENCE_COL), 2).alias("latest_actual_pm2_5"),
            F.round(F.col("prediction"), 2).alias("predicted_pm2_5"),
            F.round(F.col("prediction") - F.col(PERSISTENCE_COL), 2).alias("expected_change"),
            F.lit(model_name).alias("model_used"),
            "region",
            "state",
        )
        .orderBy("city")
    )

    out_path = ML_PATHS["predictions"] / "forecast_results"
    write_parquet(forecast_results, out_path, sort_cols=["city"])
    log.info("Persisted batch next-day forecasts for %d cities to %s", forecast_results.count(), out_path)
    return forecast_results
