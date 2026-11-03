"""Machine Learning configuration: features, targets, model parameters, and paths."""
from pathlib import Path
from src.config import (
    ANALYTICS_DIR,
    ML_DIR,
    ML_METRICS_DIR,
    ML_MODELS_DIR,
    ML_PREDICTIONS_DIR,
    ML_REPORTS_DIR,
    PROCESSED_DIR,
    SEASON_MONTHS,
)

# ---------------------------------------------------------------- directories
ML_PATHS = {
    "root": ML_DIR,
    "models": ML_MODELS_DIR,
    "metrics": ML_METRICS_DIR,
    "predictions": ML_PREDICTIONS_DIR,
    "reports": ML_REPORTS_DIR,
}

# ---------------------------------------------------------------- target & keys
TARGET_COL = "target_pm2_5"
PERSISTENCE_COL = "pm2_5_lag1"
DATE_COL = "date"
FORECAST_DATE_COL = "forecast_date"
CITY_COL = "city"

# ---------------------------------------------------------------- features
# Historical PM2.5 and pollutant features available on observation day T
POLLUTANT_LAG_FEATURES = [
    "pm2_5_lag1",       # PM2.5 on day T (lag 1 relative to T+1 target)
    "pm2_5_lag2",       # PM2.5 on day T-1 (lag 2 relative to T+1 target)
    "pm2_5_lag7",       # PM2.5 on day T-6 (lag 7 relative to T+1 target)
    "pm2_5_rolling_3d", # Rolling 3-day PM2.5 average [T-2 to T], excludes T+1
    "pm2_5_rolling_7d", # Rolling 7-day PM2.5 average [T-6 to T], excludes T+1
    "pm10_lag1",        # PM10 on day T
    "no2_lag1",         # NO2 on day T
    "so2_lag1",         # SO2 on day T
    "co_lag1",          # CO on day T (from co_8h_max)
]

# Weather features available on or before observation day T
WEATHER_FEATURES = [
    "temperature_lag1",   # Mean temperature on day T
    "humidity_lag1",      # Relative humidity on day T
    "wind_speed_lag1",    # Wind speed max on day T
    "precipitation_lag1", # Precipitation sum on day T
    "temperature_lag2",   # Mean temperature on day T-1
    "humidity_lag2",      # Relative humidity on day T-1
    "wind_speed_lag2",    # Wind speed max on day T-1
    "precipitation_lag2", # Precipitation sum on day T-1
]

# Calendar and temporal features (strictly derived for the forecast day T+1)
TEMPORAL_NUMERIC_FEATURES = [
    "target_month",
    "target_dayofweek",
    "target_dayofyear",
    "target_year",
]

CATEGORICAL_FEATURES = [
    "target_season",
]

# Assembled numeric features to feed into VectorAssembler alongside one-hot encoded season
NUMERIC_FEATURES = POLLUTANT_LAG_FEATURES + WEATHER_FEATURES + TEMPORAL_NUMERIC_FEATURES

# All raw feature columns needed before vector assembly
ALL_FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------- data splitting
CHRONOLOGICAL_SPLIT = {
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
}

# ---------------------------------------------------------------- model hyperparameters
MODEL_PARAMS = {
    "linear_regression": {
        "regParam": 0.1,
        "elasticNetParam": 0.0,  # Ridge L2 regularization to control collinearity
        "maxIter": 100,
        "standardization": True,
    },
    "decision_tree": {
        "maxDepth": 6,
        "minInstancesPerNode": 20,
        "seed": 42,
    },
    "random_forest": {
        "numTrees": 35,
        "maxDepth": 8,
        "minInstancesPerNode": 10,
        "featureSubsetStrategy": "sqrt",
        "seed": 42,
    },
}

# Metric keys evaluated across models
EVALUATION_METRICS = ["mae", "rmse", "r2"]
