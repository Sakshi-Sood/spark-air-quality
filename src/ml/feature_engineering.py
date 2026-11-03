"""Feature engineering module for PM2.5 next-day forecasting.

Constructs historical lag features, rolling statistics, weather features,
and calendar temporal features using Apache Spark window functions partitioned
by city and ordered chronologically.

Leakage Prevention Guarantees:
1. Target day T+1 measurements are strictly excluded from all input features.
2. Rolling window averages over [T-2, T] and [T-6, T] terminate at observation
   day T, never including target day T+1.
3. City-level window partitioning guarantees that no cross-city lag or rolling
   information leaks between series.
4. Chronological splitting ensures training, validation, and test sets are
   partitioned by date boundaries, preventing lookahead leakage.
"""
from typing import Dict, Tuple

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src.ml.ml_config import (
    ALL_FEATURE_COLS,
    CHRONOLOGICAL_SPLIT,
    DATE_COL,
    FORECAST_DATE_COL,
    NUMERIC_FEATURES,
    PERSISTENCE_COL,
    POLLUTANT_LAG_FEATURES,
    TARGET_COL,
    WEATHER_FEATURES,
)
from src.utils import get_logger

log = get_logger("ml.feature_engineering")


def create_forecasting_features(daily_df: DataFrame) -> DataFrame:
    """Compute lag, rolling, weather, and temporal features for next-day PM2.5 forecasting.

    Each row represents observation day T (date), and defines features up to day T.
    The prediction target is PM2.5 concentration on day T+1 (forecast_date).

    Parameters
    ----------
    daily_df : DataFrame
        Clean daily joined dataset containing city, date, pollutant, and weather fields.

    Returns
    -------
    DataFrame
        Enriched DataFrame with engineered features and target_pm2_5.
    """
    # Order by date strictly within each city
    w_city_date = Window.partitionBy("city").orderBy("date")
    w_rolling_3d = Window.partitionBy("city").orderBy("date").rowsBetween(-2, 0)
    w_rolling_7d = Window.partitionBy("city").orderBy("date").rowsBetween(-6, 0)

    # 1. Target and Forecast Date (Day T+1)
    df = (
        daily_df
        .withColumn(FORECAST_DATE_COL, F.date_add(F.col("date"), 1))
        .withColumn(TARGET_COL, F.lead("pm2_5_mean", 1).over(w_city_date))
    )

    # 2. Historical PM2.5 and Pollutant Lags
    # At day T:
    # pm2_5_lag1 is the actual PM2.5 on day T (1 day prior to day T+1 target)
    # pm2_5_lag2 is PM2.5 on day T-1 (2 days prior to day T+1 target)
    # pm2_5_lag7 is PM2.5 on day T-6 (7 days prior to day T+1 target)
    df = (
        df
        .withColumn("pm2_5_lag1", F.col("pm2_5_mean"))
        .withColumn("pm2_5_lag2", F.lag("pm2_5_mean", 1).over(w_city_date))
        .withColumn("pm2_5_lag7", F.lag("pm2_5_mean", 6).over(w_city_date))
        .withColumn("pm2_5_rolling_3d", F.round(F.avg("pm2_5_mean").over(w_rolling_3d), 3))
        .withColumn("pm2_5_rolling_7d", F.round(F.avg("pm2_5_mean").over(w_rolling_7d), 3))
        .withColumn("pm10_lag1", F.col("pm10_mean"))
        .withColumn("no2_lag1", F.col("no2_mean"))
        .withColumn("so2_lag1", F.col("so2_mean"))
        .withColumn("co_lag1", F.col("co_8h_max"))
    )

    # 3. Weather Features (Day T and Day T-1)
    df = (
        df
        .withColumn("temperature_lag1", F.col("temperature_2m_mean"))
        .withColumn("humidity_lag1", F.col("relative_humidity_2m_mean").cast("double"))
        .withColumn("wind_speed_lag1", F.col("wind_speed_10m_max"))
        .withColumn("precipitation_lag1", F.col("precipitation_sum"))
        .withColumn("temperature_lag2", F.lag("temperature_2m_mean", 1).over(w_city_date))
        .withColumn("humidity_lag2", F.lag("relative_humidity_2m_mean", 1).over(w_city_date).cast("double"))
        .withColumn("wind_speed_lag2", F.lag("wind_speed_10m_max", 1).over(w_city_date))
        .withColumn("precipitation_lag2", F.lag("precipitation_sum", 1).over(w_city_date))
    )

    # 4. Temporal Features (Derived strictly from forecast_date T+1)
    month_col = F.month(F.col(FORECAST_DATE_COL))
    df = (
        df
        .withColumn("target_month", month_col)
        .withColumn("target_dayofweek", F.dayofweek(F.col(FORECAST_DATE_COL)))
        .withColumn("target_dayofyear", F.dayofyear(F.col(FORECAST_DATE_COL)))
        .withColumn("target_year", F.year(F.col(FORECAST_DATE_COL)))
        .withColumn(
            "target_season",
            F.when(month_col.isin(1, 2), "Winter")
            .when(month_col.isin(3, 4, 5), "Pre-monsoon")
            .when(month_col.isin(6, 7, 8, 9), "Monsoon")
            .otherwise("Post-monsoon"),
        )
    )

    return df


def validate_and_filter_ml_data(features_df: DataFrame, require_target: bool = True) -> DataFrame:
    """Validate feature columns and filter rows with incomplete historical lags or missing target.

    For model training and evaluation, require_target=True ensures only rows with
    known day T+1 ground truth and complete 7-day historical lags are retained.
    For live forecasting on the latest date, require_target=False allows rows where
    T+1 has not yet occurred.

    Parameters
    ----------
    features_df : DataFrame
        DataFrame containing all engineered features.
    require_target : bool
        If True, drops rows where target_pm2_5 is null or non-positive.

    Returns
    -------
    DataFrame
        Validated and cleaned DataFrame ready for training or forecasting.
    """
    total_before = features_df.count()

    # Required feature check (all lag and rolling columns must be present)
    condition = F.lit(True)
    for col_name in ALL_FEATURE_COLS:
        condition = condition & F.col(col_name).isNotNull()

    # Validate PM2.5 values are positive
    condition = condition & (F.col(PERSISTENCE_COL) > 0.0)

    if require_target:
        condition = condition & F.col(TARGET_COL).isNotNull() & (F.col(TARGET_COL) > 0.0)

    filtered_df = features_df.filter(condition)
    total_after = filtered_df.count()
    dropped = total_before - total_after

    log.info(
        "Feature validation (require_target=%s): %d -> %d rows retained (%d dropped)",
        require_target, total_before, total_after, dropped,
    )
    return filtered_df


def split_time_series_data(
    df: DataFrame,
    train_ratio: float = CHRONOLOGICAL_SPLIT["train_ratio"],
    val_ratio: float = CHRONOLOGICAL_SPLIT["val_ratio"],
    test_ratio: float = CHRONOLOGICAL_SPLIT["test_ratio"],
) -> Tuple[DataFrame, DataFrame, DataFrame, Dict]:
    """Perform chronological train / validation / test splitting across all cities.

    Partitions distinct dates into chronological blocks so that every city on a
    given date is assigned to the exact same split, preventing lookahead leakage.

    Parameters
    ----------
    df : DataFrame
        Validated features DataFrame containing 'date'.
    train_ratio : float
        Fraction of dates for training (default 0.70).
    val_ratio : float
        Fraction of dates for validation (default 0.15).
    test_ratio : float
        Fraction of dates for testing (default 0.15).

    Returns
    -------
    Tuple[DataFrame, DataFrame, DataFrame, Dict]
        (train_df, val_df, test_df, split_metadata)
    """
    # Extract sorted distinct dates
    distinct_dates = [r.date for r in df.select(DATE_COL).distinct().orderBy(DATE_COL).collect()]
    n_dates = len(distinct_dates)

    if n_dates < 10:
        raise ValueError(f"Insufficient distinct dates ({n_dates}) for time series split.")

    n_train = int(n_dates * train_ratio)
    n_val = int(n_dates * val_ratio)
    n_test = n_dates - n_train - n_val

    train_dates = distinct_dates[:n_train]
    val_dates = distinct_dates[n_train: n_train + n_val]
    test_dates = distinct_dates[n_train + n_val:]

    train_start, train_end = str(train_dates[0]), str(train_dates[-1])
    val_start, val_end = str(val_dates[0]), str(val_dates[-1])
    test_start, test_end = str(test_dates[0]), str(test_dates[-1])

    train_df = df.filter(F.col(DATE_COL) <= F.lit(train_end))
    val_df = df.filter((F.col(DATE_COL) >= F.lit(val_start)) & (F.col(DATE_COL) <= F.lit(val_end)))
    test_df = df.filter(F.col(DATE_COL) >= F.lit(test_start))

    metadata = {
        "total_dates": n_dates,
        "splits": {
            "train": {
                "start_date": train_start,
                "end_date": train_end,
                "num_dates": len(train_dates),
                "ratio": round(len(train_dates) / n_dates, 4),
            },
            "validation": {
                "start_date": val_start,
                "end_date": val_end,
                "num_dates": len(val_dates),
                "ratio": round(len(val_dates) / n_dates, 4),
            },
            "test": {
                "start_date": test_start,
                "end_date": test_end,
                "num_dates": len(test_dates),
                "ratio": round(len(test_dates) / n_dates, 4),
            },
        },
    }

    log.info(
        "Chronological Split: Train [%s to %s, %d days], Val [%s to %s, %d days], Test [%s to %s, %d days]",
        train_start, train_end, len(train_dates),
        val_start, val_end, len(val_dates),
        test_start, test_end, len(test_dates),
    )

    return train_df, val_df, test_df, metadata
