"""Model evaluation module for PM2.5 regression forecasting.

Calculates MAE, RMSE, and R² scores for each model and the persistence baseline.
Extracts model weights, feature importances, and residual statistics.
"""
from typing import Dict, List, Optional

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.ml.ml_config import (
    EVALUATION_METRICS,
    PERSISTENCE_COL,
    TARGET_COL,
)
from src.ml.model_training import get_feature_names
from src.utils import get_logger

log = get_logger("ml.model_evaluation")


def evaluate_predictions(
    df: DataFrame,
    prediction_col: str,
    label_col: str = TARGET_COL,
) -> Dict[str, float]:
    """Calculate MAE, RMSE, and R² for a prediction column against ground truth.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing label_col and prediction_col.
    prediction_col : str
        Name of the column containing model or baseline predictions.
    label_col : str
        Name of the ground truth target column.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys 'mae', 'rmse', 'r2', and 'count'.
    """
    evaluator_mae = RegressionEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="mae")
    evaluator_rmse = RegressionEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="r2")

    mae = evaluator_mae.evaluate(df)
    rmse = evaluator_rmse.evaluate(df)
    r2 = evaluator_r2.evaluate(df)
    count = df.count()

    return {
        "mae": round(float(mae), 3),
        "rmse": round(float(rmse), 3),
        "r2": round(float(r2), 4),
        "count": int(count),
    }


def evaluate_models_and_baseline(
    models: Dict[str, PipelineModel],
    eval_df: DataFrame,
    dataset_name: str = "validation",
) -> Dict[str, Dict]:
    """Evaluate all trained models and the persistence baseline on an evaluation split.

    Parameters
    ----------
    models : Dict[str, PipelineModel]
        Dictionary of trained model pipelines.
    eval_df : DataFrame
        Validation or test DataFrame.
    dataset_name : str
        Label for logging ("validation" or "test").

    Returns
    -------
    Dict[str, Dict]
        Dictionary mapping model names to evaluation metrics and metadata.
    """
    results = {}

    # 1. Persistence Baseline: Predicted(T+1) = Actual(T)
    log.info("Evaluating Persistence Baseline on %s set...", dataset_name)
    results["persistence"] = {
        "model_name": "Persistence Baseline",
        "description": "Naïve baseline: Predicted PM2.5(T+1) = Actual PM2.5(T)",
        "metrics": evaluate_predictions(eval_df, prediction_col=PERSISTENCE_COL),
    }

    # 2. Machine Learning Models
    for name, pipeline_model in models.items():
        log.info("Evaluating %s on %s set...", name, dataset_name)
        preds = pipeline_model.transform(eval_df)
        display_name = name.replace("_", " ").title()
        results[name] = {
            "model_name": display_name,
            "metrics": evaluate_predictions(preds, prediction_col="prediction"),
        }

    return results


def extract_feature_importances(
    models: Dict[str, PipelineModel],
) -> Dict[str, List[Dict[str, float]]]:
    """Extract feature importances or coefficients for each model where applicable.

    Parameters
    ----------
    models : Dict[str, PipelineModel]
        Fitted model pipelines.

    Returns
    -------
    Dict[str, List[Dict[str, float]]]
        Sorted feature importance or coefficient dictionaries per model.
    """
    importances_by_model = {}
    feature_names = get_feature_names()

    for name, model in models.items():
        # The estimator is the last stage of the pipeline
        estimator = model.stages[-1]

        if hasattr(estimator, "featureImportances"):
            # Tree-based models (DecisionTree, RandomForest)
            sparse_vector = estimator.featureImportances
            weights = sparse_vector.toArray()
            feature_imp = [
                {"feature": fname, "importance": round(float(val), 4)}
                for fname, val in zip(feature_names, weights)
            ]
            feature_imp.sort(key=lambda x: x["importance"], reverse=True)
            importances_by_model[name] = feature_imp

        elif hasattr(estimator, "coefficients"):
            # Linear Regression
            coeffs = estimator.coefficients.toArray()
            feature_coeff = [
                {"feature": fname, "coefficient": round(float(val), 4)}
                for fname, val in zip(feature_names, coeffs)
            ]
            feature_coeff.sort(key=lambda x: abs(x["coefficient"]), reverse=True)
            importances_by_model[name] = feature_coeff

    return importances_by_model
