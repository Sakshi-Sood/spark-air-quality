"""Model selection, persistence, and reporting module.

Selects the best-performing model based on validation metrics (lowest RMSE / MAE),
evaluates performance on the untouched chronological test period, and persists
Spark MLlib pipeline models, test predictions, metrics, and training reports.
"""
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.ml.ml_config import (
    ALL_FEATURE_COLS,
    ML_PATHS,
    MODEL_PARAMS,
    PERSISTENCE_COL,
    TARGET_COL,
)
from src.ml.model_evaluation import (
    evaluate_models_and_baseline,
    evaluate_predictions,
    extract_feature_importances,
)
from src.utils import get_logger, write_json, write_parquet

log = get_logger("ml.model_selection")


def select_best_model(val_results: Dict[str, Dict]) -> Tuple[str, str]:
    """Select the best-performing model based on validation RMSE.

    Parameters
    ----------
    val_results : Dict[str, Dict]
        Validation evaluation results dictionary.

    Returns
    -------
    Tuple[str, str]
        (best_model_key, selection_justification)
    """
    # Exclude persistence baseline from candidate models to select the best ML model
    ml_candidates = {k: v for k, v in val_results.items() if k != "persistence"}

    # Sort candidates by validation RMSE (primary criterion) and MAE (secondary)
    ranked = sorted(
        ml_candidates.items(),
        key=lambda item: (item[1]["metrics"]["rmse"], item[1]["metrics"]["mae"]),
    )

    best_key, best_info = ranked[0]
    best_name = best_info["model_name"]
    best_rmse = best_info["metrics"]["rmse"]
    best_mae = best_info["metrics"]["mae"]
    best_r2 = best_info["metrics"]["r2"]

    persistence_rmse = val_results["persistence"]["metrics"]["rmse"]
    persistence_mae = val_results["persistence"]["metrics"]["mae"]

    rmse_improvement = round(((persistence_rmse - best_rmse) / persistence_rmse) * 100, 2)

    justification = (
        f"{best_name} achieved the best validation performance with lowest RMSE ({best_rmse:.2f} µg/m³) "
        f"and MAE ({best_mae:.2f} µg/m³), achieving an R² of {best_r2:.4f}. "
        f"It outperforms the persistence baseline (RMSE {persistence_rmse:.2f} µg/m³) by {rmse_improvement}%."
    )
    log.info("Selected best model: %s -> %s", best_key, justification)
    return best_key, justification


def save_ml_artifacts(
    models: Dict[str, PipelineModel],
    best_key: str,
    val_results: Dict[str, Dict],
    test_results: Dict[str, Dict],
    test_predictions_df: DataFrame,
    split_metadata: Dict,
    justification: str,
) -> Dict:
    """Save trained models, test predictions, metrics JSON, and comprehensive training report.

    Parameters
    ----------
    models : Dict[str, PipelineModel]
        Fitted models dictionary.
    best_key : str
        Key of the selected best model.
    val_results : Dict[str, Dict]
        Validation set metrics.
    test_results : Dict[str, Dict]
        Test set metrics.
    test_predictions_df : DataFrame
        Spark DataFrame with test set actuals, predictions, and residuals.
    split_metadata : Dict
        Date ranges and record counts.
    justification : str
        Explanation of model selection.

    Returns
    -------
    Dict
        Consolidated ML training report.
    """
    # Ensure all ML output directories exist
    for dir_path in ML_PATHS.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    # 1. Save all trained Spark MLlib pipeline models
    for name, model in models.items():
        save_path = ML_PATHS["models"] / name
        model.write().overwrite().save(save_path.as_posix())
        log.info("Saved %s model to %s", name, save_path)

    # Save best model copy
    best_model_path = ML_PATHS["models"] / "best_model"
    models[best_key].write().overwrite().save(best_model_path.as_posix())
    log.info("Saved best model (%s) to %s", best_key, best_model_path)

    # 2. Extract feature importances
    feature_importances = extract_feature_importances(models)

    # 3. Save test predictions to Parquet for fast dashboard visualization
    test_pred_path = ML_PATHS["predictions"] / "test_predictions"
    write_parquet(test_predictions_df, test_pred_path, sort_cols=["city", "date"])
    log.info("Saved test predictions to %s", test_pred_path)

    # 4. Build consolidated comparison table
    comparison_table = []
    for model_key in ["persistence", "linear_regression", "decision_tree", "random_forest"]:
        val_m = val_results[model_key]["metrics"]
        test_m = test_results[model_key]["metrics"]
        comparison_table.append({
            "model_key": model_key,
            "model_name": val_results[model_key]["model_name"],
            "val_mae": val_m["mae"],
            "val_rmse": val_m["rmse"],
            "val_r2": val_m["r2"],
            "test_mae": test_m["mae"],
            "test_rmse": test_m["rmse"],
            "test_r2": test_m["r2"],
            "is_best": (model_key == best_key),
        })

    model_comparison = {
        "timestamp": datetime.now().isoformat(),
        "best_model": best_key,
        "justification": justification,
        "comparison_table": comparison_table,
    }
    write_json(model_comparison, ML_PATHS["metrics"] / "model_comparison.json")

    evaluation_metrics = {
        "validation": val_results,
        "test": test_results,
        "feature_importances": feature_importances,
    }
    write_json(evaluation_metrics, ML_PATHS["metrics"] / "evaluation_metrics.json")

    # 5. Build and save full training report
    report = {
        "training_timestamp": datetime.now().isoformat(),
        "selected_model": best_key,
        "selected_model_display": val_results[best_key]["model_name"],
        "selection_justification": justification,
        "hyperparameters": MODEL_PARAMS,
        "dataset_split": split_metadata,
        "features": {
            "all_features": ALL_FEATURE_COLS,
            "feature_count": len(ALL_FEATURE_COLS),
        },
        "model_comparison": comparison_table,
        "feature_importances": feature_importances.get(best_key, []),
    }
    write_json(report, ML_PATHS["reports"] / "ml_training_report.json")
    log.info("Saved ML training report to %s", ML_PATHS["reports"] / "ml_training_report.json")

    return report
