"""Model training module using Apache Spark MLlib pipelines.

Constructs estimators and pipelines for:
- Linear Regression (L2 ridge regularization to handle collinear lag features)
- Decision Tree Regressor (capturing non-linear interactions without severe overfitting)
- Random Forest Regressor (ensemble of diverse trees)

All feature transformations (StringIndexer, OneHotEncoder, VectorAssembler)
are assembled into Spark Pipelines and fitted strictly on training data.
"""
from typing import Dict, List, Tuple

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import (
    DecisionTreeRegressor,
    LinearRegression,
    RandomForestRegressor,
)
from pyspark.sql import DataFrame

from src.ml.ml_config import (
    CATEGORICAL_FEATURES,
    MODEL_PARAMS,
    NUMERIC_FEATURES,
    TARGET_COL,
)
from src.utils import get_logger

log = get_logger("ml.model_training")


def build_preprocessing_stages() -> Tuple[List, str]:
    """Create StringIndexer, OneHotEncoder, and VectorAssembler stages.

    Returns
    -------
    Tuple[List, str]
        List of Spark ML pipeline transformation stages, and the output vector column name.
    """
    stages = []
    encoded_cat_cols = []

    for cat_col in CATEGORICAL_FEATURES:
        indexer = StringIndexer(
            inputCol=cat_col,
            outputCol=f"{cat_col}_idx",
            handleInvalid="keep",
        )
        encoder = OneHotEncoder(
            inputCol=f"{cat_col}_idx",
            outputCol=f"{cat_col}_vec",
            handleInvalid="keep",
        )
        stages.extend([indexer, encoder])
        encoded_cat_cols.append(f"{cat_col}_vec")

    assembler_inputs = NUMERIC_FEATURES + encoded_cat_cols
    assembler = VectorAssembler(
        inputCols=assembler_inputs,
        outputCol="features",
        handleInvalid="keep",
    )
    stages.append(assembler)
    return stages, "features"


def get_feature_names() -> List[str]:
    """Return an interpretable list of feature names corresponding to the assembled vector.

    For categorical features (e.g. season), labels the categories explicitly.
    """
    feature_names = list(NUMERIC_FEATURES)
    # Target season has 4 categories: Winter, Pre-monsoon, Monsoon, Post-monsoon
    # OneHotEncoder keeps dropLast=True by default (3 binary indicators + keep bucket)
    feature_names.extend([
        "target_season_Winter",
        "target_season_Pre-monsoon",
        "target_season_Monsoon",
    ])
    return feature_names


def train_linear_regression(train_df: DataFrame) -> PipelineModel:
    """Train a Linear Regression pipeline with L2 regularization."""
    stages, features_col = build_preprocessing_stages()
    lr_params = MODEL_PARAMS["linear_regression"]

    lr = LinearRegression(
        featuresCol=features_col,
        labelCol=TARGET_COL,
        regParam=lr_params["regParam"],
        elasticNetParam=lr_params["elasticNetParam"],
        maxIter=lr_params["maxIter"],
        standardization=lr_params["standardization"],
        predictionCol="prediction",
    )
    pipeline = Pipeline(stages=stages + [lr])
    log.info("Training Linear Regression (regParam=%s, elasticNetParam=%s)...",
             lr_params["regParam"], lr_params["elasticNetParam"])
    return pipeline.fit(train_df)


def train_decision_tree(train_df: DataFrame) -> PipelineModel:
    """Train a Decision Tree Regressor pipeline."""
    stages, features_col = build_preprocessing_stages()
    dt_params = MODEL_PARAMS["decision_tree"]

    dt = DecisionTreeRegressor(
        featuresCol=features_col,
        labelCol=TARGET_COL,
        maxDepth=dt_params["maxDepth"],
        minInstancesPerNode=dt_params["minInstancesPerNode"],
        seed=dt_params["seed"],
        predictionCol="prediction",
    )
    pipeline = Pipeline(stages=stages + [dt])
    log.info("Training Decision Tree Regressor (maxDepth=%s, minInstancesPerNode=%s)...",
             dt_params["maxDepth"], dt_params["minInstancesPerNode"])
    return pipeline.fit(train_df)


def train_random_forest(train_df: DataFrame) -> PipelineModel:
    """Train a Random Forest Regressor pipeline."""
    stages, features_col = build_preprocessing_stages()
    rf_params = MODEL_PARAMS["random_forest"]

    rf = RandomForestRegressor(
        featuresCol=features_col,
        labelCol=TARGET_COL,
        numTrees=rf_params["numTrees"],
        maxDepth=rf_params["maxDepth"],
        minInstancesPerNode=rf_params["minInstancesPerNode"],
        featureSubsetStrategy=rf_params["featureSubsetStrategy"],
        seed=rf_params["seed"],
        predictionCol="prediction",
    )
    pipeline = Pipeline(stages=stages + [rf])
    log.info("Training Random Forest Regressor (numTrees=%s, maxDepth=%s, featureSubsetStrategy=%s)...",
             rf_params["numTrees"], rf_params["maxDepth"], rf_params["featureSubsetStrategy"])
    return pipeline.fit(train_df)


def train_all_models(train_df: DataFrame) -> Dict[str, PipelineModel]:
    """Train all 3 ML models on the training dataset.

    Parameters
    ----------
    train_df : DataFrame
        Chronological training dataset.

    Returns
    -------
    Dict[str, PipelineModel]
        Dictionary mapping model names to fitted PipelineModels.
    """
    models = {
        "linear_regression": train_linear_regression(train_df),
        "decision_tree": train_decision_tree(train_df),
        "random_forest": train_random_forest(train_df),
    }
    log.info("Successfully trained all 3 regression models on %d training rows.", train_df.count())
    return models
