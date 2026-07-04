"""
model_registry.py — Model Persistence with Metadata
=====================================================
Saves and loads models with rich metadata for tracking and reproducibility.
Each model is saved as a .pkl file alongside a .json metadata file.

This is the foundation for the Signal Engine (Step 10) to automatically
load the "best model" based on validation metrics.
"""

import os
import json
import joblib
import logging
from datetime import datetime
from typing import Optional, Dict, Any

import config

logger = logging.getLogger(__name__)


def _make_model_filename(pair: str, timeframe: str,
                         model_type: str, version: str) -> str:
    """Generate consistent model filename."""
    return f"{pair}_{timeframe}_{model_type}_{version}"


def save_model(
    model,
    model_type: str,
    feature_cols: list,
    val_metrics: dict,
    train_period: str = "",
    pair: str = None,
    timeframe: str = None,
    version: str = None,
    extra_metadata: dict = None,
) -> str:
    """
    Save model with metadata.

    Parameters
    ----------
    model : trained model object
        The trained sklearn/xgboost model.
    model_type : str
        Type of model (e.g., "random_forest", "xgboost", "logistic_regression").
    feature_cols : list
        Feature column names used for training.
    val_metrics : dict
        Validation metrics (accuracy, f1_macro, etc.).
    train_period : str
        Human-readable training period (e.g., "2026-05-01 to 2026-05-22").
    pair : str
        Currency pair (default: config.PAIR).
    timeframe : str
        Timeframe (default: config.TIMEFRAME).
    version : str
        Model version (default: config.MODEL_VERSION).
    extra_metadata : dict
        Additional metadata to include.

    Returns
    -------
    str
        Path to the saved model file.
    """
    pair = pair or config.PAIR
    timeframe = timeframe or config.TIMEFRAME
    version = version or config.MODEL_VERSION

    base_name = _make_model_filename(pair, timeframe, model_type, version)
    model_path = os.path.join(config.MODELS_DIR, f"{base_name}.pkl")
    meta_path = os.path.join(config.MODELS_DIR, f"{base_name}_metadata.json")

    # Save model
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(model, model_path)

    # Build metadata
    metadata = {
        "pair": pair,
        "timeframe": timeframe,
        "model_type": model_type,
        "version": version,
        "train_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_period": train_period,
        "features": feature_cols,
        "n_features": len(feature_cols),
        "val_metrics": val_metrics,
        "model_file": os.path.basename(model_path),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    # Save metadata
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Saved model: {model_path}")
    logger.info(f"Saved metadata: {meta_path}")

    return model_path


def load_model(
    model_type: str,
    pair: str = None,
    timeframe: str = None,
    version: str = None,
) -> tuple:
    """
    Load a model and its metadata.

    Returns
    -------
    tuple
        (model, metadata_dict)
    """
    pair = pair or config.PAIR
    timeframe = timeframe or config.TIMEFRAME
    version = version or config.MODEL_VERSION

    base_name = _make_model_filename(pair, timeframe, model_type, version)
    model_path = os.path.join(config.MODELS_DIR, f"{base_name}.pkl")
    meta_path = os.path.join(config.MODELS_DIR, f"{base_name}_metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    logger.info(f"Loaded model: {model_path}")
    return model, metadata


def load_best_model(
    pair: str = None,
    timeframe: str = None,
    metric: str = "f1_macro",
) -> tuple:
    """
    Load the best model for a given pair/timeframe based on validation metric.
    Scans all metadata files in the models directory.

    Returns
    -------
    tuple
        (model, metadata_dict) of the best model.
    """
    pair = pair or config.PAIR
    timeframe = timeframe or config.TIMEFRAME

    models_dir = config.MODELS_DIR
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    best_model = None
    best_metadata = None
    best_score = -1

    for fname in os.listdir(models_dir):
        if not fname.endswith("_metadata.json"):
            continue

        meta_path = os.path.join(models_dir, fname)
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        # Filter by pair and timeframe
        if metadata.get("pair") != pair or metadata.get("timeframe") != timeframe:
            continue

        # Get the metric score
        score = metadata.get("val_metrics", {}).get(metric, -1)
        if score > best_score:
            best_score = score
            best_metadata = metadata

    if best_metadata is None:
        raise FileNotFoundError(
            f"No models found for {pair}/{timeframe} in {models_dir}"
        )

    # Load the best model
    model_file = best_metadata["model_file"]
    model_path = os.path.join(models_dir, model_file)
    model = joblib.load(model_path)

    logger.info(f"Loaded best model: {model_file} "
                f"({metric}={best_score:.4f})")

    return model, best_metadata


def list_models(pair: str = None, timeframe: str = None) -> list:
    """List all saved models with their metadata summaries."""
    pair = pair or config.PAIR
    timeframe = timeframe or config.TIMEFRAME
    models_dir = config.MODELS_DIR

    results = []
    if not os.path.exists(models_dir):
        return results

    for fname in os.listdir(models_dir):
        if not fname.endswith("_metadata.json"):
            continue

        meta_path = os.path.join(models_dir, fname)
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        if pair and metadata.get("pair") != pair:
            continue
        if timeframe and metadata.get("timeframe") != timeframe:
            continue

        results.append(metadata)

    return results
