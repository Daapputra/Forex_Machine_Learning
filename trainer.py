"""
trainer.py — Model Training (v2.0)
=====================================
UPGRADED for 5-year dataset:
  1. Random Forest (tuned)
  2. XGBoost (tuned with regularization + early stopping)
  3. Logistic Regression baseline (with StandardScaler)
  4. Rule-based baseline (RSI)
  5. NEW: Ensemble Voting Classifier (RF + XGBoost combined)

ANTI-LEAKAGE:
  - StandardScaler is fit ONLY on training data, then applied to val/test.
  - XGBoost uses validation set for early stopping (no test leakage).
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import config
import model_registry

logger = logging.getLogger(__name__)


class XGBWrapper:
    """
    Wrapper for XGBoost to handle label mapping transparently.
    Maps original labels (-1, 0, 1) to 0-indexed (0, 1, 2) internally,
    but exposes predict/predict_proba with original labels.
    This makes it compatible with VotingClassifier.
    """
    def __init__(self, xgb_model, label_map, label_map_inv):
        self.xgb_model = xgb_model
        self.label_map = label_map
        self.label_map_inv = label_map_inv
        self.classes_ = np.array(sorted(label_map.keys()))  # [-1, 0, 1]
        self.feature_importances_ = xgb_model.feature_importances_

    def predict(self, X):
        raw_pred = self.xgb_model.predict(X)
        return np.array([self.label_map_inv[p] for p in raw_pred])

    def predict_proba(self, X):
        # XGBoost returns probabilities in 0-indexed order (SELL, HOLD, BUY)
        # which maps to original order (-1, 0, 1) — same order, so no reorder needed
        return self.xgb_model.predict_proba(X)


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict = None,
) -> RandomForestClassifier:
    """Train Random Forest with balanced class weights."""
    params = params or config.RF_PARAMS

    logger.info(f"Training Random Forest: {params}")

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    logger.info(f"RF Train accuracy: {train_acc:.4f}")

    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict = None,
    early_stopping: int = None,
) -> XGBWrapper:
    """
    Train XGBoost with early stopping. Returns wrapped model that handles
    label mapping transparently.
    """
    params = params or config.XGB_PARAMS.copy()
    early_stopping = early_stopping or config.XGB_EARLY_STOPPING

    # Map labels to 0-indexed for XGBoost
    y_train_mapped = y_train.map(config.LABEL_MAP)
    y_val_mapped = y_val.map(config.LABEL_MAP)

    logger.info(f"Training XGBoost: {params}")
    logger.info(f"Early stopping: {early_stopping} rounds on validation set")

    model = XGBClassifier(**params, early_stopping_rounds=early_stopping)
    model.fit(
        X_train, y_train_mapped,
        eval_set=[(X_val, y_val_mapped)],
        verbose=False,
    )

    best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else "N/A"
    logger.info(f"XGBoost best iteration: {best_iteration}")

    # Wrap for transparent label mapping
    wrapped = XGBWrapper(model, config.LABEL_MAP, config.LABEL_MAP_INV)
    return wrapped


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    X_test: pd.DataFrame = None,
    params: dict = None,
) -> Tuple[LogisticRegression, StandardScaler, pd.DataFrame, pd.DataFrame]:
    """
    Train Logistic Regression baseline with StandardScaler.
    ANTI-LEAKAGE: Scaler is fit ONLY on X_train.
    """
    params = params or config.LR_PARAMS

    logger.info(f"Training Logistic Regression baseline: {params}")

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_val_scaled = None
    X_test_scaled = None
    if X_val is not None:
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index,
        )
    if X_test is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )

    model = LogisticRegression(**params)
    model.fit(X_train_scaled, y_train)

    train_acc = model.score(X_train_scaled, y_train)
    logger.info(f"LR Train accuracy: {train_acc:.4f}")

    return model, scaler, X_val_scaled, X_test_scaled


def train_ensemble(
    rf_model: RandomForestClassifier,
    xgb_wrapped: XGBWrapper,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> VotingClassifier:
    """
    NEW: Create an ensemble of Random Forest + XGBoost using soft voting.
    Soft voting = averages class probabilities from both models.
    This typically outperforms individual models by reducing variance.
    """
    logger.info("Training Ensemble (RF + XGBoost soft voting)...")

    # We use a VotingClassifier with pre-fitted estimators
    # The XGBWrapper already handles label mapping, but VotingClassifier
    # needs sklearn-compatible estimators. We'll do a manual ensemble instead.
    class ManualEnsemble:
        def __init__(self, models, weights=None):
            self.models = models
            self.weights = weights or [1.0] * len(models)
            self.classes_ = models[0].classes_ if hasattr(models[0], 'classes_') else np.array([-1, 0, 1])
            # Aggregate feature importances (weighted average)
            importances = []
            for m, w in zip(models, self.weights):
                if hasattr(m, 'feature_importances_'):
                    importances.append(m.feature_importances_ * w)
            if importances:
                self.feature_importances_ = np.mean(importances, axis=0)

        def predict(self, X):
            probas = self.predict_proba(X)
            pred_indices = np.argmax(probas, axis=1)
            return np.array([self.classes_[i] for i in pred_indices])

        def predict_proba(self, X):
            all_probas = []
            for model, weight in zip(self.models, self.weights):
                proba = model.predict_proba(X)
                all_probas.append(proba * weight)
            # Weighted average of probabilities
            avg_proba = np.sum(all_probas, axis=0) / sum(self.weights)
            return avg_proba

    # Weight XGBoost slightly higher (usually better calibrated)
    ensemble = ManualEnsemble([rf_model, xgb_wrapped], weights=[0.4, 0.6])

    # Verify on training data
    train_pred = ensemble.predict(X_train)
    train_acc = np.mean(train_pred == y_train)
    logger.info(f"Ensemble Train accuracy: {train_acc:.4f}")

    return ensemble


def generate_rule_based_predictions(
    df: pd.DataFrame,
    feature_cols: list,
) -> pd.Series:
    """Rule-based baseline: RSI < 30 -> BUY, RSI > 70 -> SELL, else HOLD."""
    if "RSI_14" not in df.columns:
        logger.warning("RSI_14 not found. Cannot generate rule-based predictions.")
        return pd.Series(0, index=df.index)

    predictions = pd.Series(0, index=df.index, name="rule_based_pred")
    predictions[df["RSI_14"] < 30] = 1   # BUY
    predictions[df["RSI_14"] > 70] = -1  # SELL

    logger.info(f"Rule-based predictions: "
                f"BUY={(predictions == 1).sum()}, "
                f"SELL={(predictions == -1).sum()}, "
                f"HOLD={(predictions == 0).sum()}")

    return predictions


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list,
    train_period: str = "",
) -> Dict[str, Any]:
    """
    Train all models including the new ensemble.
    """
    results = {}

    # 1. Random Forest
    logger.info("=" * 60)
    logger.info("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    results["random_forest"] = {
        "model": rf_model,
        "scaler": None,
        "X_val": X_val,
        "X_test": X_test,
        "is_xgboost": False,
    }

    # 2. XGBoost (wrapped)
    logger.info("=" * 60)
    logger.info("Training XGBoost...")
    xgb_wrapped = train_xgboost(X_train, y_train, X_val, y_val)
    results["xgboost"] = {
        "model": xgb_wrapped,
        "scaler": None,
        "X_val": X_val,
        "X_test": X_test,
        "is_xgboost": False,  # Wrapper handles mapping now
    }

    # 3. Ensemble (RF + XGBoost)
    logger.info("=" * 60)
    logger.info("Training Ensemble...")
    ensemble_model = train_ensemble(rf_model, xgb_wrapped, X_train, y_train)
    results["ensemble"] = {
        "model": ensemble_model,
        "scaler": None,
        "X_val": X_val,
        "X_test": X_test,
        "is_xgboost": False,
    }

    # 4. Logistic Regression (baseline)
    logger.info("=" * 60)
    logger.info("Training Logistic Regression baseline...")
    lr_model, scaler, X_val_scaled, X_test_scaled = train_logistic_regression(
        X_train, y_train, X_val, X_test
    )
    results["logistic_regression"] = {
        "model": lr_model,
        "scaler": scaler,
        "X_val": X_val_scaled,
        "X_test": X_test_scaled,
        "is_xgboost": False,
    }

    logger.info("=" * 60)
    logger.info("All models trained successfully.")

    return results
