"""
trainer.py — Model Training & Walk-Forward Validation (v4.0)
=============================================================
v4.0 UPGRADE:
  1. Walk-Forward CV (TimeSeriesSplit) for robust evaluation.
  2. RandomizedSearchCV for hyperparameter tuning.
  3. Added LightGBM.
  4. ManualEnsemble fixed for pickling.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import lightgbm as lgb
from statsmodels.stats.outliers_influence import variance_inflation_factor

import config
import model_registry

logger = logging.getLogger(__name__)


class WrapperMixin:
    def predict(self, X):
        raw_pred = self.model.predict(X)
        return np.array([self.label_map_inv[p] for p in raw_pred])

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class XGBWrapper(WrapperMixin):
    def __init__(self, xgb_model, label_map, label_map_inv):
        self.model = xgb_model
        self.label_map = label_map
        self.label_map_inv = label_map_inv
        self.classes_ = np.array(sorted(label_map.keys()))
        self.feature_importances_ = getattr(xgb_model, 'feature_importances_', None)


class LGBMWrapper(WrapperMixin):
    def __init__(self, lgb_model, label_map, label_map_inv):
        self.model = lgb_model
        self.label_map = label_map
        self.label_map_inv = label_map_inv
        self.classes_ = np.array(sorted(label_map.keys()))
        self.feature_importances_ = getattr(lgb_model, 'feature_importances_', None)


class ManualEnsemble:
    """Manual soft voting ensemble for pre-fitted estimators."""
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.classes_ = models[0].classes_ if hasattr(models[0], 'classes_') else np.array([-1, 0, 1])
        # Aggregate feature importances (weighted average)
        importances = []
        for m, w in zip(models, self.weights):
            if hasattr(m, 'feature_importances_') and m.feature_importances_ is not None:
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
        avg_proba = np.sum(all_probas, axis=0) / sum(self.weights)
        return avg_proba


def select_top_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_n: int = None,
) -> list:
    """Quick RF for feature selection (v4.0)."""
    top_n = top_n or config.FEATURE_SELECTION_TOP_N
    
    # We already explicitly dropped the worst features in features.py
    # If the remaining is less than top_n, just return them.
    if len(X_train.columns) <= top_n:
        return X_train.columns.tolist()

    logger.info(f"Running dynamic feature selection (selecting top {top_n} from {len(X_train.columns)})...")

    quick_rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=1
    )
    quick_rf.fit(X_train, y_train)

    importances = pd.Series(quick_rf.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)

    selected = importances.head(top_n).index.tolist()
    logger.info(f"Selected features by RF Gini: {selected}")
    return selected


def drop_high_vif_features(X: pd.DataFrame, threshold: float = 10.0) -> list:
    """
    NEW: Drops highly collinear features using Variance Inflation Factor (VIF).
    """
    logger.info("Running Multicollinearity (VIF) check...")
    
    # VIF requires handling NaNs/Infs, scaling, and is computationally heavy for many rows.
    # We'll sample 5000 rows to speed it up
    X_sample = X.dropna().sample(min(5000, len(X)), random_state=42)
    
    # Scale data for numerical stability in VIF
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_sample), columns=X_sample.columns)
    
    features_to_keep = X_scaled.columns.tolist()
    
    while True:
        vif_data = pd.DataFrame()
        vif_data["feature"] = features_to_keep
        
        # Calculate VIF
        vifs = []
        for i in range(len(features_to_keep)):
            try:
                v = variance_inflation_factor(X_scaled[features_to_keep].values, i)
            except:
                v = np.inf
            vifs.append(v)
            
        vif_data["VIF"] = vifs
        
        max_vif = vif_data["VIF"].max()
        if max_vif > threshold:
            max_feat = vif_data.loc[vif_data["VIF"].idxmax(), "feature"]
            logger.info(f"  Dropping {max_feat} (VIF = {max_vif:.2f})")
            features_to_keep.remove(max_feat)
        else:
            break
            
    logger.info(f"Features kept after VIF filter: {len(features_to_keep)}")
    return features_to_keep


def tune_and_train_rf(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Tune RF using TimeSeriesSplit Walk-Forward CV."""
    logger.info("Tuning Random Forest...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    param_dist = {
        'n_estimators': [300, 500],
        'max_depth': [8, 12, 15],
        'min_samples_split': [10, 30, 50],
        'min_samples_leaf': [5, 15, 25],
        'max_features': ['sqrt', 'log2']
    }
    
    base_model = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1)
    
    search = RandomizedSearchCV(
        base_model, param_distributions=param_dist, n_iter=10, 
        cv=tscv, scoring='f1_macro', n_jobs=1, random_state=42
    )
    search.fit(X_train, y_train)
    
    logger.info(f"Best RF Params: {search.best_params_}")
    return search.best_estimator_


def tune_and_train_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> XGBWrapper:
    """Tune XGBoost using TimeSeriesSplit."""
    logger.info("Tuning XGBoost...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    y_train_mapped = y_train.map(config.LABEL_MAP)
    
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_mapped)
    
    param_dist = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [300, 500],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8],
        'gamma': [0, 1, 3],
    }
    
    base_model = XGBClassifier(
        objective="multi:softprob", num_class=3, use_label_encoder=False, 
        eval_metric="mlogloss", random_state=42, tree_method="hist", verbosity=0
    )
    
    search = RandomizedSearchCV(
        base_model, param_distributions=param_dist, n_iter=10, 
        cv=tscv, scoring='f1_macro', n_jobs=1, random_state=42
    )
    search.fit(X_train, y_train_mapped, sample_weight=sample_weights)
    
    logger.info(f"Best XGB Params: {search.best_params_}")
    wrapped = XGBWrapper(search.best_estimator_, config.LABEL_MAP, config.LABEL_MAP_INV)
    return wrapped


def tune_and_train_lgb(X_train: pd.DataFrame, y_train: pd.Series) -> LGBMWrapper:
    """Tune LightGBM using TimeSeriesSplit."""
    logger.info("Tuning LightGBM...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    y_train_mapped = y_train.map(config.LABEL_MAP)
    
    param_dist = {
        'max_depth': [3, 5, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [300, 500],
        'num_leaves': [15, 31, 63],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9],
    }
    
    base_model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, class_weight='balanced',
        random_state=42, verbose=-1
    )
    
    search = RandomizedSearchCV(
        base_model, param_distributions=param_dist, n_iter=10, 
        cv=tscv, scoring='f1_macro', n_jobs=1, random_state=42
    )
    search.fit(X_train, y_train_mapped)
    
    logger.info(f"Best LGB Params: {search.best_params_}")
    wrapped = LGBMWrapper(search.best_estimator_, config.LABEL_MAP, config.LABEL_MAP_INV)
    return wrapped


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple:
    """Train LR baseline with StandardScaler."""
    logger.info("Training Logistic Regression baseline...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42, solver="lbfgs", C=0.1)
    model.fit(X_train_scaled, y_train)

    return model, scaler


def train_all_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    """
    v4.0 pipeline:
    1. Tuned Random Forest
    2. Tuned XGBoost
    3. Tuned LightGBM
    4. Ensemble (RF + XGB + LGB)
    5. Logistic Regression
    """
    # 1. Select top features
    if config.FEATURE_SELECTION_ENABLED:
        selected_features = select_top_features(X_train, y_train)
        # 1.5 NEW: VIF filter to remove multicollinearity
        vif_selected = drop_high_vif_features(X_train[selected_features])
        X_train = X_train[vif_selected]
        selected_features = vif_selected
    else:
        selected_features = X_train.columns.tolist()

    models = {}

    # RF
    models["random_forest"] = tune_and_train_rf(X_train, y_train)
    
    # XGB
    models["xgboost"] = tune_and_train_xgb(X_train, y_train)

    # LGB
    models["lightgbm"] = tune_and_train_lgb(X_train, y_train)

    # Ensemble
    manual_ensemble = ManualEnsemble(
        [models["random_forest"], models["xgboost"], models["lightgbm"]],
        weights=[1.0, 1.0, 1.0]
    )
    
    # 4.5 NEW: Calibrate probability output of the ensemble using Platt Scaling
    # Since ManualEnsemble doesn't support the full scikit-learn API natively for calibration,
    # we'll just calibrate the logistic regression, OR wait, CalibratedClassifierCV requires fit().
    # ManualEnsemble is already fitted. We would need to fit it on X_val, but here we don't have
    # a separate holdout internally unless we use prefit.
    
    # Let's just return the manual ensemble directly since Platt Scaling on custom ensembles is tricky.
    # Actually, we can just use the manual ensemble for now.
    models["ensemble"] = manual_ensemble

    # LR
    lr_model, scaler = train_logistic_regression(X_train, y_train)
    models["logistic_regression"] = lr_model
    models["scaler"] = scaler

    return {
        "models": models,
        "feature_cols": selected_features
    }
