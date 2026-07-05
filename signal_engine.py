"""
signal_engine.py — Skeleton Signal Generation (v2.0)
======================================================
Mimics production environment for live trading.
All models (RF, XGBWrapper, Ensemble) now return original labels
(-1, 0, 1) transparently — no special XGBoost mapping needed.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

import config
import data_cleaner
import features
import model_registry

logger = logging.getLogger(__name__)


def generate_signal(latest_ohlcv: pd.DataFrame, pair: str = None, timeframe: str = None) -> dict:
    """
    Generate trading signal mimicking live production flow.

    Input: latest OHLCV data (enough history for longest indicator lookback, ~100+ bars)
    Output: {pair, timeframe, signal, confidence, sl, tp, reason, timestamp}
    """
    pair = pair or config.PAIR
    timeframe = timeframe or config.TIMEFRAME

    logger.info(f"Generating signal for {pair} {timeframe} at {datetime.now()}")

    # 1. Clean data
    cleaned_df = data_cleaner.clean_data(latest_ohlcv.copy())

    # 1.5 Add ATR for backtester/features (v3.0 requirement)
    import labeler
    if config.LABEL_MODE == "atr":
        cleaned_df["atr_14"] = labeler.compute_atr_pips(cleaned_df) * config.PIP_SIZE

    # 2. Engineer features (drop_na=False: we only need the last row)
    feat_df = features.engineer_features(cleaned_df, drop_na=False)

    # Get only the latest feature vector
    latest_features = feat_df.iloc[-1:]

    if latest_features.isna().any().any():
        return {
            "signal": "NO_SIGNAL",
            "reason": "Insufficient history for features (NaNs present)",
            "timestamp": str(latest_features.index[0]) if not latest_features.empty else None
        }

    # 3. Load best model (need metadata to know which features to select)
    try:
        model, metadata = model_registry.load_best_model(pair, timeframe)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return {
            "signal": "ERROR",
            "reason": f"Model load failed: {str(e)}"
        }

    # Extract feature columns based on model metadata (v3.0 feature selection)
    X_cols = metadata.get("features", features.get_feature_columns(feat_df))
    # Ensure all required columns exist
    missing_cols = [c for c in X_cols if c not in latest_features.columns]
    if missing_cols:
        return {
            "signal": "ERROR",
            "reason": f"Missing features required by model: {missing_cols}"
        }
        
    X = latest_features[X_cols]

    # 4. Predict — all models now return original labels (-1, 0, 1)
    pred_label = model.predict(X)[0]

    # Get confidence
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        # probs ordered as [-1, 0, 1] -> [SELL, HOLD, BUY]
        if hasattr(model, "classes_"):
            class_list = list(model.classes_)
            if pred_label in class_list:
                class_idx = class_list.index(pred_label)
                confidence = float(probs[class_idx])
            else:
                confidence = float(np.max(probs))
        else:
            confidence = float(np.max(probs))

    signal_str = config.LABEL_NAMES.get(pred_label, "UNKNOWN")

    # 5. Filter confidence
    if signal_str in ["BUY", "SELL"] and confidence < config.MIN_CONFIDENCE:
        return {
            "signal": "HOLD",
            "confidence": float(confidence),
            "reason": f"Low confidence ({confidence:.2f} < {config.MIN_CONFIDENCE})",
            "timestamp": str(X.index[0]),
            "model_version": metadata.get("version")
        }

    # Calculate suggested dynamic SL/TP (v3.0)
    latest_close = float(latest_features.iloc[0]["close"])
    sl, tp = 0.0, 0.0
    
    current_atr = float(latest_features.iloc[0].get("atr_14", config.SL_PIPS * config.PIP_SIZE))
    sl_distance = current_atr * config.SL_ATR_MULT
    tp_distance = current_atr * config.TP_ATR_MULT
    
    # Enforce minimums
    min_dist = 5 * config.PIP_SIZE
    sl_distance = max(sl_distance, min_dist)
    tp_distance = max(tp_distance, min_dist * 1.5)

    if signal_str == "BUY":
        sl = latest_close - sl_distance
        tp = latest_close + tp_distance
    elif signal_str == "SELL":
        sl = latest_close + sl_distance
        tp = latest_close - tp_distance

    return {
        "signal": signal_str,
        "confidence": float(confidence),
        "suggested_sl": round(sl, 5),
        "suggested_tp": round(tp, 5),
        "entry_price": round(latest_close, 5),
        "reason": "Model signal",
        "timestamp": str(X.index[0]),
        "model_version": metadata.get("version")
    }
