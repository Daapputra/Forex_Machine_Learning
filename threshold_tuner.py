"""
threshold_tuner.py — Tuning Trading Thresholds Without Data Leakage (Phase 0 Fix)
================================================================================
Tunes MIN_CONFIDENCE and MIN_ADX_TREND on the Validation set so we don't
overfit to the Test set.
"""

import pandas as pd
import numpy as np
import logging
import config
import backtester

logger = logging.getLogger(__name__)

def tune_thresholds(model, X_val, y_val, df_features_val):
    """
    Grid search for best confidence and ADX thresholds on the validation set.
    Goal: Maximize Total Trades (subject to ROI > 0) or Maximize ROI.
    """
    logger.info("Tuning Trading Thresholds on Validation Set (Leakage Fix)...")
    
    # Generate predictions on validation set
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)
    
    confidences = []
    classes = model.classes_ if hasattr(model, 'classes_') else np.array([-1, 0, 1])
    for i, p in enumerate(y_pred):
        class_idx = np.where(classes == p)[0][0]
        confidences.append(y_prob[i][class_idx])
        
    df_val = df_features_val.copy()
    df_val["prediction"] = y_pred
    df_val["confidence"] = confidences
    
    best_roi = -999.0
    best_params = {"min_confidence": 0.45, "min_adx": 15.0}
    
    results = []
    
    # Temporarily mute backtester logging for grid search
    backtest_logger = logging.getLogger("backtester")
    prev_level = backtest_logger.level
    backtest_logger.setLevel(logging.WARNING)
    
    for conf in config.CONFIDENCE_GRID:
        for adx in config.ADX_GRID:
            metrics = backtester.run_backtest(df_val, model_name="val_tuning", min_confidence=conf, min_adx=adx)
            roi = metrics["roi_pct"]
            trades = metrics["total_trades"]
            
            results.append({
                "conf": conf,
                "adx": adx,
                "trades": trades,
                "roi": roi
            })
            
            # Optimization logic: we want at least 10 trades on val, max ROI
            if trades >= 10 and roi > best_roi:
                best_roi = roi
                best_params = {"min_confidence": conf, "min_adx": adx}
                
    backtest_logger.setLevel(prev_level)
    
    # Fallback if no profitable settings with >= 10 trades
    if best_roi == -999.0:
        logger.warning("Could not find highly profitable thresholds on Validation. Using fallback defaults.")
        best_params = {"min_confidence": 0.45, "min_adx": 15.0}
        
    logger.info(f"Threshold Tuning Results on Validation:")
    for res in results:
        logger.info(f"  Conf: {res['conf']:.2f}, ADX: {res['adx']}, Trades: {res['trades']}, ROI: {res['roi']:.2f}%")
        
    logger.info(f"Selected Best Thresholds: {best_params}")
    return best_params

