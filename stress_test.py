"""
stress_test.py — Phase 8 Sensitivity & Stress Testing
=======================================================
Varies thresholds and risk parameters to ensure the strategy is robust
and not overfit to a specific magical number.
"""

import os
import logging
import pandas as pd
import numpy as np

import config
import data_loader
import data_cleaner
import labeler
import features
import splitter
import trainer
import threshold_tuner
import backtester

logger = logging.getLogger("StressTest")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

def run_stress_tests():
    logger.info("Starting Phase 8 Sensitivity & Stress Test...")
    
    # 1. Base Data & Model (Skip full tuning for speed, just use a basic model or pre-trained)
    df_raw = data_loader.load_data()
    df_clean = data_cleaner.clean_data(df_raw)
    df_labeled = labeler.label_data(df_clean)
    df_features = features.engineer_features(df_labeled)
    feature_cols = features.get_feature_columns(df_features)
    
    X_train, y_train, X_val, y_val, X_test, y_test = splitter.split_data(df_features, feature_cols)
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    
    # Quick train
    from lightgbm import LGBMClassifier
    from trainer import LGBMWrapper
    
    y_train_mapped = y_train_full.map(config.LABEL_MAP)
    model = LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42, verbose=-1)
    model.fit(X_train_full[feature_cols], y_train_mapped)
    wrapped_model = LGBMWrapper(model, config.LABEL_MAP, config.LABEL_MAP_INV)
    
    X_test_selected = X_test[feature_cols]
    df_test = df_features.loc[X_test.index].copy()
    
    y_pred = wrapped_model.predict(X_test_selected)
    y_prob = wrapped_model.predict_proba(X_test_selected)
    
    confidences = []
    classes = wrapped_model.classes_
    for i, p in enumerate(y_pred):
        class_idx = np.where(classes == p)[0][0]
        confidences.append(y_prob[i][class_idx])
        
    df_test["prediction"] = y_pred
    df_test["confidence"] = confidences
    
    results = []
    
    # Baseline
    base_conf = 0.45
    base_adx = 15.0
    metrics = backtester.run_backtest(df_test, "Stress_Base", base_conf, base_adx)
    results.append({"Scenario": "Baseline", "ROI": metrics["roi_pct"], "MDD": metrics["max_drawdown_pct"]})
    
    # Sensitivity 1: Confidence +- 5%
    metrics = backtester.run_backtest(df_test, "Stress_Conf_High", base_conf + 0.05, base_adx)
    results.append({"Scenario": "Confidence +5%", "ROI": metrics["roi_pct"], "MDD": metrics["max_drawdown_pct"]})
    
    metrics = backtester.run_backtest(df_test, "Stress_Conf_Low", base_conf - 0.05, base_adx)
    results.append({"Scenario": "Confidence -5%", "ROI": metrics["roi_pct"], "MDD": metrics["max_drawdown_pct"]})
    
    # Sensitivity 2: ADX +- 3
    metrics = backtester.run_backtest(df_test, "Stress_ADX_High", base_conf, base_adx + 3.0)
    results.append({"Scenario": "ADX +3", "ROI": metrics["roi_pct"], "MDD": metrics["max_drawdown_pct"]})
    
    metrics = backtester.run_backtest(df_test, "Stress_ADX_Low", base_conf, base_adx - 3.0)
    results.append({"Scenario": "ADX -3", "ROI": metrics["roi_pct"], "MDD": metrics["max_drawdown_pct"]})
    
    # Sensitivity 3: Tighter SL/TP
    orig_sl = config.SL_ATR_MULT
    orig_tp = config.TP_ATR_MULT
    
    config.SL_ATR_MULT *= 0.5
    config.TP_ATR_MULT *= 0.5
    metrics = backtester.run_backtest(df_test, "Stress_Tight_SLTP", base_conf, base_adx)
    results.append({"Scenario": "Tighter SL/TP (50%)", "ROI": metrics["roi_pct"], "MDD": metrics["max_drawdown_pct"]})
    
    config.SL_ATR_MULT = orig_sl * 1.5
    config.TP_ATR_MULT = orig_tp * 1.5
    metrics = backtester.run_backtest(df_test, "Stress_Wide_SLTP", base_conf, base_adx)
    results.append({"Scenario": "Wider SL/TP (150%)", "ROI": metrics["roi_pct"], "MDD": metrics["max_drawdown_pct"]})
    
    # Reset
    config.SL_ATR_MULT = orig_sl
    config.TP_ATR_MULT = orig_tp
    
    df_results = pd.DataFrame(results)
    
    logger.info("\n" + "="*60)
    logger.info("STRESS TEST RESULTS")
    logger.info("="*60)
    logger.info(f"\n{df_results.to_string()}")
    
    df_results.to_csv(os.path.join(config.OUTPUTS_DIR, "stress_test_results.csv"), index=False)
    
if __name__ == "__main__":
    run_stress_tests()
