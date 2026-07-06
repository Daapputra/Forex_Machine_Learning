"""
ablation_study.py — Phase 7 Ablation Study
==========================================
Tests the contribution of individual components to the final ROI.
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

logger = logging.getLogger("AblationStudy")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


def run_ablation_experiment(df_features, feature_cols, experiment_name: str, skip_ensemble: bool = False, skip_class_weight: bool = False, disable_regime_filter: bool = False):
    """Run a single ablation experiment and return ROI."""
    
    logger.info(f"\n--- Running Ablation: {experiment_name} ---")
    
    X_train, y_train, X_val, y_val, X_test, y_test = splitter.split_data(df_features, feature_cols)
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    
    # Train Models
    # A bit hacky: we use config to communicate class_weight intent
    # But since we can't easily pass it to train_all_models without refactoring,
    # we just note it. (In a real scenario we'd pass it as an argument).
    # For now, we'll just run the experiment.
    
    result = trainer.train_all_models(X_train_full, y_train_full)
    trained_models = result["models"]
    selected_features = result["feature_cols"]
    
    # Tune Thresholds
    df_val = df_features.loc[X_val.index]
    
    model_to_use = trained_models["lightgbm"] if skip_ensemble else trained_models["ensemble"]
    
    best_thresholds = threshold_tuner.tune_thresholds(
        model=model_to_use, 
        X_val=X_val[selected_features], 
        y_val=y_val, 
        df_features_val=df_val
    )
    
    min_conf = best_thresholds["min_confidence"]
    min_adx = 0.0 if disable_regime_filter else best_thresholds["min_adx"]
    
    # Test
    X_test_selected = X_test[selected_features]
    df_test = df_features.loc[X_test.index].copy()
    
    y_pred = model_to_use.predict(X_test_selected)
    y_prob = model_to_use.predict_proba(X_test_selected)
    
    confidences = []
    classes = model_to_use.classes_
    for i, p in enumerate(y_pred):
        class_idx = np.where(classes == p)[0][0]
        confidences.append(y_prob[i][class_idx])
        
    df_test["prediction"] = y_pred
    df_test["confidence"] = confidences
    
    # Hack for disable_regime_filter
    if disable_regime_filter:
        df_test["Vol_Regime"] = 1  # Force pass
        df_test["ADX_14"] = 100    # Force pass
        
    metrics = backtester.run_backtest(
        df_test, model_name=f"ablation_{experiment_name}", 
        min_confidence=min_conf, min_adx=min_adx
    )
    
    roi = metrics["roi_pct"]
    trades = metrics["total_trades"]
    logger.info(f"Result [{experiment_name}]: {trades} trades, ROI {roi:.2f}%")
    
    return roi, trades


def run_full_ablation():
    logger.info("Starting Full Ablation Study (Phase 7)...")
    
    # 1. Base Data
    df_raw = data_loader.load_data()
    df_clean = data_cleaner.clean_data(df_raw)
    df_labeled = labeler.label_data(df_clean)
    
    results = []
    
    # Baseline (Full System)
    df_feat_full = features.engineer_features(df_labeled)
    cols_full = features.get_feature_columns(df_feat_full)
    roi_base, tr_base = run_ablation_experiment(df_feat_full, cols_full, "Baseline (Full System)")
    results.append({"Component": "Baseline (Full System)", "ROI": roi_base, "Trades": tr_base, "Diff": 0.0})
    
    # Ablation 1: No Regime Filter (Simulated in backtester)
    roi, tr = run_ablation_experiment(df_feat_full, cols_full, "No Regime Filter (ADX/Vol)", disable_regime_filter=True)
    results.append({"Component": "No Regime Filter", "ROI": roi, "Trades": tr, "Diff": roi - roi_base})
    
    # Ablation 2: No Ensemble (Use LightGBM only)
    roi, tr = run_ablation_experiment(df_feat_full, cols_full, "No Ensemble (LGBM Only)", skip_ensemble=True)
    results.append({"Component": "No Ensemble", "ROI": roi, "Trades": tr, "Diff": roi - roi_base})
    
    # Ablation 3: No MTF Features
    df_feat_no_mtf = df_feat_full.drop(columns=["D1_MA_20", "D1_Return"], errors='ignore')
    cols_no_mtf = features.get_feature_columns(df_feat_no_mtf)
    roi, tr = run_ablation_experiment(df_feat_no_mtf, cols_no_mtf, "No MTF Features")
    results.append({"Component": "No MTF Features", "ROI": roi, "Trades": tr, "Diff": roi - roi_base})
    
    # Ablation 4: No Cyclical Features
    df_feat_no_cyc = df_feat_full.drop(columns=["hour_sin", "hour_cos"], errors='ignore')
    cols_no_cyc = features.get_feature_columns(df_feat_no_cyc)
    roi, tr = run_ablation_experiment(df_feat_no_cyc, cols_no_cyc, "No Cyclical Features")
    results.append({"Component": "No Cyclical Features", "ROI": roi, "Trades": tr, "Diff": roi - roi_base})
    
    # Summary
    df_results = pd.DataFrame(results)
    
    logger.info("\n" + "="*60)
    logger.info("ABLATION STUDY RESULTS")
    logger.info("="*60)
    logger.info(f"\n{df_results.to_string()}")
    
    df_results.to_csv(os.path.join(config.OUTPUTS_DIR, "ablation_study_results.csv"), index=False)

if __name__ == "__main__":
    run_full_ablation()
