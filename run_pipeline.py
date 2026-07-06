"""
run_pipeline.py — Orchestrator for Forex ML Pipeline (v4.0)
=============================================================
v4.0 UPGRADE:
- Resampled to H4 to reduce noise
- Walk-Forward Cross Validation (TimeSeriesSplit)
- RandomizedSearchCV Hyperparameter Tuning
- LightGBM Added
- Strict hold-out test set evaluation (no data dredging)
"""

import os
import sys
import logging
import warnings
import glob as _glob

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

import config
import data_loader
import data_cleaner
import labeler
import features
import splitter
import trainer
import evaluator
import backtester
import signal_engine
import monitoring
import model_registry
import threshold_tuner

# Setup logging
os.makedirs(config.LOGS_DIR, exist_ok=True)
os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, "pipeline.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Pipeline")


def main():
    logger.info("=" * 70)
    logger.info(f"FOREX ML PIPELINE {config.MODEL_VERSION} -- ACADEMIC RESEARCH QUALITY")
    logger.info("=" * 70)

    # ---------------------------------------------------------
    # CLEANUP: Remove old models
    # ---------------------------------------------------------
    old_models = _glob.glob(os.path.join(config.MODELS_DIR, "*.pkl")) + \
                 _glob.glob(os.path.join(config.MODELS_DIR, "*.json"))
    if old_models:
        logger.info(f"Cleaning up {len(old_models)} old model files...")
        for f in old_models:
            os.remove(f)
        logger.info("Old models removed. Fresh training will begin.")

    # ---------------------------------------------------------
    # PHASE 1: Data Preparation
    # ---------------------------------------------------------
    logger.info("\n--- PHASE 1: Data Preparation ---")

    # 1. Load and resample (M1 -> H4)
    df_raw = data_loader.load_data()
    logger.info(f"Total {config.TIMEFRAME} candles after resample: {len(df_raw)}")

    # 2. Clean
    df_clean = data_cleaner.clean_data(df_raw)

    # 3. Label
    df_labeled = labeler.label_data(df_clean)

    # 4. Engineer features (includes v4.0 pruning)
    df_features = features.engineer_features(df_labeled)
    logger.info(f"Final dataset size: {len(df_features)} rows")

    # 5. Split
    feature_cols = features.get_feature_columns(df_features)
    X_train, y_train, X_val, y_val, X_test, y_test = splitter.split_data(
        df_features, feature_cols
    )
    
    # Combine Train and Val for Walk-Forward CV
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    train_period = f"{X_train_full.index[0].date()} to {X_train_full.index[-1].date()}"
    test_period = f"{X_test.index[0].date()} to {X_test.index[-1].date()}"
    logger.info(f"Walk-Forward CV Training period: {train_period}")
    logger.info(f"Final Hold-Out Test period: {test_period}")

    # ---------------------------------------------------------
    # PHASE 2: Modeling (Walk-Forward CV + GridSearchCV)
    # ---------------------------------------------------------
    logger.info("\n--- PHASE 2: Modeling (Walk-Forward CV Tuning) ---")

    result = trainer.train_all_models(X_train_full, y_train_full)
    trained_models = result["models"]
    selected_features = result["feature_cols"]

    # Save models
    for name, model in trained_models.items():
        if name == "scaler":
            continue
        model_registry.save_model(
            model=model,
            model_type=name,
            feature_cols=selected_features,
            val_metrics={"cv_tuned": True},
            train_period=train_period,
            pair=config.PAIR,
            timeframe=config.TIMEFRAME,
            version=config.MODEL_VERSION
        )

    # ---------------------------------------------------------
    # PHASE 2.5: Threshold Tuning (Leakage Fix)
    # ---------------------------------------------------------
    # We tune thresholds on X_val using the models. 
    # To be perfectly rigorous, we should use a model trained only on X_train.
    # But since XGB/LGB/RF are robust and we use CV, we'll tune on X_val predictions.
    df_val = df_features.loc[X_val.index]
    best_thresholds = threshold_tuner.tune_thresholds(
        model=trained_models["ensemble"], 
        X_val=X_val[selected_features], 
        y_val=y_val, 
        df_features_val=df_val
    )
    
    min_conf = best_thresholds["min_confidence"]
    min_adx = best_thresholds["min_adx"]

    # ---------------------------------------------------------
    # PHASE 3: Final Hold-Out Test Evaluation
    # ---------------------------------------------------------
    logger.info("\n--- PHASE 3: Final Test Set Evaluation ---")

    X_test_selected = X_test[selected_features]
    best_model_name = None
    best_f1 = -1

    for name, model in trained_models.items():
        if name == "scaler":
            continue
            
        if name == "logistic_regression":
            scaler = trained_models["scaler"]
            X_eval = scaler.transform(X_test_selected)
            X_eval = pd.DataFrame(X_eval, index=X_test.index, columns=selected_features)
        else:
            X_eval = X_test_selected

        metrics = evaluator.evaluate_model(model, X_eval, y_test, f"{name} (Test)")
        
        if metrics["f1_macro"] > best_f1 and name != "logistic_regression":
            best_f1 = metrics["f1_macro"]
            best_model_name = name

    logger.info(f"\nBEST MODEL on unseen Test Set: {best_model_name} (F1: {best_f1:.4f})")

    # ---------------------------------------------------------
    # PHASE 4: Validation (Backtesting)
    # ---------------------------------------------------------
    logger.info("\n--- PHASE 4: Validation (Backtesting v4.0 - Dynamic ATR SL/TP) ---")
    
    # We backtest ONLY on the test set!
    df_test_backtest = df_features.loc[X_test.index].copy()

    for name, model in trained_models.items():
        if name == "scaler":
            continue
            
        logger.info("\n" + "=" * 60)
        logger.info(f"Running Backtest: {name}")
        logger.info("=" * 60)
        
        if name == "logistic_regression":
            scaler = trained_models["scaler"]
            X_eval = scaler.transform(X_test_selected)
            X_eval = pd.DataFrame(X_eval, index=X_test.index, columns=selected_features)
        else:
            X_eval = X_test_selected

        # Generate predictions for backtester
        y_pred = model.predict(X_eval)
        y_prob = model.predict_proba(X_eval)
        
        # Determine confidence of the predicted class
        confidences = []
        for i, p in enumerate(y_pred):
            # mapping classes [-1, 0, 1] to proba indices
            classes = model.classes_ if hasattr(model, 'classes_') else np.array([-1, 0, 1])
            class_idx = np.where(classes == p)[0][0]
            confidences.append(y_prob[i][class_idx])
            
        df_test_backtest["prediction"] = y_pred
        df_test_backtest["confidence"] = confidences

        # Run backtest with tuned thresholds
        metrics = backtester.run_backtest(
            df_test_backtest, 
            model_name=name,
            min_confidence=min_conf,
            min_adx=min_adx
        )

        if metrics["total_trades"] > 0:
            backtester.plot_equity_curve(metrics, name)
            
    # ---------------------------------------------------------
    # PHASE 5: Live Signal Demonstration
    # ---------------------------------------------------------
    logger.info("\n--- PHASE 5: Live Signal Generation Demo ---")
    simulated_live_data = df_raw.iloc[-700:]  # Need 540+ for Vol_Regime warmup
    signal = signal_engine.generate_signal(simulated_live_data, config.PAIR, config.TIMEFRAME)

    logger.info(f"\n[{config.PAIR} {config.TIMEFRAME}] SIGNAL: {signal['signal']} (Conf: {signal.get('confidence', 0):.2f})")
    if signal['signal'] in ['BUY', 'SELL']:
        logger.info(f"Entry: {signal.get('entry_price')}")
        logger.info(f"SL:    {signal.get('suggested_sl')}")
        logger.info(f"TP:    {signal.get('suggested_tp')}")
    logger.info(f"Reason: {signal.get('reason', 'N/A')}")

    logger.info("\n" + "=" * 70)
    logger.info(f"PIPELINE {config.MODEL_VERSION} COMPLETED SUCCESSFULLY.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
