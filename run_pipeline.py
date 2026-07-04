"""
run_pipeline.py — Orchestrator for Forex ML Pipeline (v2.0)
=============================================================
UPGRADED: Handles 5-year multi-file dataset, ensemble model,
feature selection, and comprehensive backtesting.
"""

import os
import sys
import logging
import warnings

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
    logger.info("FOREX ML PIPELINE v2.0 — PRODUCTION QUALITY")
    logger.info("=" * 70)

    # ---------------------------------------------------------
    # PHASE 1: Data Preparation
    # ---------------------------------------------------------
    logger.info("\n--- PHASE 1: Data Preparation ---")

    # 1. Load and resample (M1 -> H1) — now reads ALL CSVs in data/
    df_raw = data_loader.load_data()
    logger.info(f"Total H1 candles after resample: {len(df_raw)}")

    # 2. Clean
    df_clean = data_cleaner.clean_data(df_raw)

    # 3. Label (pip-returns, BEFORE feature engineering to avoid leakage)
    df_labeled = labeler.label_data(df_clean)

    # 4. Engineer features (v2.0: ~45 features, all shifted)
    df_features = features.engineer_features(df_labeled)
    logger.info(f"Final dataset size: {len(df_features)} rows")

    # 5. Split (Chronological, NO shuffle)
    feature_cols = features.get_feature_columns(df_features)
    X_train, y_train, X_val, y_val, X_test, y_test = splitter.split_data(
        df_features, feature_cols
    )

    train_period = f"{X_train.index[0].date()} to {X_train.index[-1].date()}"
    logger.info(f"Training period: {train_period}")

    # ---------------------------------------------------------
    # PHASE 2: Modeling
    # ---------------------------------------------------------
    logger.info("\n--- PHASE 2: Modeling ---")

    # Train RF, XGB, Ensemble, LR Baseline
    trained_models = trainer.train_all_models(
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, train_period
    )

    # Generate Rule-based baseline predictions for Val and Test
    # Need to use df_features (which has RSI_14 column) for rule-based
    val_df = df_features.iloc[
        int(len(df_features) * config.TRAIN_RATIO):
        int(len(df_features) * (config.TRAIN_RATIO + config.VAL_RATIO))
    ]
    test_df = df_features.iloc[
        int(len(df_features) * (config.TRAIN_RATIO + config.VAL_RATIO)):
    ]

    rule_based_val_pred = trainer.generate_rule_based_predictions(val_df, feature_cols)
    rule_based_test_pred = trainer.generate_rule_based_predictions(test_df, feature_cols)

    # Evaluate all models
    all_results = evaluator.evaluate_all_models(
        trained_models, y_val, y_test, feature_cols,
        rule_based_val_pred, rule_based_test_pred
    )

    comparison_df = evaluator.save_evaluation_report(all_results)

    # Save models to registry (skip LR baseline)
    for model_name, info in trained_models.items():
        if model_name != "logistic_regression":
            model_registry.save_model(
                model=info["model"],
                model_type=model_name,
                feature_cols=feature_cols,
                val_metrics=all_results[model_name]["val_metrics"],
                train_period=train_period
            )

    # Identify best model based on test F1 Macro
    best_model_name = None
    best_f1 = -1
    for name, res in all_results.items():
        if name == "rule_based_rsi":
            continue
        test_f1 = res["test_metrics"]["f1_macro"]
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_model_name = name

    logger.info(f"\nBEST MODEL: {best_model_name} (Test F1 Macro: {best_f1:.4f})")

    # ---------------------------------------------------------
    # PHASE 3: Validation (Backtesting)
    # ---------------------------------------------------------
    logger.info("\n--- PHASE 3: Validation (Backtesting) ---")

    # Reconstruct raw df for test period
    df_test_raw = df_labeled.loc[X_test.index]

    # Backtest best model
    best_model = trained_models[best_model_name]["model"]
    eq_df, trades_df, bt_metrics = backtester.execute_backtest(
        best_model, best_model_name, X_test, y_test, df_test_raw
    )

    # Also backtest other ML models for comparison
    for model_name in ["random_forest", "xgboost", "ensemble"]:
        if model_name != best_model_name:
            backtester.execute_backtest(
                trained_models[model_name]["model"],
                model_name, X_test, y_test, df_test_raw
            )

    # ---------------------------------------------------------
    # Save backtest summary
    # ---------------------------------------------------------
    summary_path = os.path.join(config.OUTPUTS_DIR, "backtest_summary.csv")
    bt_summary = pd.DataFrame([bt_metrics], index=[best_model_name])
    bt_summary.to_csv(summary_path)
    logger.info(f"Backtest summary saved: {summary_path}")

    # ---------------------------------------------------------
    # Demonstration: Signal Engine & Monitoring
    # ---------------------------------------------------------
    logger.info("\n--- Demonstration: Live Signal Generation ---")
    simulated_live_data = df_raw.iloc[-200:]  # More history for indicators

    signal = signal_engine.generate_signal(simulated_live_data)
    logger.info(f"Generated Live Signal:\n{signal}")

    monitoring.log_prediction(signal)

    # ---------------------------------------------------------
    # Final Summary
    # ---------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    logger.info(f"Dataset: {len(df_features)} H1 candles")
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Best Model: {best_model_name} (F1 Macro: {best_f1:.4f})")
    logger.info(f"Backtest Trades: {bt_metrics.get('Total Trades', 0)}")
    logger.info(f"Backtest Return: {bt_metrics.get('Total Return (%)', 0):.2f}%")
    logger.info(f"Max Drawdown: {bt_metrics.get('Max Drawdown (%)', 0):.2f}%")
    logger.info(f"Sharpe Ratio: {bt_metrics.get('Sharpe Ratio', 0):.2f}")
    logger.info(f"Win Rate: {bt_metrics.get('Win Rate (%)', 0):.1f}%")
    logger.info("=" * 70)
    logger.info(f"Models saved to: {config.MODELS_DIR}")
    logger.info(f"Charts saved to: {config.OUTPUTS_DIR}")
    logger.info(f"Logs saved to: {config.LOGS_DIR}")


if __name__ == "__main__":
    main()
