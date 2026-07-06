"""
run_multipair.py — Multi-Pair Validation (Phase 1 Upgrade)
============================================================
Runs the pipeline across multiple currency pairs to prove generalization
and pools the trades to increase sample size for statistical significance.
"""

import os
import glob
import logging
import pandas as pd
import config
import data_loader
import data_cleaner
import labeler
import features
import splitter
import trainer
import threshold_tuner
import backtester

# Setup basic logging for multipair
logger = logging.getLogger("MultiPair")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

# The pairs we want to test
TARGET_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]

def check_data_exists(pair: str) -> bool:
    """Check if we have historical M1 CSV data for the pair."""
    pattern = os.path.join(config.DATA_DIR, f"DAT_ASCII_{pair}_M1_*.csv")
    files = glob.glob(pattern)
    return len(files) > 0

def run_single_pair(pair: str) -> pd.DataFrame:
    """Run the pipeline for a single pair and return the hold-out test trades."""
    logger.info(f"\n{'='*50}\nProcessing {pair}...\n{'='*50}")
    
    # Override config temporarily
    original_pair = config.PAIR
    config.PAIR = pair
    
    # 1. Prepare Data
    df_raw = data_loader.load_data()
    df_clean = data_cleaner.clean_data(df_raw)
    df_labeled = labeler.label_data(df_clean)
    df_features = features.engineer_features(df_labeled)
    feature_cols = features.get_feature_columns(df_features)
    
    X_train, y_train, X_val, y_val, X_test, y_test = splitter.split_data(
        df_features, feature_cols
    )
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    
    # 2. Train Models
    result = trainer.train_all_models(X_train_full, y_train_full)
    trained_models = result["models"]
    selected_features = result["feature_cols"]
    
    # 3. Tune Thresholds on Val
    df_val = df_features.loc[X_val.index]
    best_thresholds = threshold_tuner.tune_thresholds(
        model=trained_models["ensemble"], 
        X_val=X_val[selected_features], 
        y_val=y_val, 
        df_features_val=df_val
    )
    
    min_conf = best_thresholds["min_confidence"]
    min_adx = best_thresholds["min_adx"]
    
    # 4. Backtest on Test
    X_test_selected = X_test[selected_features]
    df_test = df_features.loc[X_test.index].copy()
    
    y_pred = trained_models["ensemble"].predict(X_test_selected)
    y_prob = trained_models["ensemble"].predict_proba(X_test_selected)
    
    confidences = []
    classes = trained_models["ensemble"].classes_
    for i, p in enumerate(y_pred):
        import numpy as np
        class_idx = np.where(classes == p)[0][0]
        confidences.append(y_prob[i][class_idx])
        
    df_test["prediction"] = y_pred
    df_test["confidence"] = confidences
    
    metrics = backtester.run_backtest(
        df_test, model_name=f"ensemble_{pair}", 
        min_confidence=min_conf, min_adx=min_adx
    )
    
    # Restore original config
    config.PAIR = original_pair
    
    if "trades_df" in metrics:
        df_trades = metrics["trades_df"]
        df_trades["pair"] = pair
        return df_trades
    return pd.DataFrame()

def pool_trades():
    all_trades = []
    for pair in TARGET_PAIRS:
        if check_data_exists(pair):
            df_trades = run_single_pair(pair)
            if not df_trades.empty:
                all_trades.append(df_trades)
        else:
            logger.warning(f"No data found for {pair}. Skipping.")
            
    if not all_trades:
        logger.error("No trades generated across any pair.")
        return
        
    pooled_df = pd.concat(all_trades, ignore_index=True)
    
    total_trades = len(pooled_df)
    wins = len(pooled_df[pooled_df["pnl_dollars"] > 0])
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    total_pnl = pooled_df["pnl_dollars"].sum()
    expectancy = total_pnl / total_trades if total_trades > 0 else 0
    
    logger.info("\n" + "="*50)
    logger.info("POOLED MULTI-PAIR RESULTS (Phase 1 Validation)")
    logger.info("="*50)
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Win Rate:     {win_rate:.2f}%")
    logger.info(f"Expectancy:   ${expectancy:.2f} per trade")
    logger.info(f"Total PnL:    ${total_pnl:.2f}")
    
    # Calculate approximate pooled ROI
    # Assuming initial capital is 10k per pair
    total_capital = config.INITIAL_CAPITAL * len(all_trades)
    roi_pct = (total_pnl / total_capital) * 100
    logger.info(f"Pooled ROI:   {roi_pct:.2f}%")
    
if __name__ == "__main__":
    pool_trades()
