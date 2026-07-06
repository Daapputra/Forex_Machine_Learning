"""
diagnostics.py — Phase 1 Analysis for v4.0 Upgrade
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import logging

import config
import data_loader
import data_cleaner
import labeler
import features
import splitter

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def run_diagnostics():
    logger.info("=== PHASE 1: V3.0 DIAGNOSTICS ===")

    # 1. Load Data
    df_raw = data_loader.load_data()
    df_clean = data_cleaner.clean_data(df_raw)
    df_labeled = labeler.label_data(df_clean)
    df_features = features.engineer_features(df_labeled)
    
    feature_cols = features.get_feature_columns(df_features)
    X_train, y_train, X_val, y_val, X_test, y_test = splitter.split_data(df_features, feature_cols)

    # 2. Load Model
    model_path = os.path.join(config.MODELS_DIR, "EURUSD_H1_random_forest_v3.0.0.pkl")
    if not os.path.exists(model_path):
        logger.error("v3.0 model not found!")
        return

    model = joblib.load(model_path)
    
    # Needs to match v3.0 selected features (from logs: 25 features)
    selected = ['hour_cos', 'hour_sin', 'is_london', 'ATR_Ratio', 'RetStd_12', 'atr_14', 'RetStd_3', 'CCI_20', 'RetStd_6', 'Return_1', 'RetStd_24', 'BB_Width', 'ATR_14', 'DI_plus', 'Upper_Wick', 'RetMean_3', 'MACD_Signal', 'Lower_Wick', 'ADX_14', 'BB_PctB', 'Body_Ratio', 'Stoch_D', 'RetMean_24', 'Return_5', 'DI_minus']
    
    X_train = X_train[selected]
    X_test = X_test[selected]
    
    # 3. Gap Analysis
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    logger.info(f"\n1. Gap Analysis (Overfitting Check):")
    logger.info(f"Train Accuracy: {train_acc:.4f}")
    logger.info(f"Test Accuracy:  {test_acc:.4f}")
    gap = train_acc - test_acc
    logger.info(f"Gap: {gap*100:.2f}% -> {'OVERFITTING' if gap > 0.1 else 'OK'}")

    # 4. Confidence Distribution
    probs = model.predict_proba(X_test)
    # classes: [-1, 0, 1] usually map to [0, 1, 2] indices
    sell_probs = probs[:, 0]
    buy_probs = probs[:, 2]
    
    max_trade_probs = []
    for i in range(len(probs)):
        pred = model.predict(X_test.iloc[i:i+1])[0]
        if pred == -1:
            max_trade_probs.append(sell_probs[i])
        elif pred == 1:
            max_trade_probs.append(buy_probs[i])
            
    logger.info(f"\n2. Confidence Distribution (on executed trades):")
    max_trade_probs = np.array(max_trade_probs)
    if len(max_trade_probs) > 0:
        logger.info(f"Mean Confidence: {np.mean(max_trade_probs):.4f}")
        logger.info(f"Median Confidence: {np.median(max_trade_probs):.4f}")
        low_conf_pct = np.mean(max_trade_probs < 0.6) * 100
        logger.info(f"Trades with < 60% confidence: {low_conf_pct:.1f}%")
    else:
        logger.info("No trades would be executed.")

    # 5. Regime Analysis (using ADX)
    logger.info(f"\n3. Regime Analysis (Trending vs Sideways via ADX):")
    df_test_full = df_features.loc[X_test.index]
    is_trending = df_test_full["ADX_14"] > 25
    is_sideways = df_test_full["ADX_14"] <= 25
    
    y_pred = model.predict(X_test)
    
    acc_trend = np.mean(y_pred[is_trending] == y_test[is_trending]) if sum(is_trending) > 0 else 0
    acc_side = np.mean(y_pred[is_sideways] == y_test[is_sideways]) if sum(is_sideways) > 0 else 0
    
    logger.info(f"Accuracy in Trending (ADX > 25): {acc_trend:.4f} (n={sum(is_trending)})")
    logger.info(f"Accuracy in Sideways (ADX <= 25): {acc_side:.4f} (n={sum(is_sideways)})")

    # 6. Feature Importance
    logger.info(f"\n4. Feature Importance (Top 5 vs Bottom 5):")
    importances = pd.Series(model.feature_importances_, index=selected).sort_values(ascending=False)
    logger.info("Top 5:")
    logger.info(importances.head(5).to_string())
    logger.info("\nBottom 5:")
    logger.info(importances.tail(5).to_string())
    
    report = f"""# v3.0 Diagnostic Report
    
## 1. Gap Analysis
Train Acc: {train_acc:.4f}
Test Acc: {test_acc:.4f}
Gap: {gap*100:.2f}% ({'OVERFITTING' if gap > 0.1 else 'OK'})

## 2. Confidence
Mean: {np.mean(max_trade_probs):.4f}
<60% Confidence: {low_conf_pct:.1f}%

## 3. Regime Analysis
Trending (ADX > 25) Acc: {acc_trend:.4f}
Sideways (ADX <= 25) Acc: {acc_side:.4f}

## 4. Feature Importance
Top 5:
{importances.head(5).to_string()}
Bottom 5:
{importances.tail(5).to_string()}
"""
    with open(os.path.join(config.OUTPUTS_DIR, "v3_diagnostics.txt"), "w") as f:
        f.write(report)

if __name__ == "__main__":
    run_diagnostics()
