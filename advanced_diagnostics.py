"""
advanced_diagnostics.py — Phase 4 Statistical Validation
======================================================
Implements Bootstrap Resampling (95% CI) and Monte Carlo Permutation testing.
"""

import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import os
import config
import shap

logger = logging.getLogger(__name__)

def bootstrap_metrics(trades_df: pd.DataFrame, n_iterations: int = 5000) -> dict:
    """
    Bootstrap resampling to compute 95% Confidence Intervals for ROI and Win Rate.
    """
    logger.info(f"Running Bootstrap Resampling ({n_iterations} iterations)...")
    
    if trades_df.empty:
        logger.warning("No trades available for bootstrap.")
        return {}
        
    n_trades = len(trades_df)
    rois = []
    win_rates = []
    
    for i in range(n_iterations):
        # Sample with replacement
        sample_trades = trades_df.sample(n=n_trades, replace=True)
        
        wins = len(sample_trades[sample_trades["pnl_dollars"] > 0])
        wr = (wins / n_trades) * 100
        
        total_pnl = sample_trades["pnl_dollars"].sum()
        roi = (total_pnl / config.INITIAL_CAPITAL) * 100
        
        rois.append(roi)
        win_rates.append(wr)
        
    # Compute 95% CI
    roi_ci_lower = np.percentile(rois, 2.5)
    roi_ci_upper = np.percentile(rois, 97.5)
    
    wr_ci_lower = np.percentile(win_rates, 2.5)
    wr_ci_upper = np.percentile(win_rates, 97.5)
    
    logger.info(f"Bootstrap 95% CI for ROI:      [{roi_ci_lower:.2f}%, {roi_ci_upper:.2f}%]")
    logger.info(f"Bootstrap 95% CI for Win Rate: [{wr_ci_lower:.2f}%, {wr_ci_upper:.2f}%]")
    
    # Plot Distribution
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(rois, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', label='Breakeven')
    plt.axvline(x=roi_ci_lower, color='green', linestyle=':', label='95% CI')
    plt.axvline(x=roi_ci_upper, color='green', linestyle=':')
    plt.title("Bootstrap Distribution of ROI")
    plt.xlabel("ROI (%)")
    plt.ylabel("Frequency")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(win_rates, bins=50, color='lightgreen', edgecolor='black')
    plt.axvline(x=wr_ci_lower, color='green', linestyle=':', label='95% CI')
    plt.axvline(x=wr_ci_upper, color='green', linestyle=':')
    plt.title("Bootstrap Distribution of Win Rate")
    plt.xlabel("Win Rate (%)")
    plt.ylabel("Frequency")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUTS_DIR, "bootstrap_distribution.png"))
    plt.close()
    
    return {
        "roi_ci": (roi_ci_lower, roi_ci_upper),
        "wr_ci": (wr_ci_lower, wr_ci_upper)
    }


def monte_carlo_permutation_test(model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, n_iterations: int = 1000):
    """
    Monte Carlo Permutation Test.
    Shuffles y_train labels, retrains a fast version of the model, and evaluates on y_test.
    If the real model F1 is not significantly higher than the permutation distribution,
    the model might be learning noise.
    """
    logger.info(f"Running Monte Carlo Permutation Test ({n_iterations} iterations)...")
    logger.warning("This can take a long time even with a fast model!")
    
    from sklearn.metrics import f1_score
    from lightgbm import LGBMClassifier
    
    # Use a very fast model for permutations
    fast_model = LGBMClassifier(n_estimators=50, max_depth=3, class_weight='balanced', verbose=-1)
    
    # Fit real model
    fast_model.fit(X_train, y_train)
    real_preds = fast_model.predict(X_test)
    real_f1 = f1_score(y_test, real_preds, average='macro')
    
    null_f1s = []
    
    y_train_array = y_train.values
    
    for i in range(n_iterations):
        if i % 100 == 0:
            logger.info(f"  Permutation {i}/{n_iterations}...")
            
        # Shuffle labels
        y_train_shuffled = np.random.permutation(y_train_array)
        
        # Fit on shuffled labels
        fast_model.fit(X_train, y_train_shuffled)
        
        # Predict on actual test features
        shuffled_preds = fast_model.predict(X_test)
        shuffled_f1 = f1_score(y_test, shuffled_preds, average='macro')
        
        null_f1s.append(shuffled_f1)
        
    p_value = np.mean(np.array(null_f1s) >= real_f1)
    
    logger.info(f"Real F1-Macro: {real_f1:.4f}")
    logger.info(f"Mean Null F1-Macro: {np.mean(null_f1s):.4f}")
    logger.info(f"Monte Carlo P-Value: {p_value:.4f}")
    
    # Plot Distribution
    plt.figure(figsize=(8, 6))
    plt.hist(null_f1s, bins=50, color='gray', edgecolor='black', alpha=0.7, label='Null Distribution')
    plt.axvline(x=real_f1, color='red', linestyle='-', linewidth=2, label=f'Real Model (p={p_value:.4f})')
    plt.title("Monte Carlo Permutation Test (F1-Macro)")
    plt.xlabel("F1-Macro Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUTS_DIR, "permutation_test.png"))
    plt.close()
    
    return p_value

def run_shap_analysis(model, X_train: pd.DataFrame):
    """
    Run SHAP analysis using a subset of data.
    """
    logger.info("Running SHAP Value Analysis...")
    
    # TreeExplainer is fast for tree-based models
    try:
        # Sample to avoid huge computation time
        X_sample = X_train.sample(min(1000, len(X_train)), random_state=42)
        
        # Use LightGBM from the ensemble as representative for SHAP
        # Since ManualEnsemble doesn't have an explainer, we extract LGBM
        if hasattr(model, 'models'):
            lgbm_model = None
            for m in model.models:
                if 'LGBM' in str(type(m)) or 'LGBMWrapper' in str(type(m)):
                    lgbm_model = m.model if hasattr(m, 'model') else m
                    break
            if lgbm_model is None:
                logger.warning("No LightGBM found in ensemble for SHAP analysis.")
                return
        else:
            lgbm_model = model
            
        explainer = shap.TreeExplainer(lgbm_model)
        shap_values = explainer.shap_values(X_sample)
        
        plt.figure()
        # Ensure shap_values format is handled (multiclass returns list of arrays)
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values, X_sample, show=False)
        else:
            shap.summary_plot(shap_values, X_sample, show=False)
            
        plt.savefig(os.path.join(config.OUTPUTS_DIR, "shap_summary.png"), bbox_inches="tight")
        plt.close()
        logger.info("SHAP analysis completed and plot saved.")
        
    except Exception as e:
        logger.error(f"SHAP analysis failed: {e}")
