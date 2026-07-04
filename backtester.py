"""
backtester.py — Strategy Validation
=====================================
Simulates trading strategy using the ML model's predictions.

Rules (from PRD Phase 3):
- Spread: 2 pips, Slippage: 0.5 pips (applied at entry).
- SL: 20 pips, TP: 40 pips.
- Capital: $10,000, max 1 position active.
- Confidence filter: > 60%.
- Max daily loss: 3%.

Outputs:
- Equity curve chart.
- Trade log CSV.
- Summary metrics (Total Return, Drawdown, Sharpe, Win Rate, etc.).
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
from typing import Tuple

import config

logger = logging.getLogger(__name__)


def run_backtest(
    df: pd.DataFrame,
    predictions: pd.Series,
    probabilities: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Run backtest simulation.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame (usually the test set).
    predictions : pd.Series
        Predicted labels (-1, 0, 1).
    probabilities : pd.DataFrame, optional
        Predicted probabilities for each class.
        If provided, confidence filtering is applied.

    Returns
    -------
    tuple
        (equity_curve_df, trade_log_df, metrics_dict)
    """
    capital = config.INITIAL_CAPITAL
    equity = capital
    equity_curve = []
    trades = []
    
    position = 0          # 1 for long, -1 for short, 0 for none
    entry_price = 0.0
    entry_time = None
    sl_price = 0.0
    tp_price = 0.0
    
    daily_start_capital = capital
    current_day = df.index[0].date()
    
    pip_val = config.PIP_SIZE * config.LOT_SIZE
    cost_pips = config.SPREAD_PIPS + config.SLIPPAGE_PIPS
    
    for i in range(len(df)):
        row = df.iloc[i]
        timestamp = df.index[i]
        
        # Check for new day to reset max daily loss
        if timestamp.date() != current_day:
            daily_start_capital = equity
            current_day = timestamp.date()
            
        # 1. Check for exits if in a position
        if position != 0:
            exit_price = 0.0
            exit_reason = ""
            
            if position == 1:  # Long
                if row["low"] <= sl_price:
                    exit_price = sl_price
                    exit_reason = "SL"
                elif row["high"] >= tp_price:
                    exit_price = tp_price
                    exit_reason = "TP"
                    
            elif position == -1:  # Short
                if row["high"] >= sl_price:
                    exit_price = sl_price
                    exit_reason = "SL"
                elif row["low"] <= tp_price:
                    exit_price = tp_price
                    exit_reason = "TP"
                    
            if exit_reason:
                # Calculate P&L
                gross_pips = (exit_price - entry_price) / config.PIP_SIZE * position
                net_pips = gross_pips - cost_pips
                pnl = net_pips * (config.LOT_SIZE * config.PIP_SIZE)  # Simple PnL, assumes base=quote or USD quote
                
                equity += pnl
                
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": timestamp,
                    "type": "LONG" if position == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "reason": exit_reason,
                    "net_pips": net_pips,
                    "pnl": pnl,
                    "equity": equity
                })
                position = 0
                
        # 2. Check for entry if flat
        # Max daily loss check
        daily_loss_pct = (daily_start_capital - equity) / daily_start_capital * 100
        if daily_loss_pct >= config.MAX_DAILY_LOSS_PCT:
            equity_curve.append({"timestamp": timestamp, "equity": equity})
            continue  # Block entries for the rest of the day

        if position == 0:
            pred = predictions.iloc[i]
            
            # Confidence filter
            if probabilities is not None:
                # For xgboost/rf with proba
                # Map pred to original label space if needed, assuming pred is -1, 0, 1
                # proba columns assume 0=SELL, 1=HOLD, 2=BUY (from LABEL_MAP)
                conf = 0.0
                if pred == 1:
                    conf = probabilities.iloc[i, config.LABEL_MAP[1]]
                elif pred == -1:
                    conf = probabilities.iloc[i, config.LABEL_MAP[-1]]
                
                if conf < config.MIN_CONFIDENCE:
                    pred = 0  # Ignore signal
                    
            if pred == 1:  # BUY
                position = 1
                entry_price = row["close"] # Assuming entry at close of signal candle
                entry_time = timestamp
                sl_price = entry_price - (config.SL_PIPS * config.PIP_SIZE)
                tp_price = entry_price + (config.TP_PIPS * config.PIP_SIZE)
            elif pred == -1:  # SELL
                position = -1
                entry_price = row["close"]
                entry_time = timestamp
                sl_price = entry_price + (config.SL_PIPS * config.PIP_SIZE)
                tp_price = entry_price - (config.TP_PIPS * config.PIP_SIZE)
                
        equity_curve.append({"timestamp": timestamp, "equity": equity})
        
    # Close open position at end of backtest
    if position != 0:
        exit_price = df.iloc[-1]["close"]
        gross_pips = (exit_price - entry_price) / config.PIP_SIZE * position
        net_pips = gross_pips - cost_pips
        pnl = net_pips * (config.LOT_SIZE * config.PIP_SIZE)
        equity += pnl
        trades.append({
            "entry_time": entry_time,
            "exit_time": df.index[-1],
            "type": "LONG" if position == 1 else "SHORT",
            "entry_price": entry_price,
            "exit_price": exit_price,
            "reason": "END_OF_DATA",
            "net_pips": net_pips,
            "pnl": pnl,
            "equity": equity
        })
        equity_curve[-1]["equity"] = equity

    eq_df = pd.DataFrame(equity_curve).set_index("timestamp")
    trades_df = pd.DataFrame(trades)
    
    metrics = calculate_metrics(eq_df, trades_df)
    
    return eq_df, trades_df, metrics


def calculate_metrics(eq_df: pd.DataFrame, trades_df: pd.DataFrame) -> dict:
    """Calculate strategy performance metrics."""
    capital = config.INITIAL_CAPITAL
    
    if len(trades_df) == 0:
        return {
            "Total Return (%)": 0.0,
            "Max Drawdown (%)": 0.0,
            "Total Trades": 0,
            "Win Rate (%)": 0.0,
            "Profit Factor": 0.0,
            "Sharpe Ratio": 0.0
        }
        
    final_equity = eq_df["equity"].iloc[-1]
    total_return = (final_equity - capital) / capital * 100
    
    # Drawdown
    roll_max = eq_df["equity"].cummax()
    drawdown = (eq_df["equity"] - roll_max) / roll_max * 100
    max_drawdown = drawdown.min()
    
    # Trade stats
    winning_trades = trades_df[trades_df["pnl"] > 0]
    losing_trades = trades_df[trades_df["pnl"] <= 0]
    
    win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
    
    gross_profit = winning_trades["pnl"].sum()
    gross_loss = abs(losing_trades["pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    
    # Sharpe Ratio (Approximate for H1 data, assuming 252*24 hours/year, risk-free rate 0)
    returns = eq_df["equity"].pct_change().dropna()
    mean_return = returns.mean()
    std_return = returns.std()
    
    if std_return > 0:
        # Annualization factor for hourly data
        annualization_factor = np.sqrt(252 * 24)
        sharpe = (mean_return / std_return) * annualization_factor
    else:
        sharpe = 0.0
        
    metrics = {
        "Total Return (%)": total_return,
        "Max Drawdown (%)": max_drawdown,
        "Total Trades": len(trades_df),
        "Win Rate (%)": win_rate,
        "Profit Factor": profit_factor,
        "Sharpe Ratio": sharpe
    }
    
    return metrics


def plot_equity_curve(eq_df: pd.DataFrame, output_dir: str = None, title: str = "Strategy Equity Curve"):
    """Plot and save the equity curve."""
    output_dir = output_dir or config.OUTPUTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(eq_df.index, eq_df["equity"], label="Equity", color="blue")
    plt.axhline(y=config.INITIAL_CAPITAL, color="red", linestyle="--", label="Initial Capital")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(True)
    
    # Use model name in filename so each model gets its own chart
    safe_title = title.replace(" ", "_").replace("-", "").lower()
    filepath = os.path.join(output_dir, f"equity_curve_{safe_title}.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved equity curve: {filepath}")
    

def execute_backtest(model, model_name: str, X_test: pd.DataFrame, y_test: pd.Series, df_test_raw: pd.DataFrame):
    """
    Wrapper to execute backtest for a specific model and log results.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Running Backtest: {model_name}")
    logger.info(f"{'='*60}")
    
    # Get predictions — all models (RF, XGBWrapper, Ensemble) now return
    # original labels (-1, 0, 1) directly from predict()
    y_pred = model.predict(X_test)
    predictions = pd.Series(y_pred, index=X_test.index)
    
    # Get probabilities if available
    probabilities = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        probabilities = pd.DataFrame(probs, index=X_test.index)
        
    eq_df, trades_df, metrics = run_backtest(df_test_raw, predictions, probabilities)
    
    logger.info("Backtest Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.2f}")
        else:
            logger.info(f"  {k}: {v}")
            
    plot_equity_curve(eq_df, title=f"Equity Curve - {model_name}")
    
    # Save trades
    trades_path = os.path.join(config.OUTPUTS_DIR, f"trades_{model_name}.csv")
    if not trades_df.empty:
        trades_df.to_csv(trades_path, index=False)
        logger.info(f"Saved trade log to {trades_path}")
    else:
        logger.info("No trades executed.")
        
    return eq_df, trades_df, metrics
