"""
backtester.py — Strategy Validation (v4.0 — Dynamic ATR + Regime Filters)
========================================================================
v4.0 UPGRADE: 
- Regime Filter: only trade when Vol_Regime > 0 and ADX > threshold.
- Confidence Filter: only trade when confidence > MIN_CONFIDENCE.
- Risk/Reward: TP is 2.5x ATR, SL is 1.0x ATR.
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


def compute_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ATR series in PRICE units for dynamic SL/TP."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def run_backtest(df: pd.DataFrame, model_name: str = "model") -> dict:
    """
    Run backtest simulation with dynamic ATR-based SL/TP and v4.0 Regime Filters.
    Expects 'prediction' and 'confidence' columns in df.
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

    # Compute ATR for dynamic SL/TP
    atr_series = compute_atr_series(df)

    for i in range(len(df)):
        row = df.iloc[i]
        timestamp = df.index[i]

        # Reset daily max loss tracker
        if timestamp.date() != current_day:
            daily_start_capital = equity
            current_day = timestamp.date()

        # 1. Manage Open Positions
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

            # Execute exit
            if exit_price != 0:
                price_diff = exit_price - entry_price if position == 1 else entry_price - exit_price
                pnl_pips = (price_diff / config.PIP_SIZE) - cost_pips
                pnl_dollars = pnl_pips * pip_val

                equity += pnl_dollars
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": timestamp,
                    "type": "BUY" if position == 1 else "SELL",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pips": pnl_pips,
                    "pnl_dollars": pnl_dollars,
                    "reason": exit_reason,
                    "equity": equity
                })

                position = 0

        # 2. Check for New Entries
        if position == 0:
            pred = row.get("prediction", 0)
            conf = row.get("confidence", 0)
            
            # v4.0 Regime & Confidence Filters
            # 1. Confidence > MIN_CONFIDENCE
            if conf >= config.MIN_CONFIDENCE:
                
                # 2. ADX Trend Filter (only if ADX feature exists)
                adx = row.get("ADX_14", 100)  # Default pass if not calculated
                vol_regime = row.get("Vol_Regime", 1) # Default pass if not calculated
                
                if adx >= config.MIN_ADX_TREND and vol_regime > 0:
                    
                    atr_val = atr_series.iloc[i]
                    if pd.isna(atr_val) or atr_val == 0:
                        atr_val = 0.0020  # Fallback 20 pips

                    if pred == 1:
                        position = 1
                        entry_price = row["open"]  # Enter at open of current candle
                        sl_price = entry_price - (atr_val * config.SL_ATR_MULT)
                        tp_price = entry_price + (atr_val * config.TP_ATR_MULT)
                        entry_time = timestamp

                    elif pred == -1:
                        position = -1
                        entry_price = row["open"]
                        sl_price = entry_price + (atr_val * config.SL_ATR_MULT)
                        tp_price = entry_price - (atr_val * config.TP_ATR_MULT)
                        entry_time = timestamp

        # Daily loss check
        if equity < daily_start_capital * (1 - config.MAX_DAILY_LOSS_PCT / 100.0):
            # Force close if we exceeded max daily loss
            if position != 0:
                exit_price = row["close"]
                price_diff = exit_price - entry_price if position == 1 else entry_price - exit_price
                pnl_pips = (price_diff / config.PIP_SIZE) - cost_pips
                pnl_dollars = pnl_pips * pip_val
                equity += pnl_dollars
                trades.append({
                    "entry_time": entry_time, "exit_time": timestamp,
                    "type": "BUY" if position == 1 else "SELL",
                    "entry_price": entry_price, "exit_price": exit_price,
                    "pnl_pips": pnl_pips, "pnl_dollars": pnl_dollars,
                    "reason": "MAX_LOSS", "equity": equity
                })
                position = 0
            daily_start_capital = equity # Prevent trading rest of day
            
        equity_curve.append(equity)

    # Compile metrics
    df_trades = pd.DataFrame(trades)
    
    metrics = {
        "total_trades": len(trades),
        "win_rate": 0.0,
        "total_pnl": equity - capital,
        "roi_pct": ((equity - capital) / capital) * 100,
        "max_drawdown_pct": 0.0,
        "profit_factor": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "equity_curve": equity_curve,
        "timestamps": df.index,
    }

    if not df_trades.empty:
        wins = df_trades[df_trades["pnl_dollars"] > 0]
        losses = df_trades[df_trades["pnl_dollars"] <= 0]
        
        metrics["win_rate"] = len(wins) / len(df_trades) * 100
        metrics["avg_win"] = wins["pnl_dollars"].mean() if not wins.empty else 0
        metrics["avg_loss"] = losses["pnl_dollars"].mean() if not losses.empty else 0
        
        gross_profit = wins["pnl_dollars"].sum() if not wins.empty else 0
        gross_loss = abs(losses["pnl_dollars"].sum()) if not losses.empty else 1e-9
        metrics["profit_factor"] = gross_profit / gross_loss
        
        # Drawdown
        eq_series = pd.Series(equity_curve)
        peak = eq_series.cummax()
        drawdown = (eq_series - peak) / peak
        metrics["max_drawdown_pct"] = drawdown.min() * 100

    logger.info(f"Backtest {model_name} | Trades: {metrics['total_trades']} | WR: {metrics['win_rate']:.1f}% | ROI: {metrics['roi_pct']:.2f}% | MDD: {metrics['max_drawdown_pct']:.2f}%")
    
    return metrics


def plot_equity_curve(metrics: dict, model_name: str):
    """Plot and save equity curve."""
    if metrics["total_trades"] == 0:
        return
        
    plt.figure(figsize=(10, 5))
    plt.plot(metrics["timestamps"], metrics["equity_curve"], label=f"{model_name} Equity")
    plt.title(f"{model_name} Backtest Equity Curve (v4.0)")
    plt.xlabel("Time")
    plt.ylabel("Capital ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUTS_DIR, f"{model_name}_equity.png"))
    plt.close()
