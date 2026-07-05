"""
backtester.py — Strategy Validation (v3.0 — Dynamic ATR SL/TP)
================================================================
v3.0 UPGRADE: SL/TP now dynamically adjust based on ATR (volatility).
- Volatile market: wider SL/TP (avoids premature stop-outs)
- Calm market: tighter SL/TP (locks in smaller but consistent gains)

Rules:
- Spread: 2 pips, Slippage: 0.5 pips (applied at entry).
- SL: ATR * 1.5, TP: ATR * 2.0 (dynamic per-trade)
- Capital: $10,000, max 1 position active.
- Confidence filter: > 38%.
- Max daily loss: 3%.
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


def run_backtest(
    df: pd.DataFrame,
    predictions: pd.Series,
    probabilities: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Run backtest simulation with dynamic ATR-based SL/TP.
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
                pnl = net_pips * (config.LOT_SIZE * config.PIP_SIZE)

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
                    "equity": equity,
                    "sl_pips": round(abs(sl_price - entry_price) / config.PIP_SIZE, 1),
                    "tp_pips": round(abs(tp_price - entry_price) / config.PIP_SIZE, 1),
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
                conf = 0.0
                if pred == 1:
                    conf = probabilities.iloc[i, config.LABEL_MAP[1]]
                elif pred == -1:
                    conf = probabilities.iloc[i, config.LABEL_MAP[-1]]

                if conf < config.MIN_CONFIDENCE:
                    pred = 0  # Ignore signal

            # Calculate dynamic SL/TP based on current ATR
            current_atr = atr_series.iloc[i] if not pd.isna(atr_series.iloc[i]) else config.SL_PIPS * config.PIP_SIZE
            sl_distance = current_atr * config.SL_ATR_MULT
            tp_distance = current_atr * config.TP_ATR_MULT

            # Enforce minimum SL/TP (at least 5 pips)
            min_distance = 5 * config.PIP_SIZE
            sl_distance = max(sl_distance, min_distance)
            tp_distance = max(tp_distance, min_distance * 1.5)

            if pred == 1:  # BUY
                position = 1
                entry_price = row["close"]
                entry_time = timestamp
                sl_price = entry_price - sl_distance
                tp_price = entry_price + tp_distance
            elif pred == -1:  # SELL
                position = -1
                entry_price = row["close"]
                entry_time = timestamp
                sl_price = entry_price + sl_distance
                tp_price = entry_price - tp_distance

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
            "equity": equity,
            "sl_pips": round(abs(sl_price - entry_price) / config.PIP_SIZE, 1),
            "tp_pips": round(abs(tp_price - entry_price) / config.PIP_SIZE, 1),
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
            "Sharpe Ratio": 0.0,
            "Avg Win (pips)": 0.0,
            "Avg Loss (pips)": 0.0,
        }

    final_equity = eq_df["equity"].iloc[-1]
    total_return = (final_equity - capital) / capital * 100

    # Drawdown
    roll_max = eq_df["equity"].cummax()
    drawdown = (eq_df["equity"] - roll_max) / roll_max * 100
    max_dd = drawdown.min()

    # Trade stats
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]

    total_trades = len(trades_df)
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0

    gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Sharpe ratio (annualized, hourly data)
    returns = eq_df["equity"].pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        # ~6240 trading hours per year (260 days * 24 hours)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(6240)
    else:
        sharpe = 0.0

    # Average win/loss in pips
    avg_win_pips = wins["net_pips"].mean() if len(wins) > 0 else 0
    avg_loss_pips = losses["net_pips"].mean() if len(losses) > 0 else 0

    metrics = {
        "Total Return (%)": total_return,
        "Max Drawdown (%)": max_dd,
        "Total Trades": total_trades,
        "Win Rate (%)": win_rate,
        "Profit Factor": profit_factor,
        "Sharpe Ratio": sharpe,
        "Avg Win (pips)": avg_win_pips,
        "Avg Loss (pips)": avg_loss_pips,
    }

    return metrics


def plot_equity_curve(eq_df: pd.DataFrame, output_dir: str = None, title: str = "Strategy Equity Curve"):
    """Plot and save the equity curve with enhanced visuals for paper."""
    output_dir = output_dir or config.OUTPUTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot equity line
    ax.plot(eq_df.index, eq_df["equity"], label="Portfolio Equity", color="#2196F3", linewidth=1.5)
    ax.axhline(y=config.INITIAL_CAPITAL, color="#F44336", linestyle="--", label="Initial Capital", alpha=0.7)

    # Fill green/red areas
    ax.fill_between(eq_df.index, config.INITIAL_CAPITAL, eq_df["equity"],
                     where=eq_df["equity"] >= config.INITIAL_CAPITAL,
                     color="#4CAF50", alpha=0.15, label="Profit Zone")
    ax.fill_between(eq_df.index, config.INITIAL_CAPITAL, eq_df["equity"],
                     where=eq_df["equity"] < config.INITIAL_CAPITAL,
                     color="#F44336", alpha=0.15, label="Loss Zone")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Equity ($)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Use model name in filename
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

    # Get predictions — all models return original labels (-1, 0, 1)
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
