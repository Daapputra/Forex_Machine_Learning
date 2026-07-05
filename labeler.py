"""
labeler.py — Target Variable Creation (v3.0 — Dynamic ATR-based Labeling)
===========================================================================
v3.0 UPGRADE: Labels now adapt to market volatility using ATR.
In volatile markets -> threshold is higher (harder to trigger BUY/SELL)
In calm markets -> threshold is lower (easier to trigger)
This produces MORE BALANCED labels and better model performance.

LOOK-AHEAD BIAS NOTE:
    The label uses close.shift(-1) — the NEXT candle's close price — which is
    future information. This is CORRECT for the target variable.
    ATR used for threshold is computed on CURRENT data (no leakage).
"""

import pandas as pd
import numpy as np
import logging

import config

logger = logging.getLogger(__name__)


def compute_pip_return(df: pd.DataFrame) -> pd.Series:
    """
    Compute forward pip return: (next_close - current_close) / pip_size
    This intentionally uses future data (shift(-1)) because it IS the target.
    """
    pip_return = (df["close"].shift(-1) - df["close"]) / config.PIP_SIZE
    return pip_return


def compute_atr_pips(df: pd.DataFrame, period: int = None) -> pd.Series:
    """
    Compute ATR in pips for dynamic labeling threshold.
    ATR = Average True Range over `period` candles.
    Uses CURRENT data only (no look-ahead bias).
    """
    period = period or config.LABEL_ATR_PERIOD

    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    # Convert to pips
    atr_pips = atr / config.PIP_SIZE
    return atr_pips


def assign_labels_fixed(pip_return: pd.Series) -> pd.Series:
    """Fixed threshold labeling (v2.0 fallback)."""
    labels = pd.Series(0, index=pip_return.index, name="label")
    labels[pip_return > config.LABEL_BUY_THRESHOLD] = 1
    labels[pip_return < config.LABEL_SELL_THRESHOLD] = -1
    return labels


def assign_labels_atr(pip_return: pd.Series, atr_pips: pd.Series) -> pd.Series:
    """
    Dynamic ATR-based labeling (v3.0).
    Threshold = ATR * multiplier, computed per-candle.
    This makes labels adapt to current market volatility.
    """
    threshold = atr_pips * config.LABEL_ATR_MULTIPLIER

    labels = pd.Series(0, index=pip_return.index, name="label")
    labels[pip_return > threshold] = 1
    labels[pip_return < -threshold] = -1

    return labels


def label_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full labeling pipeline:
    1. Compute pip return (forward-looking — this is the TARGET)
    2. Assign labels (BUY/SELL/HOLD) using fixed or dynamic ATR threshold
    3. Drop last row (no future close available)
    """
    logger.info(f"Creating labels (mode: {config.LABEL_MODE})...")

    df = df.copy()
    df["pip_return"] = compute_pip_return(df)

    if config.LABEL_MODE == "atr":
        atr_pips = compute_atr_pips(df)
        df["label"] = assign_labels_atr(df["pip_return"], atr_pips)

        # Store ATR for backtester to use (in price units, not pips)
        df["atr_14"] = atr_pips * config.PIP_SIZE

        # Log ATR stats
        valid_atr = atr_pips.dropna()
        threshold_pips = valid_atr * config.LABEL_ATR_MULTIPLIER
        logger.info(f"  ATR stats (pips): mean={valid_atr.mean():.1f}, "
                     f"median={valid_atr.median():.1f}, "
                     f"min={valid_atr.min():.1f}, max={valid_atr.max():.1f}")
        logger.info(f"  Dynamic threshold (pips): mean={threshold_pips.mean():.1f}, "
                     f"median={threshold_pips.median():.1f}")
    else:
        df["label"] = assign_labels_fixed(df["pip_return"])

    # Drop last row — no forward close available
    df = df.iloc[:-1]

    # Log label distribution
    label_counts = df["label"].value_counts().sort_index()
    total = len(df)
    logger.info("Label distribution:")
    for label_val, count in label_counts.items():
        label_name = config.LABEL_NAMES.get(label_val, str(label_val))
        pct = count / total * 100
        logger.info(f"  {label_name} ({label_val}): {count} ({pct:.1f}%)")

    return df
