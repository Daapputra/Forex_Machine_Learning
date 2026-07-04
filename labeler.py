"""
labeler.py — Target Variable Creation (Pip-Return Labeling)
============================================================
Creates the target label BEFORE feature engineering to prevent data leakage.

LOOK-AHEAD BIAS NOTE:
    The label uses close.shift(-1) — the NEXT candle's close price — which is
    future information. This is CORRECT for the target variable (we're predicting
    the next candle's direction). But:
    1. Labels must be created BEFORE features.
    2. The label column must NEVER be used as an input feature.
    3. The last row is dropped because it has no future close.
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

    Returns
    -------
    pd.Series
        Pip return for each candle (last candle will be NaN).
    """
    pip_return = (df["close"].shift(-1) - df["close"]) / config.PIP_SIZE
    return pip_return


def assign_labels(pip_return: pd.Series,
                  buy_threshold: float = None,
                  sell_threshold: float = None) -> pd.Series:
    """
    Assign class labels based on pip return thresholds.

    Label mapping:
        1  (BUY)  if pip_return >  buy_threshold
       -1  (SELL) if pip_return <  sell_threshold
        0  (HOLD) otherwise

    Parameters
    ----------
    pip_return : pd.Series
        Forward pip returns.
    buy_threshold : float
        Minimum pip return for BUY signal (default: config.LABEL_BUY_THRESHOLD).
    sell_threshold : float
        Maximum pip return for SELL signal (default: config.LABEL_SELL_THRESHOLD).

    Returns
    -------
    pd.Series
        Integer labels: 1 (BUY), -1 (SELL), 0 (HOLD).
    """
    buy_threshold = buy_threshold or config.LABEL_BUY_THRESHOLD
    sell_threshold = sell_threshold or config.LABEL_SELL_THRESHOLD

    labels = pd.Series(0, index=pip_return.index, name="label")
    labels[pip_return > buy_threshold] = 1
    labels[pip_return < sell_threshold] = -1

    return labels


def label_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full labeling pipeline:
    1. Compute pip return (forward-looking — this is the TARGET, not a feature)
    2. Assign labels (BUY/SELL/HOLD)
    3. Drop last row (no future close available)

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned H1 OHLCV data.

    Returns
    -------
    pd.DataFrame
        Data with 'pip_return' and 'label' columns added.
        Last row is dropped.
    """
    logger.info("Creating labels (pip-return based)...")

    df = df.copy()
    df["pip_return"] = compute_pip_return(df)
    df["label"] = assign_labels(df["pip_return"])

    # Drop last row — no forward close available
    # LOOK-AHEAD NOTE: This is necessary because the last candle has no
    # future close to compare against.
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
