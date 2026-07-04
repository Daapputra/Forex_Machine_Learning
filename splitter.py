"""
splitter.py — Chronological Train / Val / Test Split
======================================================
CRITICAL: No shuffling! Financial time series must be split chronologically
to prevent look-ahead bias. The model must only train on past data and
validate/test on future data.

Split order: [TRAIN 70%] → [VAL 15%] → [TEST 15%]
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple

import config

logger = logging.getLogger(__name__)


def split_data(
    df: pd.DataFrame,
    feature_cols: list,
    train_ratio: float = None,
    val_ratio: float = None,
) -> Tuple[
    pd.DataFrame, pd.Series,  # X_train, y_train
    pd.DataFrame, pd.Series,  # X_val, y_val
    pd.DataFrame, pd.Series,  # X_test, y_test
]:
    """
    Split data chronologically into train/val/test sets.

    ANTI-LEAKAGE: No shuffling. Data is split strictly by time order.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame with features and 'label' column.
    feature_cols : list
        List of feature column names to include in X.
    train_ratio : float
        Fraction for training (default: config.TRAIN_RATIO = 0.70).
    val_ratio : float
        Fraction for validation (default: config.VAL_RATIO = 0.15).
        Test ratio is implicitly 1 - train_ratio - val_ratio.

    Returns
    -------
    tuple
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    train_ratio = train_ratio or config.TRAIN_RATIO
    val_ratio = val_ratio or config.VAL_RATIO

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Split chronologically — NO shuffle
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    X_val = val_df[feature_cols]
    y_val = val_df["label"]
    X_test = test_df[feature_cols]
    y_test = test_df["label"]

    # Log split info
    logger.info(f"Chronological split (NO shuffle):")
    logger.info(f"  Train: {len(train_df)} rows "
                f"({train_df.index[0]} -> {train_df.index[-1]})")
    logger.info(f"  Val:   {len(val_df)} rows "
                f"({val_df.index[0]} -> {val_df.index[-1]})")
    logger.info(f"  Test:  {len(test_df)} rows "
                f"({test_df.index[0]} -> {test_df.index[-1]})")

    # Log label distribution per split
    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        counts = y.value_counts().sort_index()
        dist_str = ", ".join(
            f"{config.LABEL_NAMES.get(k, k)}: {v}" for k, v in counts.items()
        )
        logger.info(f"  {name} labels: {dist_str}")

    return X_train, y_train, X_val, y_val, X_test, y_test
