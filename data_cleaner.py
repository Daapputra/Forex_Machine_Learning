"""
data_cleaner.py — Data Validation & Cleaning
==============================================
Validates OHLCV integrity, handles missing values, and logs anomalies.
"""

import pandas as pd
import numpy as np
import logging

import config

logger = logging.getLogger(__name__)


def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate timestamps, keeping first occurrence."""
    n_dupes = df.index.duplicated().sum()
    if n_dupes > 0:
        logger.warning(f"Found {n_dupes} duplicate timestamps — dropping duplicates.")
        df = df[~df.index.duplicated(keep="first")]
    else:
        logger.info("No duplicate timestamps found.")
    return df


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate OHLCV integrity:
    - High must be >= max(Open, Close)
    - Low must be <= min(Open, Close)

    Anomalous rows are logged and flagged but NOT dropped (they may be
    legitimate wicks/spikes).
    """
    anomalies = []

    # High should be >= max(Open, Close)
    high_violation = df["high"] < df[["open", "close"]].max(axis=1)
    n_high = high_violation.sum()
    if n_high > 0:
        logger.warning(f"OHLCV violation: {n_high} rows where High < max(Open, Close)")
        anomalies.append(("high_violation", n_high))

    # Low should be <= min(Open, Close)
    low_violation = df["low"] > df[["open", "close"]].min(axis=1)
    n_low = low_violation.sum()
    if n_low > 0:
        logger.warning(f"OHLCV violation: {n_low} rows where Low > min(Open, Close)")
        anomalies.append(("low_violation", n_low))

    # Log anomalies to file
    if anomalies:
        _log_anomalies(anomalies, df[high_violation | low_violation])

    return df


def handle_missing_values(df: pd.DataFrame,
                          max_ffill: int = None) -> pd.DataFrame:
    """
    Handle missing candles:
    1. Forward-fill up to max_ffill consecutive NaN candles.
    2. Drop remaining NaN rows.

    NOTE: Weekend gaps are naturally absent from the data and should NOT be
    filled. We only fill intra-session gaps (e.g., a missing candle mid-day).
    Since our data is already resampled to H1 and weekends are not present,
    forward-filling here handles only unexpected gaps within trading hours.
    """
    max_ffill = max_ffill or config.MAX_FFILL_CANDLES

    n_before = len(df)
    n_nan_before = df.isna().any(axis=1).sum()

    if n_nan_before > 0:
        logger.info(f"Found {n_nan_before} rows with NaN values. "
                    f"Forward-filling up to {max_ffill} consecutive candles.")
        df = df.ffill(limit=max_ffill)

        # Drop any remaining NaN rows that couldn't be filled
        n_nan_after = df.isna().any(axis=1).sum()
        if n_nan_after > 0:
            df = df.dropna()
            logger.info(f"Dropped {n_nan_after} rows that couldn't be forward-filled.")
    else:
        logger.info("No missing values found.")

    n_after = len(df)
    logger.info(f"Rows: {n_before} -> {n_after} (dropped {n_before - n_after})")

    return df


def check_zero_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for zero-volume candles. In histdata.com data, volume is typically
    all zeros (tick-count based). We log this but do NOT drop these rows.
    """
    n_zero_vol = (df["volume"] == 0).sum()
    if n_zero_vol == len(df):
        logger.info("All volume values are 0 (typical for histdata.com). "
                     "Volume-based indicators will be skipped.")
    elif n_zero_vol > 0:
        logger.warning(f"{n_zero_vol}/{len(df)} candles have zero volume.")

    return df


def _log_anomalies(anomalies: list, anomaly_rows: pd.DataFrame):
    """Write anomaly details to the anomaly log file."""
    try:
        with open(config.ANOMALY_LOG_FILE, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Anomaly Report\n")
            f.write(f"{'='*60}\n")
            for anom_type, count in anomalies:
                f.write(f"  {anom_type}: {count} rows\n")
            if len(anomaly_rows) > 0:
                f.write(f"\nSample anomalous rows:\n")
                f.write(anomaly_rows.head(10).to_string())
                f.write("\n")
    except Exception as e:
        logger.error(f"Failed to write anomaly log: {e}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
    1. Remove duplicate timestamps
    2. Validate OHLCV integrity
    3. Handle missing values (forward-fill then drop)
    4. Check zero volume

    Parameters
    ----------
    df : pd.DataFrame
        Raw H1 OHLCV data.

    Returns
    -------
    pd.DataFrame
        Cleaned H1 OHLCV data with anomaly summary logged.
    """
    logger.info(f"Starting data cleaning on {len(df)} rows...")

    df = check_duplicates(df)
    df = validate_ohlcv(df)
    df = handle_missing_values(df)
    df = check_zero_volume(df)

    logger.info(f"Data cleaning complete. Final: {len(df)} valid rows.")

    return df
