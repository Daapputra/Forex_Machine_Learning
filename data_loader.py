"""
data_loader.py — Load CSV, Timezone Conversion & Resample M1 → H1
===================================================================
Handles the raw histdata.com CSV format:
    YYYYMMDD HHMMSS;open;high;low;close;volume

Timezone: Source is EST (US/Eastern), converted to UTC.
Resample: M1 → H1 with proper OHLCV aggregation.
"""

import pandas as pd
import logging

import config

logger = logging.getLogger(__name__)


import glob
import os

def load_m1_data(data_dir: str = None) -> pd.DataFrame:
    """
    Load M1 (1-minute) OHLCV data from ALL histdata.com CSV files in the data directory.
    Combines them and sorts by datetime chronologically.
    """
    data_dir = data_dir or config.DATA_DIR
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    logger.info(f"Found {len(csv_files)} CSV files in {data_dir}. Reading and combining...")

    dfs = []
    for filepath in csv_files:
        logger.info(f"Reading {os.path.basename(filepath)}...")
        df_part = pd.read_csv(
            filepath,
            sep=";",
            header=None,
            names=["datetime_str", "open", "high", "low", "close", "volume"],
            dtype={
                "open": float,
                "high": float,
                "low": float,
                "close": float,
                "volume": float,
            },
        )
        dfs.append(df_part)

    df = pd.concat(dfs, ignore_index=True)

    # Parse datetime: format is "YYYYMMDD HHMMSS"
    df["datetime"] = pd.to_datetime(df["datetime_str"], format="%Y%m%d %H%M%S")
    df.set_index("datetime", inplace=True)
    df.drop(columns=["datetime_str"], inplace=True)

    # CRITICAL: Sort by datetime chronologically (especially when combining multiple files)
    df.sort_index(inplace=True)

    logger.info(f"Loaded {len(df)} total M1 rows, range: {df.index[0]} - {df.index[-1]}")

    return df


def convert_timezone(df: pd.DataFrame,
                     source_tz: str = None,
                     target_tz: str = None) -> pd.DataFrame:
    """
    Localize naive timestamps to source timezone, then convert to target timezone.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex (naive, assumed EST).
    source_tz : str
        Source timezone (default: config.SOURCE_TZ = "US/Eastern").
    target_tz : str
        Target timezone (default: config.TARGET_TZ = "UTC").

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex converted to target timezone.
    """
    source_tz = source_tz or config.SOURCE_TZ
    target_tz = target_tz or config.TARGET_TZ

    logger.info(f"Converting timezone: {source_tz} -> {target_tz}")

    df = df.copy()
    df.index = df.index.tz_localize(source_tz).tz_convert(target_tz)

    return df


def resample_to_tf(df: pd.DataFrame, timeframe: str = None) -> pd.DataFrame:
    """
    Resample M1 data to a higher timeframe (e.g., '4h', '1h').
    """
    timeframe = timeframe or config.TIMEFRAME

    # Use '4h' or whatever config.TIMEFRAME dictates.
    # Note: 'H' is deprecated in pandas 2.2.0+, 'h' is used.
    # If config says 'H4', we convert to '4h'
    pandas_tf = timeframe.lower()
    if pandas_tf == 'h4': pandas_tf = '4h'
    elif pandas_tf == 'h1': pandas_tf = '1h'

    logger.info(f"Resampling M1 -> {timeframe}...")

    resampled_df = df.resample(pandas_tf).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })

    resampled_df.dropna(inplace=True)

    logger.info(f"{timeframe} resampled: {len(resampled_df)} candles, range: {resampled_df.index[0]} - {resampled_df.index[-1]}")

    return resampled_df


def load_data(data_dir: str = None) -> pd.DataFrame:
    """
    Complete data loading pipeline:
    1. Load raw M1 CSVs
    2. Convert timezone (EST -> UTC)
    3. Resample to target timeframe (e.g., 4H)
    """
    df_m1 = load_m1_data(data_dir)
    df_utc = convert_timezone(df_m1)
    df_tf = resample_to_tf(df_utc)

    return df_tf
