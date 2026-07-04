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

    logger.info(f"Loaded {len(df)} total M1 rows, range: {df.index[0]} — {df.index[-1]}")

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

    logger.info(f"Converting timezone: {source_tz} → {target_tz}")

    df = df.copy()
    df.index = df.index.tz_localize(source_tz).tz_convert(target_tz)

    return df


def resample_to_h1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample M1 data to H1 (1-hour) candles with proper OHLCV aggregation.

    Aggregation rules:
        - open:   first (opening price of the hour)
        - high:   max   (highest price during the hour)
        - low:    min   (lowest price during the hour)
        - close:  last  (closing price of the hour)
        - volume: sum   (total volume during the hour)

    Weekend gaps are NOT filled — they are naturally absent from the data.

    Returns
    -------
    pd.DataFrame
        H1 OHLCV DataFrame.
    """
    logger.info("Resampling M1 → H1...")

    ohlcv_agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    df_h1 = df.resample("1h").agg(ohlcv_agg).dropna()

    logger.info(f"H1 resampled: {len(df_h1)} candles, "
                f"range: {df_h1.index[0]} — {df_h1.index[-1]}")

    return df_h1


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Full data loading pipeline: Load M1 → Convert TZ → Resample to H1.

    This is the main entry point for Phase 1 data loading.

    Parameters
    ----------
    filepath : str, optional
        Path to M1 CSV file.

    Returns
    -------
    pd.DataFrame
        Clean H1 OHLCV DataFrame with UTC DatetimeIndex.
    """
    df_m1 = load_m1_data(filepath)
    df_m1 = convert_timezone(df_m1)
    df_h1 = resample_to_h1(df_m1)

    return df_h1
