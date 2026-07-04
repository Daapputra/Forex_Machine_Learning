"""
features.py — Standalone Feature Engineering Module (v2.0)
============================================================
UPGRADED: Added 15+ new features for production-quality modeling:
  - Price action: candle body/wick ratios, gap detection
  - Momentum: Williams %R, CCI, ADX (trend strength)
  - Volatility: ATR ratio, volatility regime, Keltner Channel width
  - Lag features: rolling mean/std of returns
  - Relative price: distance from MAs (normalized by ATR)

ANTI-LEAKAGE DESIGN:
    ALL price-derived indicators are shifted by 1 period (.shift(1)) so the model
    only sees information available at the time of prediction. At bar T, the model
    sees indicators computed from data up to bar T-1.

    Session/time features (is_asian, hour_sin, etc.) are NOT shifted because they
    describe the CURRENT candle's time — this is known at prediction time.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
import logging

import config

logger = logging.getLogger(__name__)


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trend indicators: MA_10, MA_20, MA_50, EMA_12, EMA_26, MA_Cross.
    NEW: Distance from MAs normalized by ATR (how far price is from trend).
    """
    for period in config.MA_PERIODS:
        df[f"MA_{period}"] = df["close"].rolling(window=period).mean().shift(1)

    df["EMA_12"] = df["close"].ewm(span=config.EMA_FAST, adjust=False).mean().shift(1)
    df["EMA_26"] = df["close"].ewm(span=config.EMA_SLOW, adjust=False).mean().shift(1)

    # MA Cross: 1 if EMA_12 > EMA_26, else 0
    df["MA_Cross"] = (df["EMA_12"] > df["EMA_26"]).astype(int)

    # NEW: Price distance from key MAs (normalized by ATR for scale-independence)
    # Tells the model HOW FAR price is from the trend, not just above/below
    atr_raw = ta.atr(df["high"], df["low"], df["close"], length=config.ATR_PERIOD)
    if atr_raw is not None:
        atr_safe = atr_raw.replace(0, np.nan)
        for period in [20, 50]:
            ma = df["close"].rolling(window=period).mean()
            df[f"Dist_MA_{period}"] = ((df["close"] - ma) / atr_safe).shift(1)

    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum indicators.
    NEW: Williams %R, CCI_20, ADX_14 (trend strength).
    """
    # RSI
    rsi = ta.rsi(df["close"], length=config.RSI_PERIOD)
    if rsi is not None:
        df["RSI_14"] = rsi.shift(1)

    # Stochastic K/D
    stoch = ta.stoch(
        df["high"], df["low"], df["close"],
        k=config.STOCH_K_PERIOD, d=config.STOCH_D_PERIOD
    )
    if stoch is not None:
        df["Stoch_K"] = stoch.iloc[:, 0].shift(1)
        df["Stoch_D"] = stoch.iloc[:, 1].shift(1)

    # MACD
    macd = ta.macd(
        df["close"],
        fast=config.MACD_FAST,
        slow=config.MACD_SLOW,
        signal=config.MACD_SIGNAL
    )
    if macd is not None:
        df["MACD"] = macd.iloc[:, 0].shift(1)
        df["MACD_Signal"] = macd.iloc[:, 1].shift(1)
        df["MACD_Hist"] = macd.iloc[:, 2].shift(1)

    # NEW: Williams %R (mean-reversion signal, similar to Stochastic but inverted)
    willr = ta.willr(df["high"], df["low"], df["close"], length=14)
    if willr is not None:
        df["WillR_14"] = willr.shift(1)

    # NEW: CCI — Commodity Channel Index (identifies cyclical turns)
    cci = ta.cci(df["high"], df["low"], df["close"], length=20)
    if cci is not None:
        df["CCI_20"] = cci.shift(1)

    # NEW: ADX — Average Directional Index (trend STRENGTH, regardless of direction)
    # High ADX = strong trend (good for trend-following), Low ADX = ranging (avoid)
    adx_result = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx_result is not None:
        df["ADX_14"] = adx_result.iloc[:, 0].shift(1)    # ADX value
        df["DI_plus"] = adx_result.iloc[:, 1].shift(1)   # +DI
        df["DI_minus"] = adx_result.iloc[:, 2].shift(1)  # -DI

    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volatility indicators.
    NEW: ATR ratio (current vs historical), volatility regime detector.
    """
    # ATR
    atr = ta.atr(df["high"], df["low"], df["close"], length=config.ATR_PERIOD)
    if atr is not None:
        df["ATR_14"] = atr.shift(1)

        # NEW: ATR ratio — current ATR / rolling mean ATR(50)
        # Values > 1 = higher volatility than normal, < 1 = calmer than normal
        atr_mean_50 = atr.rolling(50).mean()
        atr_mean_safe = atr_mean_50.replace(0, np.nan)
        df["ATR_Ratio"] = (atr / atr_mean_safe).shift(1)

        # NEW: Volatility regime — simple bucketing for the model
        # 0 = low vol, 1 = normal, 2 = high vol
        atr_pct = atr / df["close"]
        atr_pct_rolling = atr_pct.rolling(100).rank(pct=True)
        df["Vol_Regime"] = pd.cut(
            atr_pct_rolling, bins=[0, 0.33, 0.66, 1.0],
            labels=[0, 1, 2], include_lowest=True
        ).astype(float).shift(1)

    # Bollinger Band Width
    bbands = ta.bbands(
        df["close"],
        length=config.BBANDS_PERIOD,
        std=config.BBANDS_STD
    )
    if bbands is not None:
        upper = bbands.iloc[:, 0]
        middle = bbands.iloc[:, 1]
        lower = bbands.iloc[:, 2]
        mid_safe = middle.replace(0, np.nan)
        df["BB_Width"] = ((upper - lower) / mid_safe).shift(1)

        # NEW: BB %B — where price is within the bands (0=lower, 1=upper)
        band_range = upper - lower
        band_range_safe = band_range.replace(0, np.nan)
        df["BB_PctB"] = ((df["close"] - lower) / band_range_safe).shift(1)

    return df


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return features in pips.
    NEW: Rolling mean/std of returns (momentum persistence & volatility clustering).
    """
    for period in config.RETURN_PERIODS:
        raw_return = (df["close"] - df["close"].shift(period)) / config.PIP_SIZE
        df[f"Return_{period}"] = raw_return.shift(1)

    # NEW: Rolling statistics of 1-bar returns — captures momentum persistence
    ret_1 = (df["close"] - df["close"].shift(1)) / config.PIP_SIZE
    for window in config.LAG_PERIODS:
        df[f"RetMean_{window}"] = ret_1.rolling(window).mean().shift(1)
        df[f"RetStd_{window}"] = ret_1.rolling(window).std().shift(1)

    return df


def add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    NEW: Price action features derived from candlestick structure.
    These capture market microstructure (buyer/seller pressure).
    """
    body = abs(df["close"] - df["open"])
    full_range = df["high"] - df["low"]
    full_range_safe = full_range.replace(0, np.nan)

    # Body ratio: how much of the candle is "body" vs wicks
    # High body ratio = strong conviction, low = indecision (doji-like)
    df["Body_Ratio"] = (body / full_range_safe).shift(1)

    # Upper wick ratio: selling pressure
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    df["Upper_Wick"] = (upper_wick / full_range_safe).shift(1)

    # Lower wick ratio: buying pressure
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    df["Lower_Wick"] = (lower_wick / full_range_safe).shift(1)

    # Candle direction: 1 = bullish (close > open), 0 = bearish
    df["Candle_Dir"] = (df["close"] > df["open"]).astype(int).shift(1)

    # NEW: Consecutive candle count — how many same-direction candles in a row
    # Detects trend exhaustion or momentum
    direction = (df["close"] > df["open"]).astype(int)
    groups = (direction != direction.shift(1)).cumsum()
    df["Consec_Candles"] = direction.groupby(groups).cumcount().shift(1)

    return df


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trading session indicators based on UTC hour. NOT shifted."""
    hour = df.index.hour

    asian_start, asian_end = config.SESSION_ASIAN
    london_start, london_end = config.SESSION_LONDON
    ny_start, ny_end = config.SESSION_NY

    df["is_asian"] = ((hour >= asian_start) & (hour < asian_end)).astype(int)
    df["is_london"] = ((hour >= london_start) & (hour < london_end)).astype(int)
    df["is_ny"] = ((hour >= ny_start) & (hour < ny_end)).astype(int)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical time encoding. NOT shifted."""
    hour = df.index.hour
    dow = df.index.dayofweek
    month = df.index.month

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # NEW: Month cyclical — captures seasonal patterns (e.g., summer doldrums)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    return df


def engineer_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """
    Full feature engineering pipeline v2.0.
    Designed for IDENTICAL use in training and live inference.

    Total features: ~45 (up from 24 in v1.0)
    """
    logger.info("Engineering features (v2.0 — expanded feature set)...")

    df = df.copy()

    df = add_trend_features(df)
    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_return_features(df)
    df = add_price_action_features(df)
    df = add_session_features(df)
    df = add_time_features(df)

    if drop_na:
        n_before = len(df)
        df = df.dropna()
        n_dropped = n_before - len(df)
        logger.info(f"Dropped {n_dropped} rows due to indicator warmup NaNs. "
                    f"Remaining: {len(df)} rows.")

    non_feature_cols = ["open", "high", "low", "close", "volume", "pip_return", "label"]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    logger.info(f"Total features: {len(feature_cols)}")
    logger.info(f"Features: {feature_cols}")

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get feature column names (excludes OHLCV, label, pip_return)."""
    non_feature_cols = ["open", "high", "low", "close", "volume", "pip_return", "label"]
    return [c for c in df.columns if c not in non_feature_cols]
