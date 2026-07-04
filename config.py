"""
config.py — Centralized Configuration for Forex ML Pipeline (v2.0)
====================================================================
UPGRADED for 5-year dataset (2021-2025) + production-quality modeling.
All tunable parameters live here. No magic numbers in other modules.
"""

import os

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Source data directory (Pipeline reads ALL .csv files inside this folder)

# ──────────────────────────────────────────────────────────────
# Instrument & Timeframe
# ──────────────────────────────────────────────────────────────
PAIR = "EURUSD"
TIMEFRAME = "H1"
PIP_SIZE = 0.0001  # 1 pip = 0.0001 for EUR/USD

# Timezone conversion (histdata.com default is EST)
SOURCE_TZ = "US/Eastern"
TARGET_TZ = "UTC"

# ──────────────────────────────────────────────────────────────
# Labeling Thresholds (in pips)
# ──────────────────────────────────────────────────────────────
LABEL_BUY_THRESHOLD = 10    # pip_return > 10 -> BUY (1)
LABEL_SELL_THRESHOLD = -10  # pip_return < -10 -> SELL (-1)
# Otherwise -> HOLD (0)

# ──────────────────────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────────────────────
# Moving average periods
MA_PERIODS = [10, 20, 50]
EMA_FAST = 12
EMA_SLOW = 26

# Momentum
RSI_PERIOD = 14
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Volatility
ATR_PERIOD = 14
BBANDS_PERIOD = 20
BBANDS_STD = 2.0

# Return lookback (in pips)
RETURN_PERIODS = [1, 5, 10]

# Lag features — rolling windows for mean-reversion / trend strength
LAG_PERIODS = [3, 6, 12, 24]  # hours

# Session times (UTC hours)
SESSION_ASIAN = (0, 8)
SESSION_LONDON = (8, 16)
SESSION_NY = (13, 21)

# ──────────────────────────────────────────────────────────────
# Data Cleaning
# ──────────────────────────────────────────────────────────────
MAX_FFILL_CANDLES = 3

# ──────────────────────────────────────────────────────────────
# Train / Val / Test Split Ratios (chronological)
# ──────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ──────────────────────────────────────────────────────────────
# Model Hyperparameters — TUNED for 5-year dataset
# ──────────────────────────────────────────────────────────────

# Random Forest — beefier for larger dataset
RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 15,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

# XGBoost — tuned with regularization to prevent overfitting
XGB_PARAMS = {
    "max_depth": 8,
    "learning_rate": 0.05,        # slower learning for better generalization
    "n_estimators": 1000,         # more trees (early stopping will pick best)
    "min_child_weight": 10,       # prevent splits on tiny samples
    "subsample": 0.8,             # row subsampling (bagging)
    "colsample_bytree": 0.8,     # column subsampling per tree
    "gamma": 1,                   # min loss reduction to make split
    "reg_alpha": 0.1,            # L1 regularization
    "reg_lambda": 1.0,           # L2 regularization
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "use_label_encoder": False,
    "random_state": 42,
    "verbosity": 0,
    "tree_method": "hist",       # faster for large datasets
}
XGB_EARLY_STOPPING = 50  # more patience for larger dataset

# Logistic Regression baseline
LR_PARAMS = {
    "max_iter": 2000,
    "class_weight": "balanced",
    "random_state": 42,
    "solver": "lbfgs",
    "C": 0.1,                    # stronger regularization
}

# ──────────────────────────────────────────────────────────────
# Backtesting
# ──────────────────────────────────────────────────────────────
SPREAD_PIPS = 2.0
SLIPPAGE_PIPS = 0.5
SL_PIPS = 20
TP_PIPS = 40
INITIAL_CAPITAL = 10_000
MAX_POSITIONS = 1
MIN_CONFIDENCE = 0.40  # Lowered from 0.60 so the model actually takes trades
MAX_DAILY_LOSS_PCT = 3.0
LOT_SIZE = 10_000  # mini lot

# ──────────────────────────────────────────────────────────────
# Model Registry / Versioning
# ──────────────────────────────────────────────────────────────
MODEL_VERSION = "v2.0.0"

# ──────────────────────────────────────────────────────────────
# Monitoring
# ──────────────────────────────────────────────────────────────
PREDICTION_LOG_FILE = os.path.join(LOGS_DIR, "predictions.csv")
ANOMALY_LOG_FILE = os.path.join(LOGS_DIR, "anomalies.log")

# ──────────────────────────────────────────────────────────────
# Label Encoding (internal mapping)
# ──────────────────────────────────────────────────────────────
LABEL_MAP = {-1: 0, 0: 1, 1: 2}       # original -> model
LABEL_MAP_INV = {0: -1, 1: 0, 2: 1}   # model -> original
LABEL_NAMES = {-1: "SELL", 0: "HOLD", 1: "BUY"}
