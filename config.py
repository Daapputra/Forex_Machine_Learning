"""
config.py — Centralized Configuration for Forex ML Pipeline (v3.0)
====================================================================
UPGRADED v3.0: Dynamic ATR-based labeling & SL/TP, feature selection,
optimized for profitable backtesting and paper-quality results.
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

# ──────────────────────────────────────────────────────────────
# Instrument & Timeframe
# ──────────────────────────────────────────────────────────────
PAIR = "EURUSD"
TIMEFRAME = "H4"  # v4.0: H4 timeframe for higher signal-to-noise ratio
PIP_SIZE = 0.0001  # 1 pip = 0.0001 for EUR/USD

# Timezone conversion (histdata.com default is EST)
SOURCE_TZ = "US/Eastern"
TARGET_TZ = "UTC"

# ──────────────────────────────────────────────────────────────
# Labeling — v3.0: DYNAMIC ATR-based (adapts to market volatility)
# ──────────────────────────────────────────────────────────────
LABEL_MODE = "atr"   # "fixed" = old style, "atr" = dynamic v3.0

# Fixed thresholds (used only if LABEL_MODE == "fixed")
LABEL_BUY_THRESHOLD = 10    # pip_return > 10 -> BUY (1)
LABEL_SELL_THRESHOLD = -10  # pip_return < -10 -> SELL (-1)

# ATR-based thresholds (used if LABEL_MODE == "atr")
LABEL_ATR_PERIOD = 14       # ATR lookback period for labeling
LABEL_ATR_MULTIPLIER = 0.5  # v4.0: threshold = 0.5x ATR (lower = more BUY/SELL labels)
# Example: if H4 ATR=28 pips, threshold = 28*0.5 = 14 pips
# This creates MORE BUY/SELL labels in volatile markets,
# and FEWER in calm markets — exactly what we want.

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
# Feature Selection — v3.0: Keep only top-N features
# ──────────────────────────────────────────────────────────────
FEATURE_SELECTION_ENABLED = True
FEATURE_SELECTION_TOP_N = 25  # Keep top 25 most important features

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

# Random Forest
RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 12,          # v3.0: slightly shallower to reduce overfitting
    "min_samples_split": 30,  # v3.0: increased
    "min_samples_leaf": 15,   # v3.0: increased
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

# XGBoost — tuned with regularization to prevent overfitting
XGB_PARAMS = {
    "max_depth": 6,              # v3.0: reduced from 8
    "learning_rate": 0.03,       # v3.0: slower learning
    "n_estimators": 1500,        # v3.0: more trees (early stopping picks best)
    "min_child_weight": 15,      # v3.0: increased
    "subsample": 0.7,            # v3.0: more row subsampling
    "colsample_bytree": 0.7,     # v3.0: more column subsampling
    "gamma": 2,                  # v3.0: stronger min loss requirement
    "reg_alpha": 0.5,            # v3.0: stronger L1
    "reg_lambda": 2.0,           # v3.0: stronger L2
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "use_label_encoder": False,
    "random_state": 42,
    "verbosity": 0,
    "tree_method": "hist",
}
XGB_EARLY_STOPPING = 50

# Logistic Regression baseline
LR_PARAMS = {
    "max_iter": 2000,
    "class_weight": "balanced",
    "random_state": 42,
    "solver": "lbfgs",
    "C": 0.1,
}

# ──────────────────────────────────────────────────────────────
# Backtesting — v3.0: DYNAMIC ATR-based SL/TP
# ──────────────────────────────────────────────────────────────
SPREAD_PIPS = 2.0
SLIPPAGE_PIPS = 0.5

# Dynamic SL/TP (v4.0): Wider TP, tighter SL
SL_ATR_MULT = 1.0   # v4.0: SL = 1.0x ATR
TP_ATR_MULT = 2.5   # v4.0: TP = 2.5x ATR (Risk:Reward = 1:2.5)
# Risk:Reward = 1:1.33 — with a decent model, this is profitable

# Fallback fixed SL/TP (used only if ATR not available)
SL_PIPS = 20
TP_PIPS = 40

INITIAL_CAPITAL = 10_000
MAX_POSITIONS = 1

# v4.1 (Fix Leakage): Thresholds are now tuned via Grid Search on Train/Val, NOT hardcoded.
CONFIDENCE_GRID = [0.35, 0.40, 0.45, 0.50]
ADX_GRID = [10.0, 15.0, 20.0]

MAX_DAILY_LOSS_PCT = 3.0
LOT_SIZE = 10_000  # mini lot

# ──────────────────────────────────────────────────────────────
# Model Registry / Versioning
# ──────────────────────────────────────────────────────────────
MODEL_VERSION = "v4.0.0"

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
