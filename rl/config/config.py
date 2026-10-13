"""Defaults for the ``--mode`` training toolkit: output directories, dates,
features, agent hyperparameters and the default ticker list.

Most of these can be overridden from the command line (``python main.py -h``)
or through :class:`rl.autotrain.training.TrainConfig`. Benchmark tasks define
their own universes and splits in :mod:`rl.bench.tasks`.
"""

# output directories (relative to the working directory)
DATA_SAVE_DIR = "datasets"
TRAINED_MODEL_DIR = "trained_models"
TENSORBOARD_LOG_DIR = "tensorboard_log"
RESULTS_DIR = "results"

## time_fmt = '%Y-%m-%d'
START_DATE = "2000-01-01"
END_DATE = "2021-01-01"
START_TRADE_DATE = "2019-01-01"

## stockstats technical indicator column names
## check https://pypi.org/project/stockstats/ for different names
TECHNICAL_INDICATORS_LIST = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_10",
    "rsi_20",
    "cci_10",
    "cci_20",
    "dx_30",
    "close_20_sma",
    "close_60_sma",
    "close_120_sma",
    "close_20_ema",
    "close_60_ema",
    "close_120_ema",
]


## Model Parameters
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}

# Dow 30 constituents at 2019/01. Choosing today's index members for a
# backtest that starts earlier is survivorship-biased.
DOW_30_TICKER = [
    "AAPL",
    "MSFT",
    "JPM",
    "V",
    "RTX",
    "PG",
    "GS",
    "NKE",
    "DIS",
    "AXP",
    "HD",
    "INTC",
    "WMT",
    "IBM",
    "MRK",
    "UNH",
    "KO",
    "CAT",
    "TRV",
    "JNJ",
    "CVX",
    "MCD",
    "VZ",
    "CSCO",
    "XOM",
    "BA",
    "MMM",
    "PFE",
    "WBA",
    "DD",
]
