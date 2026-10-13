"""Causal, scale-free per-asset features shared by RL training and evaluation.

Every feature at session ``t`` depends only on the ``LOOKBACK`` sessions
ending at ``t``, and is computed with elementwise operations in a fixed
order. Evaluating :func:`feature_panel` over a whole history and taking row
``t`` therefore gives exactly the same numbers as :func:`latest_features` on
``history.window(LOOKBACK)`` at decision time: what an agent trains on is
what it sees when it trades.

A feature whose window contains a missing price (asset not yet listed,
halted, delisted) or extends before the start of the data is 0, and so is
every feature of an unlisted asset; the final ``listed`` flag tells the two
apart from a genuine 0.
"""

from __future__ import annotations

import numpy as np

from rl.bench.market import MarketData

LOOKBACK = 200  # sessions of history any feature looks at (the 200-day average)
FEATURE_NAMES = (
    "ret_1",
    "ret_5",
    "ret_21",
    "ret_63",
    "vol_21",
    "vol_63",
    "sma_50",
    "sma_200",
    "rsi_14",
    "volume_z",
    "listed",
)
N_FEATURES = len(FEATURE_NAMES)

_CLIP = 5.0
_DAILY_MOVE = 0.015  # typical one-day move that scales returns to order one
_TRADING_DAYS = 252
_VOLUME_WINDOW = 63


def _shifted(x: np.ndarray, lag: int) -> np.ndarray:
    """``x`` with row ``t`` replaced by row ``t - lag`` (NaN before the start)."""
    out = np.full_like(x, np.nan)
    if lag < len(x):
        out[lag:] = x[: len(x) - lag]
    return out


def _roll_sum(x: np.ndarray, window: int) -> np.ndarray:
    """Sum of the last ``window`` rows, accumulated oldest first (NaN if incomplete)."""
    out = np.full_like(x, np.nan)
    count = len(x) - window + 1
    if count > 0:
        total = x[:count].copy()
        for k in range(1, window):
            total += x[k : k + count]
        out[window - 1 :] = total
    return out


def _roll_mean_std(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling mean and population standard deviation (two-pass, fixed order)."""
    mean = _roll_sum(x, window) / window
    var = np.full_like(x, np.nan)
    count = len(x) - window + 1
    if count > 0:
        squares = np.zeros_like(x[:count])
        centre = mean[window - 1 :]
        for k in range(window):
            squares += (x[k : k + count] - centre) ** 2
        var[window - 1 :] = squares / window
    return mean, np.sqrt(var)


def feature_panel(data: MarketData) -> np.ndarray:
    """Features of every session as a ``(T, N, N_FEATURES)`` float array."""
    close = data.close
    with np.errstate(all="ignore"):
        log_close = np.log(close)
        step = log_close - _shifted(log_close, 1)
        columns = []
        for horizon in (1, 5, 21, 63):
            move = log_close - _shifted(log_close, horizon)
            columns.append(move / (_DAILY_MOVE * np.sqrt(horizon)))
        for horizon in (21, 63):
            columns.append(_roll_mean_std(step, horizon)[1] * np.sqrt(_TRADING_DAYS))
        for horizon in (50, 200):
            columns.append(np.log(close / (_roll_sum(close, horizon) / horizon)) / 0.05)
        change = close - _shifted(close, 1)
        gain = _roll_sum(np.maximum(change, 0.0), 14)
        loss = _roll_sum(np.maximum(-change, 0.0), 14)
        total = gain + loss
        # (RSI - 50) / 50 for the simple-average RSI, defined as 0 when flat.
        columns.append(np.where(total > 0, (gain - loss) / np.where(total > 0, total, 1.0), 0.0))
        log_volume = np.log1p(data.volume)
        mean, std = _roll_mean_std(log_volume, _VOLUME_WINDOW)
        columns.append(np.where(std > 0, (log_volume - mean) / np.where(std > 0, std, 1.0), 0.0))
    listed = np.isfinite(close)
    panel = np.stack(columns, axis=-1)
    panel = np.clip(np.nan_to_num(panel, nan=0.0, posinf=_CLIP, neginf=-_CLIP), -_CLIP, _CLIP)
    panel[~listed] = 0.0
    return np.concatenate([panel, listed[..., None].astype(float)], axis=-1)


def latest_features(history: MarketData) -> np.ndarray:
    """Features at the last session of ``history``, equal to ``feature_panel(history)[-1]``.

    Only the final ``LOOKBACK`` sessions are used, so callers may pass
    ``history.window(LOOKBACK)`` to keep the work constant.
    """
    return feature_panel(history.window(LOOKBACK))[-1]
