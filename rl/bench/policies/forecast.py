"""Long-only trading rule driven by an LSTM next-session return forecast.

One LSTM, shared by all assets, reads the last ``window`` sessions of causal
scale-free features of an asset (log return of the close, log high/low range
and log close/open, each standardised with statistics of the training
sessions) and predicts the asset's next executable return: the open-to-open
return from the fill session to the one after, which is what holding the
position for one decision interval earns under the benchmark's execution
model. At every close the policy puts equal weight on the listed assets whose
forecast return clears a cost hurdle (optionally only the ``top`` highest) and
holds cash when none does.

The model and training loop are those of :mod:`rl.forecast.lstm`; only the
windowing is done here because the original works on long frames.

Training labels end before ``validation_start``; validation labels (used only
to choose the epoch) begin on or after it, so no label spans the boundary.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

from rl.bench.engine import Decision
from rl.bench.market import MarketData, MarketView, as_day
from rl.bench.policy import Policy
from rl.forecast.lstm import LSTMForecaster, predict_ratio, train_forecaster

logger = logging.getLogger(__name__)

N_FEATURES = 3
CLIP = 10.0
MIN_SAMPLES = 32
_WEIGHTS_FILE = "model.pt"
_STATE_FILE = "state.json"


def _features(data: MarketData) -> np.ndarray:
    """Causal ``(sessions, assets, 3)`` features; NaN where an input price is missing."""
    close = data.close
    previous = np.vstack([np.full((1, close.shape[1]), np.nan), close[:-1]])
    return np.stack(
        [np.log(close / previous), np.log(data.high / data.low), np.log(close / data.open)],
        axis=-1,
    )


def _windows(scaled: np.ndarray, window: int) -> np.ndarray:
    """``(ends, assets, window, features)`` view; entry ``k`` ends at session ``k + window - 1``."""
    return sliding_window_view(scaled, window, axis=0).transpose(0, 1, 3, 2)


class LSTMForecastPolicy(Policy):
    """Hold the assets an LSTM expects to rise, equally weighted, else cash.

    ``transaction_cost`` is the hurdle a forecast return must exceed to buy an
    asset that is not already held; held assets stay while the forecast is
    positive. ``top`` limits the holding to the highest forecasts.
    """

    stochastic = True

    def __init__(
        self,
        window: int = 30,
        hidden_size: int = 32,
        num_layers: int = 2,
        epochs: int = 20,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        loss: str = "mape",
        top: int | None = None,
        transaction_cost: float = 0.001,
    ):
        if window < 2:
            raise ValueError("window must be at least 2")
        if top is not None and top < 1:
            raise ValueError("top must be a positive integer or None")
        if transaction_cost < 0:
            raise ValueError("transaction_cost must be nonnegative")
        self.window = int(window)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.loss = loss
        self.top = None if top is None else int(top)
        self.transaction_cost = float(transaction_cost)
        self.model: LSTMForecaster | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.history_: list[dict] = []

    def _scale(self, features: np.ndarray) -> np.ndarray:
        return np.clip((features - self.mean) / self.std, -CLIP, CLIP)

    def fit(self, history: MarketData, validation_start: np.datetime64, seed: int) -> None:
        n_sessions = len(history)
        cutoff = min(history.index_of(as_day(validation_start)), n_sessions)
        # Training labels (open[t+2] / open[t+1]) must all precede validation_start.
        last_train = cutoff - 3
        if last_train - self.window + 1 < MIN_SAMPLES:
            raise ValueError("Not enough history before validation_start to train the forecaster")

        features = _features(history)
        training = features[:cutoff]
        finite = np.isfinite(training).all(axis=-1)
        rows = training[finite]
        self.mean = rows.mean(axis=0)
        self.std = np.maximum(rows.std(axis=0), 1e-12)
        scaled = self._scale(features)

        label = np.full(history.open.shape, np.nan)
        label[:-2] = history.open[2:] / history.open[1:-1]
        windows = _windows(scaled, self.window)
        ends = np.arange(windows.shape[0]) + self.window - 1
        valid = np.isfinite(windows).all(axis=(2, 3)) & np.isfinite(label[ends])
        train_mask = valid & (ends <= last_train)[:, None]
        val_mask = valid & (ends >= cutoff - 1)[:, None]

        def gather(mask):
            k, n = np.nonzero(mask)
            return (
                np.ascontiguousarray(windows[k, n], dtype=np.float32),
                label[ends[k], n].astype(np.float32),
            )

        X, y = gather(train_mask)
        X_val, y_val = gather(val_mask)
        if len(X) < MIN_SAMPLES:
            raise ValueError("Only {} training windows; need at least {}".format(len(X), MIN_SAMPLES))
        logger.info("LSTM forecast: %d training and %d validation windows", len(X), len(X_val))

        torch.manual_seed(seed)
        self.model, table = train_forecaster(
            X,
            y,
            X_val,
            y_val,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            loss=self.loss,
        )
        self.history_ = table.to_dict("records")

    def forecast(self, history: MarketData) -> np.ndarray:
        """Predicted next open-to-open return per asset; NaN where the window is incomplete."""
        if self.model is None:
            raise RuntimeError("fit or load the policy before calling act")
        recent = history.window(self.window + 1)
        out = np.full(len(history.tickers), np.nan)
        if len(recent) < self.window + 1:
            return out
        scaled = self._scale(_features(recent)[1:])  # (window, assets, features)
        X = np.ascontiguousarray(scaled.transpose(1, 0, 2), dtype=np.float32)
        ok = np.isfinite(X).all(axis=(1, 2))
        if ok.any():
            out[ok] = predict_ratio(self.model, X[ok]) - 1.0
        return out

    def act(self, view: MarketView) -> Decision:
        expected = self.forecast(view.history)
        held = view.weights > 0
        hurdle = np.where(held, 0.0, self.transaction_cost)
        with np.errstate(invalid="ignore"):
            chosen = np.nonzero(expected > hurdle)[0]
        if self.top is not None and chosen.size > self.top:
            chosen = chosen[np.argsort(-expected[chosen], kind="stable")[: self.top]]
        weights = np.zeros(len(view.tickers))
        if chosen.size:
            weights[chosen] = 1.0 / chosen.size
        return weights

    def save(self, directory: Path) -> None:
        if self.model is None:
            raise RuntimeError("nothing to save: the policy is not fitted")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / _WEIGHTS_FILE)
        state = {
            "window": self.window,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
        (directory / _STATE_FILE).write_text(json.dumps(state, indent=2))

    def load(self, directory: Path) -> None:
        directory = Path(directory)
        state = json.loads((directory / _STATE_FILE).read_text())
        for key in ("window", "hidden_size", "num_layers"):
            if state[key] != getattr(self, key):
                raise ValueError(
                    "Saved {} is {} but the policy was built with {}".format(key, state[key], getattr(self, key))
                )
        self.mean = np.array(state["mean"], dtype=float)
        self.std = np.array(state["std"], dtype=float)
        model = LSTMForecaster(N_FEATURES, self.hidden_size, self.num_layers)
        model.load_state_dict(torch.load(directory / _WEIGHTS_FILE, map_location="cpu", weights_only=True))
        model.eval()
        self.model = model
