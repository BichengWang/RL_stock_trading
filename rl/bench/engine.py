"""The benchmark's single execution model.

A policy decides target weights at a session's close; the account trades to
them at the next session's open and is marked at that session's close.
Targets are long-only fractions of post-fee equity summing to at most one;
the remainder is cash, which earns nothing. Fees are a proportional rate on
traded notional. An asset without a price at a fill session cannot be
bought, and any holding in it is sold at its last close, paying the fee.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from rl.bench.market import MarketData, MarketView
from rl.trade.execution import rebalance_positions

Decision = Optional[np.ndarray]
WEIGHT_TOLERANCE = 1e-9


class PolicyError(ValueError):
    """A policy returned weights the execution model cannot trade."""


def check_weights(target, n_assets: int, date=None) -> np.ndarray:
    """Validate long-only, unlevered target weights."""
    weights = np.array(target, dtype=float).reshape(-1)
    where = "" if date is None else " on {}".format(date)
    if weights.size != n_assets:
        raise PolicyError("Expected {} weights{}, got {}".format(n_assets, where, weights.size))
    if not np.isfinite(weights).all():
        raise PolicyError("Weights must be finite{}".format(where))
    if (weights < -WEIGHT_TOLERANCE).any():
        raise PolicyError("Weights must be nonnegative (long-only){}".format(where))
    np.maximum(weights, 0.0, out=weights)
    total = weights.sum()
    if total > 1 + WEIGHT_TOLERANCE * max(1, n_assets):
        raise PolicyError("Weights sum to {:.6g} > 1 (no leverage){}".format(total, where))
    if total > 1:
        weights /= total
    return weights


class Account:
    """Cash and fractional share holdings traded under the execution model."""

    def __init__(self, data: MarketData, transaction_cost: float, initial: float = 1.0):
        if not np.isfinite(transaction_cost) or not 0 <= transaction_cost < 1:
            raise ValueError("transaction_cost must be in [0, 1)")
        if not np.isfinite(initial) or initial <= 0:
            raise ValueError("initial capital must be positive")
        self.data = data
        self.transaction_cost = float(transaction_cost)
        self.cash = float(initial)
        self.shares = np.zeros(len(data.tickers))
        self._values = np.empty(len(data.tickers))

    def _mark(self, t: int) -> np.ndarray:
        np.multiply(self.shares, self.data.close[t], out=self._values, where=self.shares != 0)
        self._values[self.shares == 0] = 0.0
        return self._values

    def equity(self, t: int) -> float:
        """Account value at session ``t``'s close."""
        return self.cash + float(self._mark(t).sum())

    def weights(self, t: int) -> np.ndarray:
        """Asset weights at session ``t``'s close."""
        values = self._mark(t)
        total = self.cash + values.sum()
        return values / total if total > 0 else np.zeros_like(values)

    def trade(self, t: int, target: Decision) -> tuple[float, float]:
        """Fill at session ``t``'s open; ``None`` keeps current holdings.

        Returns ``(turnover, cost)``: traded notional as a fraction of
        pre-trade equity, and fees paid.
        """
        opens = self.data.open[t]
        tradable = np.isfinite(opens)
        traded = 0.0
        cost = 0.0
        stuck = (self.shares != 0) & ~tradable
        if stuck.any():
            # Held at the previous close, so that close is the last price.
            proceeds = float(np.dot(self.shares[stuck], self.data.close[t - 1][stuck]))
            fee = proceeds * self.transaction_cost
            self.cash += proceeds - fee
            self.shares[stuck] = 0.0
            traded += proceeds
            cost += fee
        positions = np.where(tradable, self.shares * np.where(tradable, opens, 0.0), 0.0)
        before = self.cash + positions.sum() + cost
        if target is not None:
            weights = np.where(tradable, target, 0.0)
            new_positions, self.cash, fee = rebalance_positions(
                self.cash, positions, weights, self.transaction_cost
            )
            traded += float(np.abs(new_positions - positions).sum())
            cost += fee
            self.shares = np.divide(
                new_positions, opens, out=np.zeros_like(new_positions), where=tradable
            )
        return (traded / before if before > 0 else 0.0), cost


@dataclass(frozen=True, eq=False)
class Backtest:
    """An evaluated period.

    ``dates[0]`` is the decision session before the period, where equity is
    the initial capital; each later date is an evaluated session. Per-session
    arrays (``weights``, ``turnover``, ``costs``) align with ``dates[1:]``.
    """

    dates: np.ndarray
    tickers: tuple[str, ...]
    equity: np.ndarray
    weights: np.ndarray
    turnover: np.ndarray
    costs: np.ndarray

    @property
    def returns(self) -> np.ndarray:
        return self.equity[1:] / self.equity[:-1] - 1


def simulate(
    data: MarketData,
    act: Callable[[MarketView], Decision],
    start: int,
    transaction_cost: float,
    *,
    stop: Optional[int] = None,
    initial: float = 1.0,
) -> Backtest:
    """Evaluate ``act`` on sessions ``[start, stop)`` of ``data``.

    ``act`` sees history through each decision close only; the first
    decision is at session ``start - 1`` and fills at ``start``'s open.
    """
    stop = len(data) if stop is None else stop
    if not 1 <= start < stop <= len(data):
        raise ValueError("Need 1 <= start < stop <= {}, got {} and {}".format(len(data), start, stop))
    account = Account(data, transaction_cost, initial)
    n_days, n_assets = stop - start, len(data.tickers)
    equity = np.empty(n_days + 1)
    weights = np.empty((n_days, n_assets))
    turnover = np.empty(n_days)
    costs = np.empty(n_days)
    equity[0] = initial
    current = np.zeros(n_assets)
    for i, t in enumerate(range(start - 1, stop - 1)):
        current.setflags(write=False)
        decision = act(MarketView(data.rows(0, t + 1), current))
        if decision is not None:
            decision = check_weights(decision, n_assets, data.dates[t])
        turnover[i], costs[i] = account.trade(t + 1, decision)
        current = account.weights(t + 1)
        weights[i] = current
        equity[i + 1] = account.equity(t + 1)
    return Backtest(
        dates=data.dates[start - 1 : stop],
        tickers=data.tickers,
        equity=equity,
        weights=weights,
        turnover=turnover,
        costs=costs,
    )
