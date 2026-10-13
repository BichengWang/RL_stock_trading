"""Rule-based baselines every learned policy has to beat.

All baselines are deterministic and stateless: the target weights are a pure
function of the :class:`~rl.bench.market.MarketView` (history up to the
decision close and the current drifted weights), so they behave identically
when restored for the live track.

Shared conventions:

* An asset without a price at the decision session (not yet listed, halted or
  delisted) never receives weight.
* Estimators use only the trailing window named by the policy's lookback. An
  asset needs a price on every session of that window to be scored; assets
  that were listed more recently are skipped until they have the history.
  While *no* listed asset has enough history, the policy holds equal weight
  over the listed assets instead.
* Policies with a ``frequency`` decide on the first session of a new
  calendar period (day, ISO week, month or quarter) and otherwise return
  ``None`` (hold, so no trading). They also decide whenever the account is
  fully in cash, so the first position is taken immediately instead of
  waiting for the next period and a signal that appears while in cash is
  acted on at once.
"""

from __future__ import annotations

import numpy as np

from rl.bench.engine import Decision
from rl.bench.market import MarketView
from rl.bench.policy import Policy

FREQUENCIES = ("daily", "weekly", "monthly", "quarterly")
_INVESTED = 1e-9  # total asset weight above which the account counts as invested


def equal_weights(listed: np.ndarray) -> np.ndarray:
    """``1/n`` over the ``n`` listed assets, zero elsewhere (all zeros if none)."""
    weights = np.zeros(listed.size)
    if listed.any():
        weights[listed] = 1.0 / listed.sum()
    return weights


def _period(dates: np.ndarray, frequency: str) -> np.ndarray:
    """Index of the calendar period each date falls in."""
    days = dates.astype("datetime64[D]").astype(np.int64)
    if frequency == "daily":
        return days
    if frequency == "weekly":
        return (days + 3) // 7  # 1970-01-01 was a Thursday: weeks start on Monday
    months = dates.astype("datetime64[M]").astype(np.int64)
    return months if frequency == "monthly" else months // 3


def new_period(dates: np.ndarray, frequency: str) -> bool:
    """Whether the last date starts a new period relative to the one before it."""
    if dates.size < 2:
        return True
    last = _period(dates[-2:], frequency)
    return bool(last[0] != last[1])


def check_frequency(frequency: str) -> str:
    if frequency not in FREQUENCIES:
        raise ValueError("frequency must be one of {}, got {!r}".format(FREQUENCIES, frequency))
    return frequency


def check_lookback(lookback: int, minimum: int = 2) -> int:
    if not isinstance(lookback, (int, np.integer)) or lookback < minimum:
        raise ValueError("lookback must be an integer of at least {}, got {!r}".format(minimum, lookback))
    return int(lookback)


def trailing_window(view: MarketView, sessions: int) -> tuple[np.ndarray, np.ndarray]:
    """The last ``sessions`` closes and the listed assets priced on all of them.

    No asset qualifies while the history is shorter than ``sessions``.
    """
    closes = view.history.close[-sessions:]
    if len(closes) < sessions:
        return closes, np.zeros(closes.shape[1], dtype=bool)
    return closes, view.listed & np.isfinite(closes).all(axis=0)


class Cash(Policy):
    """Stay in cash: the zero-risk floor."""

    stochastic = False

    def act(self, view: MarketView) -> Decision:
        return np.zeros(len(view.tickers))


class EqualWeightBuyHold(Policy):
    """Split capital equally over the listed assets once, then never trade.

    This is the benchmark carried by every leaderboard record. Assets that
    list or recover later are not added, and proceeds from a delisted asset
    stay in cash (that is what holding means); only a fully-cash account
    invests again.
    """

    stochastic = False

    def act(self, view: MarketView) -> Decision:
        if view.weights.sum() > _INVESTED:
            return None
        return equal_weights(view.listed)


class _Rebalancing(Policy):
    """Template: recompute targets at the first session of each period."""

    stochastic = False

    def __init__(self, frequency: str = "monthly"):
        self.frequency = check_frequency(frequency)

    def act(self, view: MarketView) -> Decision:
        if view.weights.sum() > _INVESTED and not new_period(view.history.dates, self.frequency):
            return None
        return self.target(view)

    def target(self, view: MarketView) -> np.ndarray:
        raise NotImplementedError


class EqualWeightRebalance(_Rebalancing):
    """Rebalance to ``1/N`` over the listed assets every period.

    DeMiguel, Garlappi and Uppal (2009) show 1/N is hard to beat.
    """

    def target(self, view: MarketView) -> np.ndarray:
        return equal_weights(view.listed)


class InverseVolatility(_Rebalancing):
    """Weight each asset in proportion to ``1 / volatility``.

    Volatility is the standard deviation of daily simple returns over the
    last ``lookback`` sessions. Weights sum to one over assets with a full
    window.
    """

    def __init__(self, lookback: int = 63, frequency: str = "monthly"):
        super().__init__(frequency)
        self.lookback = check_lookback(lookback)

    def target(self, view: MarketView) -> np.ndarray:
        closes, ready = trailing_window(view, self.lookback + 1)
        if not ready.any():
            return equal_weights(view.listed)
        returns = closes[1:, ready] / closes[:-1, ready] - 1
        inverse = 1.0 / np.maximum(returns.std(axis=0, ddof=1), 1e-12)
        weights = np.zeros(closes.shape[1])
        weights[ready] = inverse / inverse.sum()
        return weights


def min_variance_weights(covariance: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    """Long-only, fully invested minimum-variance weights (primal active set).

    ``covariance`` must be symmetric positive semi-definite; a tiny ridge keeps
    singular matrices solvable.
    """
    covariance = np.asarray(covariance, dtype=float)
    n = covariance.shape[0]
    covariance = covariance + np.eye(n) * 1e-10 * max(np.trace(covariance) / n, 1e-12)
    weights = np.full(n, 1.0 / n)
    active = np.ones(n, dtype=bool)
    for _ in range(100 * n):
        index = np.flatnonzero(active)
        solution = np.linalg.solve(covariance[np.ix_(index, index)], np.ones(index.size))
        target = np.zeros(n)
        target[index] = solution / solution.sum()
        if (target[index] < -tolerance).any():
            # Walk toward the unconstrained optimum until an asset hits zero.
            step = target - weights
            blocking = active & (step < 0)
            ratios = np.where(blocking, weights / np.where(blocking, -step, 1.0), np.inf)
            drop = int(np.argmin(ratios))
            weights = weights + ratios[drop] * step
            weights[drop] = 0.0
            active[drop] = False
            continue
        weights = np.maximum(target, 0.0)
        gradient = covariance @ weights
        gain = gradient - weights @ gradient  # negative: adding the asset lowers variance
        gain[active] = np.inf
        add = int(np.argmin(gain))
        if gain[add] >= -tolerance * max(gradient.max(), 1e-12):
            break
        active[add] = True
    return weights / weights.sum()


class MinimumVariance(_Rebalancing):
    """Long-only global minimum-variance portfolio.

    The covariance of daily simple returns over the last ``lookback``
    sessions is shrunk toward its diagonal,
    ``(1 - shrinkage) * S + shrinkage * diag(S)``, and the weights solve
    ``min w'Σw`` with ``w >= 0`` and ``sum(w) = 1`` over assets with a full
    window.
    """

    def __init__(self, lookback: int = 252, frequency: str = "monthly", shrinkage: float = 0.1):
        super().__init__(frequency)
        self.lookback = check_lookback(lookback)
        if not 0 <= shrinkage <= 1:
            raise ValueError("shrinkage must be in [0, 1], got {!r}".format(shrinkage))
        self.shrinkage = float(shrinkage)

    def target(self, view: MarketView) -> np.ndarray:
        closes, ready = trailing_window(view, self.lookback + 1)
        weights = np.zeros(closes.shape[1])
        if not ready.any():
            return equal_weights(view.listed)
        if ready.sum() == 1:
            weights[ready] = 1.0
            return weights
        returns = closes[1:, ready] / closes[:-1, ready] - 1
        covariance = np.cov(returns, rowvar=False)
        covariance = (1 - self.shrinkage) * covariance + self.shrinkage * np.diag(np.diag(covariance))
        weights[ready] = min_variance_weights(covariance)
        return weights


class Momentum(_Rebalancing):
    """Cross-sectional momentum with an absolute filter.

    Each asset is scored by its return from ``lookback`` sessions ago to
    ``skip`` sessions ago (Jegadeesh and Titman, 1993: the recent ``skip``
    sessions are left out to avoid short-term reversal). The ``top``
    best-scoring assets get ``1/k`` each, where ``k = min(top, scored
    assets)``, but only if their score is positive; slots of assets with
    non-positive momentum stay in cash.
    """

    def __init__(self, lookback: int = 252, skip: int = 21, top: int = 3, frequency: str = "monthly"):
        super().__init__(frequency)
        self.lookback = check_lookback(lookback)
        if not isinstance(skip, (int, np.integer)) or not 0 <= skip < self.lookback:
            raise ValueError("skip must be an integer in [0, lookback), got {!r}".format(skip))
        if not isinstance(top, (int, np.integer)) or top < 1:
            raise ValueError("top must be a positive integer, got {!r}".format(top))
        self.skip = int(skip)
        self.top = int(top)

    def target(self, view: MarketView) -> np.ndarray:
        close = view.history.close
        weights = np.zeros(close.shape[1])
        if len(close) <= self.lookback:
            return equal_weights(view.listed)
        past, recent = close[-1 - self.lookback], close[-1 - self.skip]
        scored = view.listed & np.isfinite(past) & np.isfinite(recent)
        if not scored.any():
            return equal_weights(view.listed)
        score = np.where(scored, recent / np.where(scored, past, 1.0) - 1, -np.inf)
        k = min(self.top, int(scored.sum()))
        best = np.argsort(-score, kind="stable")[:k]
        best = best[score[best] > 0]
        weights[best] = 1.0 / k
        return weights


class TrendFollowing(_Rebalancing):
    """Hold assets trading above their simple moving average.

    Each asset whose close exceeds the mean of its last ``lookback`` closes
    (including today's) gets ``1/N`` of capital, ``N`` being the number of
    listed assets; the rest of the capital is cash (Moskowitz, Ooi and
    Pedersen, 2012, time-series momentum).
    """

    def __init__(self, lookback: int = 200, frequency: str = "monthly"):
        super().__init__(frequency)
        self.lookback = check_lookback(lookback)

    def target(self, view: MarketView) -> np.ndarray:
        closes, ready = trailing_window(view, self.lookback)
        if not ready.any():
            return equal_weights(view.listed)
        weights = np.zeros(closes.shape[1])
        above = closes[-1, ready] > closes[:, ready].mean(axis=0)
        weights[np.flatnonzero(ready)[above]] = 1.0 / view.listed.sum()
        return weights
