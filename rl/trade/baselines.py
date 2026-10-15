"""Rule-based portfolios to benchmark trained agents against.

Each function takes a long (date, tic) frame with finite positive ``open``
and ``close`` prices. Signals formed at a close fill at the next session's open;
the first date therefore records the initial capital entirely in cash. Each
function returns a frame with ``date`` and ``account_value`` columns, the same format
as :meth:`rl.env.env_stocktrading.StockTradingEnv.save_asset_memory`.
"""

import logging

import numpy as np
import pandas as pd

from rl.trade.execution import rebalance_positions

logger = logging.getLogger(__name__)

MIN_VARIANCE_LOOKBACK = 252


def _price_matrices(df):
    for column in ("open", "close"):
        if column not in df.columns:
            raise ValueError("Data is missing column '{}'".format(column))
    close = df.pivot(index="date", columns="tic", values="close").sort_index()
    open_prices = df.pivot(index="date", columns="tic", values="open").reindex(
        index=close.index, columns=close.columns
    )
    if close.empty:
        raise ValueError("Data is empty")
    for name, frame in (("open", open_prices), ("close", close)):
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Every date and ticker must have finite positive {} prices".format(name))
    return open_prices, close


def buy_and_hold(df, initial_amount=1_000_000, buy_cost_pct=0.0):
    """Split cash equally at the second date's open, then hold shares."""
    open_prices, close = _price_matrices(df)
    values = np.empty(len(close))
    values[0] = initial_amount
    if len(close) > 1:
        first_fill = open_prices.iloc[1].to_numpy(dtype=float)
        budget = initial_amount / len(first_fill)
        shares = budget / (first_fill * (1 + buy_cost_pct))
        values[1:] = close.to_numpy(dtype=float)[1:] @ shares
    return pd.DataFrame({"date": close.index, "account_value": values})


def equal_weight_rebalanced(
    df, initial_amount=1_000_000, transaction_cost_pct=0.0, rebalance_every=21
):
    """Equal-weight targets signaled every ``rebalance_every`` sessions.

    The first signal is on date zero and fills at date one's open. Subsequent
    rebalances also fill at the next open and pay fees on actual traded notional.
    """
    if rebalance_every < 1:
        raise ValueError("rebalance_every must be at least 1")
    open_prices, close = _price_matrices(df)
    opens = open_prices.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)
    n_days, n_tickers = closes.shape
    target = np.full(n_tickers, 1.0 / n_tickers)
    shares = np.zeros(n_tickers)
    cash = float(initial_amount)
    values = np.empty(n_days)
    values[0] = initial_amount
    for t in range(1, n_days):
        if (t - 1) % rebalance_every == 0:
            positions, cash, _ = rebalance_positions(
                cash, shares * opens[t], target, transaction_cost_pct
            )
            shares = positions / opens[t]
        values[t] = cash + np.dot(shares, closes[t])
    return pd.DataFrame({"date": close.index, "account_value": values})


def target_weights_strategy(df, weights, initial_amount=1_000_000, transaction_cost_pct=0.0):
    """Rebalance to signal-date target weights at the next session's open.

    ``weights`` has one ticker column and one row per fillable signal date,
    i.e. all dates except the final date. A supplied final row is ignored.
    Targets are fractions of post-fee equity; uninvested equity remains cash
    and weights summing to more than one borrow cash at no interest.
    Prior holdings retain their overnight P&L before each rebalance.
    """
    open_prices, close = _price_matrices(df)
    weights = weights.reindex(index=close.index[:-1], columns=close.columns)
    targets = weights.to_numpy(dtype=float)
    if not np.isfinite(targets).all():
        raise ValueError("weights must cover every signal date and ticker with finite values")
    opens = open_prices.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)
    shares = np.zeros(closes.shape[1])
    cash = float(initial_amount)
    values = np.empty(len(closes))
    values[0] = initial_amount
    for t in range(1, len(closes)):
        positions, cash, _ = rebalance_positions(
            cash, shares * opens[t], targets[t - 1], transaction_cost_pct
        )
        shares = positions / opens[t]
        values[t] = cash + np.dot(shares, closes[t])
    return pd.DataFrame({"date": close.index, "account_value": values})


def min_variance_weights(returns, ridge=1e-10):
    """Long-only, fully invested minimum-variance weights.

    ``returns`` is a (sessions, assets) array of simple returns. The weights
    minimise ``w' S w`` over ``w >= 0, sum(w) = 1`` for the sample covariance
    ``S`` (``ddof=1``) plus a ``ridge`` times its mean variance on the
    diagonal, which keeps ``S`` invertible when there are fewer sessions than
    assets. The solution is exact: an active-set method with a KKT check, so
    assets that the unconstrained solution would short get exactly zero.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 2 or returns.shape[0] < 2 or returns.shape[1] < 1:
        raise ValueError("returns must be a (sessions >= 2, assets >= 1) array")
    if not np.isfinite(returns).all():
        raise ValueError("returns must be finite")
    n = returns.shape[1]
    if n == 1:
        return np.ones(1)
    cov = np.cov(returns, rowvar=False)
    cov = cov + ridge * np.trace(cov) / n * np.eye(n)
    active = np.ones(n, dtype=bool)
    for _ in range(20 * n):
        index = np.flatnonzero(active)
        x = np.linalg.solve(cov[np.ix_(index, index)], np.ones(len(index)))
        restricted = x / x.sum()
        if restricted.min() < 0:
            active[index[np.argmin(restricted)]] = False
            continue
        weights = np.zeros(n)
        weights[index] = restricted
        gradient = cov @ weights
        variance = float(weights @ gradient)
        outside = np.flatnonzero(~active)
        if outside.size:
            slack = gradient[outside] - variance
            worst = int(np.argmin(slack))
            # An excluded asset whose marginal variance is below the portfolio's
            # would lower the variance if it were held.
            if slack[worst] < -1e-9 * variance:
                active[outside[worst]] = True
                continue
        return weights
    raise ArithmeticError("Minimum-variance active-set search did not converge")


def min_variance_portfolio(
    df, history, initial_amount=1_000_000, transaction_cost_pct=0.0,
    lookback=MIN_VARIANCE_LOOKBACK, rebalance_every=63,
):
    """Long-only minimum-variance portfolio, re-estimated every ``rebalance_every`` sessions.

    ``history`` holds the sessions before ``df`` (same columns, same tickers);
    the weights signalled at a close use the ``lookback`` simple returns
    ending at that close, so need ``lookback`` earlier closes before the first
    signal on ``df``'s first date. Like the other baselines the first signal is
    on date zero, fills at date one's open, and fees are paid on the traded
    notional. Nothing after the signal date enters the weights.
    """
    if lookback < 2 or rebalance_every < 1:
        raise ValueError("lookback must be at least 2 and rebalance_every at least 1")
    df = df.copy()
    df["date"] = df["date"].astype(str)
    history = history[history["tic"].isin(df["tic"].unique())].copy()
    history["date"] = history["date"].astype(str)
    if not history.empty and history["date"].max() >= df["date"].min():
        raise ValueError("history must end before the first date of df")
    n_prior = history["date"].nunique()
    if n_prior < lookback:
        raise ValueError(
            "Need {} sessions of history before the first signal, got {}".format(lookback, n_prior)
        )
    open_prices, close = _price_matrices(pd.concat([history, df], ignore_index=True))
    opens = open_prices.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)
    first = n_prior
    n_days = len(closes) - first
    shares = np.zeros(closes.shape[1])
    cash = float(initial_amount)
    values = np.empty(n_days)
    values[0] = initial_amount
    for t in range(1, n_days):
        if (t - 1) % rebalance_every == 0:
            signal = first + t - 1
            window = closes[signal - lookback : signal + 1]
            target = min_variance_weights(window[1:] / window[:-1] - 1)
            positions, cash, _ = rebalance_positions(
                cash, shares * opens[first + t], target, transaction_cost_pct
            )
            shares = positions / opens[first + t]
        values[t] = cash + np.dot(shares, closes[first + t])
    return pd.DataFrame({"date": close.index[first:], "account_value": values})


def baseline_account_values(
    df, initial_amount=1_000_000, transaction_cost_pct=0.0, history=None
):
    """Account values of every built-in baseline, keyed by a display name.

    With ``history`` (the sessions before ``df``) the long-only minimum-variance
    portfolio is included, unless the history is shorter than its 252-session
    look-back, which is logged.
    """
    baselines = {
        "Equal-weight buy & hold": buy_and_hold(df, initial_amount, transaction_cost_pct),
        "Equal-weight monthly rebalance": equal_weight_rebalanced(
            df, initial_amount, transaction_cost_pct, rebalance_every=21
        ),
    }
    if history is not None:
        n_prior = history.loc[history["tic"].isin(df["tic"].unique()), "date"].nunique()
        if n_prior < MIN_VARIANCE_LOOKBACK:
            logger.warning(
                "Skipping the minimum-variance baseline: %d sessions of history, need %d",
                n_prior, MIN_VARIANCE_LOOKBACK,
            )
        else:
            baselines["Min-variance"] = min_variance_portfolio(
                df, history, initial_amount, transaction_cost_pct
            )
    return baselines
