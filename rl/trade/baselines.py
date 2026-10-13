"""Rule-based portfolios to benchmark trained agents against.

Each function takes a long (date, tic) frame with finite positive ``open``
and ``close`` prices. Signals formed at a close fill at the next session's open;
the first date therefore records the initial capital entirely in cash. Each
function returns a frame with ``date`` and ``account_value`` columns, the same format
as :meth:`rl.env.env_stocktrading.StockTradingEnv.save_asset_memory`.
"""

import numpy as np
import pandas as pd

from rl.trade.execution import rebalance_positions


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


def baseline_account_values(df, initial_amount=1_000_000, transaction_cost_pct=0.0):
    """Account values of every built-in baseline, keyed by a display name."""
    return {
        "Equal-weight buy & hold": buy_and_hold(df, initial_amount, transaction_cost_pct),
        "Equal-weight monthly rebalance": equal_weight_rebalanced(
            df, initial_amount, transaction_cost_pct, rebalance_every=21
        ),
    }
