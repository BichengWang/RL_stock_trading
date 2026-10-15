"""The causal long-only minimum-variance baseline."""

import numpy as np
import pandas as pd
import pytest

from rl.trade.baselines import (
    baseline_account_values,
    min_variance_portfolio,
    min_variance_weights,
)

TICKERS = ["AAA", "BBB", "CCC", "DDD"]


def returns_with_covariance(cov, sessions=400, seed=0):
    """Returns whose sample covariance (ddof=1) is exactly ``cov`` and mean zero."""
    n = len(cov)
    z = np.random.default_rng(seed).standard_normal((sessions, n))
    z -= z.mean(axis=0)
    q, _ = np.linalg.qr(z)
    return q * np.sqrt(sessions - 1) @ np.linalg.cholesky(cov).T


def test_uncorrelated_assets_get_inverse_variance_weights():
    variances = np.array([1.0, 4.0, 0.25, 2.0]) * 1e-4
    weights = min_variance_weights(returns_with_covariance(np.diag(variances)))
    expected = (1 / variances) / (1 / variances).sum()
    np.testing.assert_allclose(weights, expected, rtol=1e-6)


def test_assets_the_unconstrained_solution_would_short_get_zero():
    sigma1, sigma2, rho = 2.0, 1.0, 0.8
    cov = np.array([[sigma1**2, rho * sigma1 * sigma2], [rho * sigma1 * sigma2, sigma2**2]]) * 1e-4
    # Unconstrained: w1 = (s2^2 - rho s1 s2) / (s1^2 + s2^2 - 2 rho s1 s2) < 0.
    assert (sigma2**2 - rho * sigma1 * sigma2) < 0
    weights = min_variance_weights(returns_with_covariance(cov))
    np.testing.assert_allclose(weights, [0.0, 1.0], atol=1e-9)


@pytest.mark.parametrize("sessions, assets", [(60, 12), (252, 30), (5, 8)])
def test_weights_are_valid_and_optimal(sessions, assets):
    rng = np.random.default_rng(sessions + assets)
    factor = rng.standard_normal((sessions, 2))
    returns = 0.01 * (factor @ rng.standard_normal((2, assets)) + rng.standard_normal((sessions, assets)))
    weights = min_variance_weights(returns)
    assert weights.shape == (assets,)
    assert (weights >= 0).all() and weights.sum() == pytest.approx(1.0)

    cov = np.cov(returns, rowvar=False)
    variance = weights @ cov @ weights
    candidates = rng.dirichlet(np.ones(assets) * 0.3, size=20_000)
    candidate_variance = np.einsum("ij,jk,ik->i", candidates, cov, candidates)
    assert variance <= candidate_variance.min() + 1e-12
    assert variance <= np.full(assets, 1 / assets) @ cov @ np.full(assets, 1 / assets) + 1e-12
    # KKT: marginal variance is equal on held assets and no lower on excluded ones.
    gradient = cov @ weights
    held = weights > 1e-12
    np.testing.assert_allclose(gradient[held], variance, rtol=1e-6)
    assert (gradient[~held] >= variance * (1 - 1e-6)).all()


def test_weights_reject_bad_input():
    with pytest.raises(ValueError):
        min_variance_weights(np.zeros((1, 3)))
    with pytest.raises(ValueError):
        min_variance_weights(np.array([[0.1, np.nan], [0.0, 0.1], [0.2, 0.1]]))
    np.testing.assert_array_equal(min_variance_weights(np.array([[0.1], [0.2], [0.0]])), [1.0])


def random_walk_panel(dates, seed, tickers=TICKERS):
    rng = np.random.default_rng(seed)
    rows = []
    scale = np.linspace(0.005, 0.02, len(tickers))
    closes = 50 * np.exp(np.cumsum(rng.standard_normal((len(dates), len(tickers))) * scale, axis=0))
    opens = closes * (1 + 0.003 * rng.standard_normal(closes.shape))
    for d, date in enumerate(dates):
        for i, tic in enumerate(tickers):
            rows.append({"date": date, "tic": tic, "open": opens[d, i], "close": closes[d, i]})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel():
    dates = pd.bdate_range("2018-01-01", periods=400).strftime("%Y-%m-%d").tolist()
    full = random_walk_panel(dates, seed=1)
    return dates, full


def split(full, dates, n_history=300):
    cut = dates[n_history]
    return full[full.date < cut], full[full.date >= cut]


def test_first_account_value_is_cash_then_the_weights_earn_the_next_session(panel):
    dates, full = panel
    history, df = split(full, dates)
    account = min_variance_portfolio(df, history, initial_amount=1000.0)
    assert account.date.tolist() == dates[300:]
    assert account.account_value.iloc[0] == 1000.0

    # Zero fees: the signal on the first date is the min-variance weights of the
    # 252 returns ending that day; they fill at the second date's open.
    closes = full.pivot(index="date", columns="tic", values="close")
    opens = full.pivot(index="date", columns="tic", values="open")
    window = closes.iloc[300 - 252 : 301].to_numpy()
    weights = min_variance_weights(window[1:] / window[:-1] - 1)
    expected = 1000.0 * (weights * closes.iloc[301].to_numpy() / opens.iloc[301].to_numpy()).sum()
    assert account.account_value.iloc[1] == pytest.approx(expected)


def test_account_does_not_depend_on_later_prices(panel):
    dates, full = panel
    history, df = split(full, dates)
    baseline = min_variance_portfolio(df, history, transaction_cost_pct=0.001)
    cut = 330  # a session index inside df
    changed = full.copy()
    later = changed.date >= dates[cut]
    changed.loc[later, ["open", "close"]] *= np.random.default_rng(9).uniform(
        0.5, 1.5, later.sum()
    )[:, None]
    history2, df2 = split(changed, dates)
    shifted = min_variance_portfolio(df2, history2, transaction_cost_pct=0.001)
    before = baseline.date < dates[cut]
    np.testing.assert_array_equal(
        baseline.account_value[before].to_numpy(), shifted.account_value[before].to_numpy()
    )
    assert not np.allclose(baseline.account_value[~before], shifted.account_value[~before])


def test_weights_are_re_estimated_only_every_63_sessions(panel):
    dates, full = panel
    history, df = split(full, dates)
    # With every ticker repriced so the held shares are worthless after the first
    # rebalance, the account only recovers at the next (63rd) rebalance.
    base = min_variance_portfolio(df, history, rebalance_every=63)
    daily = min_variance_portfolio(df, history, rebalance_every=1)
    assert not np.allclose(base.account_value, daily.account_value)
    assert len(base) == len(daily) == len(df.date.unique())


def test_fees_reduce_the_account(panel):
    dates, full = panel
    history, df = split(full, dates)
    free = min_variance_portfolio(df, history)
    paid = min_variance_portfolio(df, history, transaction_cost_pct=0.01)
    assert paid.account_value.iloc[-1] < free.account_value.iloc[-1]


def test_short_history_is_an_error_and_a_skipped_baseline(panel):
    dates, full = panel
    history, df = split(full, dates, n_history=100)
    with pytest.raises(ValueError, match="history"):
        min_variance_portfolio(df, history)
    assert "Min-variance" not in baseline_account_values(df, history=history)
    with pytest.raises(ValueError, match="before the first date"):
        min_variance_portfolio(df, full)


def test_baseline_set_includes_min_variance_only_with_history(panel):
    dates, full = panel
    history, df = split(full, dates)
    assert "Min-variance" not in baseline_account_values(df)
    with_history = baseline_account_values(df, history=history)
    assert list(with_history) == [
        "Equal-weight buy & hold", "Equal-weight monthly rebalance", "Min-variance",
    ]
