"""Behavior of the benchmark's rule-based baseline policies."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rl.bench.engine import simulate
from rl.bench.market import MarketData, MarketView
from rl.bench.policies import baselines
from rl.bench.policies.baselines import (
    Cash,
    EqualWeightBuyHold,
    EqualWeightRebalance,
    InverseVolatility,
    MinimumVariance,
    Momentum,
    TrendFollowing,
    min_variance_weights,
)
from rl.bench.submission import discover
from rl.trade.baselines import buy_and_hold

SUBMISSIONS = Path(__file__).resolve().parents[1] / "submissions"


def panel(closes, opens=None, start="2020-01-01"):
    closes = np.asarray(closes, dtype=float)
    opens = closes if opens is None else np.asarray(opens, dtype=float)
    n_dates, n_tickers = closes.shape
    return MarketData(
        dates=pd.bdate_range(start, periods=n_dates).to_numpy(dtype="datetime64[D]"),
        tickers=tuple("T{}".format(i) for i in range(n_tickers)),
        open=opens,
        high=np.fmax(opens, closes),
        low=np.fmin(opens, closes),
        close=closes,
        volume=np.where(np.isfinite(closes), 1.0, np.nan),
    )


def view_at(data, weights=None):
    return MarketView(data, np.zeros(len(data.tickers)) if weights is None else np.asarray(weights, float))


def geometric(rates, days):
    return 100 * np.exp(np.outer(np.arange(days), np.log1p(rates)))


@pytest.fixture
def walk():
    rng = np.random.default_rng(11)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, (300, 4)), axis=0))
    opens = closes * np.exp(rng.normal(0, 0.003, closes.shape))
    return panel(closes, opens)


def test_buy_and_hold_trades_once_and_matches_the_reference_equity(walk):
    fee = 0.003
    result = simulate(walk, EqualWeightBuyHold().act, start=1, transaction_cost=fee, initial=1000)
    assert result.turnover[0] == pytest.approx(1 / (1 + fee), rel=1e-3)  # the whole account, net of the fee
    assert (result.turnover[1:] == 0).all()
    reference = buy_and_hold(walk.to_frame(), 1000, fee)
    np.testing.assert_allclose(result.equity, reference.account_value.to_numpy(), rtol=1e-9)


@pytest.mark.parametrize(
    "frequency, period",
    [("daily", "D"), ("weekly", "W"), ("monthly", "M"), ("quarterly", "Q")],
)
def test_rebalances_exactly_at_period_boundaries(walk, frequency, period):
    policy = EqualWeightRebalance(frequency)
    start = 5
    result = simulate(walk, policy.act, start=start, transaction_cost=0.0)
    labels = pd.DatetimeIndex(walk.dates).to_period(period)
    # A decision on the first session of a period fills on the next session.
    decisions = [t for t in range(start, len(walk) - 1) if labels[t] != labels[t - 1]]
    expected = {start} | {t + 1 for t in decisions}
    traded = {start + i for i in np.flatnonzero(result.turnover > 1e-9)}
    assert traded == expected
    assert len(expected) > 1


def all_baselines():
    return [
        Cash(),
        EqualWeightBuyHold(),
        EqualWeightRebalance("weekly"),
        InverseVolatility(lookback=20, frequency="weekly"),
        MinimumVariance(lookback=20, frequency="weekly"),
        Momentum(lookback=30, skip=5, top=2, frequency="weekly"),
        TrendFollowing(lookback=20, frequency="weekly"),
    ]


@pytest.mark.parametrize("policy", all_baselines(), ids=lambda p: type(p).__name__)
def test_unlisted_assets_never_get_weight_and_policies_keep_running(policy):
    rng = np.random.default_rng(5)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, (150, 4)), axis=0))
    closes[:40, 3] = np.nan  # lists late
    closes[90:, 1] = np.nan  # delists
    data = panel(closes)
    decisions = []

    def act(view):
        weights = policy.act(view)
        decisions.append((view.listed.copy(), weights))
        return weights

    policy.reset()
    result = simulate(data, act, start=20, transaction_cost=0.001)
    assert len(decisions) == len(data) - 20 and np.isfinite(result.equity).all()
    for listed, weights in decisions:
        if weights is not None:
            assert weights.shape == (4,) and weights.min() >= 0 and weights.sum() <= 1 + 1e-9
            assert (weights[~listed] == 0).all()
    assert (result.weights[:, 1][70:] == 0).all()  # sold out of the delisted asset


def test_cash_stays_in_cash(walk):
    result = simulate(walk, Cash().act, start=1, transaction_cost=0.01)
    np.testing.assert_array_equal(result.equity, np.ones(len(walk)))


def test_min_variance_prefers_the_calmest_asset():
    rng = np.random.default_rng(2)
    vols = np.array([0.02, 0.005, 0.01])
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 1, (400, 3)) * vols, axis=0))
    weights = MinimumVariance(lookback=252).act(view_at(panel(closes)))
    assert weights.min() >= 0 and weights.sum() == pytest.approx(1.0)
    assert weights.argmax() == 1 and weights[1] > 0.6


def test_min_variance_respects_the_long_only_constraint():
    # A and B are 99% correlated with B slightly calmer: unconstrained, the
    # optimum shorts A; long-only must hold B alone (C is much riskier).
    covariance = np.array(
        [
            [0.0100, 0.0095, 0.0],
            [0.0095, 0.0092, 0.0],
            [0.0, 0.0, 0.0400],
        ]
    )
    unconstrained = np.linalg.solve(covariance, np.ones(3))
    assert (unconstrained / unconstrained.sum()).min() < -1
    weights = min_variance_weights(covariance)
    assert weights.min() >= 0 and weights.sum() == pytest.approx(1.0)
    assert weights[0] == pytest.approx(0, abs=1e-9)
    # No feasible portfolio does better.
    rng = np.random.default_rng(0)
    candidates = np.vstack([rng.dirichlet(np.ones(3) * a, 3000) for a in (0.2, 1, 5)])
    best = weights @ covariance @ weights
    assert (np.einsum("ij,jk,ik->i", candidates, covariance, candidates) >= best - 1e-12).all()


def test_min_variance_matches_a_brute_force_search_on_random_covariances():
    rng = np.random.default_rng(8)
    candidates = rng.dirichlet(np.ones(5) * 0.3, 20000)
    for _ in range(5):
        factors = rng.normal(size=(5, 5))
        covariance = factors @ factors.T / 5 + np.diag(rng.uniform(0.01, 0.2, 5))
        weights = min_variance_weights(covariance)
        assert weights.min() >= 0 and weights.sum() == pytest.approx(1.0)
        best = weights @ covariance @ weights
        assert (np.einsum("ij,jk,ik->i", candidates, covariance, candidates) >= best - 1e-9).all()


def test_inverse_volatility_weights_by_inverse_volatility():
    rng = np.random.default_rng(4)
    noise = rng.normal(0, 1, (80, 3))
    closes = 100 * np.exp(np.cumsum(noise * np.array([0.01, 0.02, 0.04]), axis=0))
    weights = InverseVolatility(lookback=63).act(view_at(panel(closes)))
    returns = closes[-63:] / closes[-64:-1] - 1
    expected = 1 / returns.std(axis=0, ddof=1)
    np.testing.assert_allclose(weights, expected / expected.sum())
    assert weights[0] > weights[1] > weights[2]


def test_momentum_holds_the_top_trending_assets_and_ignores_the_recent_skip():
    rates = np.array([0.003, 0.002, -0.001, 0.0, 0.0005])
    closes = geometric(rates, 120)
    closes[-5:, 3] = closes[-6, 3] * np.linspace(1.1, 1.5, 5)  # late pump inside the skip window
    policy = Momentum(lookback=100, skip=5, top=2)
    np.testing.assert_allclose(policy.act(view_at(panel(closes))), [0.5, 0.5, 0, 0, 0])
    # Without skipping, the pump (+50% over a flat asset) wins a slot.
    unskipped = Momentum(lookback=100, skip=0, top=2).act(view_at(panel(closes)))
    assert unskipped[3] > 0


def test_momentum_leaves_slots_in_cash_without_positive_momentum():
    closes = geometric(np.array([0.002, -0.001, -0.002, -0.003]), 120)
    weights = Momentum(lookback=100, skip=5, top=3).act(view_at(panel(closes)))
    np.testing.assert_allclose(weights, [1 / 3, 0, 0, 0])
    assert Momentum(lookback=100, skip=5, top=3).act(view_at(panel(closes[:, 1:]))).sum() == 0


def test_trend_following_holds_assets_above_their_moving_average():
    closes = geometric(np.array([0.002, -0.002, 0.001, -0.0005]), 80)
    weights = TrendFollowing(lookback=50).act(view_at(panel(closes)))
    np.testing.assert_allclose(weights, [0.25, 0, 0.25, 0])


def test_trend_following_counts_only_listed_assets_in_the_position_size():
    closes = geometric(np.array([0.002, -0.002, 0.001]), 80)
    closes[40:, 1] = np.nan
    weights = TrendFollowing(lookback=30).act(view_at(panel(closes)))
    np.testing.assert_allclose(weights, [0.5, 0, 0.5])


@pytest.mark.parametrize(
    "policy",
    [
        InverseVolatility(lookback=63),
        MinimumVariance(lookback=252),
        Momentum(lookback=252),
        TrendFollowing(lookback=200),
    ],
    ids=lambda p: type(p).__name__,
)
def test_short_history_falls_back_to_equal_weight_over_listed_assets(policy):
    closes = geometric(np.array([0.002, -0.002, 0.001]), 40)
    closes[:, 1] = np.nan
    weights = policy.act(view_at(panel(closes)))
    np.testing.assert_allclose(weights, [0.5, 0, 0.5])


def test_estimators_ignore_everything_outside_their_lookback():
    rng = np.random.default_rng(6)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (300, 3)), axis=0))
    altered = closes.copy()
    altered[:100] *= rng.uniform(0.5, 2, (100, 3))
    for policy in (
        InverseVolatility(lookback=60),
        MinimumVariance(lookback=60),
        TrendFollowing(lookback=60),
        Momentum(lookback=60, skip=5, top=2),
    ):
        np.testing.assert_array_equal(
            policy.act(view_at(panel(closes))), policy.act(view_at(panel(altered)))
        )


def test_rebalancing_policies_hold_between_periods_and_when_invested():
    closes = geometric(np.array([0.001, 0.002]), 60)
    data = panel(closes)  # business days from 2020-01-01: index 22 is Jan 31, index 23 is Feb 3
    policy = EqualWeightRebalance("monthly")
    invested = [0.5, 0.5]
    assert policy.act(view_at(data.rows(0, 10), invested)) is None
    assert policy.act(view_at(data.rows(0, 24), invested)) is not None
    assert policy.act(view_at(data.rows(0, 10), [0, 0])) is not None


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        EqualWeightRebalance("yearly")
    with pytest.raises(ValueError):
        MinimumVariance(shrinkage=1.5)
    with pytest.raises(ValueError):
        Momentum(lookback=10, skip=10)
    with pytest.raises(ValueError):
        TrendFollowing(lookback=1)


def test_period_detection_uses_calendar_weeks_months_and_quarters():
    dates = np.array(["2020-01-03", "2020-01-06", "2020-01-31", "2020-02-03", "2020-03-31", "2020-04-01"], "datetime64[D]")

    def flags(frequency):
        return [baselines.new_period(dates[: i + 1], frequency) for i in range(1, len(dates))]

    assert flags("weekly") == [True, True, True, True, False]  # Tue 03-31 and Wed 04-01 share a week
    assert flags("monthly") == [False, False, True, True, True]
    assert flags("quarterly") == [False, False, False, False, True]
    assert baselines.new_period(dates[:1], "monthly")


def test_every_baseline_submission_builds():
    found = discover(SUBMISSIONS)
    expected = {
        "cash": Cash,
        "equal-weight-buy-hold": EqualWeightBuyHold,
        "equal-weight-monthly": EqualWeightRebalance,
        "inverse-volatility": InverseVolatility,
        "minimum-variance": MinimumVariance,
        "momentum": Momentum,
        "trend-following": TrendFollowing,
    }
    assert set(expected) <= set(found)
    for name, policy_class in expected.items():
        submission = found[name]
        assert submission.kind == "baseline" and submission.trials == 1
        assert submission.policy_class() is policy_class
        assert not policy_class.stochastic
