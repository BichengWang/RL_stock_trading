import numpy as np
import pytest

from rl.bench.engine import simulate
from rl.bench.market import MarketData, MarketView
from rl.bench.policies.forecast import LSTMForecastPolicy

N_SESSIONS = 700
SPLIT = 450  # first validation session
TICKERS = ("AAA", "BBB", "CCC", "DDD")


def make_panel(seed=0, phi=0.6, sigma=0.01, n=N_SESSIONS, unlisted_tail=None):
    """Random walks whose daily return follows AR(1): yesterday's return predicts today's."""
    rng = np.random.default_rng(seed)
    returns = np.zeros((n, len(TICKERS)))
    shocks = rng.normal(0, sigma, returns.shape)
    for t in range(1, n):
        returns[t] = phi * returns[t - 1] + shocks[t]
    close = 100 * np.exp(np.cumsum(returns, axis=0))
    open_ = np.vstack([close[:1], close[:-1]])  # open = previous close
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    volume = np.full(close.shape, 1e6)
    if unlisted_tail is not None:  # last asset stops trading in the final sessions
        for array in (open_, high, low, close, volume):
            array[-unlisted_tail:, -1] = np.nan
    dates = np.datetime64("2020-01-01") + np.arange(n)
    return MarketData(dates, TICKERS, open_, high, low, close, volume)


def small_policy(**kwargs):
    params = dict(window=10, hidden_size=8, epochs=8, batch_size=128, learning_rate=5e-3, top=None)
    params.update(kwargs)
    return LSTMForecastPolicy(**params)


@pytest.fixture(scope="module")
def panel():
    return make_panel()


@pytest.fixture(scope="module")
def fitted(panel):
    policy = small_policy()
    policy.fit(panel.rows(0, SPLIT + 100), panel.dates[SPLIT], seed=0)
    return policy


def test_beats_cash_and_equal_weight_on_planted_pattern(panel, fitted):
    test_start = SPLIT + 100
    result = simulate(panel, fitted.act, start=test_start, transaction_cost=0.001)
    buy_hold = panel.close[-1] / panel.close[test_start - 1]
    assert result.equity[-1] > 1.2  # cash is flat at 1.0
    assert result.equity[-1] > buy_hold.mean() + 0.1


def test_save_load_round_trip_gives_identical_decisions(panel, fitted, tmp_path):
    fitted.save(tmp_path / "frozen")
    restored = small_policy()
    restored.load(tmp_path / "frozen")
    a = simulate(panel, fitted.act, start=550, transaction_cost=0.001)
    b = simulate(panel, restored.act, start=550, transaction_cost=0.001)
    np.testing.assert_array_equal(a.weights, b.weights)
    assert a.weights.sum() > 0  # the comparison is not between two all-cash runs


def test_load_rejects_mismatched_architecture(fitted, tmp_path):
    fitted.save(tmp_path)
    with pytest.raises(ValueError, match="hidden_size"):
        small_policy(hidden_size=16).load(tmp_path)


def test_fit_is_deterministic_per_seed(panel):
    a, b, c = small_policy(epochs=2), small_policy(epochs=2), small_policy(epochs=2)
    history = panel.rows(0, SPLIT + 50)
    a.fit(history, panel.dates[SPLIT], seed=3)
    b.fit(history, panel.dates[SPLIT], seed=3)
    c.fit(history, panel.dates[SPLIT], seed=4)
    probe = panel.rows(0, 300)
    np.testing.assert_array_equal(a.forecast(probe), b.forecast(probe))
    assert not np.array_equal(a.forecast(probe), c.forecast(probe))


def test_never_allocates_to_unlisted_assets():
    panel = make_panel(unlisted_tail=60)
    policy = small_policy(epochs=3)
    policy.fit(panel.rows(0, 600), panel.dates[SPLIT], seed=0)
    for end in (580, N_SESSIONS):
        view = MarketView(panel.rows(0, end), np.zeros(len(TICKERS)))
        weights = policy.act(view)
        assert weights.shape == (len(TICKERS),)
        assert (weights[~view.listed] == 0).all()
        assert weights.sum() <= 1 + 1e-12
    # The last asset has been unlisted for 60 sessions: no forecast, no weight.
    assert np.isnan(policy.forecast(panel)[-1])
    assert policy.act(view)[-1] == 0


def test_top_limits_the_number_of_holdings(panel, fitted):
    view = MarketView(panel.rows(0, 600), np.zeros(len(TICKERS)))
    top_one = small_policy(top=1)
    top_one.model, top_one.mean, top_one.std = fitted.model, fitted.mean, fitted.std
    expected = fitted.forecast(view.history)
    weights = top_one.act(view)
    if (expected > 0.001).any():
        assert (weights > 0).sum() == 1
        assert weights[np.argmax(expected)] == 1
    else:
        assert weights.sum() == 0


def test_decisions_do_not_depend_on_data_after_the_decision(panel, fitted):
    cut = 560
    short = simulate(panel.rows(0, cut), fitted.act, start=520, transaction_cost=0.001)
    long = simulate(panel, fitted.act, start=520, transaction_cost=0.001)
    n = len(short.weights)
    np.testing.assert_array_equal(short.weights, long.weights[:n])
    assert short.weights.sum() > 0


def test_training_labels_stay_before_validation_start(panel):
    """Replacing every session from validation_start on changes only selection, not the scaler."""
    history = panel.rows(0, SPLIT + 100)
    scrambled = make_panel(seed=99).rows(0, SPLIT + 100)
    mixed = MarketData(
        history.dates,
        history.tickers,
        *[np.vstack([getattr(history, f)[:SPLIT], getattr(scrambled, f)[SPLIT:]]) for f in
          ("open", "high", "low", "close", "volume")],
    )
    a, b = small_policy(epochs=1), small_policy(epochs=1)
    a.fit(history, panel.dates[SPLIT], seed=0)
    b.fit(mixed, panel.dates[SPLIT], seed=0)
    # Standardisation uses training sessions only.
    np.testing.assert_allclose(a.mean, b.mean)
    np.testing.assert_allclose(a.std, b.std)


def test_too_little_history_is_an_error(panel):
    with pytest.raises(ValueError, match="Not enough history"):
        small_policy().fit(panel.rows(0, 40), panel.dates[30], seed=0)


def test_act_before_fit_is_an_error(panel):
    view = MarketView(panel.rows(0, 100), np.zeros(len(TICKERS)))
    with pytest.raises(RuntimeError, match="fit or load"):
        small_policy().act(view)
