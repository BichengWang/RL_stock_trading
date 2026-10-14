import numpy as np
import pandas as pd
import pytest

from rl.bench import features as F
from rl.bench.engine import simulate
from rl.bench.gym_env import PortfolioEnv, actions_to_weights
from rl.bench.market import MarketData, MarketView
from rl.bench.policies import sb3
from rl.bench.policies.sb3 import SB3Policy
from rl.marketdata.synthetic import generate_synthetic_data

COST = 0.002


def make_data(n=320, seed=3):
    """Four assets: always listed, one with a halt, a late listing and a delisting."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n).values.astype("datetime64[D]")
    close = 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, (n, 4)), axis=0))
    opens = close * np.exp(rng.normal(0, 0.003, (n, 4)))
    volume = rng.integers(1_000, 100_000, (n, 4)).astype(float)
    gone = np.zeros((n, 4), dtype=bool)
    gone[150:155, 0] = True
    gone[:100, 2] = True
    gone[250:, 3] = True
    for array in (close, opens, volume):
        array[gone] = np.nan
    high = np.maximum(close, opens) * 1.01
    low = np.minimum(close, opens) * 0.99
    return MarketData(dates, ("A", "B", "C", "D"), opens, high, low, close, volume)


def test_features_are_identical_on_a_window_and_on_the_full_panel():
    data = make_data()
    panel = F.feature_panel(data)
    assert panel.shape == (len(data), 4, F.N_FEATURES)
    assert np.isfinite(panel).all()
    for t in range(len(data)):
        history = data.rows(0, t + 1)
        assert np.array_equal(F.latest_features(history), panel[t])
        assert np.array_equal(F.latest_features(history.window(F.LOOKBACK)), panel[t])


def test_features_are_causal_and_zero_for_unlisted_assets():
    data = make_data()
    panel = F.feature_panel(data)
    cut = 230
    altered = {name: getattr(data, name).copy() for name in ("open", "high", "low", "close", "volume")}
    for array in altered.values():
        array[cut:] *= 3.0
    perturbed = MarketData(data.dates, data.tickers, **altered)
    assert np.array_equal(F.feature_panel(perturbed)[:cut], panel[:cut])
    listed = np.isfinite(data.close)
    assert (panel[~listed] == 0).all()
    assert np.array_equal(panel[..., -1], listed.astype(float))
    # Too little history for the long windows: those features are zero, short ones are not.
    young = panel[20, 1]
    assert young[F.FEATURE_NAMES.index("sma_200")] == 0 and young[F.FEATURE_NAMES.index("ret_5")] != 0


def test_unlisted_assets_get_zero_weight_and_the_rest_is_a_softmax():
    listed = np.array([True, False, True])
    weights = actions_to_weights([0.0, 5.0, 1.0, 0.0], listed)
    assert weights[1] == 0
    expected = np.exp([0.0, 1.0, 0.0])
    np.testing.assert_allclose(weights[[0, 2]], (expected / expected.sum())[:2])
    assert 0 < weights.sum() < 1
    assert actions_to_weights(np.zeros(4), np.zeros(3, dtype=bool)).sum() == 0


def test_env_equity_matches_the_evaluation_engine_for_the_same_weights():
    data = make_data()
    first = 5
    rng = np.random.default_rng(0)
    logits = rng.uniform(-5, 5, (len(data), 5)).astype(np.float32)
    env = PortfolioEnv(data, COST, episode_length=None, first_session=first)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    equity, rewards, steps, truncated = [1.0], [], 0, False
    while not truncated:
        obs, reward, terminated, truncated, info = env.step(logits[steps])
        equity.append(info["equity"])
        rewards.append(reward)
        steps += 1
    assert steps == len(data) - first

    def act(view):
        return actions_to_weights(logits[len(view.history) - first], view.listed)

    backtest = simulate(data, act, first, COST)
    np.testing.assert_allclose(equity, backtest.equity, rtol=1e-12)
    np.testing.assert_allclose(np.exp(np.sum(rewards)), backtest.equity[-1], rtol=1e-9)
    assert backtest.costs.sum() > 0 and (backtest.turnover > 0).any()


def test_env_observation_shows_current_weights_and_cash_and_never_holds_unlisted_assets():
    data = make_data()
    env = PortfolioEnv(data, COST, episode_length=None, first_session=300)
    env.reset()
    obs, *_ = env.step(np.full(5, 3.0, dtype=np.float32))  # D was delisted at session 250
    held, cash = obs[-5:-1], obs[-1]
    assert held[3] == 0 and (held[:3] > 0).all()
    np.testing.assert_allclose(held.sum() + cash, 1.0)
    env = PortfolioEnv(data, COST, episode_length=None, first_session=50)
    env.reset()
    obs, *_ = env.step(np.array([0, 0, 5, 0, 0], dtype=np.float32))  # C lists at session 100
    assert obs[-5:-1][2] == 0


def test_episodes_start_at_seeded_random_sessions_and_have_fixed_length():
    data = make_data()
    env = PortfolioEnv(data, COST, episode_length=40, first_session=10)

    def run(seed):
        _, info = env.reset(seed=seed)
        n, truncated, last = 0, False, None
        while not truncated:
            _, _, terminated, truncated, last = env.step(env.action_space.sample())
            n += 1
        return info["date"], n, last["date"]

    first = run(7)
    assert run(7) == first
    assert first[1] == 40
    starts = {run(s)[0] for s in range(8)}
    assert len(starts) > 1


def synthetic_history():
    frame = generate_synthetic_data(["AAA", "BBB", "CCC"], "2015-01-01", "2017-07-01", seed=0)
    return MarketData.from_frame(frame)


def tiny(algo="ppo", **kwargs):
    extra = {"n_steps": 64, "batch_size": 32} if algo in ("ppo", "a2c") else {"learning_starts": 50}
    if algo == "a2c":
        extra = {"n_steps": 16}
    return SB3Policy(
        algo=algo, timesteps=300, episode_length=100, eval_every=100, algo_kwargs=extra,
        transaction_cost=COST, **kwargs,
    )


def sample_views(history, count=5):
    views = []
    for t in np.linspace(len(history) - 60, len(history) - 1, count).astype(int):
        views.append(MarketView(history.rows(0, t + 1), np.array([0.2, 0.1, 0.0])))
    return views


@pytest.mark.parametrize("algo", ["ppo", "sac", "td3"])
def test_fitted_policy_acts_with_valid_weights_and_restores_from_disk(tmp_path, algo):
    history = synthetic_history()
    validation_start = np.datetime64("2017-01-02")
    policy = tiny(algo)
    policy.fit(history, validation_start, seed=1)
    views = sample_views(history)
    weights = [policy.act(v) for v in views]
    assert all((w >= 0).all() and w.sum() <= 1 + 1e-9 for w in weights)
    policy.save(tmp_path)

    restored = tiny(algo)
    restored.load(tmp_path)
    for view, expected in zip(views, weights):
        assert np.array_equal(restored.act(view), expected)
    assert restored.best_step == policy.best_step and restored.best_sharpe == policy.best_sharpe
    assert sorted(p.name for p in tmp_path.iterdir()) == ["actor.npz", "meta.json"]


@pytest.mark.parametrize("algo", ["ppo", "a2c", "sac", "td3", "ddpg"])
def test_exported_actor_reproduces_the_trained_models_deterministic_actions(algo):
    policy = tiny(algo)
    policy.fit(synthetic_history(), np.datetime64("2017-01-02"), seed=2)
    size = policy.model.observation_space.shape[0]
    rng = np.random.default_rng(0)
    # Large observations push actions into the clip/squash bounds.
    for obs in [*rng.normal(size=(20, size)), *rng.normal(scale=50.0, size=(5, size))]:
        obs = obs.astype(np.float32)
        expected, _ = policy.model.predict(obs, deterministic=True)
        np.testing.assert_allclose(policy.actor(obs), expected, rtol=1e-5, atol=1e-5)


def test_fit_is_deterministic_for_a_seed_and_depends_on_it():
    history = synthetic_history()
    validation_start = np.datetime64("2017-01-02")
    views = sample_views(history, 3)

    def acts(seed):
        policy = tiny("ppo")
        policy.fit(history, validation_start, seed=seed)
        return np.array([policy.act(v) for v in views])

    assert np.array_equal(acts(5), acts(5))
    assert not np.array_equal(acts(5), acts(6))


def test_training_stops_before_validation_and_the_best_validation_checkpoint_is_kept(monkeypatch):
    history = synthetic_history()
    validation_start = np.datetime64("2017-01-02")
    split = history.index_of(validation_start)
    seen = []

    class Spy(PortfolioEnv):
        def __init__(self, data, *args, **kwargs):
            seen.append(len(data))
            super().__init__(data, *args, **kwargs)

    monkeypatch.setattr(sb3, "PortfolioEnv", Spy)
    snapshots, scores = [], iter([0.1, 0.9, 0.3, 0.2])  # PPO rounds 300 steps up to 5 rollouts of 64

    def scripted(self, model):
        snapshots.append({n: v.clone() for n, v in model.get_parameters()["policy"].items()})
        return next(scores)

    monkeypatch.setattr(SB3Policy, "_validation_sharpe", scripted)
    policy = tiny("ppo")
    policy.fit(history, validation_start, seed=0)
    assert seen == [split]
    assert (policy.best_step, policy.best_sharpe) == (200, 0.9)
    restored = policy.model.get_parameters()["policy"]
    for name, value in snapshots[1].items():
        assert np.array_equal(restored[name].numpy(), value.numpy())
    assert not all(np.array_equal(restored[n].numpy(), v.numpy()) for n, v in snapshots[2].items())
