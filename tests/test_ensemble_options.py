"""Per-agent ensemble settings, the turbulence switch and standalone-agent accounts."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from conftest import TECH, TICKERS, make_panel
from rl import cli
from rl.autotrain import r1
from rl.autotrain.training import TrainConfig, run_training
from rl.model.models import DRLEnsembleAgent, build_model, make_vec_env
from rl.env.env_stocktrading import StockTradingEnv
from rl.preprocessing.data import data_split

TINY_KWARGS = {
    "a2c": {"n_steps": 5, "learning_rate": 0.0007},
    "ppo": {"n_steps": 16, "batch_size": 16},
    "ddpg": {
        "batch_size": 16, "buffer_size": 1000, "learning_starts": 10,
        "action_noise": {"type": "ornstein_uhlenbeck", "sigma": 0.5},
    },
}
STEPS = {"a2c": 20, "ppo": 48, "ddpg": 30}


def ensemble_config(tmp_path, **overrides):
    settings = dict(
        agent="ensemble",
        total_timesteps=16,
        data_source="synthetic",
        ticker_list=TICKERS,
        start_date="2016-01-01",
        start_trade_date="2017-06-01",
        end_date="2017-07-12",
        tech_indicator_list=TECH,
        seed=0,
        run_name="ensemble_run",
        rebalance_window=15,
        validation_window=15,
        results_dir=str(tmp_path / "results"),
        trained_model_dir=str(tmp_path / "models"),
    )
    settings.update(overrides)
    return TrainConfig(**settings)


@pytest.fixture(scope="module")
def ensemble_run(tmp_path_factory):
    """One small ensemble run with per-agent settings, no turbulence and standalone accounts."""
    tmp_path = tmp_path_factory.mktemp("ensemble")
    calls = []
    original = DRLEnsembleAgent.train_model

    def spy(model, model_name, tb_log_name, iter_num, total_timesteps=5000, save=True):
        model = original(model, model_name, tb_log_name, iter_num, total_timesteps, save)
        calls.append({
            "name": model_name, "requested": total_timesteps, "trained": model.num_timesteps,
            "model": model,
        })
        return model

    DRLEnsembleAgent.train_model = staticmethod(spy)
    try:
        cfg = ensemble_config(
            tmp_path,
            use_turbulence=False,
            standalone_accounts=True,
            ensemble_timesteps=STEPS,
            ensemble_model_kwargs=TINY_KWARGS,
        )
        result = run_training(cfg)
    finally:
        DRLEnsembleAgent.train_model = staticmethod(original)
    return result, calls


def test_each_algorithm_trains_for_its_own_number_of_steps(ensemble_run):
    _, calls = ensemble_run
    assert {c["name"] for c in calls} == {"a2c", "ppo", "ddpg"}
    for call in calls:
        assert call["requested"] == STEPS[call["name"]]
        # SB3 stops at the first whole rollout / step count at or after the request.
        assert call["trained"] >= call["requested"]
    a2c = [c["trained"] for c in calls if c["name"] == "a2c"]
    ppo = [c["trained"] for c in calls if c["name"] == "ppo"]
    assert max(a2c) <= STEPS["a2c"] + 4 and min(ppo) == 48  # 3 rollouts of 16


def test_unlisted_algorithms_use_total_timesteps(tmp_path, monkeypatch):
    seen = {}

    def spy(model, model_name, tb_log_name, iter_num, total_timesteps=5000, save=True):
        seen.setdefault(model_name, set()).add(total_timesteps)
        return model

    monkeypatch.setattr(DRLEnsembleAgent, "train_model", staticmethod(spy))
    # An untrained model trades fine; only the requested steps matter here.
    run_training(ensemble_config(
        tmp_path, total_timesteps=17, ensemble_timesteps={"ppo": 64},
        ensemble_model_kwargs=TINY_KWARGS, end_date="2017-06-20",
    ))
    assert seen == {"a2c": {17}, "ppo": {64}, "ddpg": {17}}


def test_per_agent_model_kwargs_reach_the_models(ensemble_run):
    _, calls = ensemble_run
    by_name = {c["name"]: c["model"] for c in calls}
    assert by_name["ppo"].n_steps == 16 and by_name["ppo"].batch_size == 16
    sigma = by_name["ddpg"].action_noise._sigma
    np.testing.assert_allclose(sigma, 0.5)


def test_invalid_ensemble_settings_are_rejected():
    for field, value in (
        ("ensemble_timesteps", {"sac": 10}),
        ("ensemble_timesteps", {"a2c": 0}),
        ("ensemble_timesteps", {"a2c": 1.5}),
        ("ensemble_model_kwargs", {"td3": {}}),
    ):
        with pytest.raises(ValueError):
            TrainConfig(**{field: value}).validate()


def test_ddpg_noise_sigma_is_configurable(processed):
    data = data_split(processed, "2017-01-01", "2017-03-01")
    n = len(TICKERS)
    env = StockTradingEnv(data, n, 10, 1e5, 0, 0, 1e-4, 1 + 2 * n + len(TECH) * n, n, TECH)
    venv = make_vec_env(env)
    default = build_model("ddpg", venv, model_kwargs={"action_noise": "ornstein_uhlenbeck"})
    np.testing.assert_allclose(default.action_noise._sigma, 0.1)
    custom = build_model(
        "ddpg", venv,
        model_kwargs={"action_noise": {"type": "normal", "sigma": 0.3}},
    )
    np.testing.assert_allclose(custom.action_noise._sigma, 0.3)
    assert custom.action_noise._mu.shape == (n,)
    with pytest.raises(ValueError, match="type"):
        build_model("ddpg", venv, model_kwargs={"action_noise": {"sigma": 0.3}})


def test_the_turbulence_switch_is_recorded_and_the_run_completes_without_the_column(ensemble_run):
    result, _ = ensemble_run
    with open(os.path.join(result["results_dir"], "run_config.json"), encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["use_turbulence"] is False
    assert saved["resolved_ensemble_timesteps"] == STEPS
    assert len(result["account_value"]) > 2


class BuyOneThenHold:
    def get_vec_normalize_env(self):
        return None

    def predict(self, obs, deterministic=True):
        # One ticker: observation is cash, close, holdings.
        return np.array([1.0 if obs[2] == 0 else 0.0]), None


def turbulence_ensemble(tmp_path, monkeypatch, use_turbulence, with_column=True):
    panel = make_panel([[100.0]] * 13)
    dates = sorted(panel.date.unique())
    if with_column:
        # Fitting history peaks at 1.0, so the threshold is 1.0; the spike comes on day 9.
        turbulence = [0.5, 1.0, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 10.0, 10.0, 10.0, 10.0]
        panel["turbulence"] = turbulence
    ensemble = DRLEnsembleAgent(
        df=panel, train_period=(dates[0], dates[6]),
        val_test_period=(dates[6], "2021-01-01"), rebalance_window=4, validation_window=2,
        stock_dim=1, hmax=1, initial_amount=1000.0, buy_cost_pct=0.0, sell_cost_pct=0.0,
        reward_scaling=1.0, state_space=3, action_space=1, tech_indicator_list=[],
        print_verbosity=0, save_models=False, results_dir=str(tmp_path), verbose=0,
        use_turbulence=use_turbulence,
    )
    monkeypatch.setattr(ensemble, "_train", lambda *args, **kwargs: BuyOneThenHold())
    ensemble.run_ensemble_strategy({}, {}, {}, {"ppo": 1, "a2c": 1, "ddpg": 1})
    return ensemble


def trades(ensemble):
    return ensemble.executions.drop(columns=["signal_date", "fill_date"]).to_numpy()


def test_turbulence_rule_liquidates_when_on(tmp_path, monkeypatch):
    ensemble = turbulence_ensemble(tmp_path, monkeypatch, use_turbulence=True)
    assert (trades(ensemble) < 0).any()


def test_ensemble_without_turbulence_never_liquidates(tmp_path, monkeypatch):
    ensemble = turbulence_ensemble(tmp_path, monkeypatch, use_turbulence=False)
    executed = trades(ensemble)
    assert (executed >= 0).all() and executed.sum() == 1.0


def test_turbulence_column_is_needed_only_when_the_rule_is_on(tmp_path, monkeypatch):
    ensemble = turbulence_ensemble(tmp_path, monkeypatch, False, with_column=False)
    assert ensemble.account_value is not None
    with pytest.raises(ValueError, match="turbulence"):
        turbulence_ensemble(tmp_path, monkeypatch, True, with_column=False)


def tiny_ensemble(processed, tmp_path, names, standalone):
    n = len(TICKERS)

    class Restricted(DRLEnsembleAgent):
        MODEL_NAMES = tuple(names)

    return Restricted(
        df=processed, train_period=("2016-01-01", "2017-06-01"),
        val_test_period=("2017-06-01", "2017-07-12"), rebalance_window=15, validation_window=15,
        stock_dim=n, hmax=100, initial_amount=1e6, buy_cost_pct=0.001, sell_cost_pct=0.001,
        reward_scaling=1e-4, state_space=1 + 2 * n + len(TECH) * n, action_space=n,
        tech_indicator_list=TECH, print_verbosity=10, normalize_obs=True, seed=3,
        save_models=False, results_dir=str(tmp_path), verbose=0, standalone_accounts=standalone,
    )


def test_standalone_accounts_equal_running_each_agent_alone(processed, tmp_path):
    ensemble = tiny_ensemble(processed, tmp_path / "all", ("ppo", "a2c", "ddpg"), True)
    ensemble.run_ensemble_strategy(
        TINY_KWARGS["a2c"], TINY_KWARGS["ppo"], TINY_KWARGS["ddpg"], STEPS
    )
    accounts = ensemble.agent_account_values
    assert set(accounts) == {"a2c", "ppo", "ddpg"}
    for name, account in accounts.items():
        assert account["date"].tolist() == ensemble.account_value["date"].tolist()
        # Alone, the single algorithm is trivially the winner of every window.
        alone = tiny_ensemble(processed, tmp_path / name, (name,), False)
        alone.run_ensemble_strategy(
            TINY_KWARGS["a2c"], TINY_KWARGS["ppo"], TINY_KWARGS["ddpg"], STEPS
        )
        pd.testing.assert_frame_equal(account, alone.account_value)
    final = [a["account_value"].iloc[-1] for a in accounts.values()]
    assert len({round(v, 6) for v in final}) > 1
    # The ensemble is one of the three in each window, so its account starts equal to them.
    assert ensemble.account_value["account_value"].iloc[0] == 1e6


def test_standalone_accounts_are_off_by_default(processed, tmp_path):
    ensemble = tiny_ensemble(processed, tmp_path, ("ppo",), False)
    ensemble.run_ensemble_strategy(
        TINY_KWARGS["a2c"], TINY_KWARGS["ppo"], TINY_KWARGS["ddpg"], STEPS
    )
    assert ensemble.agent_account_values is None


def test_run_reports_agents_and_baselines_side_by_side(ensemble_run):
    result, _ = ensemble_run
    stats = result["stats"]
    assert {"ENSEMBLE", "A2C", "PPO", "DDPG", "Min-variance"} <= set(stats.columns)
    accounts = pd.read_csv(os.path.join(result["results_dir"], "agent_accounts.csv"))
    assert list(accounts.columns) == ["date", "ENSEMBLE", "A2C", "PPO", "DDPG"]
    assert accounts["date"].tolist() == result["account_value"]["date"].tolist()
    np.testing.assert_allclose(
        accounts["ENSEMBLE"], result["account_value"]["account_value"]
    )


def test_cli_ensemble_options():
    parser = cli.build_parser()
    options = parser.parse_args([
        "--mode", "ensemble", "--ensemble-timesteps", "a2c=30000", "ppo=100000", "ddpg=10000",
        "--standalone-agents", "--no-turbulence",
    ])
    cfg = cli._train_config(options)
    assert cfg.ensemble_timesteps == {"a2c": 30000, "ppo": 100000, "ddpg": 10000}
    assert cfg.standalone_accounts and not cfg.use_turbulence
    default = cli._train_config(parser.parse_args(["--mode", "ensemble"]))
    assert default.ensemble_timesteps is None and not default.standalone_accounts
    for bad in ("sac=10", "a2c=0", "a2c", "a2c=x"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--ensemble-timesteps", bad])


def test_r1_driver_sets_the_authors_recipe():
    cfg = r1.r1_config("prices.csv", seed=4, use_turbulence=False, ddpg_noise_sigma=0.25)
    assert cfg.agent == "ensemble" and cfg.data_source == "csv" and cfg.data_file == "prices.csv"
    assert cfg.tech_indicator_list == ["macd", "rsi_30", "cci_30", "dx_30"]
    assert cfg.ensemble_timesteps == {"a2c": 30_000, "ppo": 100_000, "ddpg": 10_000}
    assert cfg.ensemble_model_kwargs["ppo"] == {"n_steps": 128, "batch_size": 16, "ent_coef": 0.005}
    assert cfg.ensemble_model_kwargs["ddpg"]["action_noise"] == {
        "type": "ornstein_uhlenbeck", "sigma": 0.25,
    }
    assert cfg.seed == 4 and not cfg.use_turbulence and cfg.standalone_accounts
    assert (cfg.buy_cost_pct, cfg.sell_cost_pct, cfg.initial_amount, cfg.hmax) == (
        0.001, 0.001, 1_000_000, 100,
    )
    assert (cfg.rebalance_window, cfg.validation_window) == (63, 63)
    assert (cfg.start_date, cfg.start_trade_date, cfg.end_date) == (
        "2009-01-02", "2016-01-04", "2026-01-01",
    )


def test_r1_driver_runs_from_a_csv_file(raw_prices, tmp_path):
    data_file = tmp_path / "prices.csv"
    raw_prices.to_csv(data_file, index=False)
    result = r1.run_r1(
        str(data_file), seed=0, start_date="2016-01-01", start_trade_date="2017-06-01",
        end_date="2017-06-20", results_dir=str(tmp_path / "results"), run_name="r1_test",
        ensemble_timesteps={"a2c": 10, "ppo": 16, "ddpg": 10},
        tech_indicator_list=TECH, use_turbulence=False, rebalance_window=10,
        validation_window=10,
    )
    assert set(result["agent_account_values"]) == {"a2c", "ppo", "ddpg"}
