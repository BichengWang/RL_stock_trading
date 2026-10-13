"""Effective CLI fees and portable, immutable frozen-policy stress evaluation."""

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sysconfig
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from rl import cli
from rl.autotrain.experiment import main, run_cost_sensitivity
from rl.autotrain.training import TrainConfig, env_kwargs, make_env
from rl.forecast.lstm import LSTMForecaster
from rl.model.models import build_model, make_vec_env
from rl.preprocessing.data import data_split


@pytest.fixture
def market_file(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=20).strftime("%Y-%m-%d").tolist()
    prices = pd.DataFrame({"date": dates, "tic": "AAA", "open": 10.0, "high": 10.0,
                           "low": 10.0, "close": 10.0, "volume": 1000.0})
    path = tmp_path / "prices.csv"
    prices.to_csv(path, index=False)
    return prices, dates, path


@pytest.fixture
def fixed_forecaster(tmp_path, market_file):
    _, dates, path = market_file
    cfg = TrainConfig(
        agent="lstm", data_source="csv", data_file=str(path), ticker_list=["AAA"],
        start_date=dates[0], start_trade_date=dates[15],
        end_date=(pd.Timestamp(dates[-1]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        tech_indicator_list=[], use_turbulence=False, initial_amount=50.0,
        buy_cost_pct=0.02, sell_cost_pct=0.02,
        lstm_window=3, lstm_hidden_size=1, lstm_num_layers=1, run_name="fixed_forecaster",
    )
    model_dir = tmp_path / "forecaster"
    model_dir.mkdir()
    model = LSTMForecaster(1, hidden_size=1, num_layers=1)
    # A genuine serialized network whose forecast is always above the close.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head.bias.fill_(0.1)
    torch.save(model.state_dict(), model_dir / "model.pt")
    saved = dataclasses.asdict(cfg)
    saved.update(tickers=["AAA"], eligibility_cutoff=dates[13], feature_scaler={},
                 execution_timing="next_open")
    (model_dir / "run_config.json").write_text(json.dumps(saved))
    return model_dir


@pytest.mark.parametrize("fee", [None, 0.0, 0.001, 0.1])
def test_backtest_cli_uses_saved_or_explicit_fee_in_accounting(tmp_path, fixed_forecaster, fee):
    original = (fixed_forecaster / "run_config.json").read_bytes()
    args = ["--mode", "backtest", "--model-dir", str(fixed_forecaster),
            "--results-dir", str(tmp_path / "replay")]
    if fee is not None:
        args += ["--transaction-cost", str(fee)]
    result = cli.main(args)
    effective_fee = 0.02 if fee is None else fee
    # Flat prices and a full allocation: only the first next-open purchase pays fees.
    np.testing.assert_allclose(result["account_value"].account_value.iloc[1:], 50 / (1 + effective_fee))
    assert result["account_value"].account_value.iloc[0] == 50
    saved = json.loads((Path(result["results_dir"]) / "run_config.json").read_text())
    assert saved["buy_cost_pct"] == saved["sell_cost_pct"] == effective_fee
    assert (fixed_forecaster / "run_config.json").read_bytes() == original
    assert result["stats"].loc["Final value", "Equal-weight buy & hold"] == pytest.approx(
        50 / (1 + effective_fee)
    )


def test_installed_command_exits_successfully_after_backtest(tmp_path, fixed_forecaster):
    command = Path(sysconfig.get_path("scripts")) / (
        "rl-pricing.exe" if os.name == "nt" else "rl-pricing"
    )
    result = subprocess.run(
        [str(command), "--mode", "backtest", "--model-dir", str(fixed_forecaster),
         "--results-dir", str(tmp_path / "replay")],
        cwd=tmp_path, capture_output=True, text=True, check=False, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert len(list((tmp_path / "replay").glob("*/perf_stats.csv"))) == 1


@pytest.fixture
def frozen_experiment(tmp_path, market_file):
    prices, dates, path = market_file
    source = tmp_path / "experiment"
    source.mkdir()
    content = path.read_bytes()
    (source / "input_prices.csv").write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    end = (pd.Timestamp(dates[-1]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    manifest = {
        "data_sha256": digest, "tickers": ["AAA"], "seeds": [0, 1],
        "final_start": dates[15], "end_date": end, "candidates": {"lr_3e-4": 0.0003},
        "limitations": ["Synthetic accounting fixture, not performance evidence"],
        "versions": {"numpy": importlib.metadata.version("numpy")},
    }
    (source / "manifest.json").write_text(json.dumps(manifest))
    (source / "selection.json").write_text(json.dumps({
        "data_sha256": digest, "selected": {"ppo": "lr_3e-4", "sac": "lr_3e-4"},
    }))
    records = []
    for agent in ("ppo", "sac"):
        for seed in (0, 1):
            name = f"final_{agent}_seed{seed}"
            cfg = TrainConfig(
                agent=agent, seed=seed, data_source="csv", data_file="unavailable/original.csv",
                ticker_list=["AAA"], start_date=dates[0], start_trade_date=dates[15], end_date=end,
                tech_indicator_list=[], use_turbulence=False, normalize_observations=False,
                initial_amount=50.0, hmax=1, run_name=name,
                model_kwargs={"learning_rate": 0.0003},
            )
            env = make_env(cfg, data_split(prices, dates[0], dates[13]), env_kwargs(cfg, 1))
            kwargs = {"learning_rate": 0.0003}
            kwargs.update({"n_steps": 8, "batch_size": 8} if agent == "ppo" else {"buffer_size": 100})
            model = build_model(agent, make_vec_env(env), model_kwargs=kwargs, verbose=0, seed=seed)
            # Valid SB3 checkpoints with fixed buys: four next-open fills of one share.
            with torch.no_grad():
                for parameter in model.policy.parameters():
                    parameter.zero_()
                if agent == "ppo":
                    model.policy.action_net.bias.fill_(1.0)
                else:
                    model.policy.actor.mu.bias.fill_(20.0)
            model_dir = source / "models" / name
            model_dir.mkdir(parents=True)
            model.save(model_dir / "model.zip")
            saved = dataclasses.asdict(cfg)
            saved.update(tickers=["AAA"], eligibility_cutoff=dates[13], execution_timing="next_open",
                         resolved_turbulence_threshold=None, data_sha256=digest)
            (model_dir / "run_config.json").write_text(json.dumps(saved))
            for phase in ("development", "final"):
                records.append({
                    "phase": phase, "agent": agent, "candidate": "lr_3e-4", "seed": seed,
                    "validation_sharpe": 1.0, "model_dir": "retired/location/models/" + name,
                })
    pd.DataFrame(records).to_csv(source / "validation_runs.csv", index=False)
    return source


def test_cost_stress_relocates_bundle_preserves_models_and_accounts_for_every_fee(tmp_path, frozen_experiment):
    source = frozen_experiment
    originals = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    result = run_cost_sensitivity(source, [0.0, 0.01, 0.1], output_dir=tmp_path / "stress")
    records = pd.read_csv(Path(result["output_dir"]) / "performance_runs.csv")
    for fee in (0.0, 0.01, 0.1):
        scenario = records[records.transaction_cost == fee]
        agents = scenario[scenario.strategy.isin(["PPO", "SAC"])]
        assert set(zip(agents.strategy, agents.seed)) == {("PPO", 0), ("PPO", 1), ("SAC", 0), ("SAC", 1)}
        np.testing.assert_allclose(agents["Final value"], 50 - 4 * 10 * fee)
        baselines = scenario[~scenario.strategy.isin(["PPO", "SAC"])]
        assert set(baselines.strategy) == {"Equal-weight buy & hold", "Equal-weight monthly rebalance"}
        np.testing.assert_allclose(baselines["Final value"], 50 / (1 + fee))
    for path, content in originals.items():
        assert path.read_bytes() == content
    manifest = json.loads((Path(result["output_dir"]) / "manifest.json").read_text())
    assert manifest["data_sha256"] == hashlib.sha256((source / "input_prices.csv").read_bytes()).hexdigest()
    for model in manifest["models"]:
        for name, digest in model["sha256"].items():
            assert hashlib.sha256((Path(model["model_dir"]) / name).read_bytes()).hexdigest() == digest


def test_cost_stress_rejects_changed_input_before_creating_results(tmp_path, frozen_experiment):
    path = frozen_experiment / "input_prices.csv"
    prices = pd.read_csv(path)
    prices.loc[0, "close"] *= 2
    prices.to_csv(path, index=False)
    out = tmp_path / "stress"
    with pytest.raises(ValueError):
        run_cost_sensitivity(frozen_experiment, output_dir=out)
    assert not out.exists()


def test_cost_stress_cannot_silently_drop_a_selected_seed(tmp_path, frozen_experiment):
    path = frozen_experiment / "validation_runs.csv"
    records = pd.read_csv(path)
    records.drop(records[records.phase == "final"].index[-1]).to_csv(path, index=False)
    out = tmp_path / "stress"
    with pytest.raises(ValueError):
        run_cost_sensitivity(frozen_experiment, output_dir=out)
    assert not out.exists()


@pytest.mark.parametrize("fees", [[], [0.001, 0.001], [-0.001], [np.nan], [np.inf], [1.0]])
def test_invalid_fee_scenarios_are_rejected_before_opening_inputs(tmp_path, fees):
    with pytest.raises(ValueError):
        run_cost_sensitivity(tmp_path / "missing", fees)


@pytest.mark.parametrize("args", [
    ["--experiment-dir", "missing", "--seeds", "99"],
    ["--data-file", "missing.csv", "--tickers", "AAA", "--costs", "0.001"],
])
def test_cost_replay_cli_rejects_training_or_selection_controls(args):
    with pytest.raises(SystemExit):
        main(args)
