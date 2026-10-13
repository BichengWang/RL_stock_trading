"""Saved-model CLI overrides must reflect which options the user supplied."""

import sys

import pytest

from rl import cli
from rl.autotrain import training
from rl.config import config


@pytest.fixture
def backtest_call(monkeypatch):
    call = {}

    def replay(model_dir, **overrides):
        call.update(model_dir=model_dir, overrides=overrides)
        return {"results_dir": "replay"}

    monkeypatch.setattr(training, "run_backtest", replay)
    return call


def test_backtest_preserves_saved_settings_when_options_are_omitted(backtest_call):
    cli.main(["--mode", "backtest", "--model-dir", "saved"])
    assert backtest_call == {"model_dir": "saved", "overrides": {}}


@pytest.mark.parametrize(
    "flag, value, field, expected",
    [
        ("--start-trade-date", config.START_TRADE_DATE, "start_trade_date", config.START_TRADE_DATE),
        ("--end-date", config.END_DATE, "end_date", config.END_DATE),
        ("--results-dir", config.RESULTS_DIR, "results_dir", config.RESULTS_DIR),
        ("--leverage", "1.0", "leverage", 1.0),
    ],
)
def test_explicit_parser_defaults_override_saved_settings(backtest_call, flag, value, field, expected):
    cli.main(["--mode", "backtest", "--model-dir", "saved", flag, value])
    assert backtest_call["overrides"] == {field: expected}


def test_explicit_overrides_support_equals_and_abbreviated_options(backtest_call):
    cli.main([
        "--mode", "backtest", "--model-dir", "saved",
        "--leverage=1.0", "--end-d=" + config.END_DATE,
    ])
    assert backtest_call["overrides"] == {"leverage": 1.0, "end_date": config.END_DATE}


def test_explicit_overrides_from_system_arguments(backtest_call, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "rl-pricing", "--mode", "backtest", "--model-dir", "saved", "--leverage", "1.0",
    ])
    cli.main()
    assert backtest_call["overrides"] == {"leverage": 1.0}


def test_backtest_routes_data_benchmark_and_fee_overrides(backtest_call):
    cli.main([
        "--mode", "backtest", "--model-dir", "saved",
        "--data-file", "prices.csv", "--benchmark", "SPY",
        "--turbulence-threshold", "0", "--transaction-cost", "0",
    ])
    assert backtest_call["overrides"] == {
        "data_source": "csv", "data_file": "prices.csv", "benchmark_ticker": "SPY",
        "turbulence_threshold": 0.0, "buy_cost_pct": 0.0, "sell_cost_pct": 0.0,
    }


def test_training_keeps_its_default_settings(monkeypatch):
    configs = []

    def train(cfg):
        configs.append(cfg)
        return {"results_dir": "training"}

    monkeypatch.setattr(training, "run_training", train)
    cli.main([])
    cfg = configs[0]
    assert (cfg.start_date, cfg.start_trade_date, cfg.end_date) == (
        config.START_DATE, config.START_TRADE_DATE, config.END_DATE,
    )
    assert cfg.results_dir == config.RESULTS_DIR
    assert cfg.leverage == 1.0
    assert cfg.buy_cost_pct == cfg.sell_cost_pct == 0.001
