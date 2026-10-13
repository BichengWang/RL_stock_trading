"""Effective fees when replaying a saved model through the backtest CLI."""

import dataclasses
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
from rl.autotrain.training import TrainConfig
from rl.forecast.lstm import LSTMForecaster


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
