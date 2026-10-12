"""Fit-only universe selection, frozen evaluation panels and LSTM boundaries."""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from rl.autotrain import training
from rl.marketdata.synthetic import generate_synthetic_data


@pytest.fixture
def market(monkeypatch, tmp_path):
    raw = generate_synthetic_data(["AAA", "BBB"], "2020-01-01", "2020-05-01", seed=7)
    dates = sorted(raw.date.unique())
    cfg = training.TrainConfig(
        data_source="synthetic", ticker_list=["AAA", "BBB"],
        start_date=dates[0], start_trade_date=dates[60], end_date="2020-05-01",
        tech_indicator_list=["macd", "rsi_10"], use_turbulence=False,
        run_name="causal", trained_model_dir=str(tmp_path / "models"),
        results_dir=str(tmp_path / "results"),
    ).validate()
    monkeypatch.setattr(training, "load_market_data", lambda _: raw.copy())
    return cfg, raw, dates


def test_future_listings_do_not_change_fitting_universe(market, monkeypatch):
    cfg, raw, dates = market
    before, train, _ = training.prepare_data(cfg)
    listing = raw[(raw.tic == "AAA") & (raw.date >= dates[60])].assign(tic="NEW")
    future = pd.concat([raw, listing], ignore_index=True)
    monkeypatch.setattr(training, "load_market_data", lambda _: future.copy())
    after, changed_train, _ = training.prepare_data(cfg)
    assert before.attrs["eligibility_tickers"] == after.attrs["eligibility_tickers"] == ["AAA", "BBB"]
    assert after.attrs["eligibility_cutoff"] == dates[60]
    pd.testing.assert_frame_equal(train, changed_train)


def test_future_disappearance_errors_instead_of_shrinking_universe(market, monkeypatch):
    cfg, raw, dates = market
    damaged = raw[~((raw.tic == "BBB") & (raw.date >= dates[60]))]
    monkeypatch.setattr(training, "load_market_data", lambda _: damaged.copy())
    with pytest.raises(ValueError, match="BBB.*" + dates[60]):
        training.prepare_data(cfg)


@pytest.mark.parametrize("agent,settings,n_validation", [
    ("ppo", {"validation_days": 12}, 12),
    ("ensemble", {"validation_window": 12}, 12),
    ("lstm", {"lstm_validation_fraction": 0.2}, 12),
])
def test_raw_validation_sessions_do_not_select_tickers(market, monkeypatch, agent, settings, n_validation):
    cfg, raw, dates = market
    cfg = dataclasses.replace(cfg, agent=agent, **settings).validate()
    cutoff = dates[60 - n_validation]
    listing = raw[(raw.tic == "AAA") & (raw.date >= cutoff)].assign(tic="NEW")
    monkeypatch.setattr(training, "load_market_data", lambda _: pd.concat([raw, listing], ignore_index=True))
    processed, train, trade = training.prepare_data(cfg)
    for frame in (processed, train, trade):
        assert frame.attrs["eligibility_cutoff"] == cutoff
        assert frame.attrs["validation_start_date"] == cutoff
        assert frame.attrs["eligibility_tickers"] == ["AAA", "BBB"]
    assert sorted(train.date.unique()) == dates[:60]
    assert sorted(trade.date.unique()) == dates[60:]


def test_validation_disappearance_is_an_error(market, monkeypatch):
    cfg, raw, dates = market
    cfg = dataclasses.replace(cfg, validation_days=12)
    damaged = raw[~((raw.tic == "BBB") & (raw.date >= dates[48]))]
    monkeypatch.setattr(training, "load_market_data", lambda _: damaged.copy())
    with pytest.raises(ValueError, match="BBB.*" + dates[48]):
        training.prepare_data(cfg)


def test_future_price_changes_leave_fitting_features_unchanged(market, monkeypatch):
    cfg, raw, dates = market
    cfg = dataclasses.replace(cfg, validation_days=12)
    original, _, _ = training.prepare_data(cfg)
    changed = raw.copy()
    changed.loc[changed.date >= dates[48], ["open", "high", "low", "close"]] *= 10
    monkeypatch.setattr(training, "load_market_data", lambda _: changed.copy())
    processed, _, _ = training.prepare_data(cfg)
    pd.testing.assert_frame_equal(original[original.date < dates[48]], processed[processed.date < dates[48]])


def test_saved_tickers_bypass_eligibility_selection(market, monkeypatch):
    cfg, raw, dates = market
    # BBB would fail a new coverage fit, but it is already a trained ticker.
    damaged_history = raw[~((raw.tic == "BBB") & raw.date.isin(dates[10:30]))]
    monkeypatch.setattr(training, "load_market_data", lambda _: damaged_history.copy())
    processed, _, _ = training.prepare_data(cfg, tickers=["BBB", "AAA"], eligibility_cutoff=dates[30])
    assert processed.attrs["eligibility_tickers"] == ["AAA", "BBB"]
    assert processed.attrs["eligibility_cutoff"] == dates[30]
    assert sorted(processed[processed.date < dates[30]].date.unique()) == dates[:10]
    assert sorted(processed[processed.date >= dates[30]].date.unique()) == dates[30:]


def test_saved_ticker_missing_at_replay_errors(market, monkeypatch):
    cfg, raw, dates = market
    missing = raw[~((raw.tic == "BBB") & (raw.date == dates[65]))]
    monkeypatch.setattr(training, "load_market_data", lambda _: missing.copy())
    with pytest.raises(ValueError, match="BBB.*" + dates[65]):
        training.prepare_data(cfg, tickers=["AAA", "BBB"], eligibility_cutoff=dates[48])


def test_filtering_symbols_cannot_erase_an_observed_evaluation_date(market, monkeypatch):
    cfg, raw, dates = market
    missing = raw[raw.date != dates[65]]
    unrelated = raw[(raw.date == dates[65]) & (raw.tic == "AAA")].assign(tic="NEW")
    monkeypatch.setattr(training, "load_market_data", lambda _: pd.concat([missing, unrelated], ignore_index=True))
    with pytest.raises(ValueError, match="AAA.*" + dates[65]):
        training.prepare_data(cfg, tickers=["AAA", "BBB"])


def test_fitting_balance_drops_gap_dates_without_moving_validation_boundary(market, monkeypatch):
    cfg, raw, dates = market
    cfg = dataclasses.replace(cfg, validation_days=12)
    gap = raw[~((raw.tic == "BBB") & (raw.date == dates[10]))]
    monkeypatch.setattr(training, "load_market_data", lambda _: gap.copy())
    processed, train, _ = training.prepare_data(cfg)
    fitting, validation = training.split_validation(train, 12)
    assert processed.attrs["eligibility_cutoff"] == dates[48]
    assert dates[10] not in set(fitting.date)
    assert sorted(validation.date.unique()) == dates[48:60]
    assert fitting.date.max() == dates[47]


def test_lstm_scaler_and_labels_exclude_heldout_outcomes(market, monkeypatch):
    import torch
    from rl.forecast import lstm

    cfg, raw, dates = market
    cfg = dataclasses.replace(cfg, agent="lstm", lstm_window=3, lstm_validation_fraction=0.2)
    # Deliberately unmistakable held-out prices and indicator values.
    raw.loc[raw.date >= dates[48], ["open", "high", "low", "close"]] *= 100
    processed, train, _ = training.prepare_data(cfg)
    captured = {}
    actual_fit_scaler = lstm.fit_scaler

    def scaler(features, indicators):
        captured["scaler_dates"] = sorted(features.date.unique())
        captured["scaler"] = actual_fit_scaler(features, indicators)
        return captured["scaler"]

    def forecaster(X, y, X_val, y_val, **kwargs):
        captured.update(X=X.copy(), y=y.copy(), X_val=X_val.copy(), y_val=y_val.copy())
        return torch.nn.Linear(1, 1), pd.DataFrame({"epoch": [1]})

    monkeypatch.setattr(lstm, "fit_scaler", scaler)
    monkeypatch.setattr(lstm, "train_forecaster", forecaster)
    monkeypatch.setattr(training, "_seed_everything", lambda _: None)
    monkeypatch.setattr(training, "_lstm_trade", lambda *args, **kwargs: {})
    result = training.run_lstm(cfg)
    assert captured["scaler_dates"] == dates[:48]
    expected_scaler = actual_fit_scaler(lstm.raw_features(train[train.date < dates[48]], cfg.tech_indicator_list), cfg.tech_indicator_list)
    assert captured["scaler"] == expected_scaler
    # Constructing windows on fitting history alone cannot see validation closes.
    _, fit_X, fit_y, _ = training._lstm_inputs(cfg, train[train.date < dates[48]], expected_scaler)
    keep = np.isfinite(fit_y)
    np.testing.assert_allclose(captured["X"], fit_X[keep])
    np.testing.assert_allclose(captured["y"], fit_y[keep])
    assert len(captured["y"]) == 2 * (48 - cfg.lstm_window)
    assert len(captured["y_val"]) == 2 * (12 - 1)
    with open(result["model_dir"] + "/run_config.json", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["eligibility_cutoff"] == dates[48]
    assert saved["tickers"] == ["AAA", "BBB"]
    assert saved["execution_timing"] == "next_open"


@pytest.mark.parametrize("timing", [None, "same_close"])
def test_legacy_execution_artifacts_are_rejected(tmp_path, timing):
    saved = {"agent": "ppo"}
    if timing is not None:
        saved["execution_timing"] = timing
    (tmp_path / "run_config.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="retrain legacy artifacts"):
        training.load_run_config(str(tmp_path))


def test_covariance_warmup_preserves_heldout_sessions_at_minimum_fit_history(market):
    cfg, _, dates = market
    cfg = dataclasses.replace(cfg, task="portfolio", validation_days=12, cov_lookback=46)
    _, train, trade = training.prepare_data(cfg)
    fitting, validation = training.split_validation(train, 12)
    assert sorted(fitting.date.unique()) == dates[46:48]
    assert sorted(validation.date.unique()) == dates[48:60]
    assert sorted(trade.date.unique()) == dates[60:]
    with pytest.raises(ValueError):
        training.prepare_data(dataclasses.replace(cfg, cov_lookback=47))
