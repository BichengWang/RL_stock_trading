"""Consumer-visible rolling ensemble accounting."""

import numpy as np

from conftest import make_panel
from rl.model.models import DRLEnsembleAgent


class BuyOneThenHold:
    def get_vec_normalize_env(self):
        return None

    def predict(self, obs, deterministic=True):
        # One ticker: observation is cash, close, holdings.
        return np.array([1.0 if obs[2] == 0 else 0.0]), None


def test_ensemble_carries_gap_returns_and_trades_final_partial_window(tmp_path, monkeypatch):
    panel = make_panel([[100.0]] * 6 + [[100.0]] * 5 + [[200.0], [250.0]])
    panel["turbulence"] = 0.0
    dates = sorted(panel.date.unique())
    ensemble = DRLEnsembleAgent(
        df=panel, train_period=(dates[0], dates[6]),
        val_test_period=(dates[6], "2021-01-01"), rebalance_window=4, validation_window=2,
        stock_dim=1, hmax=1, initial_amount=1000.0, buy_cost_pct=0.1, sell_cost_pct=0.1,
        reward_scaling=1.0, state_space=3, action_space=1, tech_indicator_list=[],
        print_verbosity=0, save_models=False, results_dir=str(tmp_path), verbose=0,
    )
    monkeypatch.setattr(ensemble, "_train", lambda *args, **kwargs: BuyOneThenHold())
    summary = ensemble.run_ensemble_strategy({}, {}, {}, {"ppo": 1, "a2c": 1, "ddpg": 1})
    values = ensemble.account_value
    assert values.date.tolist() == dates[6:]
    # Buy at the next open for 100+10 fees; later gaps accrue to the held share.
    np.testing.assert_allclose(values.account_value, [1000, 990, 990, 990, 990, 1090, 1140])
    assert summary.Transitions.tolist() == [4, 2]
    assert summary["Val End"].lt(summary["Trade Start"]).all()
    assert ensemble.actions.index.tolist() == dates[6:-1]
    assert ensemble.executions.signal_date.tolist() == dates[6:-1]
    assert ensemble.executions.fill_date.tolist() == dates[7:]
    assert ensemble.actions.iloc[:, 0].sum() == 1.0
