"""Gymnasium environment that trains agents under the benchmark's execution model.

Every trade goes through :class:`rl.bench.engine.Account`, the account the
evaluation harness uses, so an agent is trained on exactly the fills, fees and
delisting rules it will be scored with. The agent decides at a session's
close and trades at the next open; the reward is the log change in equity
over that session, net of fees.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

from rl.bench.engine import Account
from rl.bench.features import LOOKBACK, N_FEATURES, feature_panel
from rl.bench.market import MarketData

LOGIT_BOUND = 5.0  # actions are logits in [-LOGIT_BOUND, LOGIT_BOUND]
_MIN_EQUITY = 1e-12


def actions_to_weights(action, listed) -> np.ndarray:
    """Target asset weights from ``N + 1`` logits (one per asset, then cash).

    A softmax over the listed assets and cash; unlisted assets get zero weight.
    The remainder after the asset weights is held as cash.
    """
    logits = np.asarray(action, dtype=float).reshape(-1)
    listed = np.asarray(listed, dtype=bool)
    if logits.size != listed.size + 1:
        raise ValueError("Expected {} logits, got {}".format(listed.size + 1, logits.size))
    keep = np.append(listed, True)
    scores = np.where(keep, logits, -np.inf)
    exp = np.exp(scores - scores.max())
    return (exp / exp.sum())[:-1]


def observation(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Flattened features, current asset weights and the cash weight."""
    weights = np.asarray(weights, dtype=float)
    return np.concatenate([np.ravel(features), weights, [1.0 - weights.sum()]]).astype(np.float32)


def observation_size(n_assets: int) -> int:
    return n_assets * N_FEATURES + n_assets + 1


class PortfolioEnv(gym.Env):
    """Long-only portfolio allocation over a :class:`MarketData` panel.

    Parameters
    ----------
    data :
        Sessions the agent may train on; features and fills use only these.
    transaction_cost :
        Proportional fee rate, normally the task's.
    episode_length :
        Sessions per episode, starting at a random session drawn from the
        generator seeded by ``reset(seed=...)``; ``None`` runs once through
        the whole usable range. A length beyond the usable range is clipped
        to it.
    first_session :
        First session an episode may trade on (the first decision is at the
        previous close). Default :data:`LOOKBACK` when ``data`` has at least
        ``2 * LOOKBACK`` sessions, so features are complete, otherwise 1.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: MarketData,
        transaction_cost: float,
        episode_length: Optional[int] = 252,
        first_session: Optional[int] = None,
    ):
        super().__init__()
        n_sessions, n_assets = len(data), len(data.tickers)
        if first_session is None:
            first_session = LOOKBACK if n_sessions >= 2 * LOOKBACK else 1
        if not 1 <= first_session < n_sessions:
            raise ValueError("first_session must be in [1, {}), got {}".format(n_sessions, first_session))
        if episode_length is not None and episode_length < 1:
            raise ValueError("episode_length must be positive")
        self.data = data
        self.transaction_cost = transaction_cost
        self.first_session = first_session
        usable = n_sessions - first_session
        self.episode_length = usable if episode_length is None else min(episode_length, usable)
        self._random_start = episode_length is not None and self.episode_length < usable
        self._features = feature_panel(data)
        self._listed = np.isfinite(data.close)
        self.action_space = gym.spaces.Box(-LOGIT_BOUND, LOGIT_BOUND, (n_assets + 1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (observation_size(n_assets),), dtype=np.float32)
        self._account: Optional[Account] = None

    def _observe(self) -> np.ndarray:
        return observation(self._features[self._decision], self._account.weights(self._decision))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        last_start = len(self.data) - self.episode_length
        start = int(self.np_random.integers(self.first_session, last_start + 1)) if self._random_start else self.first_session
        self._decision = start - 1  # the close the next action is decided at
        self._stop = start + self.episode_length  # first session not traded
        self._account = Account(self.data, self.transaction_cost)
        self._equity = self._account.equity(self._decision)
        return self._observe(), {"date": self.data.dates[self._decision]}

    def step(self, action):
        if self._account is None:
            raise RuntimeError("Call reset() before step()")
        fill = self._decision + 1
        target = actions_to_weights(action, self._listed[self._decision])
        turnover, cost = self._account.trade(fill, target)
        equity = self._account.equity(fill)
        reward = float(np.log(max(equity, _MIN_EQUITY) / max(self._equity, _MIN_EQUITY)))
        self._equity = equity
        self._decision = fill
        truncated = fill + 1 >= self._stop
        info = {"date": self.data.dates[fill], "equity": equity, "turnover": turnover, "cost": cost}
        return self._observe(), reward, False, truncated, info
