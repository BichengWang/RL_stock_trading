"""Deep RL policies trained with Stable-Baselines3 on the benchmark's execution model.

An agent observes causal scale-free features of every asset
(:mod:`rl.bench.features`), its current weights and cash, and emits ``N + 1``
logits that are softmaxed into long-only target weights over the listed
assets and cash (:func:`rl.bench.gym_env.actions_to_weights`). Training runs
on :class:`rl.bench.gym_env.PortfolioEnv`, which trades through the
evaluation engine's account, so fills and fees match scoring.

Model selection uses the validation sessions only: at regular intervals the
deterministic policy is run through :func:`rl.bench.engine.simulate` over
the sessions from ``validation_start`` on, with all earlier history visible,
and the checkpoint with the best net-of-fees Sharpe ratio is restored at the
end of training. Training itself never sees a validation session.

Pass the task's ``transaction_cost`` to the constructor: the policy has no
other way to learn the fee it will be scored with.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed

from rl.bench.engine import Decision, simulate
from rl.bench.features import LOOKBACK, latest_features
from rl.bench.gym_env import PortfolioEnv, actions_to_weights, observation
from rl.bench.market import MarketData, MarketView
from rl.bench.policy import Policy

logger = logging.getLogger(__name__)

ALGORITHMS = {"ppo": PPO, "a2c": A2C, "sac": SAC, "td3": TD3, "ddpg": DDPG}
_OFF_POLICY = ("sac", "td3", "ddpg")
_EXPLORATION_NOISE = 0.1  # action noise (in units of the action range) for DDPG and TD3
_MODEL_FILE = "model"  # SB3 appends ".zip"
_META_FILE = "meta.json"


@contextlib.contextmanager
def _single_thread():
    """Run torch on one thread so training does not depend on the machine's core count."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


class _Selection(BaseCallback):
    """Tracks the best validation checkpoint while training."""

    def __init__(self, policy: SB3Policy, every: int):
        super().__init__()
        self.policy = policy
        self.every = every
        self.best_sharpe = -np.inf
        self.best_step = 0
        self.best_parameters = None
        self._last = None

    def _evaluate(self):
        self._last = self.num_timesteps
        sharpe = self.policy._validation_sharpe(self.model)
        logger.info("step %d validation Sharpe %.4f", self.num_timesteps, sharpe)
        if sharpe > self.best_sharpe:
            self.best_sharpe, self.best_step = sharpe, self.num_timesteps
            self.best_parameters = copy.deepcopy(self.model.get_parameters())

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every == 0:
            self._evaluate()
        return True

    def _on_training_end(self) -> None:
        if self._last != self.num_timesteps:
            self._evaluate()


class SB3Policy(Policy):
    """A Stable-Baselines3 agent (``ppo``, ``a2c``, ``sac``, ``td3`` or ``ddpg``).

    Parameters
    ----------
    algo, timesteps :
        Algorithm and number of environment steps to train for.
    episode_length :
        Sessions per training episode (random start); ``None`` trains on the
        whole training period in one episode.
    learning_rate, net_arch, gamma :
        Passed to the algorithm; ``net_arch`` is the hidden layer sizes.
    eval_every :
        Environment steps between validation evaluations (default
        ``timesteps // 10``).
    algo_kwargs :
        Extra keyword arguments for the SB3 algorithm class.
    transaction_cost :
        Fee rate used for training and validation. The runner must pass the
        task's cost, otherwise the agent trains under the wrong fees.
    """

    stochastic = True

    def __init__(
        self,
        algo: str = "ppo",
        timesteps: int = 100_000,
        episode_length: Optional[int] = 252,
        learning_rate: float = 3e-4,
        net_arch=(64, 64),
        gamma: float = 0.99,
        eval_every: Optional[int] = None,
        algo_kwargs: Optional[dict] = None,
        transaction_cost: float = 0.001,
    ):
        algo = str(algo).lower()
        if algo not in ALGORITHMS:
            raise ValueError("algo must be one of {}, got {!r}".format(sorted(ALGORITHMS), algo))
        if timesteps < 1:
            raise ValueError("timesteps must be positive")
        self.algo = algo
        self.timesteps = int(timesteps)
        self.episode_length = episode_length
        self.learning_rate = learning_rate
        self.net_arch = tuple(int(n) for n in net_arch)
        self.gamma = gamma
        self.eval_every = max(1, self.timesteps // 10) if eval_every is None else int(eval_every)
        self.algo_kwargs = dict(algo_kwargs or {})
        self.transaction_cost = transaction_cost
        self.model = None
        self._validation = None
        self.best_step: Optional[int] = None
        self.best_sharpe: Optional[float] = None

    def _decide(self, model, view: MarketView) -> np.ndarray:
        features = latest_features(view.history.window(LOOKBACK))
        obs = observation(features, view.weights)
        if obs.shape != model.observation_space.shape:
            raise ValueError(
                "Model expects observations of shape {}, got {}: it was trained on a different universe".format(
                    model.observation_space.shape, obs.shape
                )
            )
        action, _ = model.predict(obs, deterministic=True)
        return actions_to_weights(action, view.listed)

    def _validation_sharpe(self, model) -> float:
        """Per-session Sharpe ratio of the deterministic policy on the validation sessions."""
        history, start = self._validation
        result = simulate(
            history, lambda view: self._decide(model, view), start, self.transaction_cost
        )
        returns = result.returns
        spread = returns.std(ddof=1) if returns.size > 1 else 0.0
        return float(returns.mean() / spread) if spread > 1e-12 else 0.0

    def fit(self, history: MarketData, validation_start: np.datetime64, seed: int) -> None:
        split = history.index_of(validation_start)
        if split < 2:
            raise ValueError("Need at least two sessions before validation_start to train")
        env = PortfolioEnv(history.rows(0, split), self.transaction_cost, self.episode_length)
        has_validation = split < len(history)
        self._validation = (history, split) if has_validation else None
        kwargs = dict(self.algo_kwargs)
        if self.algo in _OFF_POLICY:
            kwargs.setdefault("buffer_size", min(self.timesteps, 1_000_000))
        if self.algo in ("td3", "ddpg"):
            size = env.action_space.shape[0]
            kwargs.setdefault(
                "action_noise", NormalActionNoise(np.zeros(size), np.full(size, _EXPLORATION_NOISE))
            )
        with _single_thread():
            set_random_seed(int(seed))
            model = ALGORITHMS[self.algo](
                "MlpPolicy",
                env,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                policy_kwargs={"net_arch": list(self.net_arch)},
                seed=int(seed),
                device="cpu",
                verbose=0,
                **kwargs,
            )
            if has_validation:
                selection = _Selection(self, self.eval_every)
                model.learn(self.timesteps, callback=selection)
                model.set_parameters(selection.best_parameters, exact_match=True)
                self.best_step, self.best_sharpe = selection.best_step, selection.best_sharpe
            else:
                logger.warning("No validation sessions after %s: keeping the final checkpoint", validation_start)
                model.learn(self.timesteps)
                self.best_step, self.best_sharpe = model.num_timesteps, None
        self._validation = None
        self.model = model

    def act(self, view: MarketView) -> Decision:
        if self.model is None:
            raise RuntimeError("SB3Policy has no model: call fit() or load() first")
        return self._decide(self.model, view)

    def save(self, directory: Path) -> None:
        if self.model is None:
            raise RuntimeError("SB3Policy has no model: call fit() or load() first")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(directory / _MODEL_FILE)
        meta = {"algo": self.algo, "best_step": self.best_step, "best_sharpe": self.best_sharpe}
        (directory / _META_FILE).write_text(json.dumps(meta, indent=2))

    def load(self, directory: Path) -> None:
        directory = Path(directory)
        meta = json.loads((directory / _META_FILE).read_text())
        if meta["algo"] != self.algo:
            raise ValueError("Saved model is {!r} but this policy is {!r}".format(meta["algo"], self.algo))
        self.model = ALGORITHMS[self.algo].load(directory / (_MODEL_FILE + ".zip"), device="cpu")
        self.best_step, self.best_sharpe = meta["best_step"], meta["best_sharpe"]
