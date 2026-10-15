import copy
import importlib.util
import logging
import os
import time

import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.config import config
from rl.env.env_stocktrading import StockTradingEnv
from rl.preprocessing.data import data_split
from rl.trade import metrics

logger = logging.getLogger(__name__)

MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}

MODEL_KWARGS = {x: getattr(config, f"{x.upper()}_PARAMS") for x in MODELS}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}

ACTION_NOISE_SIGMA = 0.1


def make_action_noise(spec, n_actions):
    """Build an exploration-noise object from a name or a settings dict.

    ``spec`` is ``"normal"`` / ``"ornstein_uhlenbeck"`` (sigma
    ``ACTION_NOISE_SIGMA``) or a dict ``{"type": ..., "sigma": ..., **kwargs}``
    whose extra entries (e.g. ``theta``, ``dt``) go to the noise class. The
    sigma is the same for every action dimension.
    """
    settings = {"type": spec} if isinstance(spec, str) else dict(spec)
    kind = settings.pop("type", None)
    if kind not in NOISE:
        raise ValueError(
            "action_noise type must be one of {}, got {!r}".format(", ".join(NOISE), kind)
        )
    sigma = float(settings.pop("sigma", ACTION_NOISE_SIGMA))
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("action_noise sigma must be finite and nonnegative")
    return NOISE[kind](mean=np.zeros(n_actions), sigma=sigma * np.ones(n_actions), **settings)


def _tensorboard_log(model_name):
    # SB3 refuses a log directory when tensorboard is not installed.
    if importlib.util.find_spec("tensorboard") is None:
        return None
    return f"{config.TENSORBOARD_LOG_DIR}/{model_name}"


def build_model(
    model_name,
    env,
    policy="MlpPolicy",
    policy_kwargs=None,
    model_kwargs=None,
    verbose=1,
    seed=None,
):
    """Create a stable-baselines3 model with this project's defaults.

    ``model_kwargs`` defaults to the ``<NAME>_PARAMS`` dict in config.py and is
    copied, so neither it nor the config is modified. An ``action_noise`` entry
    may name a noise type ("normal" or "ornstein_uhlenbeck") or give it as a dict
    with a ``sigma`` (see :func:`make_action_noise`).
    """
    if model_name not in MODELS:
        raise NotImplementedError(
            "Unknown model {!r}; choose from {}".format(model_name, ", ".join(MODELS))
        )
    kwargs = copy.deepcopy(MODEL_KWARGS[model_name] if model_kwargs is None else model_kwargs)
    if isinstance(kwargs.get("action_noise"), (str, dict)):
        kwargs["action_noise"] = make_action_noise(
            kwargs["action_noise"], env.action_space.shape[-1]
        )
    logger.info("Building %s with %s", model_name.upper(), kwargs)
    return MODELS[model_name](
        policy=policy,
        env=env,
        tensorboard_log=_tensorboard_log(model_name),
        verbose=verbose,
        policy_kwargs=policy_kwargs,
        seed=seed,
        **kwargs,
    )


def make_vec_env(env, normalize_obs=False, seed=None):
    """Wrap an environment instance for stable-baselines3.

    With ``normalize_obs`` the observations are standardised with running
    statistics (``VecNormalize``); cash, prices, share counts and indicators
    otherwise differ by several orders of magnitude.
    """
    venv = DummyVecEnv([lambda: env])
    if seed is not None:
        venv.seed(seed)
    if normalize_obs:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return venv


class EpisodeStatsCallback(BaseCallback):
    """Record the environment's end-of-episode statistics in the SB3 logger."""

    def _on_step(self):
        for info in self.locals.get("infos", []):
            for key, value in info.get("episode_stats", {}).items():
                self.logger.record(f"environment/{key}", value)
        return True


def predict_episode(model, environment, deterministic=True, obs_normalizer="auto"):
    """Run ``model`` through one full episode of ``environment``.

    ``obs_normalizer`` is a ``VecNormalize`` whose statistics are applied to
    observations; by default the one the model was trained with (if any).
    Returns the environment's account value and action frames.
    """
    if isinstance(obs_normalizer, str) and obs_normalizer == "auto":
        obs_normalizer = model.get_vec_normalize_env()
    obs, _ = environment.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        model_obs = obs_normalizer.normalize_obs(obs) if obs_normalizer is not None else obs
        action, _ = model.predict(model_obs, deterministic=deterministic)
        obs, _, terminated, truncated, _ = environment.step(action)
    return environment.save_asset_memory(), environment.save_action_memory()



class ValidationCallback(BaseCallback):
    """Keep the model that trades a held-out validation period best.

    Every ``eval_freq`` training steps (and once more when training ends) the
    current policy trades one episode of ``make_eval_env()`` deterministically.
    Whenever its annualised Sharpe ratio is the best so far, the model is saved
    to ``<save_dir>/model.zip`` together with its observation-normalisation
    statistics (``vecnormalize.pkl``), so the files always hold the best
    checkpoint. ``history`` lists every evaluation.
    """

    def __init__(self, make_eval_env, eval_freq, save_dir, verbose=0):
        super().__init__(verbose)
        if eval_freq < 1:
            raise ValueError("eval_freq must be at least 1")
        self.make_eval_env = make_eval_env
        self.eval_freq = eval_freq
        self.save_dir = save_dir
        self.history = []
        self.best = None
        self._last_eval = 0

    def _evaluate(self):
        self._last_eval = self.num_timesteps
        values, _ = predict_episode(self.model, self.make_eval_env())
        account = values["account_value"]
        record = {
            "timesteps": self.num_timesteps,
            "sharpe": metrics.sharpe_ratio(account.pct_change()),
            "total_return": float(account.iloc[-1] / account.iloc[0] - 1),
        }
        self.history.append(record)
        self.logger.record("validation/sharpe", record["sharpe"])
        self.logger.record("validation/total_return", record["total_return"])
        score = -np.inf if np.isnan(record["sharpe"]) else record["sharpe"]
        best_score = None if self.best is None else self.best["score"]
        if best_score is None or score > best_score:
            self.best = dict(record, score=score)
            os.makedirs(self.save_dir, exist_ok=True)
            self.model.save(os.path.join(self.save_dir, "model.zip"))
            normalizer = self.model.get_vec_normalize_env()
            if normalizer is not None:
                normalizer.save(os.path.join(self.save_dir, "vecnormalize.pkl"))
        logger.info(
            "Validation at %d steps: Sharpe %.3f, return %.2f%% (best so far: %d steps)",
            record["timesteps"],
            record["sharpe"],
            100 * record["total_return"],
            self.best["timesteps"],
        )

    def _on_step(self):
        if self.num_timesteps - self._last_eval >= self.eval_freq:
            self._evaluate()
        return True

    def _on_training_end(self):
        if self.num_timesteps != self._last_eval or not self.history:
            self._evaluate()


def load_model(model_dir, model_name, env=None):
    """Load ``model.zip`` and, if saved, its ``VecNormalize`` statistics.

    ``env`` (an unwrapped environment matching the model's spaces) is needed
    to restore the statistics. Returns ``(model, normalizer or None)``.
    """
    model = MODELS[model_name].load(os.path.join(model_dir, "model.zip"))
    normalizer = None
    stats_path = os.path.join(model_dir, "vecnormalize.pkl")
    if os.path.exists(stats_path):
        if env is None:
            raise ValueError("env is required to load the normalisation statistics")
        normalizer = VecNormalize.load(stats_path, DummyVecEnv([lambda: env]))
        normalizer.training = False
    return model, normalizer


class DRLAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class

    Methods
    -------
        get_model()
            build an A2C, PPO, DDPG, TD3 or SAC model
        train_model()
            train a model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    @staticmethod
    def DRL_prediction(model, environment, deterministic=True, obs_normalizer="auto"):
        """Trade one episode of ``environment`` (an unwrapped env instance)."""
        return predict_episode(
            model, environment, deterministic=deterministic, obs_normalizer=obs_normalizer
        )

    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        model_name,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
        seed=None,
    ):
        return build_model(
            model_name,
            self.env,
            policy=policy,
            policy_kwargs=policy_kwargs,
            model_kwargs=model_kwargs,
            verbose=verbose,
            seed=seed,
        )

    def train_model(self, model, tb_log_name, total_timesteps=5000, callback=None):
        callbacks = [EpisodeStatsCallback()] + ([callback] if callback is not None else [])
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=CallbackList(callbacks),
        )
        return model


class DRLEnsembleAgent:
    """Rolling-window ensemble of A2C, PPO and DDPG.

    Every ``rebalance_window`` trading days, each algorithm is trained on all
    data before a validation window, scored by its validation Sharpe ratio,
    and the winner is retrained up to the trading window and used to trade
    it. Positions carry over from one trading window to the next.

    With ``standalone_accounts`` each algorithm also trades every window with
    its own account (positions carried over, same windows and turbulence
    threshold) using its model refit on all data before the window, exactly as
    the winner is; the winner's refit model is shared, so the ensemble and the
    standalone accounts differ only in which algorithm trades each window.
    The accounts are in ``agent_account_values``. With ``use_turbulence=False``
    no liquidation threshold is used and no ``turbulence`` column is needed.
    """

    # Ties are broken in this order, as in the original implementation.
    MODEL_NAMES = ("ppo", "a2c", "ddpg")

    @staticmethod
    def get_model(
        model_name,
        env,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
        seed=None,
    ):
        return build_model(
            model_name,
            env,
            policy=policy,
            policy_kwargs=policy_kwargs,
            model_kwargs=model_kwargs,
            verbose=verbose,
            seed=seed,
        )

    @staticmethod
    def train_model(model, model_name, tb_log_name, iter_num, total_timesteps=5000, save=True):
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=EpisodeStatsCallback(),
        )
        if save:
            os.makedirs(config.TRAINED_MODEL_DIR, exist_ok=True)
            model.save(
                f"{config.TRAINED_MODEL_DIR}/{model_name.upper()}_{total_timesteps // 1000}k_{iter_num}"
            )
        return model

    def __init__(
        self,
        df,
        train_period,
        val_test_period,
        rebalance_window,
        validation_window,
        stock_dim,
        hmax,
        initial_amount,
        buy_cost_pct,
        sell_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        print_verbosity,
        *,
        normalize_obs=False,
        reward_type="asset_change",
        seed=None,
        save_models=True,
        results_dir=config.RESULTS_DIR,
        verbose=1,
        use_turbulence=True,
        standalone_accounts=False,
    ):
        if use_turbulence and "turbulence" not in df.columns:
            raise ValueError("The ensemble strategy needs a 'turbulence' column")
        self.df = df
        self.train_period = train_period
        self.val_test_period = val_test_period

        # Like data_split: the start date is included and the end date is not.
        self.unique_trade_date = np.sort(
            df[(df.date >= val_test_period[0]) & (df.date < val_test_period[1])].date.unique()
        )
        if rebalance_window < 1 or validation_window < 2:
            raise ValueError("rebalance_window must be positive and validation_window at least 2")
        self.rebalance_window = rebalance_window
        self.validation_window = validation_window

        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.print_verbosity = print_verbosity
        self.normalize_obs = normalize_obs
        self.reward_type = reward_type
        self.seed = seed
        self.save_models = save_models
        self.results_dir = results_dir
        self.verbose = verbose
        self.use_turbulence = use_turbulence
        self.standalone_accounts = standalone_accounts
        self.account_value = None
        self.actions = None
        self.executions = None
        self.agent_account_values = None

    def _env(self, data, **kwargs):
        return StockTradingEnv(
            data,
            self.stock_dim,
            self.hmax,
            self.initial_amount,
            self.buy_cost_pct,
            self.sell_cost_pct,
            self.reward_scaling,
            self.state_space,
            self.action_space,
            self.tech_indicator_list,
            print_verbosity=self.print_verbosity,
            reward_type=self.reward_type,
            results_dir=self.results_dir,
            **kwargs,
        )

    def _train(self, model_name, data, model_kwargs, total_timesteps, tb_log_name, iter_num):
        venv = make_vec_env(self._env(data), normalize_obs=self.normalize_obs, seed=self.seed)
        model = self.get_model(
            model_name, venv, model_kwargs=model_kwargs, verbose=self.verbose, seed=self.seed
        )
        return self.train_model(
            model,
            model_name,
            tb_log_name=tb_log_name,
            iter_num=iter_num,
            total_timesteps=total_timesteps,
            save=self.save_models,
        )

    def DRL_validation(self, model, test_data, turbulence_threshold, iteration, model_name):
        """Trade the validation window and return its annualised Sharpe ratio."""
        env = self._env(
            test_data,
            turbulence_threshold=turbulence_threshold,
            iteration=iteration,
            model_name=model_name.upper(),
            mode="validation",
        )
        account_value, _ = predict_episode(model, env)
        return metrics.sharpe_ratio(account_value["account_value"].pct_change())

    def DRL_prediction(
        self, model, name, last_state, start_index, end_index, turbulence_threshold, initial
    ):
        """Trade inclusive signal/valuation endpoints, preserving every transition."""
        trade_data = data_split(
            self.df,
            start=self.unique_trade_date[start_index],
            end=(pd.Timestamp(self.unique_trade_date[end_index]) + pd.Timedelta(days=1)).strftime(
                "%Y-%m-%d"
            ),
        )
        trade_env = self._env(
            trade_data,
            turbulence_threshold=turbulence_threshold,
            initial=initial,
            previous_state=last_state,
            model_name=name,
            mode="trade",
            iteration=start_index,
        )
        account_value, _ = predict_episode(model, trade_env)
        last_state = list(trade_env.render())
        os.makedirs(self.results_dir, exist_ok=True)
        pd.DataFrame({"last_state": last_state}).to_csv(
            os.path.join(self.results_dir, "last_state_{}_{}.csv".format(name, start_index)),
            index=False,
        )
        return (
            last_state, account_value, trade_env.save_action_memory(),
            trade_env.save_execution_memory(),
        )

    def run_ensemble_strategy(
        self, A2C_model_kwargs, PPO_model_kwargs, DDPG_model_kwargs, timesteps_dict
    ):
        """Select on preceding validation dates and trade every requested transition."""
        logger.info("============Start Ensemble Strategy============")
        model_kwargs = {"a2c": A2C_model_kwargs, "ppo": PPO_model_kwargs, "ddpg": DDPG_model_kwargs}
        all_dates = np.sort(self.df[self.df.date >= self.train_period[0]].date.unique())
        daily_turbulence = (
            self.df.drop_duplicates(subset=["date"]).set_index("date")["turbulence"].sort_index()
            if self.use_turbulence
            else None
        )
        last_state = []
        agent_states = {name: [] for name in self.MODEL_NAMES}
        agent_values = {name: [] for name in self.MODEL_NAMES}
        records, account_values, actions, executions = [], [], [], []
        start = time.time()
        self.agent_account_values = None
        for first in range(0, len(self.unique_trade_date) - 1, self.rebalance_window):
            last = min(first + self.rebalance_window, len(self.unique_trade_date) - 1)
            signal_date = self.unique_trade_date[first]
            preceding = all_dates[all_dates < signal_date]
            if len(preceding) < self.validation_window + 2:
                raise ValueError(
                    "Need at least {} dates before {} for fitting and validation".format(
                        self.validation_window + 2, signal_date
                    )
                )
            validation_start = preceding[-self.validation_window]
            train = data_split(self.df, self.train_period[0], validation_start)
            validation = data_split(self.df, validation_start, signal_date)
            # Both the reference quantile and regime decision use fitting history only.
            threshold = None
            if self.use_turbulence:
                fitted_turbulence = daily_turbulence[
                    (daily_turbulence.index >= self.train_period[0])
                    & (daily_turbulence.index < validation_start)
                ]
                reference = float(fitted_turbulence.quantile(0.90))
                recent_mean = fitted_turbulence.iloc[-63:].mean()
                threshold = reference if recent_mean > reference else float(fitted_turbulence.max())
                threshold = threshold if threshold > 0 else None
            sharpes = {}
            for name in self.MODEL_NAMES:
                model = self._train(
                    name, train, model_kwargs[name], timesteps_dict[name], f"{name}_{first}", first
                )
                sharpes[name] = self.DRL_validation(
                    model, validation, threshold, first, name
                )
            best = max(
                self.MODEL_NAMES,
                key=lambda name: -np.inf if np.isnan(sharpes[name]) else sharpes[name],
            )
            train_full = data_split(self.df, self.train_period[0], signal_date)
            if self.standalone_accounts:
                # Every algorithm is refit as the winner is; the winner's model is reused.
                refit = {
                    name: self._train(
                        name, train_full, model_kwargs[name], timesteps_dict[name],
                        f"{name}_full_{first}", first,
                    )
                    for name in self.MODEL_NAMES
                }
                model = refit[best]
                for name in self.MODEL_NAMES:
                    agent_states[name], own_values, _, _ = self.DRL_prediction(
                        model=refit[name], name=name, last_state=agent_states[name],
                        start_index=first, end_index=last, turbulence_threshold=threshold,
                        initial=first == 0,
                    )
                    agent_values[name].append(own_values.iloc[1:] if first else own_values)
            else:
                model = self._train(
                    best, train_full, model_kwargs[best], timesteps_dict[best],
                    f"ensemble_{first}", first,
                )
            last_state, values, window_actions, window_executions = self.DRL_prediction(
                model=model, name="ensemble", last_state=last_state,
                start_index=first, end_index=last, turbulence_threshold=threshold,
                initial=first == 0,
            )
            # Adjacent windows share the signal close, not a return or a fill.
            if account_values:
                values = values.iloc[1:]
            account_values.append(values)
            actions.append(window_actions)
            executions.append(window_executions)
            records.append({
                "Iter": first,
                "Val Start": validation_start,
                "Val End": preceding[-1],
                "Trade Start": signal_date,
                "Trade End": self.unique_trade_date[last],
                "Transitions": last - first,
                "Model Used": best.upper(),
                "A2C Sharpe": sharpes.get("a2c", np.nan),
                "PPO Sharpe": sharpes.get("ppo", np.nan),
                "DDPG Sharpe": sharpes.get("ddpg", np.nan),
            })
        logger.info("Ensemble Strategy took %.1f minutes", (time.time() - start) / 60)
        if account_values:
            self.account_value = pd.concat(account_values, ignore_index=True)
            self.actions = pd.concat(actions)
            self.executions = pd.concat(executions, ignore_index=True)
            if self.standalone_accounts:
                self.agent_account_values = {
                    name: pd.concat(frames, ignore_index=True)
                    for name, frames in agent_values.items()
                }
        return pd.DataFrame(records)
