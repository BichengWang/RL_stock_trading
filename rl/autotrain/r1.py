"""Driver for the R1 runs: the toolkit's ensemble with the FinRL authors' recipe.

Sets what the toolkit lets us match from Yang, Liu, Zhong and Walid (ICAIF
2020): the four indicators (``macd``, ``rsi_30``, ``cci_30``, ``dx_30``), the
training steps per algorithm (A2C 30,000, PPO 100,000, DDPG 10,000), PPO with
``ent_coef`` 0.005 and 128-step rollouts in minibatches of 16 (the authors'
``nminibatches=8``), and DDPG with Ornstein-Uhlenbeck action noise of sigma
0.5. A2C and the rest of DDPG keep the toolkit defaults, as does everything
else (0.1% fees per side, $1,000,000, ``hmax`` 100, reward scaling 1e-4,
63-session rebalance and validation windows, next-open fills, observation
normalisation, refit of the winner). Each algorithm also trades its own
account (``standalone_accounts``).

Prices come from a CSV in the toolkit format (``date,tic,open,high,low,close,
volume``), e.g. from :mod:`rl.marketdata.sharadar`::

    python -m rl.autotrain.r1 --data-file prices.csv --seed 0 --run-name r1_a_s0
"""

import argparse
import copy
import logging

from rl.autotrain.training import TrainConfig, run_ensemble
from rl.config import config

AUTHORS_INDICATORS = ("macd", "rsi_30", "cci_30", "dx_30")
AUTHORS_TIMESTEPS = {"a2c": 30_000, "ppo": 100_000, "ddpg": 10_000}
AUTHORS_DDPG_NOISE_SIGMA = 0.5


def authors_model_kwargs(ddpg_noise_sigma=AUTHORS_DDPG_NOISE_SIGMA):
    """Per-algorithm model settings (each replaces the config.py default)."""
    return {
        "a2c": copy.deepcopy(config.A2C_PARAMS),
        # 128 steps in 8 minibatches; every other PPO setting is stable-baselines3's default.
        "ppo": {"n_steps": 128, "batch_size": 16, "ent_coef": 0.005},
        "ddpg": {
            **copy.deepcopy(config.DDPG_PARAMS),
            "action_noise": {"type": "ornstein_uhlenbeck", "sigma": ddpg_noise_sigma},
        },
    }


def r1_config(
    data_file,
    *,
    seed,
    use_turbulence=True,
    ticker_list=None,
    start_date="2009-01-02",
    start_trade_date="2016-01-04",
    end_date="2026-01-01",
    ddpg_noise_sigma=AUTHORS_DDPG_NOISE_SIGMA,
    ensemble_timesteps=None,
    standalone_accounts=True,
    run_name=None,
    results_dir=config.RESULTS_DIR,
    **overrides,
):
    """The :class:`TrainConfig` of one R1 ensemble run.

    ``end_date`` is exclusive (the default includes 2025-12-31). ``overrides``
    set any other :class:`TrainConfig` field.
    """
    settings = dict(
        agent="ensemble",
        task="trading",
        data_source="csv",
        data_file=data_file,
        ticker_list=ticker_list,
        start_date=start_date,
        start_trade_date=start_trade_date,
        end_date=end_date,
        tech_indicator_list=list(AUTHORS_INDICATORS),
        use_turbulence=use_turbulence,
        ensemble_timesteps=dict(AUTHORS_TIMESTEPS if ensemble_timesteps is None else ensemble_timesteps),
        ensemble_model_kwargs=authors_model_kwargs(ddpg_noise_sigma),
        standalone_accounts=standalone_accounts,
        seed=seed,
        run_name=run_name,
        results_dir=results_dir,
    )
    settings.update(overrides)
    return TrainConfig(**settings).validate()


def run_r1(data_file, *, seed, **options):
    """Run the ensemble for ``seed`` (see :func:`r1_config`); returns :func:`run_ensemble`'s dict."""
    return run_ensemble(r1_config(data_file, seed=seed, **options))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-file", required=True, help="CSV of date,tic,open,high,low,close,volume")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tickers", nargs="+", help="default: every ticker in the file")
    parser.add_argument("--no-turbulence", action="store_true", help="arm B: no liquidation rule")
    parser.add_argument("--start-date", default="2009-01-02")
    parser.add_argument("--start-trade-date", default="2016-01-04")
    parser.add_argument("--end-date", default="2026-01-01", help="exclusive")
    parser.add_argument("--ddpg-noise-sigma", type=float, default=AUTHORS_DDPG_NOISE_SIGMA)
    parser.add_argument("--run-name")
    parser.add_argument("--results-dir", default=config.RESULTS_DIR)
    options = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    return run_r1(
        options.data_file,
        seed=options.seed,
        use_turbulence=not options.no_turbulence,
        ticker_list=options.tickers,
        start_date=options.start_date,
        start_trade_date=options.start_trade_date,
        end_date=options.end_date,
        ddpg_noise_sigma=options.ddpg_noise_sigma,
        run_name=options.run_name,
        results_dir=options.results_dir,
    )


if __name__ == "__main__":
    main()
