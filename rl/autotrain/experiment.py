"""Frozen-data, chronological PPO/SAC evaluation: python -m rl.autotrain.experiment."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rl.autotrain.training import TrainConfig, normalize_date, run_training


# Deliberately small search, identical networks and budgets within each algorithm.
CANDIDATES = {"lr_3e-4": 3e-4, "lr_1e-3": 1e-3}
METRICS = ("Cumulative returns", "Sharpe ratio", "Max drawdown", "Final value")


def select_candidates(validation):
    """Rank only development validation scores, never development/test returns."""
    development = validation[validation["phase"] == "development"].copy()
    development["validation_sharpe"] = development["validation_sharpe"].where(
        np.isfinite(development["validation_sharpe"])
    )
    selected = {}
    for agent in ("ppo", "sac"):
        scores = development[development.agent == agent].groupby("candidate", sort=True)[
            "validation_sharpe"
        ].agg(["mean", "count", "size"])
        # A candidate must have a defined score on every scheduled fold/seed.
        scores = scores[(scores["count"] == scores["size"]) & (scores["count"] > 0)]
        if scores.empty:
            raise ValueError("No candidate has finite validation Sharpe on every run for " + agent)
        selected[agent] = scores["mean"].idxmax()
    return selected


def run_experiment(
    data_file, tickers, fold_starts, final_start, end_date, *, start_date="2010-01-01",
    seeds=(0, 1, 2), total_timesteps=20_000, validation_days=126,
    output_dir="results/evaluation",
):
    """Freeze inputs, select settings on past validation, then open the final test.

    Each development fold trains on earlier dates, holds out the immediately
    preceding validation period, and evaluates the following fold. Only the
    validation scores select the learning rate. The final year is evaluated
    after selection is written to disk. Seeds measure training variability,
    not independent market samples.
    """
    start_date = normalize_date(start_date)
    fold_starts = [normalize_date(date) for date in fold_starts]
    final_start, end_date = normalize_date(final_start), normalize_date(end_date)
    boundaries = [start_date] + fold_starts + [final_start, end_date]
    if len(fold_starts) < 2 or any(a >= b for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("Need at least two increasing development fold starts before the final test")
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least two distinct training seeds")
    if validation_days < 2 or total_timesteps < 1:
        raise ValueError("validation_days must be at least 2 and timesteps positive")
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("Provide an explicit, unique ticker universe")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "input_prices.csv"
    content = Path(data_file).read_bytes()
    snapshot.write_bytes(content)
    manifest = {
        "data_sha256": hashlib.sha256(content).hexdigest(),
        "source_file": str(Path(data_file).resolve()),
        "tickers": sorted(tickers), "start_date": start_date,
        "fold_starts": fold_starts, "final_start": final_start, "end_date": end_date,
        "seeds": list(seeds), "requested_timesteps": total_timesteps,
        "validation_days": validation_days, "candidates": CANDIDATES,
        "selection_metric": "mean development validation Sharpe across all folds and seeds",
        "execution_timing": "next_open",
        "limitations": [
            "Explicit universe is not a point-in-time constituent history.",
            "Adjusted OHLC models reinvested distributions, not raw exchange fills.",
            "No bid-ask spread, market impact, or interest on cash/borrowing.",
            "Seed variability is not independent evidence across market regimes.",
        ],
        "versions": {name: importlib.metadata.version(name) for name in (
            "numpy", "pandas", "torch", "gymnasium", "stable-baselines3", "stockstats"
        )},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    torch.set_num_threads(1)
    validation_records, performance = [], []
    baseline_folds = set()

    def run(phase, fold, fold_start, fold_end, agent, candidate, seed):
        name = f"{phase}_{fold}_{agent}_{candidate}_seed{seed}"
        kwargs = {"learning_rate": CANDIDATES[candidate]}
        if agent == "ppo":
            kwargs.update(n_steps=256, batch_size=64)
        else:
            kwargs.update(learning_starts=1000, buffer_size=100_000, batch_size=128)
        cfg = TrainConfig(
            agent=agent, data_source="csv", data_file=str(snapshot), ticker_list=list(tickers),
            start_date=start_date, start_trade_date=fold_start, end_date=fold_end,
            total_timesteps=total_timesteps, validation_days=validation_days,
            eval_freq=max(1, total_timesteps // 4), seed=seed, model_kwargs=kwargs,
            initial_amount=100_000, reward_type="log_return", reward_scaling=1.0,
            run_name=name, results_dir=str(out / "runs"), trained_model_dir=str(out / "models"),
        )
        result = run_training(cfg)
        config_path = Path(result["model_dir"]) / "run_config.json"
        saved = json.loads(config_path.read_text())
        saved["data_sha256"] = manifest["data_sha256"]
        config_path.write_text(json.dumps(saved, indent=2) + "\n")
        validation_records.append({
            "phase": phase, "fold": fold, "agent": agent, "candidate": candidate, "seed": seed,
            "validation_sharpe": saved["best_validation_sharpe"],
            "checkpoint_timesteps": saved["best_validation_timesteps"],
            "eligibility_cutoff": saved["eligibility_cutoff"],
            "model_dir": result["model_dir"],
        })
        for strategy in result["stats"].columns:
            is_agent = strategy == agent.upper()
            if not is_agent and (phase, fold) in baseline_folds:
                continue
            performance.append({
                "phase": phase, "fold": fold, "start": fold_start, "end": fold_end,
                "strategy": strategy, "candidate": candidate if is_agent else "baseline",
                "seed": seed if is_agent else None,
                **{metric: float(result["stats"].loc[metric, strategy]) for metric in METRICS},
            })
        baseline_folds.add((phase, fold))
        pd.DataFrame(validation_records).to_csv(out / "validation_runs.csv", index=False)
        pd.DataFrame(performance).to_csv(out / "performance_runs.csv", index=False)

    ends = fold_starts[1:] + [final_start]
    for fold, (begin, end) in enumerate(zip(fold_starts, ends)):
        for agent in ("ppo", "sac"):
            for candidate in CANDIDATES:
                for seed in seeds:
                    run("development", fold, begin, end, agent, candidate, seed)
    selected = select_candidates(pd.DataFrame(validation_records))
    # Persist selection before any model sees final-period prices in evaluation.
    (out / "selection.json").write_text(json.dumps({
        "selected": selected, "criterion": manifest["selection_metric"],
        "data_sha256": manifest["data_sha256"],
    }, indent=2) + "\n")
    for agent, candidate in selected.items():
        for seed in seeds:
            run("final", 0, final_start, end_date, agent, candidate, seed)
    frame = pd.DataFrame(performance)
    grouped = frame.groupby(["phase", "fold", "strategy", "candidate"], sort=True)[list(METRICS)]
    summary = grouped.agg(["mean", "std", "min", "max"])
    summary.columns = [f"{metric} {stat}" for metric, stat in summary.columns]
    summary.to_csv(out / "summary.csv")
    return {"selected": selected, "summary": summary, "output_dir": str(out)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--start-date", default="2010-01-01")
    parser.add_argument("--fold-starts", nargs="+", default=["2018-01-01", "2020-01-01", "2022-01-01"])
    parser.add_argument("--final-start", default="2024-01-01")
    parser.add_argument("--end-date", default="2025-01-01")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--output-dir", default="results/evaluation")
    options = vars(parser.parse_args(argv))
    options["total_timesteps"] = options.pop("timesteps")
    result = run_experiment(**options)
    print("Selected settings:", result["selected"])
    print(result["summary"].loc["final"].to_string())
    print("Artifacts:", result["output_dir"])


if __name__ == "__main__":
    main()
