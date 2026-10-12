"""Frozen-data, chronological PPO/SAC evaluation: python -m rl.autotrain.experiment."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rl.autotrain.training import (
    TrainConfig, load_run_config, normalize_date, run_backtest, run_training,
)


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


def run_cost_sensitivity(
    experiment_dir, costs=(0.001, 0.0025, 0.005), *, output_dir=None
):
    """Replay the selected final policies under symmetric fees, without retraining.

    This is a diagnostic on an already-opened final period, not a new holdout
    or another policy-selection stage. Policies and normalization stay frozen;
    their actions can respond to the different cash/holdings caused by fees.
    """
    fees = [float(cost) for cost in costs]
    if not fees or len(set(fees)) != len(fees):
        raise ValueError("Provide at least one cost and no duplicate costs")
    if any(not np.isfinite(fee) or not 0 <= fee < 1 for fee in fees):
        raise ValueError("Transaction costs must be finite and in [0, 1)")
    source = Path(experiment_dir).resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    selection = json.loads((source / "selection.json").read_text())
    content = (source / "input_prices.csv").read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != manifest["data_sha256"] or digest != selection["data_sha256"]:
        raise ValueError("Frozen input SHA-256 does not match the experiment and selection")
    validation = pd.read_csv(source / "validation_runs.csv")
    selected = selection["selected"]
    if selected != select_candidates(validation):
        raise ValueError("Saved selection disagrees with development validation scores")
    final = validation[validation.phase == "final"]
    expected = {(agent, seed) for agent in selected for seed in manifest["seeds"]}
    actual = set(zip(final.agent, final.seed))
    if actual != expected or len(final) != len(expected):
        raise ValueError("Final model cohort must contain each selected agent/seed exactly once")
    models, identities = [], []
    for row in final.itertuples(index=False):
        # Use the bundle layout, not machine-specific paths stored in the CSV.
        model_dir = source / "models" / Path(row.model_dir).name
        cfg, saved = load_run_config(model_dir)
        if (
            row.candidate != selected[row.agent]
            or cfg.agent != row.agent or cfg.seed != row.seed
            or cfg.start_trade_date != manifest["final_start"]
            or cfg.end_date != manifest["end_date"]
            or saved.get("data_sha256") != digest
            or saved["tickers"] != manifest["tickers"]
            or (cfg.model_kwargs or {}).get("learning_rate") != manifest["candidates"][row.candidate]
        ):
            raise ValueError("Final model metadata disagrees with its selected cohort: " + str(model_dir))
        artifacts = ["model.zip", "run_config.json"]
        if cfg.normalize_observations:
            artifacts.append("vecnormalize.pkl")
        identities.append({
            "agent": row.agent, "seed": int(row.seed), "candidate": row.candidate,
            "model_dir": str(model_dir),
            "sha256": {
                name: hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
                for name in artifacts
            },
        })
        models.append((row.agent, int(row.seed), row.candidate, model_dir))
    out = (source / "cost_sensitivity" if output_dir is None else Path(output_dir)).resolve()
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "input_prices.csv"
    snapshot.write_bytes(content)
    stress_manifest = {
        "source_experiment": str(source), "data_sha256": digest,
        "selected": selected, "seeds": manifest["seeds"], "models": identities,
        "transaction_costs": fees, "execution_timing": "next_open",
        "final_start": manifest["final_start"], "end_date": manifest["end_date"],
        "policy_selection": "unchanged; diagnostic replay only",
        "limitations": manifest["limitations"],
        "versions": {name: importlib.metadata.version(name) for name in manifest["versions"]},
    }
    (out / "manifest.json").write_text(json.dumps(stress_manifest, indent=2) + "\n")
    torch.set_num_threads(1)
    records = []
    for fee in fees:
        baseline_recorded = False
        for agent, seed, candidate, model_dir in models:
            result = run_backtest(
                str(model_dir), data_source="csv", data_file=str(snapshot),
                buy_cost_pct=fee, sell_cost_pct=fee,
                results_dir=str(out / "runs" / f"fee_{fee!r}" / model_dir.name),
            )
            for strategy in result["stats"].columns:
                is_agent = strategy == agent.upper()
                if not is_agent and baseline_recorded:
                    continue
                records.append({
                    "transaction_cost": fee, "strategy": strategy,
                    "candidate": candidate if is_agent else "baseline",
                    "seed": seed if is_agent else None,
                    "results_dir": result["results_dir"],
                    **{metric: float(result["stats"].loc[metric, strategy]) for metric in METRICS},
                })
            baseline_recorded = True
        pd.DataFrame(records).to_csv(out / "performance_runs.csv", index=False)
    frame = pd.DataFrame(records)
    summary = frame.groupby(["transaction_cost", "strategy"], sort=True)[list(METRICS)].agg(
        ["mean", "std", "min", "max"]
    )
    summary.columns = [f"{metric} {stat}" for metric, stat in summary.columns]
    summary.to_csv(out / "summary.csv")
    return {"selected": selected, "summary": summary, "output_dir": str(out)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-file", help="train a new frozen-data experiment")
    source.add_argument("--experiment-dir", help="replay an existing experiment's final policies")
    training = parser.add_argument_group("training (with --data-file)")
    training.add_argument("--tickers", nargs="+")
    training.add_argument("--start-date")
    training.add_argument("--fold-starts", nargs="+")
    training.add_argument("--final-start")
    training.add_argument("--end-date")
    training.add_argument("--seeds", nargs="+", type=int)
    training.add_argument("--timesteps", type=int)
    training.add_argument("--validation-days", type=int)
    parser.add_argument("--costs", nargs="+", type=float, help="symmetric fee fractions for replay")
    parser.add_argument("--output-dir", help="new output directory; never overwrites prior results")
    options = vars(parser.parse_args(argv))
    experiment_dir = options.pop("experiment_dir")
    data_file = options.pop("data_file")
    costs = options.pop("costs")
    output_dir = options.pop("output_dir")
    if experiment_dir is not None:
        if any(value is not None for value in options.values()):
            parser.error("Training arguments cannot change an existing experiment's cost replay")
        kwargs = {} if costs is None else {"costs": costs}
        result = run_cost_sensitivity(experiment_dir, output_dir=output_dir, **kwargs)
        table = result["summary"]
    else:
        if costs is not None:
            parser.error("--costs requires --experiment-dir; it is not a training selection option")
        if not options["tickers"]:
            parser.error("--tickers is required with --data-file")
        options = {key: value for key, value in options.items() if value is not None}
        options["fold_starts"] = options.get("fold_starts", ["2018-01-01", "2020-01-01", "2022-01-01"])
        options["final_start"] = options.get("final_start", "2024-01-01")
        options["end_date"] = options.get("end_date", "2025-01-01")
        if "timesteps" in options:
            options["total_timesteps"] = options.pop("timesteps")
        if output_dir is not None:
            options["output_dir"] = output_dir
        result = run_experiment(data_file=data_file, **options)
        table = result["summary"].loc["final"]
    print("Selected settings (unchanged during cost replay):", result["selected"])
    print(table.to_string())
    print("Artifacts:", result["output_dir"])


if __name__ == "__main__":
    main()
