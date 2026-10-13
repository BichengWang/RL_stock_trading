# RL-pricing: deep learning for stock trading

A research toolkit for training and backtesting multi-stock trading strategies.
Compare deep reinforcement learning agents (A2C, PPO, DDPG, TD3 and SAC), a
rolling ensemble, and an LSTM forecast-driven strategy against rule-based
portfolios. The current Python CLI supports Yahoo Finance, local CSV files and
synthetic data, with saved models, run settings and performance reports.
Active strategies use close-time signals and next-session open fills; ticker
eligibility is fitted on past history and frozen for held-out evaluation.

**Start with the offline example below** to learn the workflow without fetching
market data. Synthetic returns demonstrate the pipeline, not investment skill.
The 2021 paper and saved notebook results are historical research artifacts;
they are not evidence that the current code beats portfolio managers or will
deliver similar future returns. This is a backtesting project, not a live
broker-execution system or investment advice.

## Navigation

- [Setup](#setup) → [offline quick start](#quick-start-offline) → [outputs](#outputs)
- [Real market data and CSV input](#real-market-data)
- [Validation](#choosing-the-checkpoint-on-validation-data) · [saved-model backtests](#backtesting-a-saved-model)
- [Portfolio allocation](#portfolio-allocation) · [LSTM](#lstm-forecasting-strategy) · [ensemble](#ensemble-strategy)
- [Frozen-data, multi-seed evaluation](#frozen-data-multi-seed-evaluation)
- [CLI options](#options) · [Python API](#python-api) · [environment mechanics](#the-trading-environment)
- [Changes since 2021](#changes-from-the-2021-version) · [research and limitations](#research-and-limitations)
- [Tests](#tests) · [pull requests](#pull-requests) · [license](#license)

## Repository layout

| Path | Contents |
| --- | --- |
| `main.py` | Command-line entry point (`--mode train`, `backtest`, `ensemble`, `lstm`, `download_data`; `--task trading` or `portfolio`) |
| `rl/` | Trading library adapted from [FinRL](https://github.com/AI4Finance-Foundation/FinRL) |
| `rl/marketdata/` | Yahoo Finance downloader and a synthetic price generator for offline runs |
| `rl/preprocessing/` | Panel cleaning, technical indicators, turbulence index, covariance features |
| `rl/env/` | Trading environments (gymnasium API) |
| `rl/forecast/lstm.py` | LSTM price forecaster (PyTorch) and the forecast-driven trading rule |
| `rl/model/models.py` | DRL agents (A2C, PPO, DDPG, SAC, TD3) and the rolling ensemble |
| `rl/trade/` | Performance metrics, rule-based baselines and backtest plots |
| `rl/autotrain/training.py` | End-to-end train / backtest / ensemble pipelines (`TrainConfig`) |
| `rl/autotrain/experiment.py` | Frozen-CSV, multi-seed PPO/SAC evaluation with validation-only selection |
| `rl/config/config.py` | Default dates, ticker lists, technical indicators and agent hyperparameters |
| `tests/` | Offline test suite (synthetic data, no network needed) |
| `dl/` | Original TensorFlow LSTM code used by the notebooks |
| [`rl_portfolio_trading.ipynb`](rl_portfolio_trading.ipynb) | Historical 2021 experiment and saved outputs |
| [`rl_portfolio_trading_phase2.ipynb`](rl_portfolio_trading_phase2.ipynb) | Phase 2 experiment |
| `notebooks/`, `*.ipynb` | Other experiments |
| `img/` | Backtest and performance figures |
| `docs/` | Paper copy |

## Setup

Python 3.10 or newer. Run these commands from a terminal:

```bash
git clone https://github.com/BichengWang/RL-pricing.git
cd RL-pricing
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, activate with `.venv\Scripts\activate` instead. PyTorch is installed
as a dependency; use its platform-specific installation instructions if you need
a particular CPU/CUDA build. The virtual environment activation shown above is
for macOS/Linux.

[`requirements.in`](requirements.in) declares the dependency constraints used by
[`setup.py`](setup.py); [`requirements.txt`](requirements.txt) provides a pinned
dependency snapshot. The installed `rl-pricing` command accepts the same options
as `python main.py`. The legacy notebooks and `dl/` code use a different,
2021-era stack; the editable install is intended for the current `rl/` pipeline.

## Usage

### Quick start (offline)

After installing dependencies, the synthetic data source generates correlated
random-walk prices, so this training-and-backtest run needs no network access.
The small training budget is for learning the workflow, not tuning a strategy:

```bash
python main.py --mode train --data-source synthetic --tickers AAA BBB CCC DDD \
    --start-date 2015-01-01 --start-trade-date 2019-01-01 --end-date 2020-01-01 \
    --agent ppo --timesteps 2000 --seed 0 --run-name offline-ppo
```

Training uses dates before 2019; the saved policy then trades the held-out
2019 period using close signals and next-open fills. Inspect
`results/offline-ppo/perf_stats.csv`, `results/offline-ppo/executions.csv` and
`results/offline-ppo/backtest.png`, and keep `trained_models/offline-ppo/`
for a later backtest:

```bash
python main.py --mode backtest --model-dir trained_models/offline-ppo
```

Replay uses the saved data source, dates and frozen tickers, and writes to a
new `results/offline-ppo_backtest_<timestamp>/` directory. Use a different
`--run-name` for each training experiment to avoid overwriting its artifacts.
A seed controls randomness, but does not guarantee identical results across
dependency versions or hardware.

### Real market data

```bash
# Train SAC on the Dow 30 (2000-2018) and trade 2019-2020, compared with the
# Dow Jones index as well as the built-in baselines
python main.py --mode train --agent sac --timesteps 80000 --benchmark ^DJI

# Save prices once, then train from the file (repeatable, no re-download)
python main.py --mode download_data --tickers AAPL MSFT JPM --output datasets/prices.csv
python main.py --mode train --data-file datasets/prices.csv --agent a2c
```

Yahoo downloads (including `--benchmark`) require network access and available
symbols. For your own CSV, use one row per ticker and date with columns
`date,tic,open,high,low,close,volume`; dates should be `YYYY-MM-DD`. Supplying
`--data-file` selects the CSV source automatically, and without `--tickers`
the CSV pipeline considers all tickers in the file for fitting-history
eligibility. Ensure the file covers both training and trading dates; data
outside `[--start-date, --end-date)` are ignored. Open and close prices must
be finite, positive and on the same adjustment basis.

### Choosing the checkpoint on validation data

RL training is noisy, and the model at the end of training is often not the
best one. With `--validation-days N` the last N days of the training period
are held out: every `--eval-freq` steps the agent trades them, and the
checkpoint with the best Sharpe ratio there is kept and used for trading:

```bash
python main.py --mode train --agent ppo --timesteps 200000 \
    --validation-days 252 --eval-freq 10000
```

The test period is never used for this choice. Each evaluation is listed in
`results/<run>/validation.csv`, and `run_config.json` records which checkpoint
was chosen.

Ticker eligibility is decided from fitting history only, before validation.
The sorted universe is then frozen for validation, trading and saved-model
replay. Missing or invalid open/close prices in these held-out panels raise
a ticker/date error instead of silently changing the universe or dropping
sessions. The split uses observed raw session dates; it does not infer an
exchange calendar for sessions missing from every ticker.

### Backtesting a saved model

Single-agent training saves its model, settings and, when enabled,
observation-normalisation statistics, so it can be re-run on the same or a
different period. Substitute the directory produced by your own training run:

```bash
python main.py --mode backtest --model-dir trained_models/<run-name> \
    --start-trade-date 2020-01-01 --end-date 2021-01-01
```
Artifacts must record `execution_timing: "next_open"`; older same-close models
must be retrained, even if their stable-baselines3 version is otherwise compatible.

Omitting `--transaction-cost` preserves the checkpoint's saved buy/sell fees.
An explicit value overrides both, including `0` and `0.001`:

```bash
python main.py --mode backtest --model-dir trained_models/<run-name> \
    --transaction-cost 0.005
```

Fees must be finite fractions in `[0, 1)`. Backtests write their effective
settings to `results/<run>/run_config.json`; the saved checkpoint settings
remain unchanged.

Replay preserves saved settings for omitted options. Explicitly supplied
options override them even when they match the training CLI defaults; for
example, `--leverage 1.0` uses unleveraged targets for an LSTM checkpoint.


### Portfolio allocation

`--task portfolio` swaps share trading for portfolio allocation: every day the
agent picks long-only weights at the close, rebalances at the next session's
open, and values its positions at that session's close. Existing holdings keep
their overnight returns, and transaction costs are paid on actual turnover.
Its observation is the return covariance matrix of the last `--cov-lookback`
days stacked on technical indicators. Covariance warm-up must leave at least
two fitting sessions; it cannot consume validation or trading sessions.

```bash
python main.py --mode train --task portfolio --agent ppo --timesteps 50000
python main.py --mode train --task portfolio --data-source synthetic \
    --tickers AAA BBB CCC DDD --start-date 2015-01-01 --start-trade-date 2019-01-01 \
    --end-date 2020-01-01 --agent ppo --timesteps 20000 --seed 0
```

`actions.csv` holds the target weights chosen at each signal close instead of
share counts; `executions.csv` pairs each signal with its next-session fill. A saved
portfolio model is backtested like any other (`--mode backtest --model-dir
...`). `--hmax` and the turbulence options do not apply to this task, and the
ensemble supports only `--task trading`.

### LSTM forecasting strategy

The paper's other approach: instead of learning actions end to end, a
two-layer LSTM forecasts every ticker's next close. At each signal close,
the rule targets `1/N` of the account, times `--leverage`, for each stock whose
forecast is at or above that close, and keeps the rest in cash. These targets
fill at the next open, not at the price used to make the forecast. It is
benchmarked against the same baselines as the agents:

```bash
python main.py --mode lstm --lstm-epochs 30
python main.py --mode lstm --data-source synthetic --tickers AAA BBB CCC DDD \
    --start-date 2015-01-01 --start-trade-date 2019-01-01 --end-date 2020-01-01 --seed 0

# The paper's leveraged variant, from the same saved model
python main.py --mode backtest --model-dir trained_models/<run-name> --leverage 2
```

As in the paper, one model is trained on all tickers together with a mean
absolute percentage error (MAPE) loss (`--lstm-loss mse` is the alternative),
so it fits low- and high-price periods equally. The inputs are the last
`--lstm-window` days of each ticker's log returns and technical indicators,
made scale free (moving averages and bands as ratios to the close; other
indicators standardised with fitting-period statistics). The most recent
10% of raw training sessions are held out, and the epoch with the lowest
validation loss is kept. Scaling excludes validation, and a fitting signal
whose next-close label lies in validation is excluded from fitting.

The run also writes `predictions.csv` (each day's close, forecast, next close
and signal), `forecast_metrics.csv` (MAPE and directional accuracy over the
trading period) and `training_history.csv` (losses per epoch); `actions.csv`
holds fillable signal-date weights, `executions.csv` records signal/fill dates,
and the model is saved as `model.pt`. The last forecast has no subsequent
session within the requested period and therefore produces no trade.

### Ensemble strategy

The ensemble walks forward through the trading period. Every
`--rebalance-window` days it trains A2C, PPO and DDPG, picks the one with the
best Sharpe ratio on the preceding `--validation-window` days, retrains it and
trades the next window, carrying positions over:

```bash
python main.py --mode ensemble --timesteps 20000 --rebalance-window 63 --validation-window 63
```

Validation precedes the first requested trading date. Each rebalance window
owns its price transitions, sharing only its boundary close with the next
window. The final partial window is included. `ensemble_summary.csv` records
validation dates, inclusive signal/valuation endpoints and transition counts;
`results/<run>/run_config.json` records the frozen universe and eligibility cutoff.

### Frozen-data, multi-seed evaluation

Use a fixed CSV rather than re-downloading between agents or seeds:

```bash
python main.py --mode download_data --tickers SPY TLT GLD \
    --start-date 2010-01-01 --end-date 2025-01-01 \
    --output datasets/evaluation_prices.csv
python -m rl.autotrain.experiment --data-file datasets/evaluation_prices.csv \
    --tickers SPY TLT GLD --fold-starts 2018-01-01 2020-01-01 2022-01-01 \
    --final-start 2024-01-01 --end-date 2025-01-01 --seeds 0 1 2 \
    --timesteps 20000 --validation-days 126 --output-dir results/evaluation
```

This runs PPO and SAC with learning rates `3e-4` and `1e-3` on three expanding
development folds and three training seeds. Each fit holds out its preceding
126 sessions for checkpoint selection. Mean development **validation** Sharpe
selects a learning rate per algorithm; development trading returns do not
select settings. Selection is written before evaluating the untouched final
2024 test period. A candidate with undefined validation Sharpe on any run
cannot win. This is a small comparison, not proof of statistical superiority.

The output directory must be new. It contains the exact `input_prices.csv`,
a SHA-256/version/settings `manifest.json`, `selection.json`,
`validation_runs.csv`, `performance_runs.csv`, per-fold/seed variability in
`summary.csv`, and each run's models and normal backtest artifacts. Baselines
are deterministic and recorded once per fold. Preserve the copied input and
manifest: later Yahoo downloads may revise adjusted historical prices.

The explicit ETF universe is not a point-in-time constituent history. Prices
use a consistent adjusted OHLC basis, modeling reinvested distributions rather
than raw exchange fills. Spread, impact and cash/borrowing interest are not
modeled. Training seeds measure policy variability, not independent market
samples. Synthetic data verifies the pipeline, not profitability.

The recorded reference run used `results/evaluation_2010_2024_next_open/`:
36 development runs and six final runs, with 20,000 requested steps per fit.
Validation selected PPO at `3e-4` and SAC at `1e-3`. Its frozen input SHA-256 is
`c580fdbaa445c1731ba2cfb07caffa7f73fdaa9cffcf8a812f6551800082737e`.
That run's `summary.csv` records exact returns, Sharpe, drawdowns and seed
dispersion. Generated `results/` directories are ignored by Git and are not
included in a fresh checkout; the path above is an artifact location, not a
bundled report.
On the final year, PPO's seed-average return and Sharpe exceeded the baselines,
but its drawdown was larger; SAC's mean return was lower and varied more by seed.
These are simulation results for this snapshot, not a general alpha claim.

### Transaction-cost sensitivity of frozen policies

Replay the selected final PPO/SAC checkpoints without retraining:

```bash
python -m rl.autotrain.experiment \
    --experiment-dir results/evaluation_2010_2024_next_open \
    --costs 0.001 0.0025 0.005 \
    --output-dir results/cost_sensitivity_2024
```

The costs are 10, 25 and 50 basis points **per buy and per sell**, applied
equally to the agents and both baselines. Omit `--costs` for these defaults.
The replay verifies the frozen input SHA-256, development-validation selection
and complete final agent/seed cohort before creating its output directory.
Keep the original experiment's `models/`, copied input and metadata together:
the bundle can be moved without relying on its old CSV/model paths.

The output directory must be new; its default is `<experiment-dir>/cost_sensitivity`.
It contains the copied input, a manifest with checkpoint/config/normalizer
hashes and replay software versions, per-fee/seed backtest artifacts and effective
settings, `performance_runs.csv`, and seed mean/std/min/max in `summary.csv`.
Each baseline is recorded once per fee. Duplicate, non-finite or out-of-range
costs are rejected.

Policies and observation-normalisation statistics are frozen, but their actions
can change as higher costs alter cash and holdings; this is policy replay,
not subtraction of estimated fees from an existing equity curve. Realized
returns therefore need not be monotonic in the fee. These are **post-test
diagnostics**, not a new untouched holdout or another checkpoint-selection round.
Training options cannot be mixed with `--experiment-dir`. Spread, market impact
and financing costs remain unmodeled.

The completed reference replay is in `results/cost_sensitivity_2024/`: 18
policy replays plus two baselines per fee. At 10 bps it reproduces the original
final-period results. PPO's mean-return lead persists at 50 bps, with larger
drawdowns than buy-and-hold; SAC's mean return remains below buy-and-hold.

### Options

Run `python main.py --help` for the full list. The most useful ones:

| Option | Default | Meaning |
| --- | --- | --- |
| `--task` | `trading` | `trading` (buy and sell shares) or `portfolio` (daily portfolio weights) |
| `--agent` | `sac` | `a2c`, `ppo`, `ddpg`, `td3` or `sac` |
| `--timesteps` | 80000 | training steps (per model for the ensemble) |
| `--data-source` / `--data-file` | `yahoo` | `yahoo`, `csv` or `synthetic` |
| `--tickers` | Dow 30 | tickers to trade |
| `--start-date`, `--start-trade-date`, `--end-date` | 2000-01-01, 2019-01-01, 2021-01-01 | training period is `[start, start-trade)`, trading period `[start-trade, end)` |
| `--seed` | none | controls random seeds; exact reproducibility depends on the software and hardware |
| `--transaction-cost` | training: 0.001; backtest: saved setting | finite fraction in `[0, 1)` paid on every buy and sell; explicit backtest values override both saved fees |
| `--hmax` | 100 | maximum shares traded per stock per day |
| `--reward-type` | `asset_change` | or `log_return` |
| `--cov-lookback` | 252 | portfolio task: days of returns in each covariance matrix |
| `--turbulence-quantile` / `--turbulence-threshold` / `--no-turbulence` | 0.99 quantile | when to liquidate during market turmoil |
| `--no-normalize` | off | feed raw observations to the agent |
| `--validation-days` / `--eval-freq` | 0 / 10000 | hold out the last N training days and keep the best checkpoint on them |
| `--lstm-window`, `--lstm-epochs`, `--lstm-hidden-size`, `--lstm-loss` | 20, 30, 32, `mape` | LSTM strategy settings |
| `--leverage` | 1 | LSTM strategy: fraction of the account invested when every stock is forecast to rise |
| `--benchmark` | none | Yahoo ticker to add to the comparison, e.g. `^DJI` or `SPY` |

### Outputs

A run named `<run>` (default: timestamp and agent) writes:

| Path | Contents |
| --- | --- |
| `trained_models/<run>/model.zip` | the trained stable-baselines3 model (the best validation checkpoint with `--validation-days`) |
| `trained_models/<run>/vecnormalize.pkl` | observation-normalisation statistics, when normalisation is enabled |
| `trained_models/<run>/run_config.json` | every setting, frozen tickers, eligibility cutoff, execution timing and turbulence threshold |
| `results/<run>/run_config.json` | saved-model backtest: effective settings and source checkpoint directory, including any overridden fees |
| `results/<run>/account_value.csv` | daily account value over the trading period |
| `results/<run>/actions.csv` | signal-date-indexed signed executed shares; target weights for portfolio/LSTM |
| `results/<run>/executions.csv` | signal date, next-session fill date and signed shares or target weights |
| `results/<run>/perf_stats.csv` | agent, baselines and benchmark side by side |
| `results/<run>/backtest.png` | cumulative return and drawdown chart |
| `results/<run>/validation.csv` | with `--validation-days`: validation Sharpe and return at each evaluation |

The statistics are those of pyfolio's `perf_stats` (annual return, Sharpe,
Sortino, Calmar, max drawdown, ...) plus the final account value. The agent is
always compared with an equal-weight buy-and-hold portfolio and an equal-weight
portfolio rebalanced monthly. Both start in cash on the first valuation date,
buy at the next open and include transaction costs in reported returns.
A `--benchmark` index is a hypothetical buy-and-hold account with the same
timing, starting capital and buy cost, making its value comparable with the agent's.

### Python API

```python
from rl.autotrain.training import TrainConfig, run_training, run_backtest

result = run_training(TrainConfig(agent="ppo", total_timesteps=50_000, data_source="synthetic",
                                  ticker_list=["AAA", "BBB", "CCC"], seed=0))
print(result["stats"])
```

Core building-block names remain (`YahooDownloader`, `FeatureEngineer`,
`data_split`, `StockTradingEnv`, `DRLAgent`, `backtest_stats`, `backtest_plot`).
Active environments and baselines require finite positive `open` and `close`
prices on the same adjustment basis. Notebook outputs and the legacy
cash-penalty/stop-loss experiments are historical, not results of this workflow.

## The trading environment

`rl.env.env_stocktrading.StockTradingEnv` holds cash and whole shares. Each day
the agent observes the close and indicators, then outputs a value in [-1, 1]
per stock, scaled by `hmax` into shares to sell (negative) or buy (positive).
Orders fill at the next session's open. Sells execute first, then buys from
the largest order down, limited by cash **including** transaction costs.
Signal-day turbulence at or above the threshold schedules liquidation at the
next open and blocks buying; fill-day indicators do not influence that decision.
Prior holdings retain overnight P&L. The account is marked at each close,
including the final session. Observation layout stays
`[cash, prices, holdings, indicators]`, standardised by running statistics.

`rl.env.env_portfolio.StockPortfolioEnv` (used by `--task portfolio`) holds a
fractional, long-only portfolio. The agent outputs one score in [0, 1] per
stock and a softmax turns the scores into weights, so no stock gets more than
about e times the weight of another: the agent tilts away from an equal-weight
portfolio rather than concentrating in a few names. Between rebalances the
weights drift with prices, including the overnight move before rebalancing.
Fees and post-fee target notionals are solved together so cash plus holdings
remains self-financing. The account starts in cash; its first allocation pays
fees at the next open, just like the baselines' initial purchase.

## Changes from the 2021 version

- **Causal evaluation.** Eligibility is fitted before validation and frozen
  thereafter; incomplete held-out panels fail explicitly. LSTM scaling and
  fitting labels exclude validation outcomes.
- **Next-open execution.** Active agents, forecasting strategy, baselines and
  optional benchmark use close signals and next-open fills. Yahoo OHLC now
  shares one adjustment basis. Execution CSVs distinguish signal/fill dates.
- **Complete rolling accounting.** The ensemble includes its partial final
  window and preserves every return and fill at window boundaries.
- **Reproducible comparison.** Frozen-data PPO/SAC evaluation records settings,
  software versions, input identity, validation-only choices and seed variability.
- **Modern dependencies.** Runs on current Python, pandas, NumPy, gymnasium and
  stable-baselines3 2.x. Environments use the gymnasium API: `reset()` returns
  `(obs, info)` and `step()` returns `(obs, reward, terminated, truncated, info)`.
  pyfolio is no longer needed; `rl/trade/metrics.py` computes the same
  statistics.
- **Fixed train mode**, which crashed calling `DRL_prediction` with the wrong
  arguments.
- **Environment accounting fixes.** Buys could spend more cash than available
  (costs were ignored when sizing orders); forced turbulence sales recorded no
  transaction cost; failed buys were counted as trades.
- **Causal indicator filling.** Missing indicators used to be back-filled across
  the whole frame, pulling values from later rows and other tickers. Each ticker
  is now forward-filled from its own past only. Universe coverage is fitted
  only on fitting history, not on the full train/test panel.
- **Deterministic evaluation.** Trading uses the policy's deterministic action
  (it previously sampled random actions).
- **Hyperparameter dicts are not modified** when building a model (reusing a
  dict with `"action_noise": "normal"` for a second model used to fail).
- **Ensemble.** The turbulence lookback is 63 trading days rather than 63 data
  rows (about two days with 30 tickers), validation Sharpe ratios are
  annualised with sqrt(252), and the stitched account values are returned for
  benchmarking.
- **Portfolio environment** charges its transaction costs and rewards the
  change in portfolio value instead of its level.
- **New:** command-line options, backtest and ensemble modes, CSV and synthetic
  data sources, rule-based baselines, saved run configurations, and an offline
  test suite with CI.

Models without `execution_timing: "next_open"` cannot be replayed; retrain
them under the causal execution contract. The original 2021 models also used
stable-baselines3 0.11. Archived notebooks retain their historical outputs.

## Research and limitations

- **Paper:** Bicheng Wang and Xinyi Zhang, *Applying Deep Learning to Stock
  Trading: LSTM Forecasting and Deep Reinforcement Learning* (June 8, 2021).
  Read the [local PDF](docs/Wang_Zhang_2021_Deep_Learning_Stock_Trading_arxiv.pdf)
  or the [original shared document](https://drive.google.com/file/d/1-ogSAAA5L0sdy1iLt8HQKwV2kcN1UuCc/view?usp=sharing).
- **Presentation:** [YouTube explanation](https://www.youtube.com/watch?v=bE8MFq4sB2k).
- **Historical results:** the paper evaluates a selected 20-stock universe,
  training on 2000–2018 and backtesting on 2019–2020. The CLI's default Dow 30
  universe and modernised implementation are not an exact reproduction of
  that experiment. Consult the paper for reported results and methodology,
  rather than treating them as current benchmark guarantees.
- **Execution assumptions:** active strategies use daily close signals,
  next-open fills and proportional transaction costs on a consistent adjusted
  OHLC basis. Adjusted prices model reinvested distributions, not attainable
  raw exchange fills. Spread, market impact and cash/borrowing interest are
  not modeled, including financing costs for leveraged LSTM strategies.
- **Data and evaluation:** freezing eligibility from fitting history prevents
  held-out prices from changing the universe, but does not make a retrospective
  ticker list a point-in-time constituent history or eliminate selection bias.
  A fixed historical test period, validation-based model selection and
  synthetic tests cannot establish performance in future market regimes.
  Training seeds measure policy variability, not independent market samples.
  Retain frozen CSV inputs and manifests: Yahoo availability and historical
  price revisions can change a later download.

## Tests

The tests use synthetic data and offline fixtures, so they need no network:

```bash
pip install -e ".[test]"
python -m pytest
```

CI also lints the code (pyflakes-style checks, configured in `ruff.toml`):

```bash
pip install -e ".[lint]"
ruff check .
```

They cover accounting with overnight/intraday gaps, exact fee-adjusted
allocation, causal universe selection and features, validation/LSTM label
boundaries, complete ensemble transitions, metrics and baselines, Yahoo's
adjusted OHLC handling, saved-model replay and short offline runs of all
active pipelines, including successful completion of the installed CLI.
Experiment-selection tests exclude final-period scores.

CI also builds a wheel from the source archive, verifies dependency metadata
and checks the installed command outside the checkout. To build both artifacts
locally:

```bash
python -m pip install build
python -m build
```

The artifacts are written to `dist/`. The source archive includes the dependency
declarations and test support files needed to build and verify it independently.

## Pull requests

Use `type(scope): title` for PR titles, for example
`feat(workspace): add workspace switching` or
`docs(repo): add pull request guidelines`. Choose the type and scope that
describe the change.

The [PR template](.github/pull_request_template.md) asks for a short summary
and verification results. Review the diff and wait for CI to pass before merging.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
The trading environment, preprocessing and training code are adapted from the
MIT-licensed [FinRL](https://github.com/AI4Finance-Foundation/FinRL) library.
