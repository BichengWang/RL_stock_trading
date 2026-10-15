# R0 pre-registration — POST HOC

> **POST HOC.** This document was written on 2026-10-14, *after* the runs it
> describes (benchmark records created 2026-10-13 06:55–07:54 UTC,
> `leaderboard/historical/paper-2021/*.json`, `created`; the evidence file of the
> share-environment runs carries no timestamp) and after the headline numbers had
> been published in the main README. R0 predates the lab protocol
> (`.plan/lab-protocol.md`), so it cannot carry the evidence a real
> pre-registration carries: the git commit time of this file is **not** proof
> that the plan came first. The thresholds below were written with the
> results in view and are best read as a precise description of how the
> verdicts were derived, not as a test that could have embarrassed us before
> the data came in. Reports from R1 on are pre-registered before the first run.

R0 is also the one report where the claims are the authors' own paper: Bicheng
Wang (repository owner) and Xinyi Zhang wrote
*Applying Deep Learning to Stock Trading: LSTM Forecasting and Deep
Reinforcement Learning* (arXiv:2609.31870; local copy
`docs/Wang_Zhang_2021_Deep_Learning_Stock_Trading_arxiv.pdf`, dated June 8,
2021). We test our own claims; Xinyi Zhang gets the same notice as any author
(see `reply.md`).

## What was run

| # | Setup | Level | What | Where the results are |
| --- | --- | --- | --- | --- |
| A | Share env + turbulence rule | L2 | `python main.py --mode train` (the paper's share-trading environment) with next-open fills and 0.1% fees; 20 stocks, train 2000–2018, trade 2019–2020; A2C, PPO, DDPG, TD3, SAC; 80,000 timesteps, default hyperparameters; seeds 0–4 | `evidence/paper-2021-share-env.json`, `runs` with `turbulence_rule: true` (25 runs) |
| B | Share env, no rule | L2 | The same with `--no-turbulence` | same file, `turbulence_rule: false` (25 runs) |
| C | Benchmark task `paper-2021` | L2 | This repository's portfolio-weights environment (`PortfolioEnv`) and engine; same 20 stocks and 2019–2020 window; fee 0.1% per side; A2C, PPO, DDPG, TD3, SAC and the `lstm-forecast` submission, 5 seeds each; validation 2017–2018 chooses the checkpoint; plus 1/N buy and hold and six other baselines | `leaderboard/historical/paper-2021/*.json` (commit `68c03c7f69c6e1ccb0eecaf6ba6d3c25cde6e2c0`) |

No L1 replication was run: the paper's PDF gives neither fill timing nor
transaction costs and our runs fix both (next open, 0.1% per side), so none of
our verdicts rests on L1. The paper's LSTM price-forecasting pipeline
(`rl/forecast/lstm.py`) was not run either; setup C's `lstm-forecast` is a
different strategy (next-session return forecast, equal weight in the assets
expected to rise).

Inputs: share env 105,680 rows, SHA-256
`1d65373352576fa8f30db279f13dd06201bb1c2105054e9adfcbc1cbd96f544f` (row count
equals the paper's); benchmark data SHA-256
`d16bbbd2b2cf0835a66d51029d449b1e62fe01c1221a27bae652dc8ba0df09e8`
(`leaderboard/historical/paper-2021/*.json`, `data.sha256`). Both are Yahoo
Finance daily prices for AAPL, MSFT, AMZN, BRK-B, JPM, JNJ, UNH, HD, PG, NVDA,
DIS, BAC, CMCSA, XOM, VZ, T, ADBE, PFE, CSCO, INTC; no prices are
redistributed. The two files come from different loaders, hence two hashes.

## Claims

Quoted verbatim from the PDF (page numbers are the PDF's). All are made for
the 2019–2020 test set, benchmark S&P 500. Ids match `claims.json`.

| Id | Quote | Source |
| --- | --- | --- |
| `ddpg-annual-return` | "DDPG attains the highest annual return (55.51%)" | Table 4 and Section 5.2, p. 6; abstract: "(55.5%)", p. 1 |
| `ddpg-sharpe` | "DDPG attains the highest annual return (55.51%), cumulative return (142.26%), Sharpe ratio (1.38)" | Table 4 and Section 5.2, p. 6; abstract, p. 1 |
| `td3-sharpe` | "TD3 and SAC offer a better balance between return and risk, with Sharpe ratios of 1.37 and 1.33" | Table 4 and Section 5.2, p. 6 |
| `sac-sharpe` | same sentence | Table 4 and Section 5.2, p. 6 |
| `ppo-sharpe` | Table 4 row "PPO 27.89% 63.71% 1.09 −25.33% 0.05 0.91" | Table 4, p. 6 |
| `a2c-sharpe` | Table 4 row "A2C 20.47% 45.23% 0.82 −31.11% −0.03 1.02" | Table 4, p. 6 |
| `lstm-max-drawdown` | "its Sharpe ratio of 1.19 lies in the middle of the range, and it has the smallest maximum drawdown (−9.57%)" | Table 4 and Section 5.2, p. 6; abstract: "(−9.6%)", p. 1 |
| `lstm-sharpe` | "its Sharpe ratio of 1.19 lies in the middle of the range" | Table 4 and Section 5.2, p. 6 |

Claims in Table 4 not tested: rolling retraining, alpha and beta, leveraged
LSTM, cumulative return, and the Sharpe/return/drawdown of TD3, SAC, PPO and
A2C other than the ones listed (annual return and drawdown of those agents
are shown for context only).

## Metrics, window, baseline

- **Metrics**: annual return (CAGR over the window; the paper's 142.26%
  cumulative over two years is 55.6% a year, so the two definitions agree),
  Sharpe ratio (mean/std × √252, risk-free rate 0), maximum drawdown.
  Setup C metrics come from `rl.bench.scoring.metrics`; setups A and B report
  the same three metrics per run in the evidence file (computed by the
  repository's evaluation code at run time) and the daily account values in
  `evidence/paper-2021-share-env-equity.json` reproduce them to within 5e-5
  (`evidence/analyze.py` asserts this); seed-averaged statistics are computed
  from those series.
- **Window**: 2019-01-02 to 2020-12-31 (setup C: 505 sessions, 505 daily returns; setups A and B: 505 account values, 504 daily returns).
- **Baseline**: the paper's stated benchmark is the S&P 500; no index series
  is in our evidence, so the baseline used is 1/N buy and hold of the same 20
  stocks (A/B: Sharpe 1.1049, annual return 29.66%, `equal_weight_buy_and_hold`
  in the evidence file; C: Sharpe 1.11, 29.9%, `leaderboard/historical/paper-2021/equal-weight-buy-hold.json`). This is a
  deviation from the paper.

## Verdict rule (as applied)

Every value is signed so that larger is better (a −20% drawdown is better
than −30%). Per claim and per setup, over the 5 seeds:

1. **✗ Does not hold**: the sign is opposite, or no single seed reaches the
   claimed value.
2. **✅ Holds**: the claimed value lies inside the 95% Student-t interval of
   the seed mean (4 degrees of freedom) **and** the seed-averaged Sharpe beats
   1/N with one-sided paired stationary-bootstrap p < 0.05
   (`rl.bench.scoring.sharpe_difference`, as in `rl.bench.leaderboard`).
3. **◐ Partly holds**: otherwise, i.e. a seed reaches the claim but the
   interval excludes it or significance is absent or untestable.
4. **? Inconclusive**: the setup that ran is not the paper's pipeline.

The protocol's "magnitude shrinks by more than half" clause is not used: no
claim fell in the range where it would change a verdict.

**Primary setup.** The verdict in `claims.json` for the five agents is the
one from setup A (share env + turbulence rule): it is the default of the
authors' own training script and the setup most favourable to the claims
(its seed-mean Sharpe is at or above setup B's for every agent). Setup B and C
verdicts are listed in `report.md`. In the first version of this document the
share-env evidence held per-run summary metrics only, so rule 2's significance
clause could not be evaluated for A and B. The daily account values
(`evidence/paper-2021-share-env-equity.json`) were added later on 2026-10-14;
rule 2 is now evaluated in every setup, and no verdict changed. Setup C has the return series, but its
environment differs from the paper's, so its verdicts are a robustness check.
The LSTM claims are tested only in setup C and get rule 4.

## Seeds, costs, trials

5 seeds per learned policy (0–4); fee 0.1% per side; fills at the next open.
Configurations tried: one per agent in each setup (`trials: 1` in setup C); the
authors' own count is unknown, and the paper says DDPG was "individually
tuned" (Section 5.2), so a deflated Sharpe against their trials cannot be
computed.

## Smallest detectable effect

Over these 505 sessions the bootstrap standard error of the annualised Sharpe
ratio of 1/N is 0.69 (stationary bootstrap, 2,000 resamples, mean block 8;
Lo (2002) formula: 0.90). A one-sided 5% test of a single Sharpe ratio would
need about 1.13. The paper's claimed Sharpe edge over 1/N (1.38 − 1.10) is
0.28, well below that, so the 2019–2020 window cannot confirm or reject a
Sharpe advantage of that size. It can reject a point value that no seed reaches
(rule 1), and that is the only way a claim is rejected here. See
`evidence/summary.json`, key `detectability`.

## Planned analyses

Seed-averaged intervals and per-seed tables for all three setups; paired
bootstrap against 1/N in setup C; verdicts by setup. The turbulence-rule
comparison (A against B) was not planned in advance; it is reported as an
observation, not tested. Everything in
`evidence/analyze.py` is deterministic and rerunnable.
