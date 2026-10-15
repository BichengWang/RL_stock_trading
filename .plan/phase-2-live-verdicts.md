# Phase 2: live verdicts (2027-01-01 → 2027-03-31)

Goal: the first prospective verdict on a famous LLM trading agent, a fourth
report, and a site that makes the reports easy to share.

| ID | Task | Owner | Depends on | Acceptance |
| --- | --- | --- | --- | --- |
| P2.1 | Publish R2 (TradingAgents replication and contamination) | agent | P1.2 | Published **2027-01-05** with reply |
| P2.2 | **R3 TradingAgents live verdict**: window from the first ledger entry (target 2026-10-30) to 2027-01-29, about 63 sessions, scored against buy-and-hold of each ticker and the paper's rule-based baselines | agent | P0.5 | Draft to authors **2027-02-08**; published **2027-02-15**; the report states the smallest detectable effect for 63 sessions |
| P2.3 | R4 (AlphaMix+ or RD-Agent(Q), per P1.7) | agent | P1.7 | Draft 2027-03-08; published **2027-03-15** |
| P2.4 | Probability of backtest overfitting (combinatorially purged cross-validation) in `rl/bench/scoring.py`, used in R4 | agent | – | Matches a published worked example; reported where configurations were selected |
| P2.5 | GitHub Pages site: verdict table, one page per report, leaderboard, live charts; built by CI from the repository | agent | P0.2 | Site deploys on every merge; every report has a shareable URL and a social card |
| P2.6 | arXiv v2 of Wang & Zhang 2609.31870 with the R0 findings | owner | P0.3 | v2 posted |
| P2.7 | ai-hedge-fund as a live ledger entrant (not a report); only if the API budget allows | agent | P0.4 | Ledger entries daily; appears in the live table |
| P2.8 | Monthly live reports for January, February and March | CI | P0.8 | Committed on the 2nd of each month |

## Exit criteria

- R2, R3, R4 published on schedule.
- Site live.
- At least two outside citations or write-ups of a report.
