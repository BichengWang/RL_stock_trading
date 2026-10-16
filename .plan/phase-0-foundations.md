# Phase 0: foundations and launch (2026-10-15 → 2026-11-05)

Goal: the lab exists, R0 is published, the TradingAgents live clock is running,
and the launch carries the first live numbers.

Critical path: P0.4 ledger → P0.5 TradingAgents live start by **2026-10-30**.
Every day of delay pushes R3.

| ID | Task | Owner | Depends on | Acceptance |
| --- | --- | --- | --- | --- |
| P0.1 | Set the GitHub description, topics and social preview (command in the launch kit) | owner | – | `gh repo view` shows description and 10 topics; preview image set |
| P0.2 | Lab scaffolding: `reports/` with `TEMPLATE.md`, `claims.json` schema, generator for the verdict table into `reports/README.md` and a README block; move `leaderboard/reproductions/` into `reports/r0-wang-zhang-2026/evidence/` | agent | – | `rl-pricing bench reports` renders the table from every `claims.json`; tests cover the verdict rendering and a malformed `claims.json` |
| P0.3 | R0 report: `preregistration.md` (marked post hoc), `claims.json` (return, Sharpe per agent, LSTM drawdown), `report.md` from the existing 75 runs | agent; owner reviews | P0.2 | Every number traces to an evidence file; draft sent to Xinyi Zhang by **2026-10-27** |
| P0.4 | Decision ledger MVP: `rl/bench/ledger.py`, `.github/workflows/ledger.yml` (12:00 UTC on trading days), entries `submissions/<id>/ledger/YYYY-MM-DD.json` holding weights, inputs' hash, model ids, the workflow run id and its start time; scorer that replays entries through the engine; a missed day is recorded as hold | agent | – | A baseline runs through the ledger for 5 sessions on a branch, and its record equals the engine's direct simulation; PRs editing ledger files are rejected by CI |
| P0.5 | TradingAgents adapter and live start: pin the TradingAgents commit and model ids, map BUY/SELL/HOLD to long-only weights (one ticker per sleeve, a third of the equity each), store API keys as Actions secrets, log cost per run, stop the job at a monthly cap | agent; owner supplies keys and the cap | P0.4 | R3 `preregistration.md` committed **before** the first ledger entry; first entry committed before the open on or before **2026-10-30**; cost per day measured |
| P0.6 | Licensed data decision for survivorship-free US equities (candidates: Sharadar via Nasdaq Data Link, Norgate, Tiingo, Databento); confirm the licence allows publishing derived results | agent researches; owner buys | – | Decision and licence excerpt in `decisions.md` by **2026-10-31** |
| P0.7 | Declared hyperparameter sweep for the reference RL agents: about 8 configurations each for PPO and SAC, selected on validation only, `trials` set accordingly | agent | – | New historical records with `trials` > 1 and DSR shown; README findings updated with whatever they show |
| P0.8 | Monthly live report generator (`rl/bench/report.py`) wired into the monthly job: live table, change since last month, figure | agent | – | `leaderboard/reports/2026-11.md` committed by CI on **2026-11-02** |
| P0.9 | README restructured lab-first (verdict table, then findings, then leaderboard); launch kit rewritten for the lab identity | agent; owner approves | P0.2, P0.3, P0.8 | README opens with the verdict table; launch kit updated |
| P0.10 | Launch: outreach to R1 and R2 authors one day before; Show HN, X, Reddit per launch kit | owner | P0.9 | Posts live **2026-11-04**; emails sent 2026-11-03 |

## Exit criteria

- R0 published with verdicts; co-author notified at least 7 days earlier.
- TradingAgents has at least 3 live ledger entries.
- November live report committed by CI.
- Launch done.
