# Decisions

Newest first. Each entry says what would reopen it.

| Date | Decision | Why | Reopen if |
| --- | --- | --- | --- |
| 2026-10-14 | **P0.6 data: Sharadar Prices, Full History ($299/yr, or $39 for one month if a single bulk pull is enough), licensed to the owner as an individual**, pending Sharadar's written confirmation that publishing aggregate backtest outputs (metrics, equity curves) in this public repository is within their Personal terms. Raw or adjusted prices are never committed; Sharadar data never enters public CI. Fallbacks in order: Sharadar via Nasdaq Data Link (quote-only), Norgate Platinum ($630/yr; clearest backtest terms, but not usable in CI), Tiingo Power (terms unclear) | Only candidate with explicit carve-outs for backtest results and summary statistics (§6, §10); history to 1997; the Dow 30 as of 2016-01-01 is a fixed list (28 names + UTX + DD), so no constituents feed is needed. Personal terms also list "publication" as prohibited (§2), hence the confirmation | Sharadar declines, or the UTX/DD history cannot be mapped |
| 2026-10-14 | **Identity: reproducibility lab.** Lead with monthly re-tests of published claims; the leaderboard and live track are instruments | Credibility gap in the field is documented; R0 shows we publish against ourselves; targets bring their own audiences | Two consecutive reports get no engagement |
| 2026-10-14 | **Targets chosen by the maintainer agent** on quality and fit (owner delegated: "find best quality and best fit"). R1 FinRL ensemble, R2/R3 TradingAgents, R4 AlphaMix+ or RD-Agent(Q) | Scored rubric in [`targets.md`](targets.md) | A target's code or data becomes unavailable, or authors request a delay |
| 2026-10-14 | **Authors get the draft 7 days before publication; their reply is printed verbatim** | Builds allies, reduces backlash, catches our errors | — |
| 2026-10-14 | **Budget: LLM APIs and licensed market data** | Needed for TradingAgents (R2, R3) and survivorship-free US equities (R1) | Monthly spend exceeds the cap set in P0.5 |
| 2026-10-14 | **Launch after the 2026-11-02 live report** | Launch carries real live numbers and R0 | — |
| 2026-10-14 | **Keep the code layout**; add `reports/`, ledger and adapters only | No churn for existing users | A package rename becomes necessary for PyPI |
| 2026-10-14 | **Plan lives in `.plan/`, committed** | Public roadmap signals seriousness | — |
| 2026-10-14 | 90-day priorities: left open by the owner; derived from the identity as research credibility first, then the integrity work reports depend on | — | Owner sets priorities |
| 2026-10-14 | Frozen RL agents stored as numpy actors; all 39 freezes re-published together (PR #85) | Replay must survive Stable-Baselines3 upgrades for years | — |
| 2026-10-13 | Live publication is stamped by CI on `main`; scoring starts after it; PRs cannot edit `leaderboard/publications.json` or `leaderboard/live/` | Self-reported freeze times are not evidence | — |
| 2026-10-12 | Domain: trading and portfolio RL, name stays RL-pricing | Owner choice | Search traffic stays negligible after launch |
| 2026-10-12 | Money: consulting and reputation, not proprietary trading | Everything can be public | — |
| 2026-10-12 | Full cutover: 2021 notebooks, `dl/`, legacy envs removed | Focus | — |
| 2026-10-12 | Publish every result, including negative ones | Credibility is the product | — |
| 2026-10-12 | Data: US equities (Yahoo, never redistributed) and crypto (Binance Vision, CC BY-NC-SA 4.0, attributed) | Free and reproducible | Commercial audits need licensed data |
| 2026-10-12 | Anti-overfitting: fixed historical split plus live track | Live results cannot be tuned | — |
