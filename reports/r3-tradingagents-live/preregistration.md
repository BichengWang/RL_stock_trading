# R3 pre-registration: TradingAgents, three months live

Written 2026-10-14, before the first ledger decision. The git commit time of
this file is the evidence that the plan came first; the first entry under
`submissions/tradingagents/ledger/` must be committed after it. This report
follows `.plan/lab-protocol.md` (levels, pre-registration, verdict table,
contamination rules, author notice). It is level **L3**: the agent runs
prospectively on the live decision ledger. The historical replication of the
paper's 2024 window is a separate report (R2); nothing here rests on it.

## 1. Question

The TradingAgents paper reports, for three stocks over 2024-01-01 to
2024-03-29, returns, Sharpe ratios and drawdowns far above buy and hold. The
authors' stated next step is "deploying the framework in a live trading
environment". We deploy the released code, before every US open, for the same
three stocks and the same length of time (about three months), and ask whether
the claimed pattern shows up on data nobody could have trained on.

## 2. Facts about the agent and the paper (read 2026-10-14)

| Item | Finding | Source |
| --- | --- | --- |
| Paper | Xiao, Sun, Luo, Wang, *TradingAgents: Multi-Agents LLM Financial Trading Framework*, arXiv 2412.20138, v7 dated 2025-06-03 (the version quoted here) | <https://arxiv.org/abs/2412.20138>, <https://arxiv.org/pdf/2412.20138v7> |
| Code, licence | Apache-2.0; 111,203 stars (GitHub API, 2026-10-14) | <https://github.com/TauricResearch/TradingAgents> |
| Latest release | **v0.6.0**, published 2026-10-03, tag commit `1394a3f72aa4393e1a98f51b382434c4b4c2d972` (merge of PR #1478). Pinned. Not on PyPI (`pypi.org/pypi/tradingagents/json` returns 404), so the `agents` extra installs that commit from GitHub. Requires Python >= 3.11 and pandas >= 3 | <https://github.com/TauricResearch/TradingAgents/releases/tag/v0.6.0>, `pyproject.toml` at the tag |
| Python API | `TradingAgentsGraph(selected_analysts=("market","social","news","fundamentals"), debug=False, config=dict, callbacks=list)`, then `propagate(company_name, trade_date, asset_type="stock", portfolio=None)` returning `(final_state, signal)`. `trade_date` must be `YYYY-MM-DD` and not in the future. `signal` is one of `Buy`, `Overweight`, `Hold`, `Underweight`, `Sell`, or `REVIEW` when the decision text has no parsable rating. `final_state["final_trade_decision"]` is the Portfolio Manager's text | `README.md` ("Python Usage"), `tradingagents/graph/trading_graph.py`, `tradingagents/agents/rating.py` at the tag |
| Config keys used | `llm_provider`, `deep_think_llm`, `quick_think_llm`, `max_debate_rounds` (default 1), `max_risk_discuss_rounds` (1), `max_tool_rounds` (20), `data_vendors` (`core_stock_apis` and `technical_indicators`: yfinance; `fundamental_data`: `sec_edgar,yfinance`; `news_data`: yfinance; `macro_data`: fred; `prediction_markets`: polymarket), provider reasoning knobs such as `openai_reasoning_effort`, `temperature`, `results_dir`, `data_cache_dir`, `memory_log_path`. Release defaults: provider `openai`, deep `gpt-6-sol`, quick `gpt-6-luna`. The quick tier serves the analysts, researchers, debaters and trader; the deep tier serves the Research Manager and Portfolio Manager | `tradingagents/default_config.py`, `README.md` at the tag |
| Keys and secrets | `OPENAI_API_KEY` (any LLM provider needs its own key). `FRED_API_KEY` (free; without it the macro tool reports the vendor unavailable and the run continues). SEC EDGAR needs no key but asks for a contact in `SEC_EDGAR_USER_AGENT`. yfinance, Polymarket, StockTwits and Reddit are keyless | `.env.example`, `README.md`, `dataflows/vendors/fred.py`, `sec_edgar.py` |
| State between runs | The memory log (`memory_log_path`) records each decision, settles it against later prices with an LLM reflection, and feeds lessons to later same-ticker runs | `README.md` ("Memory log"), `tradingagents/memory/` |
| Reproducibility warning from the authors | Runs are not reproducible at fixed settings; reasoning models ignore temperature; "Backtest results are not guaranteed to match any published figure" | `README.md` ("Reproducibility") |
| Paper's models | Quick: gpt-4o-mini and gpt-4o; deep: o1-preview. "all analyst nodes rely on deep-thinking models" (this differs from v0.6.0, where analysts use the quick tier) | paper §4.3, p. 9 |
| Paper's window and assets | "January 1st to March 29th, 2024", text names Apple, Nvidia, Microsoft, Meta, Google; Table 1 reports AAPL, GOOGL, AMZN only. 2024-03-29 was Good Friday (market closed), so the last session is 2024-03-28 | paper §5.1, p. 10; Table 1, p. 11 |
| Short positions | Not specified in the text, which says only "buy, sell, or hold". The captions of Fig. 6, S1 and S3 call the green and red arrows "Long / Short positions" | paper §5.1; Fig. 6 p. 11; Fig. S1, S3 pp. 20-21 |
| Transaction costs, slippage, position size, initial capital, risk-free rate value | Not stated anywhere in the paper | paper §5, §S1.2 |
| Baselines | Buy and Hold ("equal amounts in all selected stocks, holding them throughout"), MACD, KDJ and RSI, ZMR ("Zero Mean Reversion"), SMA, each described in one sentence with no parameters. No baseline code is in the v0.6.0 repository (searched for KDJ, ZMR, MACDStrategy) | paper §S1.1, p. 19 |
| Cost per prediction | "11 LLM calls & 20+ tool calls/prediction"; the authors tested 3 months "due to intensive LLM and tool use" | paper footnote on p. 12 |
| Authors' own caveat on Sharpe | "The highest Sharpe Ratio exceeds our expected empirical range ... We believe the exceptionally high SR resulted from the phenomenon that there were few pullbacks in TradingAgents during that period." | paper footnote, p. 12 |
| Model prices | gpt-6-sol $2.00 in / $0.20 cached / $10.00 out; gpt-6-luna $0.10 / $0.01 / $0.50, per million tokens, standard tier, <= 272K context | <https://developers.openai.com/api/docs/pricing> |

## 3. Claims, quoted from the paper

Table 1 (p. 11), TradingAgents row, and the buy-and-hold row beside it
(CR = cumulative return %, ARR = annualised return %, SR = Sharpe ratio, MDD =
maximum drawdown %):

| | AAPL CR / ARR / SR / MDD | GOOGL CR / ARR / SR / MDD | AMZN CR / ARR / SR / MDD |
| --- | --- | --- | --- |
| TradingAgents | 26.62 / 30.5 / 8.21 / 0.91 | 24.36 / 27.58 / 6.39 / 1.69 | 23.21 / 24.90 / 5.60 / 2.11 |
| B&H | -5.23 / -5.09 / -1.29 / 11.90 | 7.78 / 8.09 / 1.35 / 13.04 | 17.1 / 17.6 / 3.53 / 3.80 |

The rule baselines are in the paper's table and not repeated here (§6).
Three claims are tested, plus one text claim of the level of returns:

- **R3-C1 (beats buy and hold on return).** §6.1.1, p. 11: "TradingAgents
  achieves at least a 23.21% cumulative return and a 24.90% annual return on
  the three sampled stocks, surpassing the best-performing baselines by a
  margin of 6.1%." Table 1's excess over buy and hold is +31.85, +16.58 and
  +6.11 points (AAPL, GOOGL, AMZN), mean **+18.18 points**.
- **R3-C2 (level of return).** The same sentence: cumulative return of
  **at least 23.21%** per stock over the window.
- **R3-C3 (risk-adjusted).** §6.1.2, p. 12: "TradingAgents outperforms market
  benchmarks such as Buy-and-Hold and rule-based strategies consistently";
  Table 1 Sharpe ratios 8.21, 6.39, 5.60 against buy and hold -1.29, 1.35,
  3.53: differences +9.50, +5.04, +2.07, mean **+5.54**.
- **R3-C4 (drawdown).** §6.1.3, pp. 12-13: "TradingAgents maintained a
  relatively low maximum drawdown compared to many baselines ... ensured that
  the maximum drawdown remained within a manageable limit, not exceeding 2."
  (Table 1 lists 2.11 for AMZN, above the text's 2; we use the table.) Buy
  and hold's drawdown minus TradingAgents': 10.99, 11.35, 1.69 points, mean
  **+8.01 points**.

The paper's claims are about its 2024 window; we test the pattern, not the
numbers, in a different market. A claim about the 2024 window itself can't be
tested live and is outside R3.

## 4. Design

**Agent.** `rl.bench.policies.tradingagents:TradingAgentsPolicy`, parameters
in `submissions/tradingagents/submission.json` (provider `openai`, deep tier
`gpt-6-sol`, quick tier `gpt-6-luna`, one debate round per team, all four
analysts, reasoning effort medium, default data vendors, TradingAgents v0.6.0
at commit `1394a3f72aa4393e1a98f51b382434c4b4c2d972`, `trials` = 1: nothing was
tuned). The agent code, the model ids and the config are fixed for the whole
window; any change is a deviation (§10). Model ids are aliases, so the model
name the provider reports on every call is stored in each entry
(`inputs.token_usage`) and a change of snapshot is reported.

**Tasks.** `us-aapl`, `us-googl`, `us-amzn`: one Yahoo-priced stock each, fee
0.0005 per side, one account per stock, 100% in the stock or in cash.

**Decision rule.** Before the US open (12:00 UTC, `.github/workflows/ledger.yml`)
the policy calls `propagate(ticker, date)` with `date` = the last closed
session. The five-tier rating maps to a target weight: Buy and Overweight to
1.0 (long), Sell and Underweight to 0.0 (cash), Hold and `REVIEW` keep the
current position. The engine fills at the next session's open and charges the
fee on turnover.

**Window.** The fill session after the first verified entry (the first entry
is to be committed before the US open on or before 2026-10-30) to the close of
2027-01-29. With 2026-10-30 as the first fill that is **62 sessions** (NYSE
holidays: 2026-11-26, 2026-12-25, 2027-01-01, 2027-01-18); a later start gives
fewer. A session with no verified entry is a hold, never back-filled, and is
listed. Entries count only if their Actions run is a run of the ledger
workflow on `main` that started before that session's open
(`rl/bench/ledger.py`).

**Benchmark and baselines.** Buy and hold per stock over exactly the same
window (the ledger record's benchmark). Of the paper's rule baselines, only
buy and hold can be implemented as published. MACD, KDJ and RSI, ZMR and SMA
have no stated parameters or code, so any version we wrote would be our
invention under the paper's name; they are **not run**, and R3's verdicts don't
use them. The paper's claim of beating them is therefore not tested, and the
report says so. The lab's own deterministic baselines on these tasks
(`cash`, `trend-following`, `momentum` and the others on the live track) are
shown for context; they start at their own freeze dates, so they aren't
compared on the same window.

**Seeds.** None. The agent's sampling can't be seeded, and the point of the
ledger is a single prospective path. One decision per stock per session, no
re-runs, no selection among runs. Per-day decisions, token counts and decision
hashes are published.

**Data.** Prices from Yahoo through the lab's loader; each entry stores the
SHA-256 of the data it used. Prices are not redistributed (protocol). The
agent's own inputs (yfinance, SEC EDGAR, FRED, Polymarket, StockTwits, Reddit)
are fetched by its tools at run time and aren't archived; the decision text's
SHA-256 and first 400 characters are in each entry.

**Software.** This repository at the commit of the final record, TradingAgents
at the pinned commit above (the policy refuses to run if the installed commit
differs).

## 5. Metrics

Per stock and pooled, from the ledger record's daily returns: cumulative
return, annualised Sharpe (mean/std with ddof=1, times sqrt(252), risk-free 0;
the paper's risk-free rate is not stated), maximum drawdown, turnover, share
of sessions long, rating counts. "Pooled" means the **mean of the three
per-stock statistics**, the form in which the paper reports them. Intervals and
p-values come from the paired stationary bootstrap (Politis and Romano 1994,
mean block `round(n^(1/3))` = 4, 10,000 resamples, seed 0) that resamples the
same dates across the three stocks and across the agent and its benchmark, so
cross-correlation and serial dependence are kept. Annualised return is
reported for reference only: annualising three months is not meaningful.

## 6. Verdict thresholds

Each claim is turned into a quantity $x$ where larger is better, with null
value $b = 0$ and the paper's value $c$:

| Claim | $x$ (mean over the three stocks) | $c$ | Form |
| --- | --- | --- | --- |
| R3-C1 | cumulative return of the agent minus buy and hold, points | +18.18 | equals |
| R3-C2 | cumulative return of the agent, % | 23.21 | at least |
| R3-C3 | annualised Sharpe of the agent minus buy and hold | +5.54 | equals |
| R3-C4 | maximum drawdown of buy and hold minus the agent's, points | +8.01 | equals |

Let $\hat{x}$ be our estimate and $[l, u]$ its bootstrap 95% interval. Apply in
this order:

1. **? Inconclusive** if $l \le 0 \le u$ and $l \le c \le u$: the window can't
   tell "no effect" from "the paper's effect". The report then says what
   would decide it (a longer window or more stocks).
2. **✗ Does not hold** if $\hat{x} \le 0$, or $u < c/2$ (even the top of our
   interval is under half the claimed value).
3. **✅ Holds** if $l > 0$ and $c$ is inside $[l, u]$ (for the "at least" claim
   R3-C2: $l > 0$ and $u \ge c$). For R3-C1 and R3-C3 the one-sided
   bootstrap p-value (agent not better than buy and hold) is also below 0.05,
   which is implied by $l > 0$ at the same level and is reported.
4. **◐ Partly holds** in every remaining case, which all have $\hat{x} > 0$:
   the direction matches and either significance is lost or the value is
   outside the interval.

This is the single-path form of the protocol table: with one live path there
are no seeds, so "no seed reaches the claimed value" becomes "the top of the
interval is below half the claim". It is fixed here, before any data exists.
Per-stock numbers are published, but only the pooled quantity gets a verdict:
one stock over 62 sessions can't carry one (§7). Results from three stocks
and four claims get a multiple-comparison note in the report; the four claims
are not independent, since they share one return series.

## 7. Smallest effect this test can detect (estimate, to be recomputed by bootstrap)

- **Sharpe.** The standard error of an annualised Sharpe ratio is about
  $\sqrt{(1 + SR^2/2)/\text{years}}$ (Lo, 2002). 62 sessions is 0.246 years:
  SE is 2.0 at SR 0, 2.4 at SR 1 and 8.2 at SR 5.6. A simulation of 200,000
  paths of 63 sessions with 1.3% to 2.0% daily volatility gives a null Sharpe
  standard deviation of 2.0. R3-C3 is therefore very likely **Inconclusive**
  unless the agent does far worse than buy and hold (we expect the interval
  for a difference of $\hat{x} \approx 0$ to contain both 0 and +5.54). We
  say this now so that nobody reads an inconclusive Sharpe as a finding.
- **Return versus buy and hold.** The same simulation with a no-skill long/flat
  strategy (long 50% to 70% of days in persistent runs of about five
  sessions) gives an excess cumulative return over buy and hold with standard
  deviation 5.8 to 11.6 points per stock and a 95% null range from about
  -26 to +20 points at 2.0% daily volatility. For the mean of three stocks
  at correlation 0.6 the standard deviation is about 0.86 times that, 5 to 10
  points, so $l > 0$ needs a point estimate above roughly 10 to 20 points.
  The paper's mean excess (+18.2) is therefore at the edge: if it were the
  true effect the power to exclude zero would be about 45% to 95%,
  depending on volatility. R3-C1 and R3-C4 are the informative claims.
  AMZN's +6.1 alone is below anything this window can resolve.
- **Drawdown.** Three months of a long-only stock typically contain a
  drawdown of 5% to 15%; the claimed 0.9% to 2.1% needs a nearly flat or
  perfectly timed agent, so R3-C4 can fail or hold clearly.

These figures use assumed volatilities and a toy signal model; the report
replaces them with the bootstrap on the realised path.

## 8. Contamination

The window starts after the release dates of the models (GPT-6 Sol and Luna:
reported released 2026-09-22; a model can't have seen later data), so a live
decision cannot recall its outcome: **uncontaminated by construction**. Two
leaks are still checked and reported: (a) a silent change of the model
snapshot behind an alias (stored model names per entry); (b) look-ahead
through the agent's tools, which are asked for data as of the decision date
and, in this release, clamp to it (`dataflows/date_window.py`). The protocol's
recall probe is run once before the first decision: each model is asked,
without tools, for the closing prices and main events of the three stocks in
the week before the window, to document where its knowledge ends relative to
the window start; the provider's stated training cutoff is
read from its model page and quoted in the report. The paper's own 2024
window lies inside the training data of today's models; that is R2's problem,
not R3's.

## 9. Cost and cap

Estimate, **labeled as such, not measured**: no model call has been made. Per
stock per decision, release 0.6.0 runs about 20 to 30 model calls: four
analysts with 2 to 5 turns each (each tool round re-sends the report so far),
two debaters, the Research Manager, the Trader, three risk debaters and the
Portfolio Manager, against 11 calls stated in the paper.

| Tier | Input tokens | Output tokens | Price (per M, in / out) | Cost |
| --- | --- | --- | --- | --- |
| Quick, gpt-6-luna | ~230,000 | ~20,000 | $0.10 / $0.50 | $0.023 + $0.010 |
| Deep, gpt-6-sol | ~30,000 | ~8,000 (incl. reasoning at medium) | $2.00 / $10.00 | $0.060 + $0.080 |
| **Per stock per day** | | | | **about $0.17** |

No caching is assumed. Three stocks per day is about $0.52; 21 to 22 sessions
a month is about $11; 62 sessions is about $32. The error bars are wide: a
factor of three upward is plausible (more tool rounds, up to the default 20
per analyst; more reasoning tokens), so plan for $35 to $100 in total.
Caps: `LEDGER_MONTHLY_BUDGET_USD` = **40** (USD per month for this submission,
all three stocks together; the ledger holds with reason "budget cap" once the
month's recorded cost reaches it, so one day's cost can overshoot), and a
total of **$120** for the window; a cap-induced hold is a missed session and is
reported as such. The first dry run before 2026-10-30 replaces this table with
a measurement.

## 10. Deviations from the paper's setup (fixed in advance)

1. **Window.** Live, 2026-10-30 to 2027-01-29, not 2024-01-01 to 2024-03-29;
   a different market regime. The comparison is of patterns, not levels.
2. **Models and code.** v0.6.0 with gpt-6-sol / gpt-6-luna, not the paper's
   o1-preview, gpt-4o and gpt-4o-mini. The paper (v7, 2025-06-03) predates
   this release's five-tier ratings, Portfolio Manager, SEC EDGAR fundamentals,
   point-in-time data rules and memory log, and its analysts use the deep tier
   while the release's use the quick tier. We test the released framework, not
   the paper's code, which isn't public as a snapshot we can identify.
3. **Data vendors.** yfinance, SEC EDGAR, FRED, Polymarket, StockTwits and
   Reddit, against the paper's Bloomberg, Yahoo, EODHD, FinnHub, Reddit and X.
4. **Long-only.** The engine can't short, so Sell and Underweight mean cash.
   If the paper's backtest shorted, our Sell is weaker than theirs (§11).
5. **Rating mapping.** Overweight counts as Buy and Underweight as Sell: the
   paper has three actions, the release five.
6. **Position size.** 100% in or out; the agent's own sizing text is ignored.
   The paper doesn't state a size.
7. **Costs and timing.** Decide on the close, fill at the next open, 0.05%
   per side. The paper states no costs and no fill timing.
8. **Decision date.** `propagate` is called with the last closed session, as
   the ledger contract requires. The release then treats the run as dated in
   the past and withholds live-only data: company profile and valuation
   multiples for non-SEC sources, insider transactions and Polymarket odds
   (`dataflows/date_window.py`, `withhold_live_profile`). Statements still come
   from SEC EDGAR as filed. The agent is therefore slightly less informed than
   a run dated today.
9. **Memory off.** Each decision runs in a fresh temporary directory, so the
   memory log is empty and the agent does not learn from earlier days. The
   paper mentions "a reflective agent" and the release's memory is on by
   default.
10. **Reasoning effort** is pinned to `medium` and temperature left at the
    provider default; reasoning models ignore temperature, and the authors say
    no setting makes output reproducible.
11. **Sharpe** uses risk-free 0; the paper names T-bills without a value.
12. **Three stocks** (the three in Table 1), not the five in the text.
13. **Single run per day**, no ensembling of repeated runs.

## 11. Planned analyses

Primary: §5 and §6. Secondary, reported beside and never used for verdicts:

- Fee sensitivity at 0, 5 and 10 bp per side.
- **Short shadow:** the same logged signals with Sell and Underweight treated
  as -1, no borrow cost, computed offline from `inputs.signal`; this brackets
  the paper's possible use of shorts.
- Decision distribution, run lengths, turnover and an agreement table between
  the three stocks' ratings.
- Day-by-day ledger, missed and unverified sessions, per-month cost against the cap.
- Per-stock cumulative return, Sharpe and drawdown with intervals.
- Lab baselines on the live track for context (§4).

Anything added after the first decision is listed in the report under
"Deviations", with the reason and the date.

## 12. Run prerequisites and risks

Repository secrets and variables the owner must add (the workflow needs them
under these exact names):

| Name | Kind | Needed for |
| --- | --- | --- |
| `OPENAI_API_KEY` | secret | all model calls (already passed by the workflow) |
| `FRED_API_KEY` | secret | FRED macro tool; free key at fred.stlouisfed.org; runs continue without it but the News analyst loses macro data, which would be a silent deviation from §4 |
| `SEC_EDGAR_USER_AGENT` | variable | contact string `Name email@domain` that SEC requires for EDGAR requests (the package sends a placeholder address otherwise) |
| `LEDGER_MONTHLY_BUDGET_USD` | variable | set to `40` |

`.github/workflows/ledger.yml` passes all of these to the decision step and
installs TradingAgents at the pinned commit (`INSTALL_REQUIREMENT` in
`rl/bench/policies/tradingagents.py`) on Python 3.13.

Risks recorded now: vendor outages (a tool failure degrades the agent's inputs
silently; the entry stores the decision text hash, not the tool outputs);
provider rate limits or model deprecation mid-window (a run error is a hold);
GitHub Actions scheduling delays (a late run misses the open and is
unverifiable, so counts as a hold). The agent's pandas >= 3 requirement matches
the lab's environment (pandas 3.0.6); installing the pinned commit and
building its graph were checked locally on Python 3.12 without a model call,
and the first dry run checks a full decision.

## 13. Authors

Seven days before publication the draft report, the evidence and the
reproduction command go to the paper's authors (Yijia Xiao, Edward Sun, Di Luo,
Wei Wang; contact route to be confirmed from the repository and arXiv page).
Their reply is printed verbatim in `reply.md`, or "no reply by <date>" is
recorded. Errors they find are fixed and logged under "Corrections" with
credit.
