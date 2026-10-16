# R1 pre-registration: the FinRL ensemble (Yang, Liu, Zhong, Walid, ICAIF 2020)

Written 2026-10-14, before any training run of ours. The git commit time of
this file is the evidence that the plan came first; no file under
`reports/r1-finrl-ensemble/evidence/` and no result of ours may be committed
before it. The report follows `.plan/lab-protocol.md` (levels, pre-registration,
verdict table, author notice). It covers all three levels: **L1** (the
authors' code on their data), **L2** (this repository's ensemble mode under
the lab's honest protocol, with a turbulence-rule ablation) and **L3** (the same
method on 2020-05-11 to 2025-12-31).

What was looked at before writing, so nobody has to guess: the paper (two
versions), the authors' repository at the pinned commit including its
committed results, the toolkit's source, and one throughput probe of the
toolkit on synthetic prices (§9). §3 lists what the authors' own committed
artifacts show. No run of the method on real data has been made by us, and the
thresholds in §7 were written without seeing any of our results. Items marked
**[UNVERIFIED]** could not be checked from here, and each says why.

## 1. Question

The paper reports that an ensemble that retrains PPO, A2C and DDPG every
quarter and trades the one with the best validation Sharpe ratio beats the
three agents, the DJIA and a min-variance portfolio on the Dow 30 over
2016-01-04 to 2020-05-08, with Sharpe 1.30 and a maximum drawdown of -9.7%
against -37.1% for the index. It is the most-cited deep-RL trading result and
the template many later papers copy (Dow 30, turbulence liquidation, one run,
fills at the signal close). We ask three things: (L1) do the authors' own code
and data give their numbers, (L2) does the result survive next-open fills, the
stated universe, several seeds and a test of the turbulence rule, (L3) does it
hold afterwards.

## 2. Facts about the paper, the code and the data (read 2026-10-14)

### 2.1 Paper

| Item | Finding | Source |
| --- | --- | --- |
| Versions | ICAIF '20 camera-ready, 8 pages, DOI 10.1145/3383455.3422540, 15 Oct 2020 (Crossref: 185 citations on 2026-10-14). SSRN 3690996. arXiv repost 2511.12120v1, submitted 15 Nov 2025 by Hongyang Yang. We read the arXiv repost and the Columbia-hosted camera-ready; **Table II and Table 1 are identical in both**. The SSRN page returned HTTP 403, so the SSRN text was **[UNVERIFIED]** | <https://arxiv.org/abs/2511.12120>, <https://arxiv.org/pdf/2511.12120>, <https://openfin.engineering.columbia.edu/sites/default/files/content/publications/ensemble.pdf>, <https://api.crossref.org/works/10.1145/3383455.3422540> |
| Authors and contact | Hongyang Yang (AI4Finance LLC and Columbia, hy2500@columbia.edu), Xiao-Yang Liu (Columbia, **corresponding author**, "co-primary author with equal contribution", xl2427@columbia.edu), Shan Zhong (Wormpex AI Research, sz2495@columbia.edu), Anwar Walid (Nokia Bell Labs, anwar.walid@nokia-bell-labs.com). The arXiv repost lists contact@ai4finance.org for all four. The addresses date from 2020 and 2025; affiliations may have changed | camera-ready p. 1; arXiv v1 p. 1 |
| Universe | "We select the Dow Jones 30 constituent stocks (at 01/01/2016) as our trading stock pool." The paper does **not list the tickers** | arXiv p. 6 (§VI-A); camera-ready §6.1, p. 5 |
| Data | "historical daily data from 01/01/2009 to 05/08/2020 ... downloaded from the Compustat database through the Wharton Research Data Services (WRDS)" | same |
| Splits | "Data from 01/01/2009 to 09/30/2015 is used for training, and the data from 10/01/2015 to 12/31/2015 is used for validation and tuning of parameters ... unseen out-of-sample data from 01/01/2016 to 05/08/2020"; Table II header and Fig. 5 caption: 2016/01/04-2020/05/08, initial portfolio value $1,000,000 | arXiv p. 6, p. 7 |
| Online learning | "To better exploit the trading data, we continue training our agent while in the trading stage" | arXiv p. 6 |
| Rolling scheme | Growing training window; "we retrain our three agents at every three months"; "3-month validation rolling window after training window"; best validation Sharpe picks the agent for the next quarter; "We also adjust risk-aversion by using turbulence index in our validation stage" | arXiv §V-D, p. 5; camera-ready §5.4 |
| State | 181 dimensions: balance, 30 adjusted close prices, 30 holdings, then MACD, RSI, CCI, ADX for 30 stocks | arXiv §IV-A1, p. 4 |
| Action | per stock an integer in {-k..k}, k <= hmax, normalised to [-1, 1]. hmax value **not stated** | arXiv §IV-A2, p. 4 |
| Reward | change of portfolio value net of cost, eq. (4); the paper gives no reward scaling | arXiv §III-C, p. 3 |
| Costs and fills | "transaction costs to be 0.1% of the value of each trade (either buy or sell)"; "the orders can be rapidly executed at the close price" | arXiv §III-B, p. 3 |
| Turbulence | eq. (3), Mahalanobis distance of the day's returns from historical returns; above "a threshold" the agent halts buying and sells all shares. Threshold **value not stated**; "We can tune the turbulence index threshold lower for higher risk aversion". Fig. 6 shows the index on an axis from about 100 to 400 | arXiv p. 3, p. 7 |
| Sharpe | eq. (19) with r_f, and "subtracting the annualized risk free rate from the annualized return, and the dividing by the annualized volatility". r_f **value not stated**. Annualised return is "the geometric average amount of money earned by the agent each year"; max drawdown is "the maximum percentage loss during the trading period" | arXiv §V-D, p. 5; §VI-B1, p. 6 |
| Algorithms | PPO, A2C, DDPG; hyperparameters, timesteps and library versions **not stated** (only "adjusting key parameters, such as learning rate, number of episodes") | arXiv §VI-A, p. 6 |
| Seeds, trials | Not stated; a single run is reported | whole paper |
| Baselines | DJIA and "min-variance portfolio allocation of DJIA", with reference to Yang, Liu and Wu (TrustCom 2018). No parameters; **no min-variance code in the repository** | arXiv p. 6-7 |
| Picks | Table 1 lists validation Sharpe ratios and the picked agent for 18 quarters, 2016/01-2016/03 to 2020/04-2020/05: PPO 7, A2C 6, DDPG 5 | arXiv p. 6 |

### 2.2 Code and data (the L1 target)

| Item | Finding | Source |
| --- | --- | --- |
| Repository | The paper's link is `AI4Finance-LLC/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020` (camera-ready footnote 2); the arXiv repost links the `AI4Finance-Foundation` path, which returns 404 today. The old path redirects (HTTP 301) to <https://github.com/AI4Finance-Foundation/FinRL-Trading>, i.e. the repository was repurposed: it is 180 commits ahead of the pin and stands at "FinRL-X". `Jung132914/...ICAIF-2020` is a fork (32 stars) frozen at the same last commit | `curl -sI` and GitHub API, 2026-10-14 |
| **Pinned commit** | **`5dbb44dfcb8a01a94762dc44fca2bcb2c626d1dc`** (2021-01-21, README edit; code identical to `846ca18f69fe2465ddb398405072fb1d445fae25` of 2020-11-16). It is on the `master` history of `AI4Finance-Foundation/FinRL-Trading`. It contains `done_data.csv`, the preprocessed input, so WRDS access is not needed. The closest commit to the paper is `6fada3586f86ca610783750636650cf7c24e1dc5` (2020-10-12, "updated key hyperparameters to fix the underfitting issue"); it differs in the turbulence look-back code (date range replaced by `validation_window*30` rows) and has no `done_data.csv`. We pin the final state and report the difference | `git diff 6fada35 5dbb44d`, GitHub API compare |
| Entry point | `python run_DRL.py`: `rebalance_window = 63`, `validation_window = 63` (trading days, not months), `unique_trade_date` = dates after 2015-10-01 up to 2020-07-07 (18 trade windows from 2016-01-04; the paper stops at 2020-05-08) | `run_DRL.py` L33-39 |
| Timesteps | A2C 30,000; PPO 100,000; DDPG 10,000 per fit, three fits per quarter (140,000 steps) | `model/models.py` L227, L235, L243 |
| Hyperparameters | A2C `A2C('MlpPolicy', env)` defaults. PPO `PPO2('MlpPolicy', env, ent_coef=0.005, nminibatches=8)`. DDPG defaults plus Ornstein-Uhlenbeck action noise sigma 0.5. All else is stable-baselines defaults of the installed version | `model/models.py` L28-72, L56-60 |
| Selection | PPO if its validation Sharpe is at least both others', else A2C if strictly higher than both, else DDPG. **The picked model trades as trained up to the validation start; it is not refit on the validation window.** Validation Sharpe is `sqrt(4) * mean / std` of daily returns (an argmax is unaffected by the constant, and Table 1's magnitudes, -0.58 to 0.71, match it) | `model/models.py` L147, L255 |
| Library pins | `requirements.txt`: numpy 1.16.4, pandas 1.0.3, scikit-learn 0.21.0, gym 0.15.3, tensorflow 1.15.4, joblib 0.15.1, `stockstats` and `stable-baselines[mpi]` unpinned; README: Python >= 3.6. We pin stable-baselines **2.10.1** (PyPI 2020-08-05; 2.10.2 is dated 2021-04-06, after the code's last change) | `requirements.txt`; <https://pypi.org/pypi/stable-baselines/json> |
| Environment constants | `HMAX_NORMALIZE = 100`, `INITIAL_ACCOUNT_BALANCE = 1000000`, `TRANSACTION_FEE_PERCENT = 0.001`, `REWARD_SCALING = 1e-4`, `STOCK_DIM = 30` | `env/EnvMultipleStock_trade.py` L13-23 |
| Inputs | `data/dow_30_2009_2020.csv`: 87,780 rows, 30 tickers, 2009-01-02 to 2020-08-17, Compustat fields (`prccd`, `ajexdi`, `prcod`, `prchd`, `prcld`, `cshtrd`), SHA-256 `1fa004731c88bc2e202ec30e5f59b4f35cb40ced0566135bdd564fdd8e2a63eb`. `done_data.csv` (same rows plus indicators and turbulence): `1969cdec86154daa57df8cafb027defe85693adbaa1883928922780a86052372`. `data/^DJI.csv`: `6676701a9ec0a01d8474a89282a0ffdcdd447fd8863988944ec171b2c48d88d3`. The hashes of every other file of the checkout go into the evidence manifest | SHA-256 computed 2026-10-14 on the checkout of the pin |
| Price basis | `adjcp = prccd / ajexdi` (split-adjusted close, no dividends); the DJIA baseline in `^DJI.csv` is the price index, so the paper compares price returns with price returns | `preprocessing/preprocessors.py` L35-37 |
| Indicators | `macd`, `rsi_30`, `cci_30`, `dx_30` from `stockstats` (the paper says RSI, CCI, ADX; the code column named `adx` holds `dx_30`) | `preprocessing/preprocessors.py` L67-84 |
| Turbulence rule in code | Computed on **price levels** with an **expanding** history, zero for the first 252 dates, not on returns as eq. (3) says. Reference = 90th percentile of in-sample turbulence for dates before 2015-10 (fixed for all quarters). Per quarter, if the mean turbulence of the last `validation_window*30` rows (63 dates) before the validation start exceeds the reference, the threshold is the reference; otherwise it is the in-sample **maximum** (`np.quantile(..., 1)`). At or above the threshold the environment sells every share at that day's close and blocks buying | `preprocessing/preprocessors.py` L127-133; `model/models.py` L167-204; `env/EnvMultipleStock_trade.py` L155-156 |
| Fills | The agent observes day-t close, and its order fills at the same close (`state` uses `adjcp[t]`; sells then buys at `state[index+1]`) | `env/EnvMultipleStock_trade.py` L67-110 |
| Seeds | None. No `seed` argument to any model and none to the environments; `gym.seeding.np_random(None)` | `grep seed` over the repository |
| Baseline code | None for min-variance; DJIA only as `data/^DJI.csv`, the notebook computes pyfolio tear sheets | `backtesting.ipynb` |
| Backtest metrics | `backtesting.ipynb`: Sharpe `sqrt(252) * mean / std` of daily account-value returns, no risk-free rate; pyfolio for the rest | `backtesting.ipynb` cells 3 and 5 |
| Provenance of the committed results | `results/firstRun/*` were added on 2020-11-07 (commit 9319f86, "adding results"; the same day's 9d0817a and fc08aef are "first run" commits) by a contributor who is not a paper author, three weeks after the conference (2020-10-15) | `git log` of the pin |

### 2.3 Universe: stated, shipped and today's data

The Dow 30 on 2016-01-01 is the list set on 2015-03-19 (Apple replaces AT&T);
the next change was 2017-09-01, so it holds for the whole of 2016. [Wikipedia
wikitext of "Historical components of the Dow Jones Industrial Average",
section 2015-03-19; change dates for 2017-09-01, 2018-06-26, 2019-04-02 and
2020-04-06 in the same page; PR Newswire notices cited there.]

- **Stated universe (2016-01-01), 30 names:** AAPL, AXP, BA, CAT, CSCO, CVX,
  DD, DIS, GE, GS, HD, IBM, INTC, JNJ, JPM, KO, MCD, MMM, MRK, MSFT, NKE, PFE,
  PG, TRV, UNH, **UTX**, V, VZ, WMT, XOM.
- **Shipped universe (the 30 tickers in the authors' CSV):** the same list with
  **GE replaced by WBA** and UTX carried under its later ticker RTX: AAPL, AXP,
  BA, CAT, CSCO, CVX, DD, DIS, GS, HD, IBM, INTC, JNJ, JPM, KO, MCD, MMM, MRK,
  MSFT, NKE, PFE, PG, RTX, TRV, UNH, V, VZ, WBA, WMT, XOM. That is the Dow of
  mid-2020 (WBA joined on 2018-06-26 in GE's place; the CSV ends 2020-08-17,
  before the 2020-08-31 change) with DD standing in for DOW, which has no
  history before 2019. FinRL's own paper says its Dow 30 was "accessed at
  2020/07/01" (arXiv 2111.09395 §4.4). So the paper's "(at 01/01/2016)" does
  not describe the data it ships: GE was a Dow member for the first 2.5 years
  of the test window and is absent, and WBA is present before it joined. That
  the CSV's "DD" series is the 2016 DuPont is **[UNVERIFIED]** (Compustat
  identifiers are not in the file).
- **Yahoo today (checked 2026-10-14 with yfinance):** `UTX` returns no data;
  `RTX` returns the history from 2009-01-02; `DD` returns rows from 2009-01-02
  with a 2016-01-04 close of 89.20 (the authors' CSV has an `adjcp` of 149.80
  for the same day; both differ from the raw price, so the series are vendor
  adjusted and not comparable; **[UNVERIFIED]** which adjustment each applies).
  `WBA` returns a 404.
- **Survivorship:** every company of the stated universe is expected to
  trade today, so a point-in-time list has no delisting problem (yfinance
  confirmed GE, RTX and DD; the other names were not queried, which makes
  the rest **[INFERENCE]**). The bias in the authors' setup is the membership
  (WBA for GE), not missing prices.
- **Sharadar (licensed, local only, never committed):** per vendor
  documentation the SEP table has `open`, `high`, `low`, `close` (split
  adjusted), `closeadj` (split, cash-dividend and spin-off adjusted),
  `closeunadj` and `volume`; delisted stocks are included; TICKERS carries a
  stable `permaticker` (<https://sharadar.com/docs/stocks>). How UTX and
  the 2016 DuPont map onto current tickers, and that no 2016 name has a gap in
  its series, are **[UNVERIFIED]** until the extract is downloaded; the
  mapping, keyed by `permaticker`, and the extract's hash go into the evidence.

## 3. What the authors' own committed artifacts show (observed before the first run)

Computed 2026-10-14 from the checkout of the pin, without running the method.
They are disclosed because they shape the plan and so the report cannot be
accused of choosing its stance after the runs. The scripts and outputs are
archived in `evidence/prereg-observations/` with the report.

1. **The committed ensemble results do not reproduce Table II.** Concatenating
   `results/firstRun/account_value_trade_ensemble_{126..1197}.csv` as
   `backtesting.ipynb` does (18 files of 63 rows, dates from the code's
   `unique_trade_date`, 2016-01-04 to 2020-07-06) and cutting at 2020-05-08
   (1,094 returns) gives Sharpe 1.53 (sqrt(252), r_f = 0), cumulative return
   64.3%, annual return 12.1%, volatility 7.7%, max drawdown -5.6%. Table II:
   1.30, 70.4%, 13.0%, 9.7%, -9.7%. The notebook's stored output cell prints a
   Sharpe of 1.5921 for the same expression; the committed files give 1.5161
   for the full series. So there are three different Sharpe values for one
   strategy (1.30 paper, 1.59 notebook output, 1.52 to 1.53 recomputed). Which
   run produced the paper's table is **[UNVERIFIED]**; the committed results
   post-date the paper (§2.2).
2. **The committed run's account value does not move on a third of sessions**
   (consistent with being all cash). 428 of the 1,133
   daily returns are exactly zero; 17 are the seams between the 18 files
   (the first value of each file repeats the last of the previous), leaving 411
   flat sessions (36%). Per quarter they are 0 to 7 in 2016-2017; from 2018Q1
   nine of the ten quarters have 18 to 58 (2019Q1 has 1), up to 58 of 63 in
   2019Q4. In 2020Q1 and Q2 (52 and 55 flat sessions of 63) the account barely
   moves (1.657M on 2020-01-02, 1.643M at 2020-04-03). Over the DJIA's
   crash (peak 2020-02-12, trough 2020-03-23) the committed ensemble moves
   -0.8%. **[INFERENCE]** The cause is the price-level, expanding-window
   turbulence rule (§2.2) liquidating and staying liquid; the committed files
   do not record the threshold or the turbulence values, so this was not checked.
   It is why the turbulence ablation of L2 is part of the design. R0 found the
   same behaviour in this repository's own environment: all 25 runs with the
   rule share one identical set of cash sessions (2020-03-17, 03-20, 03-23,
   03-24, 03-25) (`reports/r0-wang-zhang-2026/report.md`, "Sessions with
   unchanged equity").
3. **The DJIA column is only partly reproducible from the authors' file.**
   `data/^DJI.csv`, 2016-01-04 to 2020-05-08, price index: volatility 20.1%
   (paper 20.1%), max drawdown -37.09% (paper -37.1%; it is the crash, peak
   2020-02-12 to trough 2020-03-23), but cumulative return 41.9% (paper 38.6%),
   annual return 8.4% (7.8%), Sharpe 0.50 (0.47). Closes on 2019-01-08,
   2020-04-24, 2020-05-04 and 2020-05-12 give 38.5% to 38.7%; 2020-05-08 gives 41.9%. The
   cause is **[UNVERIFIED]**.
4. **Text and code disagree on several settings**: validation window 3 months
   (paper) versus "12-month" (repository README) versus 63 trading days (code);
   turbulence on returns (eq. 3) versus on price levels (code); "ADX" versus
   `dx_30`; "we continue training our agent while in the trading stage"
   versus no training in the trading loop; window end 2020-05-08 versus
   2020-07-07 in `run_DRL.py`.
5. **The authors' search is not documented.** The repository history shows
   at least these changes to timesteps, hyperparameters and the turbulence
   threshold between 2020-09-27 and 2020-10-12 (commit messages "updated
   turbulence model" x4 on 2020-09-27/28 and "updated key hyperparameters to
   fix the underfitting issue" on 2020-10-12: A2C 10k to 30k steps, PPO 50k to
   100k plus `ent_coef=0.005` and `nminibatches=8`, DDPG 20k to 10k, the
   fallback turbulence quantile 0.99 to 1.0). The paper says validation was
   used for "adjusting key parameters"; whether the 2016-2020 trading window
   was consulted is not stated. For the deflated Sharpe the authors' number of
   trials is unknown and at least 3.

## 4. Claims, quoted from the paper

Page numbers: arXiv repost v1 first, camera-ready section second. Values are
Table II (arXiv p. 7; camera-ready p. 7), window 2016/01/04-2020/05/08. Ids
match `claims.json`, with a `-l1`, `-l2` or `-l3` suffix per level.

| | Ensemble | PPO | A2C | DDPG | Min-Variance | DJIA |
| --- | --- | --- | --- | --- | --- | --- |
| Cumulative Return | 70.4% | 83.0% | 60.0% | 54.8% | 31.7% | 38.6% |
| Annual Return | 13.0% | 15.0% | 11.4% | 10.5% | 6.5% | 7.8% |
| Annual Volatility | 9.7% | 13.6% | 10.4% | 12.3% | 17.8% | 20.1% |
| Sharpe Ratio | 1.30 | 1.10 | 1.12 | 0.87 | 0.45 | 0.47 |
| Max Drawdown | -9.7% | -23.7% | -10.2% | -14.8% | -34.3% | -37.1% |

(Checked against the rendered page, not only the extracted text.)

| Id | Quote (verbatim) | Source | Metric | Claimed | Baseline |
| --- | --- | --- | --- | --- | --- |
| `ens-sharpe` | "the ensemble strategy achieves a Sharpe ratio 1.30, which is much higher than the Sharpe ratio of 0.47 for DJIA" | §VI-B4, p. 7; §6.2.4, pp. 6-7; Table II | `sharpe` | 1.30; gap to DJIA +0.83 | DJIA 0.47 |
| `ens-sharpe-vs-minvar` | "... and 0.45 for the min-variance portfolio allocation." | same sentence | `sharpe_gap_vs_minvar` | gap +0.85 (1.30 - 0.45) | min-variance 0.45 |
| `ens-sharpe-vs-agents` | "The ensemble strategy also outperforms A2C with a Sharpe ratio of 1.12, PPO with a Sharpe ratio of 1.10, and DDPG with a Sharpe ratio of 0.87, respectively." Abstract: "shown to outperform the three individual algorithms and two baselines in terms of the risk-adjusted return measured by the Sharpe ratio" | §VI-B4, p. 7; abstract, p. 1; §VII | `sharpe_gap_vs_best_agent` | gap +0.18 (1.30 - 1.12, the best of the three) | best of A2C, PPO, DDPG |
| `ens-cumulative-return` | Table II, "Cumulative Return" 70.4% against DJIA 38.6% and min-variance 31.7%; "The annualized return of the ensemble strategy is also much higher" | Table II and §VI-B4, p. 7 | `cumulative_return` | 70.4%; gap +31.8 points | DJIA 38.6% |
| `ens-annual-return` | Table II, "Annual Return" 13.0% against DJIA 7.8% and min-variance 6.5% | Table II, p. 7 | `annual_return` | 13.0%; gap +5.2 points | DJIA 7.8% |
| `ens-max-drawdown` | Table II, "Max Drawdown" -9.7% against DJIA -37.1% and min-variance -34.3% | Table II, p. 7 | `max_drawdown` | -9.7%; gap +27.4 points of drawdown depth | DJIA -37.1% |
| `ens-crash-2020` | "By incorporating the turbulence index, the agents are able to cut losses and successfully survive the stock market crash in March 2020." | §VI-B3, p. 7 (camera-ready §6.2.3, p. 6); Fig. 6 | `crash_drawdown` | no number; we fix "survive" as a loss of at most half the DJIA's over the same peak-to-trough dates | DJIA's peak-to-trough loss |

Annual volatility (9.7% against 20.1% and 17.8%) is reported beside the claims
and gets no verdict: the paper's statement of it ("the annual volatility is
much lower") is part of the sentence quoted under `ens-cumulative-return`,
and it is not a separate claim with a threshold. The paper does not claim the
ensemble has the highest return (PPO is higher in Table II: 83.0% against
70.4%), and nothing here tests that.

## 5. Design

### 5.1 L1: the authors' code on the authors' data

- **Code and data:** the pinned commit (§2.2), `done_data.csv` as shipped, no
  change to any file. The DJIA benchmark is `data/^DJI.csv`. Vendor prices are
  neither committed nor republished; the evidence has hashes only.
- **Environment:** the authors' stack needs CPython <= 3.7 and TensorFlow
  1.15.4, whose PyPI wheels exist only for CPython 3.5 to 3.7 on x86_64
  (<https://pypi.org/pypi/tensorflow/1.15.4/json>). This laptop is arm64
  (Apple M5) with no Docker, so L1 does not run here. It runs on GitHub-hosted
  `ubuntu-22.04` x86_64 runners, where CPython 3.7.17 is in the
  `actions/python-versions` manifest (checked 2026-10-14), with `libopenmpi-dev`
  for DDPG, the library pins of §2.2 and stable-baselines 2.10.1. Whether the
  dependency set resolves is **[UNVERIFIED]** until the first build; see §12.
- **Runs:** the code is unseeded, so L1 is repeated: **5 independent runs** with
  no seed control, all published. Each run is the unmodified `run_DRL.py`
  (trading to 2020-07-06); the account value is cut at 2020-05-08 for the
  comparison with Table II. Seams between files are removed as the protocol of
  §6 states.
- **Metrics:** §6 definitions, computed by our code from each run's account
  values; the authors' notebook recipe is applied to the same files as a check.
- **Outcome measured:** the five Table II quantities for the ensemble and the
  DJIA from `^DJI.csv`; and the picked model per quarter against Table 1.
- **If L1 cannot be built** by 2026-11-17 (a full quarter of fits completes in
  the environment), the report says L1 was **not run**, gives the reason, and
  rests on L2 and L3 and on §3's recomputation, which is labeled "L1-lite" and
  earns no verdict.

### 5.2 L2: this repository's ensemble mode, honest protocol

- **Engine:** `python main.py --mode ensemble` / `rl.model.models.DRLEnsembleAgent`
  (`docs/toolkit.md`), signal at the close, **fill at the next open**, **0.1%
  per side** on every buy and sell, `hmax` 100, initial capital $1,000,000,
  reward scaling 1e-4, `rebalance_window` = `validation_window` = 63 sessions,
  growing training window from 2009-01-02, selection on validation Sharpe.
- **Data:** Sharadar SEP for the **stated universe** (§2.3), 2009-01-02 to
  2025-12-31. Prices are `closeadj`-scaled OHLC (total return), the engine's one
  adjustment basis. A price-only run is **not** made: spin-offs (DD and UTX in
  2019-2020, others later) would show as price crashes in `close`. The
  paper's numbers are price returns, so level claims (Sharpe, cumulative and
  annual return) are flattered by dividends on our basis. The bridge is
  reported: the same window for the DJIA price index (`^DJI`, Yahoo) and for
  its total-return proxy (DIA adjusted close), and both are shown.
- **Baselines:** (a) the DJIA total-return proxy, DIA adjusted close (Sharadar
  SFP if the licence covers it, else Yahoo, personal-research use), with
  `^DJI` as a price reference; (b) a long-only minimum-variance portfolio of the
  same 30 names, weights from the sample covariance of the trailing 252
  returns, re-estimated every 63 sessions, filled at the next open with 0.1%
  per side (**ours**: the paper gives neither parameters nor code, so
  `ens-sharpe-vs-minvar` is tested against our version and says so); (c) the
  three standalone agents (below); (d) the protocol's 1/N buy-and-hold of the
  30 names and the engine's monthly-rebalanced 1/N, shown for context.
- **Standalone agents (for `ens-sharpe-vs-agents`):** each of A2C, PPO, DDPG
  trades every quarter with its own account, refit on train plus validation the
  way the ensemble's winner is, so that the ensemble and the agents differ only
  in selection.
- **Seeds:** **10 seeds (0-9)** for each arm, run in the order 0-4 first, then
  5-9; the protocol's minimum is 5. If only 0-4 finish, the report says so and
  uses those. One seed sets the Python, NumPy, PyTorch, environment and
  model seeds (`--seed`); the same seed is used across arms.
- **Arms (all declared; the configurations we try):**
  - **A: rule on** (primary). The ensemble's turbulence rule as implemented: a
    per-quarter threshold from fitting-history turbulence (90th percentile if the
    last 63 sessions' mean exceeds it, else the maximum), liquidation scheduled
    at the next open when signal-day turbulence is at or above it.
  - **B: rule off** (the ablation, motivated by R0's finding in §3.2): the same
    seeds, validation and trading with no liquidation.
  - **C: shipped universe, rule on** (sensitivity, 5 seeds, 0-4): the authors'
    shipped list (WBA for GE) instead of the stated one, to isolate the effect of
    the membership choice.
  Number of configurations for the deflated Sharpe: 3 (A, B, C) for us.
- **Recipe matching.** Where the engine exposes it the authors' recipe is kept;
  nothing is tuned on any trading window.

| Item | Authors (pin) | Toolkit default | L2 setting |
| --- | --- | --- | --- |
| Indicators | `macd`, `rsi_30`, `cci_30`, `dx_30` (181-dim state) | 14 (`macd`, `boll_ub`, `boll_lb`, `rsi_10`, `rsi_20`, `cci_10`, `cci_20`, `dx_30`, three SMA, three EMA) | the authors' four, via `TrainConfig.tech_indicator_list` |
| Timesteps A2C / PPO / DDPG | 30,000 / 100,000 / 10,000 | one `--timesteps` for all three | 30,000 / 100,000 / 10,000 |
| A2C | stable-baselines 2 defaults | `n_steps` 5, `ent_coef` 0.01, `learning_rate` 0.0007 | toolkit default |
| PPO | `ent_coef` 0.005, `nminibatches` 8, rest defaults (PPO2 `n_steps` 128) | `n_steps` 2048, `ent_coef` 0.01, `learning_rate` 0.00025, `batch_size` 64 | `ent_coef` 0.005, `n_steps` 128, `batch_size` 16, rest SB3 defaults |
| DDPG | OU action noise sigma 0.5 | `batch_size` 128, `buffer_size` 50,000, `learning_rate` 0.001, no noise | OU noise sigma 0.5 (an object in `model_kwargs`) |
| Observation scaling | none (raw state) | running normalisation on | toolkit default (on); deviation |
| Winner refit | **none**: the model trained to the validation start trades | refit on train plus validation | toolkit behaviour; deviation |
| Turbulence | price levels, expanding, reference fixed to pre-2015-10 | returns, 252-session look-back, reference from fitting history | toolkit behaviour; deviation |
| Fills | same close | next open | next open |

- **Windows:** one continuous run per seed and arm from **2016-01-04 to
  2025-12-31** (about 2,514 sessions, 40 windows). L2 metrics use the sessions
  up to **2020-05-08** (1,095 sessions, 18 windows, the last one partial); L3
  metrics use **2020-05-11 to 2025-12-31** (about 1,419 sessions). Windows
  before 2020-05-08 do not depend on later data, and each fit is seeded with
  the run's seed, so the L2 prefix should equal what a run ended on 2020-05-08
  would produce. This is checked, not assumed: before the first
  reported run, a run ended on 2020-05-08 and the long run must give identical
  account values up to that date for one seed; if they do not, L2 uses
  separate runs ended on 2020-05-08 and L3 the long ones.

### 5.3 L3: later window

Same runs, sessions 2020-05-11 to 2025-12-31, same baselines (DIA, our
min-variance, standalone agents, 1/N). The paper's levels do not apply to a
different window, so L3 tests the **gap** form of each claim only (§7): that
the ensemble beats DJIA and min-variance in Sharpe, beats the best agent, has
a higher return and a smaller drawdown than the DJIA, and that the rule
shortens a crash drawdown.

### 5.4 Software and data record

This repository at the commit of the final record (the first-run commit is
logged in the evidence; the toolkit was read at `cf53260`, branch
`feat/phase0-followups`, with stable-baselines3 2.9.0 from `requirements.txt`);
the authors' code at `5dbb44df...`; the Sharadar extract, ticker map, Yahoo
`^DJI` and DIA files each with SHA-256 once downloaded; the evidence also
holds the lock files of both environments.

## 6. Metrics

From daily account values (close to close), per run:

- Cumulative return `V_T / V_0 - 1`; annual return `(V_T / V_0)^(252/n) - 1` for n
  daily returns (geometric, as the paper defines it); annual volatility
  `sqrt(252) * std(r, ddof=1)`; **Sharpe `sqrt(252) * mean(r) / std(r, ddof=1)`
  with r_f = 0**, which is what the authors' notebook and pyfolio compute (the
  paper's r_f value is not stated); maximum drawdown the largest peak-to-trough
  fall of the account value, reported as a depth in percent.
- Sharpe with r_f = the 3-month T-bill rate (FRED DTB3) is a secondary figure.
- Crash drawdown: peak-to-trough fall of the account value between the dates of
  the DJIA's own peak and trough in the window (2020-02-12 and 2020-03-23 in
  `^DJI.csv`, recomputed from the L2 data).
- **Joins:** the first value of each trading window is the previous window's
  last value, so one return is not counted twice; in L1 the 17 duplicate values at
  file seams are dropped (§3.2).
- **Seed averaging.** A claim's estimate is the **mean over seeds of the per-seed
  metric**, not the metric of an averaged return series (which diversification
  across seeds inflates; that Sharpe is reported as a secondary, R0-style figure).
  For L1 the mean is over the 5 unseeded runs.
- **Intervals and p-values:** paired stationary bootstrap (Politis and Romano
  1994), mean block `round(n^(1/3))` = 10 for L2's n = 1,094 returns, 10,000
  resamples, seed 0. Each resample draws one set of dates and applies it to
  every seed, the baseline and every comparator at once, recomputes each
  metric, averages over seeds, and takes the difference to the baseline. The
  95% interval is the 2.5th and 97.5th percentiles; the one-sided p-value is the
  share of resamples with a gap at or below zero. The date bootstrap captures
  market-sample noise, not seed noise; seed spread (min, max, sd) is always
  published beside it.
- **Multiplicity:** Holm-adjusted p-values over the seven gap tests of
  the claims are printed next to the verdicts; verdicts use the
  unadjusted p < 0.05 of the protocol and say if the adjusted value is
  above 0.05. Deflated Sharpe is reported with N = 3 for us and "at least 3, not
  documented" for the authors (§3.5).

## 7. Verdict thresholds

Every quantity is oriented so that larger is better (drawdown as the negative
of its depth in the level, and as depth of the baseline minus depth of the
ensemble in the gap). Per claim, with seed-averaged estimates over the
bootstrap:

- level `m` (the ensemble's own value, claimed `c`), with 95% interval
  `[l_m, u_m]`; per-seed values `m_s`;
- gap `d` to the claim's baseline on the same basis, claimed `c_d`
  (the table below), 95% interval `[l_d, u_d]`, one-sided bootstrap p-value `p_d`
  for `d <= 0`.

Both are "at least" claims: the paper reports what the strategy achieved, so a
result above the claim counts as the claim holding (the report flags "exceeds").
`ens-sharpe-vs-minvar`, `ens-sharpe-vs-agents` and `ens-crash-2020` have a
gap only; their level criterion is dropped below.

| Id | Level `m`, claimed `c` | Gap `d`, claimed `c_d` | Gap baseline |
| --- | --- | --- | --- |
| `ens-sharpe` | Sharpe, 1.30 | Sharpe minus DJIA's, +0.83 | DIA total return (price `^DJI` shown) |
| `ens-sharpe-vs-minvar` | none | Sharpe minus our min-variance's, +0.85 | our min-variance |
| `ens-sharpe-vs-agents` | none | Sharpe minus the best standalone agent's, +0.18 | best of A2C, PPO, DDPG in the same pipeline and seed set (best by seed-mean Sharpe) |
| `ens-cumulative-return` | cumulative return, 70.4% | minus DJIA's, +31.8 points | DIA total return |
| `ens-annual-return` | annual return, 13.0% | minus DJIA's, +5.2 points | DIA total return |
| `ens-max-drawdown` | minus depth, -9.7% | DJIA depth minus ensemble depth, +27.4 points | DIA total return |
| `ens-crash-2020` | none | DJIA crash drawdown minus the ensemble's, with `c_d` = 0.5 x the DJIA's own crash drawdown (about 18.5 points on `^DJI.csv`; recomputed) | DIA total return over the same dates |

Apply in this order:

1. **✅ Holds** if `p_d < 0.05`, `u_d >= c_d`, and (where there is a level)
   `u_m >= c`.
2. **? Inconclusive** if not ✅ and `l_d <= 0 < c_d <= u_d`: the interval can
   neither exclude "no advantage" nor exclude the paper's advantage. The
   report then says what would decide it.
3. **✗ Does not hold** if `d_hat <= 0` with `u_d < c_d`, or (where there is a
   level) no seed has `m_s >= c` **and** `u_m < c`. "No seed reaches the claimed
   value" (protocol) is read with statistical evidence: all seeds below the claim
   and the top of the seed-mean interval below it.
4. **◐ Partly holds** in every remaining case, all of which have `d_hat > 0`:
   significance is lost, or the claimed value lies above the interval while some
   seed reaches it, or the magnitude is below half the claim.

The protocol table does not say what to do when seeds fall below a claim but
its value lies inside the interval, or when significance holds but the claim
lies above the interval; rules 3 and 4 settle both in advance. The levels
rest on different quantities: L1 on 5 unseeded runs of the authors' code, L2
on the seeds of arm A, L3 on the same runs after 2020-05-08, where the level
criterion is dropped for every claim. The ablation (arm B) and sensitivity
(arm C) get no verdict of their own; they are reported as differences from
arm A with paired intervals.

## 8. Smallest effect this test can detect

Window 2016-01-04 to 2020-05-08: 1,094 daily returns, 4.34 years.

- **Sharpe, Lo (2002):** `sqrt((1 + SR^2/2) / years)`: 0.48 at SR 0, 0.51 at the
  DJIA's 0.47, **0.65 at the claimed 1.30**.
- **Bootstrap on the authors' committed series** (§3.1, a stand-in until our
  runs exist; stationary bootstrap, block 10, 3,000 resamples, seed 0): the
  standard deviation of the ensemble's Sharpe is **0.46**, of the DJIA's 0.49,
  and of the **paired difference 0.54** (correlation of the two return series
  only 0.36, because the committed ensemble is in cash a third of the time).
- **Consequences for the claims.** With a paired standard error near 0.54, the
  two-sided 95% minimum detectable Sharpe gap to the DJIA is about **1.05**
  (80% power: about 1.5). The paper's gap is 0.83: the test would exclude zero
  with a probability of about 0.34 (two-sided) to 0.46 (one-sided) even if the
  paper's figure were the true effect. `ens-sharpe` can therefore land on ✅
  only if our estimate is well above the paper's, and ? is the most likely
  verdict unless the effect is large or absent.
- **Ensemble against the best agent** (`ens-sharpe-vs-agents`, gap 0.18): the
  agents' returns are highly correlated. Assuming a correlation of 0.9 (an
  assumption; the paper gives no return series), the standard error of the Sharpe
  difference is about 0.29, so the minimum detectable gap is about **0.57**, three
  times the claim, and the power to see 0.18 is about 0.09. The report states
  that this claim **cannot be resolved by this window** whatever the seeds
  show; seeds shrink algorithmic noise, not the market-sample noise that
  dominates. We say it now so that nobody reads an inconclusive result as a
  finding.
- **Drawdown and cumulative return:** the drawdown is a single path statistic;
  its bootstrap interval is wide, and the 27.4-point gap to the DJIA is
  dominated by one event (the 2020 crash), so `ens-max-drawdown` and
  `ens-crash-2020` rest on one episode and are not independent of each other.
- **L3 window** (about 5.6 years, 1,419 sessions): the Lo standard error at SR 1
  is 0.52 and the paired Sharpe-difference standard error is expected near 0.47
  if the L2 correlation carries over; the combined 2016-2025 window (about 10
  years) has about 0.35. These are estimates; the report replaces them with the
  bootstrap on the realised series.

## 9. Compute (estimate, from a measured throughput probe)

**Probe**, 2026-10-14, this laptop (Apple M5, 10 cores, 32 GiB), the toolkit's
stable-baselines3 models with its default hyperparameters, 30 synthetic tickers
(481-dimensional observation), one thread, process CPU time, the machine at a
load average of 28 to 39 (oversubscribed; real figures on an idle machine are
probably lower). Marginal cost from two run lengths, which removes fixed set-up:

| Agent | Steps (short, long) | CPU seconds (short, long) | Marginal cost |
| --- | --- | --- | --- |
| A2C | 4,000; 12,000 | 6.4; 12.3 | 0.74 ms/step |
| PPO | 4,096; 12,288 | 4.3; 9.6 | 0.65 ms/step |
| DDPG | 2,000; 6,000 | 13.9; 39.9 | 6.5 ms/step |

**Per window of 63 sessions** with the authors' timesteps: selection fits
A2C 30,000 x 0.74 ms = 22 s, PPO 100,000 x 0.65 ms = 65 s, DDPG 10,000 x 6.5 ms
= 65 s, 152 s in all; a refit of the three on train plus validation another
152 s; validation and trading inference about 20 s: **about 5.4 CPU-minutes,
6 fits and 280,000 steps** (the toolkit refits only the winner, about 51 s on
the paper's pick frequencies 6 / 7 / 5, which is 3.7 CPU-minutes and 4 fits; the
extra 2 refits are the standalone agents of §5.2).

| Quantity | Value |
| --- | --- |
| Windows per continuous run, 2016-01-04 to 2025-12-31 | 40 |
| One run, seed and arm | about 3.6 CPU-hours, 240 fits, 11.2 million steps |
| Core set: arms A and B, seeds 0-4 | 10 runs, 36 CPU-hours; about 4.5 hours of wall time at 8 effective parallel processes on 10 cores |
| Everything pre-registered: A and B seeds 0-9, C seeds 0-4 | 25 runs, about 90 CPU-hours, 6,000 fits, 280 million steps; about 11 to 12 hours of wall time |
| Without the standalone agents | 2.5 CPU-hours per run, 62 CPU-hours for all 25 |
| L1, authors' code, one run | 18 windows x 140,000 steps = 2.52 million steps: A2C 6.7 min + PPO 19.5 min + DDPG 19.5 min at our SB3 rates = about 46 minutes, plus inference and pandas overhead: **about 1 hour on one x86 core, unmeasured** (TensorFlow 1 cannot run here) |
| L1, 5 runs | about 5 to 8 core-hours, on 5 runner jobs in parallel |

Memory was not measured. The whole of §9 is an estimate; the first completed
window replaces it with a measurement in the evidence.

## 10. Deviations from the paper's setup (known now)

**L1 against the paper:**

1. Authors' code at the final commit (2021-01-21), not at an identified
   submission version; stable-baselines 2.10.1 chosen, not given.
2. Their shipped data (Compustat, to 2020-08-17) rather than a fresh WRDS pull;
   it is the authors' own file and ships a universe that differs from the stated
   one (§2.3).
3. Their unseeded code run 5 times instead of one run; the account value cut at
   2020-05-08 rather than run to it.
4. Metrics from their notebook recipe (Sharpe with r_f = 0), since the paper's
   r_f is not stated.
5. The DJIA from `data/^DJI.csv`; min-variance cannot be run (no code).
6. Table 1's picks are compared to our runs' but cannot be matched exactly.

**L2 and L3 against the paper:** next-open fills (the paper fills at the signal
close); the stated 2016 universe on Sharadar instead of the shipped list;
total-return prices instead of split-adjusted close (§5.2); PyTorch
stable-baselines3 2.9.0 instead of TensorFlow stable-baselines 2.10.1, with
the settings in the §5.2 table; observation scaling on; winner refit on
train plus validation; returns-based turbulence with a 252-session look-back;
10 seeds; our own min-variance; the DJIA as a total-return ETF proxy; a
window running to 2025-12-31 (L3); the engine charges fees on turnover, as the
paper's 0.1% per trade does, but fills differently. The paper's "continue
training in the trading stage" is not implemented by the toolkit (nor by the
authors' code, §3.4). The paper's turbulence threshold is not stated and
ours is the toolkit's rule, which follows the authors' code in structure.

## 11. Planned analyses

Primary: §6 and §7. Secondary, reported beside the verdicts and never used for
them:

- Per-seed and per-run tables for every arm and level (Sharpe, annual return,
  drawdown, cumulative return, share of sessions in cash).
- **Turbulence ablation** (arm B against arm A): paired bootstrap of the
  difference in Sharpe, return and drawdown; the share of sessions in cash; the
  number of distinct sets of cash sessions across seeds (R0 found one); the
  2020 crash drawdown with and without the rule.
- **Universe sensitivity** (arm C against arm A).
- **Picks:** the picked agent per quarter against Table 1 (agreement count; the
  chance level is 6 of 18, standard deviation 2 quarters), and agreement of the
  pick between seeds.
- Baselines in context: 1/N, monthly 1/N, DIA, `^DJI` price index, our
  min-variance, standalone agents; Sharpe with r_f = FRED DTB3; the blended-return
  Sharpe; the dividend bridge (DIA total return minus `^DJI` price return over
  the same window).
- L1 against the authors' committed results (§3.1) and against L2's prefix.
- Deflated Sharpe with N = 3 (ours) and the authors' unknown N (§3.5).

Anything added after the first run is listed in the report under
"Deviations", with the date and the reason.

## 12. Prerequisites and risks

**Code that does not exist yet and must be committed before the first L2 run**
(each as a separate commit after this file; none changes a verdict rule):

1. Per-agent timesteps for the ensemble (`run_ensemble` passes one
   `--timesteps` to all three agents).
2. `--no-turbulence` honoured by the ensemble (`run_ensemble` forces
   `use_turbulence=True`).
3. Standalone-agent accounts: each of A2C, PPO, DDPG trading its own account,
   refit as the winner is.
4. A driver (Python API) that sets the authors' four indicators, the PPO and
   DDPG settings of §5.2 (the toolkit builds noise objects with a hard-coded
   sigma of 0.1) and Sharadar data via `--data-file`.
5. A min-variance baseline and the Sharadar loader with ticker map.

If any of 1 to 4 is not in place at the first run, the affected claim is
reported as not tested and listed: items 1 and 4 affect every L2 arm, item 2
arm B, item 3 `ens-sharpe-vs-agents`, item 5 `ens-sharpe-vs-minvar`.

**Risks recorded now:**

- L1 environment: TensorFlow 1.15 on CPython 3.7 with MPI may not resolve even
  on the runner; mitigation in §5.1 (L1 reported as not run, §3 recomputation
  labeled L1-lite). This is the one thing that can make L1 infeasible; nothing
  else in the authors' repository needs WRDS or a missing file.
- Sharadar: a mapping gap for UTX or DuPont, or a missing day, would change the
  universe; the report lists it and arm C is unaffected.
- The licence: results are derived metrics, equity curves and hashes only; the
  vendor files stay local. Whether Sharadar's terms allow publishing equity
  curves of a strategy is **[UNVERIFIED]** from here; checked by the owner
  before 2026-12-01.
- Compute contention: this machine is shared with other jobs; wall time may be
  two to four times the §9 figures.
- A late fix of the toolkit during the runs: any change to the engine after the
  first L2 run restarts the affected runs and is reported.

## 13. Prior re-tests (searched 2026-10-14)

Not an exhaustive review. The AI4Finance team's own later papers "reproduce"
the strategy on other windows: FinRL (arXiv 2111.09395, Table 3, test period
2020-07-01 to 2021-06-30: ensemble Sharpe 2.81 against DJIA 2.02, "FinRL
successfully reproduces the ensemble strategy") and FinRL-Meta (arXiv 2211.03107,
2020-07-01 to 2022-03-31, ensemble Sharpe 1.53 against DJIA 1.32, as quoted by a
search summary; **not read in full**). Neither checks Table II. Kong and So
(Applied Sciences 2023, doi 10.3390/app13010633) re-implement the ensemble and
add "Remake Ensemble"; only the abstract was read (the full text was blocked), so
whether it reports seeds is **[UNVERIFIED]**. FinRL issue 190 ("Reproducibility
of DRLs") reports a Sharpe range of 0.1 to 2.7 across re-runs of a tutorial,
as quoted by a search summary (**not read**). "Unstable Gains" (J. Finance
and Data Science, 2026, 20 seeds, uses a FinRL fork) was blocked (HTTP 403)
and is **[UNVERIFIED]**. We found no published multi-seed, next-open re-test of the
ICAIF 2020 ensemble on its own window; the report re-checks before publication.

## 14. Schedule and authors

| Date | Step |
| --- | --- |
| 2026-11-10 | this file committed (deadline); its commit hash is recorded in the report |
| after the commit | download and hash the Sharadar, Yahoo and authors' files; commit §12's code |
| 2026-11-17 | L1 build go/no-go (§5.1) |
| 2026-11-12 to 2026-11-20 | runs (L1 5 jobs; L2 and L3 core set first, then the remaining seeds and arm C) |
| 2026-11-24 | draft report, evidence and reproduction command emailed to the authors |
| 2026-12-01 | publication, with `reply.md` verbatim or "no reply by 2026-12-01" |

Corresponding author: **Xiao-Yang Liu**, xl2427@columbia.edu (camera-ready, p. 1),
with Hongyang Yang, Shan Zhong and Anwar Walid in copy at their addresses in
§2.1, and contact@ai4finance.org (arXiv v1). The Columbia lab page that lists
Liu's current role returned HTTP 403, so the 2020 address was not re-checked;
the draft goes to all of them and a GitHub issue on the pinned repository is
the fallback channel. Their reply is printed verbatim in `reply.md`. Errors they
find are fixed and logged under "Corrections" with the date and credit, as the
protocol says.
