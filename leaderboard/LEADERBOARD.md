# Leaderboard

Can deep RL beat 1/N? Every row is traded by the same causal execution model (signal at close *t*, fill at open *t+1*, proportional fees, long-only, no leverage). Negative results stay in.

## Historical track

Fixed public split: fit before the test window, then trade it.

### Crypto majors — `crypto-majors`

2023-01-01 to 2025-12-31 (1096 sessions) · fee 0.1% per side · BTCUSDT, ETHUSDT, BNBUSDT, EOSUSDT, BCCUSDT, NEOUSDT, ADAUSDT, LTCUSDT, XRPUSDT, ONTUSDT

| # | Strategy | Kind | Sharpe | CAGR | Max DD | Turnover/yr | Beta | Alpha (t) | ΔSharpe vs 1/N | DSR | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [Minimum variance](../submissions/minimum-variance/) | baseline | 1.64 | 90.9% | -33.3% | 2.6× | 0.70 | 43.1% (t=3.8) | +0.86 (p<0.001) | 1.00 | 1 |
| 2 | [Inverse volatility](../submissions/inverse-volatility/) | baseline | 0.88 | 40.7% | -46.8% | 2.5× | 0.97 | 6.7% (t=1.4) | +0.10 (p=0.089) | 0.94 | 1 |
| 3 | [Momentum (top 3)](../submissions/momentum/) | baseline | 0.84 | 37.8% | -47.9% | 5.3× | 0.89 | 7.8% (t=0.6) | +0.06 (p=0.401) | 0.93 | 1 |
| 4 | **1/N buy and hold** ◆ | baseline | 0.78 | 32.7% | -44.5% | 0.3× | 1.00 | 0.0% | – | 0.91 | – |
| 5 | [1/N monthly rebalance](../submissions/equal-weight-monthly/) | baseline | 0.72 | 28.4% | -51.7% | 1.5× | 1.03 | -3.0% (t=-0.6) | -0.06 (p=0.753) | 0.89 | 1 |
| 6 | [A2C](../submissions/a2c/) | rl | 0.70 ± 0.03 | 26.4% | -45.6% | 15.9× | 0.91 | -3.3% (t=-0.7) | -0.07 (p=0.822) | 0.89 | 5 |
| 7 | [TD3](../submissions/td3/) | rl | 0.66 ± 0.25 | 25.1% | -54.4% | 216.5× | 0.89 | -2.1% (t=-0.2) | -0.08 (p=0.657) | 0.89 | 5 |
| 8 | [PPO](../submissions/ppo/) | rl | 0.66 ± 0.06 | 23.6% | -54.1% | 145.5× | 0.91 | -4.3% (t=-0.6) | -0.10 (p=0.778) | 0.88 | 5 |
| 9 | [DDPG](../submissions/ddpg/) | rl | 0.53 ± 0.17 | 14.6% | -53.8% | 214.7× | 0.85 | -9.9% (t=-0.8) | -0.22 (p=0.920) | 0.83 | 5 |
| 10 | [SAC](../submissions/sac/) | rl | 0.44 ± 0.20 | 10.0% | -59.1% | 326.4× | 0.84 | -11.6% (t=-0.7) | -0.27 (p=0.899) | 0.81 | 5 |
| 11 | [Trend following (200d SMA)](../submissions/trend-following/) | baseline | 0.10 | -4.7% | -50.6% | 3.3× | 0.61 | -23.6% (t=-1.9) | -0.68 (p=0.971) | 0.57 | 1 |
| 12 | [LSTM forecast](../submissions/lstm-forecast/) | forecast | -0.32 ± 0.25 | -26.6% | -73.3% | 246.7× | 0.58 | -44.1% (t=-1.9) | -1.13 (p=0.996) | 0.27 | 5 |
| 13 | [Cash](../submissions/cash/) | baseline | – | 0.0% | 0.0% | 0.0× | 0.00 | 0.0% | – | – | 1 |

### US multi-asset allocation — `us-multi-asset`

2018-01-02 to 2025-12-31 (2011 sessions) · fee 0.05% per side · SPY, QQQ, IWM, EFA, EEM, TLT, IEF, LQD, GLD, VNQ

| # | Strategy | Kind | Sharpe | CAGR | Max DD | Turnover/yr | Beta | Alpha (t) | ΔSharpe vs 1/N | DSR | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **1/N buy and hold** ◆ | baseline | 0.73 | 8.9% | -26.6% | 0.1× | 1.00 | 0.0% | – | 0.98 | – |
| 2 | [Momentum (top 3)](../submissions/momentum/) | baseline | 0.72 | 9.9% | -23.7% | 5.0× | 0.87 | 2.4% (t=0.7) | -0.01 (p=0.506) | 0.98 | 1 |
| 3 | [1/N monthly rebalance](../submissions/equal-weight-monthly/) | baseline | 0.69 | 8.0% | -25.8% | 0.4× | 0.95 | -0.4% (t=-0.8) | -0.04 (p=0.827) | 0.97 | 1 |
| 4 | [A2C](../submissions/a2c/) | rl | 0.66 ± 0.02 | 7.1% | -23.8% | 7.2× | 0.87 | -0.6% (t=-1.2) | -0.06 (p=0.940) | 0.97 | 5 |
| 5 | [Inverse volatility](../submissions/inverse-volatility/) | baseline | 0.66 | 6.0% | -24.2% | 1.1× | 0.72 | -0.4% (t=-0.4) | -0.07 (p=0.777) | 0.97 | 1 |
| 6 | [LSTM forecast](../submissions/lstm-forecast/) | forecast | 0.60 ± 0.08 | 8.4% | -30.7% | 34.9× | 1.09 | -1.0% (t=-0.6) | -0.10 (p=0.868) | 0.96 | 5 |
| 7 | [PPO](../submissions/ppo/) | rl | 0.60 ± 0.01 | 6.5% | -24.2% | 24.1× | 0.89 | -1.3% (t=-2.3) | -0.12 (p=0.999) | 0.95 | 5 |
| 8 | [Trend following (200d SMA)](../submissions/trend-following/) | baseline | 0.57 | 4.2% | -15.6% | 2.4× | 0.49 | -0.1% (t=-0.1) | -0.16 (p=0.774) | 0.94 | 1 |
| 9 | [Minimum variance](../submissions/minimum-variance/) | baseline | 0.54 | 3.3% | -21.4% | 1.0× | 0.34 | 0.3% (t=0.2) | -0.19 (p=0.771) | 0.93 | 1 |
| 10 | [SAC](../submissions/sac/) | rl | 0.52 ± 0.15 | 5.5% | -27.2% | 45.9× | 0.88 | -2.1% (t=-2.1) | -0.20 (p=1.000) | 0.93 | 5 |
| 11 | [TD3](../submissions/td3/) | rl | 0.27 ± 0.06 | 2.8% | -30.2% | 148.9× | 0.89 | -4.6% (t=-2.0) | -0.43 (p=1.000) | 0.80 | 5 |
| 12 | [DDPG](../submissions/ddpg/) | rl | 0.21 ± 0.19 | 2.0% | -32.1% | 155.4× | 0.92 | -5.6% (t=-2.8) | -0.50 (p=1.000) | 0.74 | 5 |
| 13 | [Cash](../submissions/cash/) | baseline | – | 0.0% | 0.0% | 0.0× | 0.00 | 0.0% | – | – | 1 |

### US sector rotation — `us-sectors`

2018-01-02 to 2025-12-31 (2011 sessions) · fee 0.05% per side · XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY

| # | Strategy | Kind | Sharpe | CAGR | Max DD | Turnover/yr | Beta | Alpha (t) | ΔSharpe vs 1/N | DSR | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [1/N monthly rebalance](../submissions/equal-weight-monthly/) | baseline | 0.69 | 11.6% | -36.7% | 0.4× | 1.00 | 0.2% (t=0.3) | +0.01 (p=0.449) | 0.97 | 1 |
| 2 | **1/N buy and hold** ◆ | baseline | 0.68 | 11.5% | -35.5% | 0.1× | 1.00 | 0.0% | – | 0.97 | – |
| 3 | [A2C](../submissions/a2c/) | rl | 0.67 ± 0.01 | 10.3% | -34.3% | 10.0× | 0.91 | -0.2% (t=-0.3) | -0.01 (p=0.650) | 0.97 | 5 |
| 4 | [Inverse volatility](../submissions/inverse-volatility/) | baseline | 0.67 | 10.8% | -35.7% | 0.8× | 0.96 | -0.2% (t=-0.3) | -0.02 (p=0.664) | 0.97 | 1 |
| 5 | [Momentum (top 3)](../submissions/momentum/) | baseline | 0.67 | 11.7% | -31.6% | 5.7× | 0.97 | 0.8% (t=0.3) | -0.02 (p=0.522) | 0.97 | 1 |
| 6 | [PPO](../submissions/ppo/) | rl | 0.61 ± 0.06 | 9.1% | -33.7% | 28.8× | 0.90 | -1.2% (t=-1.5) | -0.08 (p=0.979) | 0.95 | 5 |
| 7 | [LSTM forecast](../submissions/lstm-forecast/) | forecast | 0.60 ± 0.04 | 9.8% | -38.9% | 41.2× | 0.95 | -0.9% (t=-0.5) | -0.07 (p=0.827) | 0.96 | 5 |
| 8 | [Trend following (200d SMA)](../submissions/trend-following/) | baseline | 0.59 | 6.0% | -17.4% | 2.5× | 0.51 | -0.0% (t=-0.0) | -0.10 (p=0.695) | 0.95 | 1 |
| 9 | [Minimum variance](../submissions/minimum-variance/) | baseline | 0.55 | 7.7% | -32.9% | 1.6× | 0.76 | -1.0% (t=-0.4) | -0.13 (p=0.820) | 0.94 | 1 |
| 10 | [SAC](../submissions/sac/) | rl | 0.52 ± 0.12 | 7.8% | -33.2% | 74.3× | 0.90 | -2.3% (t=-1.3) | -0.15 (p=0.987) | 0.93 | 5 |
| 11 | [DDPG](../submissions/ddpg/) | rl | 0.31 ± 0.14 | 4.2% | -39.5% | 134.6× | 0.97 | -6.2% (t=-2.7) | -0.36 (p=1.000) | 0.82 | 5 |
| 12 | [TD3](../submissions/td3/) | rl | 0.31 ± 0.16 | 4.0% | -40.4% | 120.9× | 0.94 | -6.2% (t=-2.7) | -0.37 (p=1.000) | 0.81 | 5 |
| 13 | [Cash](../submissions/cash/) | baseline | – | 0.0% | 0.0% | 0.0× | 0.00 | 0.0% | – | – | 1 |

### Re-evaluation: Wang & Zhang (2021) setup — `paper-2021`

2019-01-02 to 2020-12-31 (505 sessions) · fee 0.1% per side · AAPL, MSFT, AMZN, BRK-B, JPM, JNJ, UNH, HD, PG, NVDA, DIS, BAC, CMCSA, XOM, VZ, T, ADBE, PFE, CSCO, INTC

*Survivorship-biased universe (chosen in hindsight): this re-tests a published setup with this benchmark's execution model and is not a ranking.*

| # | Strategy | Kind | Sharpe | CAGR | Max DD | Turnover/yr | Beta | Alpha (t) | ΔSharpe vs 1/N | DSR | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| – | [Momentum (top 3)](../submissions/momentum/) | baseline | 1.31 | 48.6% | -30.4% | 6.7× | 1.19 | 10.5% (t=1.0) | +0.20 (p=0.240) | 0.96 | 1 |
| – | **1/N buy and hold** ◆ | baseline | 1.11 | 29.9% | -29.4% | 0.5× | 1.00 | 0.0% | – | 0.94 | – |
| – | [1/N monthly rebalance](../submissions/equal-weight-monthly/) | baseline | 1.04 | 26.7% | -29.5% | 1.0× | 0.96 | -1.5% (t=-0.6) | -0.07 (p=0.777) | 0.93 | 1 |
| – | [A2C](../submissions/a2c/) | rl | 1.03 ± 0.01 | 25.1% | -28.2% | 9.2× | 0.92 | -1.8% (t=-0.8) | -0.08 (p=0.825) | 0.92 | 5 |
| – | [Inverse volatility](../submissions/inverse-volatility/) | baseline | 1.01 | 24.5% | -28.9% | 1.7× | 0.91 | -2.1% (t=-0.8) | -0.10 (p=0.843) | 0.92 | 1 |
| – | [PPO](../submissions/ppo/) | rl | 0.98 ± 0.04 | 23.9% | -28.1% | 22.5× | 0.92 | -2.9% (t=-1.2) | -0.13 (p=0.898) | 0.91 | 5 |
| – | [Minimum variance](../submissions/minimum-variance/) | baseline | 0.81 | 15.6% | -25.8% | 2.7× | 0.70 | -4.2% (t=-0.7) | -0.31 (p=0.850) | 0.87 | 1 |
| – | [LSTM forecast](../submissions/lstm-forecast/) | forecast | 0.75 ± 0.29 | 17.9% | -26.5% | 79.1× | 0.81 | -4.4% (t=-0.5) | -0.27 (p=0.831) | 0.88 | 5 |
| – | [Trend following (200d SMA)](../submissions/trend-following/) | baseline | 0.74 | 11.3% | -20.1% | 2.6× | 0.58 | -5.2% (t=-1.5) | -0.38 (p=0.960) | 0.85 | 1 |
| – | [DDPG](../submissions/ddpg/) | rl | 0.70 ± 0.17 | 16.2% | -29.3% | 98.7× | 0.95 | -9.9% (t=-2.2) | -0.40 (p=0.985) | 0.84 | 5 |
| – | [TD3](../submissions/td3/) | rl | 0.70 ± 0.21 | 15.7% | -29.3% | 102.4× | 0.91 | -9.4% (t=-1.7) | -0.40 (p=0.977) | 0.84 | 5 |
| – | [SAC](../submissions/sac/) | rl | 0.66 ± 0.26 | 15.1% | -30.2% | 122.8× | 0.95 | -10.8% (t=-2.2) | -0.43 (p=0.998) | 0.83 | 5 |
| – | [Cash](../submissions/cash/) | baseline | – | 0.0% | 0.0% | 0.0× | 0.00 | 0.0% | – | – | 1 |

◆ marks the 1/N buy-and-hold benchmark (equal weights at the start of the period, never rebalanced); every ΔSharpe is measured against it over the same sessions.

**Metrics.** Computed from daily simple returns with a zero risk-free rate. *Sharpe*: mean/std × √(periods per year); with several seeds, mean ± standard deviation of per-seed Sharpe ratios. *CAGR*, *Max DD* (peak-to-trough, including the starting value) and *Beta*/*Alpha* (OLS of daily returns on the benchmark's; alpha annualised, *t* is Newey-West) are means across seeds. *Turnover/yr*: annualised one-way traded fraction of equity. *ΔSharpe*: annualised Sharpe of the seed-averaged returns minus the benchmark's, with a one-sided paired stationary-bootstrap p-value (H1: better than 1/N). *DSR*: deflated Sharpe ratio of the seed-averaged returns, i.e. the probability the true Sharpe is positive after accounting for the submission's `trials` and for the dispersion of Sharpe ratios across this table (Bailey & López de Prado, 2014).
