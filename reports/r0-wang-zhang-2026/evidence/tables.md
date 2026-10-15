### Sharpe ratio, mean ± sd over 5 seeds [min, max]

| Agent | Reported | Share env + turbulence rule | Share env, no rule | Benchmark `paper-2021` |
| --- | --- | --- | --- | --- |
| DDPG | 1.38 | 1.16 ± 0.19 [0.92, 1.43] | 1.06 ± 0.18 [0.80, 1.24] | 0.70 ± 0.17 [0.57, 0.97] |
| TD3 | 1.37 | 1.10 ± 0.34 [0.57, 1.37] | 1.00 ± 0.30 [0.53, 1.37] | 0.70 ± 0.21 [0.43, 1.01] |
| SAC | 1.33 | 1.23 ± 0.13 [1.07, 1.42] | 1.23 ± 0.18 [1.01, 1.49] | 0.66 ± 0.26 [0.19, 0.82] |
| PPO | 1.09 | 1.17 ± 0.21 [0.80, 1.32] | 0.95 ± 0.21 [0.74, 1.21] | 0.98 ± 0.04 [0.93, 1.02] |
| A2C | 0.82 | 1.09 ± 0.15 [0.91, 1.27] | 0.97 ± 0.15 [0.80, 1.16] | 1.03 ± 0.01 [1.01, 1.05] |
| LSTM forecast | 1.19 | – | – | 0.75 ± 0.29 [0.45, 1.08] |
| *1/N buy and hold* | – | *1.10* | *1.10* | *1.11* |

### Annual return, mean ± sd over 5 seeds [min, max]

| Agent | Reported | Share env + turbulence rule | Share env, no rule | Benchmark `paper-2021` |
| --- | --- | --- | --- | --- |
| DDPG | 55.51% | 24.6% ± 5.9 [15.9%, 30.4%] | 30.6% ± 8.3 [18.5%, 40.5%] | 16.2% ± 5.4 [12.0%, 25.0%] |
| TD3 | 42.00% | 23.3% ± 9.4 [8.8%, 31.3%] | 28.7% ± 12.3 [10.8%, 44.9%] | 15.7% ± 5.8 [8.5%, 24.4%] |
| SAC | 38.95% | 25.1% ± 2.5 [21.1%, 27.8%] | 38.5% ± 7.0 [27.8%, 46.0%] | 15.1% ± 7.6 [1.6%, 19.7%] |
| PPO | 27.89% | 16.0% ± 4.7 [10.6%, 22.3%] | 25.3% ± 7.9 [15.6%, 33.3%] | 23.9% ± 1.2 [22.3%, 25.4%] |
| A2C | 20.47% | 21.5% ± 5.1 [16.5%, 28.2%] | 26.0% ± 6.5 [19.0%, 35.4%] | 25.1% ± 0.4 [24.7%, 25.8%] |
| LSTM forecast | 16.99% | – | – | 17.9% ± 9.5 [9.1%, 28.3%] |
| *1/N buy and hold* | – | *29.7%* | *29.7%* | *29.9%* |

### Maximum drawdown, mean over 5 seeds [worst, best]

| Agent | Reported | Share env + turbulence rule | Share env, no rule | Benchmark `paper-2021` |
| --- | --- | --- | --- | --- |
| DDPG | -33.52% | -18.0% [-18.6%, -17.4%] | -27.8% [-30.8%, -25.7%] | -29.3% [-31.7%, -27.4%] |
| TD3 | -25.24% | -19.0% [-20.2%, -17.5%] | -28.2% [-32.7%, -25.0%] | -29.3% [-33.7%, -26.5%] |
| SAC | -24.19% | -17.7% [-21.8%, -14.8%] | -27.0% [-32.2%, -22.5%] | -30.2% [-33.7%, -28.1%] |
| PPO | -25.33% | -9.2% [-11.7%, -6.7%] | -27.4% [-32.0%, -25.3%] | -28.1% [-28.6%, -27.6%] |
| A2C | -31.11% | -17.2% [-18.9%, -14.5%] | -29.0% [-31.2%, -26.5%] | -28.2% [-28.3%, -28.0%] |
| LSTM forecast | −9.57% | – | – | -26.5% [-28.8%, -23.8%] |
| *1/N buy and hold* | – | – | – | *-29.4%* |

### Sharpe against 1/N, seed-averaged returns (one-sided paired stationary bootstrap)

Share env (A, B): from `paper-2021-share-env-equity.json`, alpha, beta and the Newey-West t of the
seed-averaged daily returns. Benchmark (C): alpha, beta and t as in `leaderboard/LEADERBOARD.md`
(per-seed values averaged). 'Seeds above' counts seeds whose own Sharpe exceeds that setup's 1/N Sharpe.

| Setup | Agent | Sharpe of seed-avg returns | ΔSharpe vs 1/N | p | Alpha | Alpha t (HAC) | Beta | Seeds above 1/N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Share env + turbulence rule | DDPG | 1.22 | +0.12 | 0.320 | 7.4% | 0.9 | 0.56 | 3/5 |
| Share env + turbulence rule | TD3 | 1.20 | +0.09 | 0.328 | 6.4% | 0.9 | 0.55 | 3/5 |
| Share env + turbulence rule | SAC | 1.33 | +0.22 | 0.218 | 8.9% | 1.2 | 0.52 | 4/5 |
| Share env + turbulence rule | PPO | 1.26 | +0.15 | 0.339 | 6.7% | 1.1 | 0.30 | 4/5 |
| Share env + turbulence rule | A2C | 1.14 | +0.04 | 0.398 | 5.4% | 0.7 | 0.54 | 2/5 |
| Share env, no rule | DDPG | 1.11 | +0.00 | 0.496 | 1.2% | 0.3 | 1.00 | 3/5 |
| Share env, no rule | TD3 | 1.06 | -0.04 | 0.654 | -0.2% | -0.0 | 0.99 | 1/5 |
| Share env, no rule | SAC | 1.32 | +0.22 | 0.133 | 7.5% | 1.6 | 1.00 | 4/5 |
| Share env, no rule | PPO | 1.00 | -0.11 | 0.713 | -2.0% | -0.4 | 0.95 | 1/5 |
| Share env, no rule | A2C | 0.99 | -0.11 | 0.792 | -2.3% | -0.6 | 0.98 | 1/5 |
| Benchmark `paper-2021` | DDPG | 0.71 | -0.40 | 0.985 | -9.9% | -2.2 | 0.95 | 0/5 |
| Benchmark `paper-2021` | TD3 | 0.71 | -0.40 | 0.977 | -9.4% | -1.7 | 0.91 | 0/5 |
| Benchmark `paper-2021` | SAC | 0.68 | -0.43 | 0.998 | -10.8% | -2.2 | 0.95 | 0/5 |
| Benchmark `paper-2021` | PPO | 0.98 | -0.13 | 0.898 | -2.9% | -1.2 | 0.92 | 0/5 |
| Benchmark `paper-2021` | A2C | 1.03 | -0.08 | 0.825 | -1.8% | -0.8 | 0.92 | 0/5 |
| Benchmark `paper-2021` | LSTM forecast | 0.84 | -0.27 | 0.831 | -4.4% | -0.5 | 0.81 | 0/5 |

### Sessions with unchanged equity (|daily return| < 1e-9)

| Setting | Runs | Runs with no such session | Distinct sets of such sessions | Sessions |
| --- | --- | --- | --- | --- |
| Share env + turbulence rule | 25 | 0 | 1 | 2020-03-17, 2020-03-20, 2020-03-23, 2020-03-24, 2020-03-25 |
| Share env, no rule | 25 | 25 | 1 | – |

### Per-seed Sharpe ratio

| Setup | Agent | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
| --- | --- | --- | --- | --- | --- | --- |
| Share env + turbulence rule | DDPG | 1.43 | 1.08 | 1.18 | 1.20 | 0.92 |
| Share env + turbulence rule | TD3 | 1.37 | 0.98 | 0.57 | 1.28 | 1.31 |
| Share env + turbulence rule | SAC | 1.42 | 1.07 | 1.27 | 1.21 | 1.20 |
| Share env + turbulence rule | PPO | 1.32 | 1.19 | 0.80 | 1.32 | 1.21 |
| Share env + turbulence rule | A2C | 1.27 | 1.09 | 1.20 | 1.00 | 0.91 |
| Share env, no rule | DDPG | 1.24 | 0.98 | 1.11 | 1.18 | 0.80 |
| Share env, no rule | TD3 | 1.37 | 0.96 | 0.53 | 1.05 | 1.07 |
| Share env, no rule | SAC | 1.49 | 1.01 | 1.33 | 1.17 | 1.14 |
| Share env, no rule | PPO | 1.09 | 0.76 | 0.74 | 0.94 | 1.21 |
| Share env, no rule | A2C | 1.16 | 0.98 | 1.03 | 0.80 | 0.85 |
| Benchmark `paper-2021` | DDPG | 0.57 | 0.61 | 0.97 | 0.77 | 0.59 |
| Benchmark `paper-2021` | TD3 | 0.72 | 1.01 | 0.61 | 0.72 | 0.43 |
| Benchmark `paper-2021` | SAC | 0.19 | 0.82 | 0.71 | 0.77 | 0.80 |
| Benchmark `paper-2021` | PPO | 0.93 | 1.02 | 0.98 | 0.97 | 1.01 |
| Benchmark `paper-2021` | A2C | 1.02 | 1.02 | 1.05 | 1.01 | 1.03 |
| Benchmark `paper-2021` | LSTM forecast | 0.63 | 0.56 | 1.08 | 1.05 | 0.45 |

### Per-seed annual return

| Setup | Agent | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
| --- | --- | --- | --- | --- | --- | --- |
| Share env + turbulence rule | DDPG | 30.4% | 21.9% | 25.7% | 29.0% | 15.9% |
| Share env + turbulence rule | TD3 | 31.3% | 18.8% | 8.8% | 28.9% | 28.7% |
| Share env + turbulence rule | SAC | 27.8% | 21.1% | 25.3% | 26.2% | 25.1% |
| Share env + turbulence rule | PPO | 22.3% | 12.2% | 10.6% | 17.3% | 17.5% |
| Share env + turbulence rule | A2C | 28.2% | 21.9% | 24.5% | 16.5% | 16.6% |
| Share env, no rule | DDPG | 35.2% | 27.1% | 31.8% | 40.5% | 18.5% |
| Share env, no rule | TD3 | 44.9% | 25.9% | 10.8% | 29.9% | 32.3% |
| Share env, no rule | SAC | 46.0% | 27.8% | 42.3% | 40.6% | 35.8% |
| Share env, no rule | PPO | 32.8% | 15.6% | 19.3% | 25.3% | 33.3% |
| Share env, no rule | A2C | 35.4% | 26.3% | 28.1% | 19.0% | 21.0% |
| Benchmark `paper-2021` | DDPG | 12.0% | 12.3% | 25.0% | 17.6% | 14.0% |
| Benchmark `paper-2021` | TD3 | 16.1% | 24.4% | 13.7% | 15.9% | 8.5% |
| Benchmark `paper-2021` | SAC | 1.6% | 19.5% | 17.3% | 17.3% | 19.7% |
| Benchmark `paper-2021` | PPO | 22.3% | 25.4% | 23.6% | 23.5% | 24.5% |
| Benchmark `paper-2021` | A2C | 24.9% | 25.0% | 25.8% | 24.7% | 25.3% |

### Verdict by setup (same rule everywhere)

| Claim | Share env + turbulence rule | Share env, no rule | Benchmark `paper-2021` |
| --- | --- | --- | --- |
| `ddpg-annual-return` | ✗ | ✗ | ✗ |
| `ddpg-sharpe` | ◐ | ✗ | ✗ |
| `td3-sharpe` | ◐ | ◐ | ✗ |
| `sac-sharpe` | ◐ | ◐ | ✗ |
| `ppo-sharpe` | ◐ | ◐ | ✗ |
| `a2c-sharpe` | ◐ | ◐ | ◐ |
