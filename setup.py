from pathlib import Path

from setuptools import setup, find_packages

# Read requirements.in, ignore comments
REQUIRES = []
with Path(__file__).with_name("requirements.in").open(encoding="utf-8") as f:
    for line in f:
        line = line.split("#", 1)[0].strip()
        if line:
            REQUIRES.append(line)

REPOSITORY = "https://github.com/BichengWang/RL-pricing"

# The PyPI page. The repository README carries the verdicts and leaderboard,
# which CI rewrites between releases, so they are linked rather than copied.
LONG_DESCRIPTION = """\
# RL-pricing

A reproducibility lab for trading AI. RL-pricing re-tests published claims
about deep RL agents, LLM trading agents and forecasters under one
pre-registered protocol. This package holds the lab's instruments:

- `rl-pricing bench`: a causal backtesting benchmark. Policies decide at the
  close and fill at the next open with fees; learned policies run over
  several seeds; frozen policies are scored on a live forward-test track; and
  a decision ledger records LLM agents' trades before each US market open.
- `rl-pricing --mode train|backtest|ensemble|lstm`: the share-trading toolkit
  that trains and backtests Stable-Baselines3 agents (A2C, PPO, DDPG, TD3,
  SAC), a rolling ensemble and an LSTM forecaster on daily equity data.

```bash
pip install rl-pricing
rl-pricing bench tasks             # list the benchmark tasks
rl-pricing bench data us-sectors   # download a task's data and print its hash
```

The import package is `rl`, so this distribution can't share an environment
with the unrelated `rl` (GNU Readline bindings) distribution. Leaderboard
submissions, results and reports live in the
[repository](https://github.com/BichengWang/RL-pricing): clone it to run the
submissions, or pass `--submissions-root` to use your own.

- [Verdicts, findings and leaderboard](https://github.com/BichengWang/RL-pricing#readme)
- [Paper: Wang & Zhang, arXiv:2609.31870](https://arxiv.org/abs/2609.31870)

The training toolkit's environment, preprocessing and training code are
adapted from [FinRL](https://github.com/AI4Finance-Foundation/FinRL), which is
MIT-licensed.
"""

setup(
    name="rl-pricing",
    version="0.5.0",
    include_package_data=True,
    author="Bicheng Wang",
    url=REPOSITORY,
    project_urls={
        "Source": REPOSITORY,
        "Issues": REPOSITORY + "/issues",
        "Leaderboard": REPOSITORY + "/blob/main/leaderboard/LEADERBOARD.md",
        "Paper": "https://arxiv.org/abs/2609.31870",
    },
    license_expression="MIT",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=REQUIRES,
    extras_require={"test": ["pytest>=7.0"], "lint": ["ruff>=0.5"]},
    entry_points={"console_scripts": ["rl-pricing=rl.cli:entrypoint"]},
    description=(
        "A reproducibility lab for trading AI: causal backtests, a live forward-test "
        "track and a pre-open decision ledger for RL, LLM and forecasting policies."
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.org/classifiers/
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="reinforcement learning, portfolio management, backtesting, benchmark, leaderboard",
    platforms=["any"],
    python_requires=">=3.10",
)
