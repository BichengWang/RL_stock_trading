from pathlib import Path

from setuptools import setup, find_packages

# Read requirements.in, ignore comments
REQUIRES = []
with Path(__file__).with_name("requirements.in").open(encoding="utf-8") as f:
    for line in f:
        line = line.split("#", 1)[0].strip()
        if line:
            REQUIRES.append(line)

setup(
    name="rl",
    version="0.4.0",
    include_package_data=True,
    author="Bicheng Wang",
    url="https://github.com/BichengWang/RL-pricing",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=REQUIRES,
    extras_require={"test": ["pytest>=7.0"], "lint": ["ruff>=0.5"]},
    entry_points={"console_scripts": ["rl-pricing=rl.cli:entrypoint"]},
    description="An honest, live-forward-tested benchmark for reinforcement learning portfolio policies.",
    long_description="""RL-pricing evaluates portfolio policies (deep RL agents, forecasters and
    rule-based baselines) under one causal execution model: signals at the close,
    fills at the next open, proportional fees, survivorship-free universes, multiple
    seeds and a live forward-test track scored only on data after each policy is frozen.

    The training toolkit's environment, preprocessing and training code are adapted
    from FinRL (https://github.com/AI4Finance-Foundation/FinRL), which is MIT-licensed.
    """,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    keywords="reinforcement learning, portfolio management, backtesting, benchmark, leaderboard",
    platforms=["any"],
    python_requires=">=3.10",
)
