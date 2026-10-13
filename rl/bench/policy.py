"""The interface every benchmarked strategy implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import numpy as np

from rl.bench.engine import Decision
from rl.bench.market import MarketData, MarketView


class Policy(ABC):
    """Maps point-in-time market history to target portfolio weights.

    Lifecycle: the harness constructs the policy from JSON parameters, calls
    :meth:`fit` once with history that ends before the evaluated period,
    then :meth:`reset` and :meth:`act` at every decision close in order.
    A policy frozen for live evaluation is restored by constructing it with
    the same parameters and calling :meth:`load` instead of :meth:`fit`; it
    must then act exactly as after :meth:`save`.

    ``act`` returns long-only weights (one per ticker, summing to at most
    one, the remainder held as cash) or ``None`` to keep current holdings.
    """

    #: Whether :meth:`fit` depends on ``seed``. Stochastic policies are
    #: evaluated over several seeds; deterministic ones once.
    stochastic: ClassVar[bool] = False

    def fit(self, history: MarketData, validation_start: np.datetime64, seed: int) -> None:
        """Learn from ``history``.

        Sessions on or after ``validation_start`` are reserved for model
        selection: train on the earlier sessions and use the later ones only
        to choose among candidates (for example with
        :func:`rl.bench.engine.simulate`).
        """

    def reset(self) -> None:
        """Clear per-episode state before an evaluation starts."""

    @abstractmethod
    def act(self, view: MarketView) -> Decision:
        """Target weights decided at ``view.date``'s close."""

    def save(self, directory: Path) -> None:
        """Write fitted state needed to restore the policy."""

    def load(self, directory: Path) -> None:
        """Restore fitted state written by :meth:`save`."""
