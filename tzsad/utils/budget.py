"""A wall-clock budget, so a batch run finishes tidily instead of being killed.

Kaggle's scheduled ("Save & Run All") notebooks are terminated at a hard limit -
9h or 12h depending on the accelerator - and anything still buffered when that
happens is lost. Worse, the notebook's *output* is what persists between scheduled
runs, so a run killed mid-write can leave nothing to resume from.

A stage therefore asks the budget whether it has room for another unit of work and
stops cleanly while there is still time to flush checkpoints and write reports.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .logging import get_logger

log = get_logger("utils.budget")


@dataclass
class TimeBudget:
    """Wall-clock budget with a reserve held back for shutdown work.

    Parameters
    ----------
    total_seconds:
        Everything this process is allowed, measured from construction.
    reserve_seconds:
        Held back for flushing checkpoints, writing reports and saving figures.
        :meth:`should_stop` turns true once only the reserve remains.
    """

    total_seconds: float
    reserve_seconds: float = 600.0
    started: float = field(default_factory=time.monotonic)
    _unit_times: list[float] = field(default_factory=list)

    @property
    def elapsed(self) -> float:
        """Seconds since the budget started."""
        return time.monotonic() - self.started

    @property
    def remaining(self) -> float:
        """Seconds left before the hard limit."""
        return max(self.total_seconds - self.elapsed, 0.0)

    @property
    def usable(self) -> float:
        """Seconds left before the reserve must be respected."""
        return max(self.remaining - self.reserve_seconds, 0.0)

    def should_stop(self) -> bool:
        """Whether to stop starting new units of work."""
        return self.usable <= 0.0

    def record_unit(self, seconds: float) -> None:
        """Log how long one unit took, for the room-for-another estimate."""
        self._unit_times.append(seconds)

    @property
    def mean_unit(self) -> float:
        """Mean duration of recent units (the last 20), or 0 if none recorded."""
        recent = self._unit_times[-20:]
        return sum(recent) / len(recent) if recent else 0.0

    def has_room(self, safety: float = 1.3) -> bool:
        """Whether another unit is likely to fit, using the observed unit time.

        Stopping *before* starting a unit that cannot finish is what keeps the
        checkpoint consistent - a half-scored image is not a resumable state.
        """
        if self.should_stop():
            return False
        if not self._unit_times:
            return True
        return self.usable >= self.mean_unit * safety

    def report(self, label: str = "") -> str:
        """Human-readable budget line for the log."""
        return (f"{label}elapsed {self.elapsed / 60:.1f}m, remaining "
                f"{self.remaining / 60:.1f}m (usable {self.usable / 60:.1f}m)")


class UnitTimer:
    """Context manager that records a unit's duration into a budget."""

    def __init__(self, budget: TimeBudget | None) -> None:
        self.budget = budget
        self._t0 = 0.0

    def __enter__(self) -> "UnitTimer":
        self._t0 = time.monotonic()
        return self

    def __exit__(self, *exc: object) -> None:
        if self.budget is not None:
            self.budget.record_unit(time.monotonic() - self._t0)
