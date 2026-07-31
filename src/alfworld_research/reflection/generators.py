"""Pluggable trajectory reflection contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from alfworld_research.trajectory.trajectory import Trajectory


@dataclass(frozen=True)
class Reflection:
    """Structured reflection suitable for later preference construction."""

    analysis: str
    suggested_action: str | None = None
    confidence: float | None = None


class ReflectionGenerator(Protocol):
    """Generate an interpretable reflection from one completed trajectory."""

    def generate(self, trajectory: Trajectory) -> Reflection:
        """Return reflection information without mutating the trajectory."""


class TerminalReflectionGenerator:
    """Deterministic baseline useful for smoke tests and pipeline wiring."""

    def generate(self, trajectory: Trajectory) -> Reflection:
        """Describe whether the observed episode reached a successful terminal state."""
        if trajectory.succeeded:
            return Reflection("The trajectory succeeded; preserve the successful action pattern.", confidence=1.0)
        action = trajectory.steps[-1].action if trajectory.steps else None
        return Reflection("The trajectory failed; inspect the final decision and state transition.", action, 0.0)
