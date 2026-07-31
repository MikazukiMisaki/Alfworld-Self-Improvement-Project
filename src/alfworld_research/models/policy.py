"""Model policy interfaces independent of a serving backend."""

from __future__ import annotations

from typing import Protocol


class ActionPolicy(Protocol):
    """Select an action given an observation and preceding interaction."""

    def act(self, observation: str, history: tuple[str, ...]) -> str:
        """Return one environment-valid textual action."""
