"""Typed immutable records for reproducible interaction traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class Step:
    """One environment transition plus optional model diagnostics."""

    observation: str
    action: str
    reward: float
    done: bool
    reasoning: str | None = None
    confidence: float | None = None
    entropy: float | None = None
    info: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(frozen=True)
class Trajectory:
    """A complete episode that can be serialized without model dependencies."""

    initial_observation: str
    steps: tuple[Step, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_reward(self) -> float:
        """Return the undiscounted episode reward."""
        return sum(step.reward for step in self.steps)

    @property
    def succeeded(self) -> bool:
        """Use explicit success metadata when available, otherwise final reward."""
        if "success" in self.metadata:
            return bool(self.metadata["success"])
        return bool(self.steps and self.steps[-1].reward > 0)

    def to_dict(self) -> dict[str, Any]:
        """Produce a JSON-serializable representation."""
        return {"initial_observation": self.initial_observation, "steps": [asdict(s) for s in self.steps], "metadata": self.metadata}
