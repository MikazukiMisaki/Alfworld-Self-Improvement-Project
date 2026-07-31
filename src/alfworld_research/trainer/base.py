"""Backend-independent interfaces for preference optimization algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from alfworld_research.preference.builder import PreferenceExample


@dataclass(frozen=True)
class TrainingResult:
    """Stable result object returned by all training implementations."""

    loss: float
    examples_seen: int
    checkpoint_path: str | None = None


class PreferenceTrainer(Protocol):
    """Contract shared by DPO and future preference-based algorithms."""

    def train(self, examples: Sequence[PreferenceExample]) -> TrainingResult:
        """Train on preference examples and return reproducible run information."""
