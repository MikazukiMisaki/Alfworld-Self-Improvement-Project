"""Environment contracts shared by collection and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Task:
    """A stable description of one environment task."""

    task_id: str
    text: str
    split: str
    family: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResetResult:
    """Initial state returned when an environment episode starts."""

    observation: str
    task: Task
    valid_actions: tuple[str, ...] | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Transition:
    """One normalized environment transition."""

    observation: str
    reward: float
    done: bool
    truncated: bool
    valid_actions: tuple[str, ...] | None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class InteractiveEnvironment(Protocol):
    """Contract implemented by ALFWorld and future environment adapters."""

    def reset(self, *, seed: int | None = None) -> ResetResult:
        """Start an episode and return the normalized initial state."""

    def step(self, action: str) -> Transition:
        """Apply an action and return a normalized transition."""

    def get_task(self) -> Task:
        """Return the task associated with the active episode."""

    def get_valid_actions(self) -> tuple[str, ...] | None:
        """Return currently admissible actions, if the environment exposes them."""
