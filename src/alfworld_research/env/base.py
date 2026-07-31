"""Interfaces for embodied interactive environments."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class InteractiveEnvironment(Protocol):
    """Minimal contract required by collectors and evaluators.

    An ALFWorld adapter should normalize its API to this contract while keeping
    task-specific metadata in the returned information dictionaries.
    """

    def reset(self, *, seed: int | None = None) -> tuple[str, dict[str, Any]]:
        """Start an episode and return its initial observation and metadata."""

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        """Apply an action and return observation, reward, completion, metadata."""
