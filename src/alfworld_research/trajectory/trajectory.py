"""Canonical, JSON-serializable records for baseline interactions."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from alfworld_research.env.base import Task
from alfworld_research.models.policy import TokenStatistics


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for an interaction artifact."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Step:
    """One policy decision and the transition it caused."""

    index: int
    observation: str
    action: str
    model_output: str
    reward: float
    done: bool
    timestamp: str
    token_statistics: TokenStatistics | None = None
    reasoning: str | None = None
    parser_status: str = "unknown"
    valid_actions: tuple[str, ...] | None = None
    action_valid: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Trajectory:
    """An immutable episode record with task, seed, and model provenance."""

    trajectory_id: str
    task: Task
    model_version: str
    seed: int | None
    initial_observation: str
    steps: tuple[Step, ...]
    started_at: str
    completed_at: str
    truncated: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_reward(self) -> float:
        """Return the undiscounted episode reward."""
        return sum(step.reward for step in self.steps)

    @property
    def succeeded(self) -> bool:
        """Return whether ALFWorld emitted a positive terminal reward."""
        return bool(self.steps and self.steps[-1].done and self.steps[-1].reward > 0)

    @property
    def episode_length(self) -> int:
        """Return the number of executed environment actions."""
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        """Produce a stable JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class RunManifest:
    """Provenance shared by all trajectories collected in one baseline run."""

    run_id: str
    created_at: str
    model_version: str
    environment: str
    resolved_config: dict[str, Any]
    seed_schedule: tuple[int, ...]
    git_revision: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Produce a JSON-serializable manifest."""
        return asdict(self)


def trajectory_id(task: Task, seed: int | None, initial_observation: str) -> str:
    """Derive a deterministic identifier from episode-defining inputs."""
    payload = f"{task.task_id}\0{seed}\0{initial_observation}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]
