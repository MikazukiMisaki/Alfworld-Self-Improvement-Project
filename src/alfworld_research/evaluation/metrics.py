"""Metrics computed over complete episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from alfworld_research.trajectory.trajectory import Trajectory


@dataclass(frozen=True)
class EvaluationMetrics:
    """Aggregate metrics that remain meaningful across environment adapters."""

    episodes: int
    success_rate: float
    mean_reward: float

    @classmethod
    def from_trajectories(cls, trajectories: Sequence[Trajectory]) -> "EvaluationMetrics":
        """Aggregate success and undiscounted reward, rejecting an empty set."""
        if not trajectories:
            raise ValueError("at least one trajectory is required")
        return cls(
            episodes=len(trajectories),
            success_rate=sum(t.succeeded for t in trajectories) / len(trajectories),
            mean_reward=sum(t.total_reward for t in trajectories) / len(trajectories),
        )
