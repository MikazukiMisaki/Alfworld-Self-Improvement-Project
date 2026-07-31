"""Aggregate metrics for reproducible baseline evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from trajectory.trajectory import Trajectory


@dataclass(frozen=True)
class EvaluationMetrics:
    """Metrics reported for a fixed task and seed schedule."""

    episodes: int
    success_rate: float
    mean_reward: float
    mean_episode_length: float
    invalid_action_rate: float
    mean_generated_tokens: float

    @classmethod
    def from_trajectories(cls, trajectories: Sequence[Trajectory]) -> "EvaluationMetrics":
        """Aggregate metrics, rejecting empty evaluation input."""
        if not trajectories:
            raise ValueError("at least one trajectory is required")
        steps = [step for trajectory in trajectories for step in trajectory.steps]
        invalid_actions = sum(step.action_valid is False for step in steps)
        generated_tokens = sum(
            step.token_statistics.generated_tokens
            for step in steps
            if step.token_statistics is not None
        )
        return cls(
            episodes=len(trajectories),
            success_rate=sum(trajectory.succeeded for trajectory in trajectories) / len(trajectories),
            mean_reward=sum(trajectory.total_reward for trajectory in trajectories) / len(trajectories),
            mean_episode_length=sum(trajectory.episode_length for trajectory in trajectories) / len(trajectories),
            invalid_action_rate=invalid_actions / len(steps) if steps else 0.0,
            mean_generated_tokens=generated_tokens / len(trajectories),
        )
