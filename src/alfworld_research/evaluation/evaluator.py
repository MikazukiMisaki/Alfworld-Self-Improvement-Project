"""Evaluation orchestration with injectable environment factories."""

from __future__ import annotations

from collections.abc import Callable

from alfworld_research.env.base import InteractiveEnvironment
from alfworld_research.models.policy import ActionPolicy
from alfworld_research.trajectory.collector import collect_episode

from .metrics import EvaluationMetrics


def evaluate(
    environment_factory: Callable[[], InteractiveEnvironment],
    policy: ActionPolicy,
    *,
    episodes: int,
    max_steps: int = 50,
) -> EvaluationMetrics:
    """Evaluate a policy in fresh environments to avoid episode-state leakage."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    trajectories = [collect_episode(environment_factory(), policy, max_steps=max_steps, seed=index) for index in range(episodes)]
    return EvaluationMetrics.from_trajectories(trajectories)
