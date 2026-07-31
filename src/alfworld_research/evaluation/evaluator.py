"""Evaluation orchestration over fresh environment instances."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from alfworld_research.env.base import InteractiveEnvironment
from alfworld_research.models.policy import ActionPolicy
from alfworld_research.trajectory.collector import collect_episode
from alfworld_research.trajectory.trajectory import Trajectory

from .metrics import EvaluationMetrics


@dataclass(frozen=True)
class EvaluationReport:
    """Episode records and their aggregate metrics."""

    trajectories: tuple[Trajectory, ...]
    metrics: EvaluationMetrics


def evaluate(
    environment_factory: Callable[[], InteractiveEnvironment],
    policy: ActionPolicy,
    *,
    seeds: tuple[int, ...],
    max_steps: int,
) -> EvaluationReport:
    """Evaluate a policy in fresh environments for the supplied seed schedule."""
    if not seeds:
        raise ValueError("at least one seed is required")
    trajectories = tuple(
        collect_episode(environment_factory(), policy, max_steps=max_steps, seed=seed)
        for seed in seeds
    )
    return EvaluationReport(trajectories, EvaluationMetrics.from_trajectories(trajectories))
