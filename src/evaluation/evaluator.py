"""Evaluation orchestration over fresh environment instances."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from env.base import InteractiveEnvironment
from models.policy import ActionPolicy
from trajectory.collector import collect_episode
from trajectory.trajectory import Trajectory

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
    # ALFWorld loads all game files during construction. Reuse its initialized
    # environment across the seed schedule, as in the proven legacy runner.
    environment = environment_factory()
    trajectories = tuple(
        collect_episode(environment, policy, max_steps=max_steps, seed=seed) for seed in seeds
    )
    return EvaluationReport(trajectories, EvaluationMetrics.from_trajectories(trajectories))
