"""Episode collection decoupled from an environment or model implementation."""

from __future__ import annotations

from typing import Any

from alfworld_research.env.base import InteractiveEnvironment
from alfworld_research.models.policy import ActionPolicy

from .trajectory import Step, Trajectory


def collect_episode(
    environment: InteractiveEnvironment,
    policy: ActionPolicy,
    *,
    max_steps: int = 50,
    seed: int | None = None,
) -> Trajectory:
    """Collect one bounded episode and preserve all environment metadata."""
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    initial_observation, reset_info = environment.reset(seed=seed)
    observation = initial_observation
    history: list[str] = []
    steps: list[Step] = []
    final_info: dict[str, Any] = dict(reset_info)
    for _ in range(max_steps):
        action = policy.act(observation, tuple(history))
        next_observation, reward, done, info = environment.step(action)
        steps.append(Step(observation=observation, action=action, reward=reward, done=done, info=info))
        history.extend((observation, action))
        observation, final_info = next_observation, info
        if done:
            break
    return Trajectory(initial_observation=initial_observation, steps=tuple(steps), metadata=final_info)
