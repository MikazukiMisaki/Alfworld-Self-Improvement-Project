"""Shared bounded collector for reproducible interactive trajectories."""

from __future__ import annotations

from typing import Any

from env.base import InteractiveEnvironment
from models.policy import ActionPolicy, ActionRequest

from .trajectory import Step, Trajectory, trajectory_id, utc_timestamp


def collect_episode(
    environment: InteractiveEnvironment,
    policy: ActionPolicy,
    *,
    max_steps: int,
    seed: int | None,
) -> Trajectory:
    """Collect a single bounded episode without writing artifacts."""
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    reset = environment.reset(seed=seed)
    observation = reset.observation
    valid_actions = reset.valid_actions
    history: list[tuple[str, str]] = []
    steps: list[Step] = []
    started_at = utc_timestamp()
    terminal = False
    for index in range(max_steps):
        decision = policy.act(
            ActionRequest(
                task=reset.task,
                observation=observation,
                history=tuple(history),
                valid_actions=valid_actions,
            )
        )
        transition = environment.step(decision.action)
        action_valid = (
            decision.action in valid_actions
            if valid_actions is not None and decision.action
            else None
        )
        steps.append(
            Step(
                index=index,
                observation=observation,
                action=decision.action,
                model_output=decision.raw_output,
                reward=transition.reward,
                done=transition.done,
                timestamp=utc_timestamp(),
                token_statistics=decision.token_statistics,
                reasoning=decision.reasoning,
                parser_status=decision.parser_status,
                valid_actions=valid_actions,
                action_valid=action_valid,
                metadata={
                    "policy": decision.metadata,
                    "environment": transition.metadata,
                    "transition_truncated": transition.truncated,
                },
            )
        )
        history.append((observation, decision.action))
        observation = transition.observation
        valid_actions = transition.valid_actions
        terminal = transition.done or transition.truncated
        if terminal:
            break
    return Trajectory(
        trajectory_id=trajectory_id(reset.task, seed, reset.observation),
        task=reset.task,
        model_version=policy.model_version,
        seed=seed,
        initial_observation=reset.observation,
        steps=tuple(steps),
        started_at=started_at,
        completed_at=utc_timestamp(),
        truncated=not terminal,
        metadata={"reset": reset.metadata, "max_steps": max_steps},
    )
