"""Shared bounded collector for reproducible interactive trajectories."""

from __future__ import annotations

from typing import Any

from env.base import InteractiveEnvironment
from models.action_parser import is_valid_action
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
    parser_failure = False
    for index in range(max_steps):
        decision = policy.act(
            ActionRequest(
                task=reset.task,
                observation=observation,
                history=tuple(history),
                valid_actions=valid_actions,
            )
        )
        action_valid = is_valid_action(decision.action, valid_actions)
        invalid_reason = _invalid_action_reason(decision, action_valid)
        if invalid_reason is None:
            transition = environment.step(decision.action)
        else:
            # Never turn malformed generation into an arbitrary environment
            # command. Preserve the failed decision and stop this episode.
            transition = None
            parser_failure = True
        steps.append(
            Step(
                index=index,
                observation=observation,
                action=decision.action,
                model_output=decision.raw_output,
                reward=transition.reward if transition is not None else 0.0,
                done=transition.done if transition is not None else False,
                timestamp=utc_timestamp(),
                token_statistics=decision.token_statistics,
                reasoning=decision.reasoning,
                parser_status=decision.parser_status,
                valid_actions=valid_actions,
                action_valid=action_valid,
                metadata={
                    "policy": decision.metadata,
                    "environment": transition.metadata if transition is not None else {},
                    "transition_truncated": transition.truncated if transition is not None else True,
                    "debug": {
                        "raw_model_output": decision.raw_output,
                        "parsed_action": decision.action,
                        "valid_actions": list(valid_actions) if valid_actions is not None else None,
                        "parser_status": decision.parser_status,
                        "invalid_action_reason": invalid_reason,
                        "generated_token_count": (
                            decision.token_statistics.generated_tokens
                            if decision.token_statistics is not None
                            else None
                        ),
                    },
                },
            )
        )
        if parser_failure:
            break
        history.append((observation, decision.action))
        assert transition is not None
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
        truncated=parser_failure or not terminal,
        metadata={
            "reset": reset.metadata,
            "max_steps": max_steps,
            "termination_reason": "parser_failure" if parser_failure else "environment_terminal" if terminal else "max_steps",
        },
    )


def _invalid_action_reason(decision: Any, action_valid: bool | None) -> str | None:
    """Describe why an action is not safe to send to a constrained environment."""
    parser = decision.metadata.get("parser", {}) if isinstance(decision.metadata, dict) else {}
    if decision.parser_status != "grounded" and action_valid is not None:
        return str(parser.get("invalid_reason") or decision.parser_status)
    if action_valid is False:
        return "not in valid actions"
    return None
