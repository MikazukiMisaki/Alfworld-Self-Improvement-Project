"""Paired continuation utilities for the fixed Sprint 2B recovery pilot."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from env.base import InteractiveEnvironment, ResetResult
from models.action_parser import is_valid_action
from models.policy import ActionDecision, ActionPolicy, ActionRequest

from .fixed import loop_indicators


def prefix_hash(trajectory: dict[str, Any], action_count: int) -> str:
    """Hash the immutable recorded state and actions defining a branch point."""
    target = trajectory["steps"][action_count]
    payload = {
        "task_id": trajectory["task"]["task_id"],
        "seed": trajectory["seed"],
        "action_count": action_count,
        "actions": [step["action"] for step in trajectory["steps"][:action_count]],
        "observation": target["observation"],
        "valid_actions": target["valid_actions"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def history_at_prefix(
    trajectory: dict[str, Any], action_count: int
) -> tuple[tuple[str, str], ...]:
    """Recover policy-visible history from immutable trajectory records."""
    return tuple(
        (str(step["observation"]), str(step["action"]))
        for step in trajectory["steps"][:action_count]
    )


def run_branch(
    environment: InteractiveEnvironment,
    reset: ResetResult,
    policy: ActionPolicy,
    trajectory: dict[str, Any],
    *,
    action_count: int,
    branch_observation: str,
    branch_valid_actions: tuple[str, ...],
    remaining_horizon: int,
    first_decision: Callable[[ActionRequest], ActionDecision],
) -> dict[str, Any]:
    """Run one fixed-first-action branch, then resume the unchanged base policy."""
    if remaining_horizon < 1:
        raise ValueError("remaining_horizon must be positive")
    observation = branch_observation
    valid_actions = branch_valid_actions
    history = list(history_at_prefix(trajectory, action_count))
    steps: list[dict[str, Any]] = []
    terminal = False
    termination_reason = "remaining_horizon"
    recovery_calls = 0

    for offset in range(remaining_horizon):
        request = ActionRequest(
            task=reset.task,
            observation=observation,
            history=tuple(history),
            valid_actions=valid_actions,
        )
        if offset == 0:
            decision = first_decision(request)
            recovery_calls = int(
                decision.metadata.get("recovery_operator_version") is not None
            )
        else:
            decision = policy.act(request)
        selection = decision.metadata.get("action_selection", {})
        status = selection.get("selection_status")
        action_valid = is_valid_action(decision.action, valid_actions)
        failure_reason = None
        if status != "selected":
            failure_reason = str(selection.get("failure_reason") or status)
        elif action_valid is not True:
            failure_reason = "mapped action is not currently admissible"
        transition = environment.step(decision.action) if failure_reason is None else None
        steps.append(
            {
                "offset": offset,
                "observation": observation,
                "valid_actions": list(valid_actions),
                "action": decision.action,
                "raw_output": decision.raw_output,
                "parser_status": decision.parser_status,
                "selection_status": status,
                "failure_reason": failure_reason,
                "reward": transition.reward if transition is not None else 0.0,
                "done": transition.done if transition is not None else False,
                "truncated": transition.truncated if transition is not None else True,
                "token_statistics": (
                    decision.token_statistics.to_dict()
                    if decision.token_statistics is not None
                    else None
                ),
                "policy_metadata": decision.metadata,
            }
        )
        if transition is None:
            termination_reason = "selection_failure"
            break
        history.append((observation, decision.action))
        observation = transition.observation
        valid_actions = transition.valid_actions or ()
        terminal = transition.done or transition.truncated
        if terminal:
            termination_reason = "environment_terminal"
            break

    actions = [step["action"] for step in steps]
    total_return = sum(float(step["reward"]) for step in steps)
    success = bool(
        steps and steps[-1]["done"] and float(steps[-1]["reward"]) > 0
    )
    return {
        "first_action": actions[0] if actions else None,
        "return": total_return,
        "success": success,
        "remaining_episode_length": len(steps),
        "remaining_horizon": remaining_horizon,
        "termination_reason": termination_reason,
        "recovery_calls": recovery_calls,
        "steps": steps,
        "loop_indicators": loop_indicators(actions),
        "terminal": terminal,
    }
