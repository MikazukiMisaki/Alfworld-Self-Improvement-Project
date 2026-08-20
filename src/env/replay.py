"""Deterministic action-prefix replay checks for recorded trajectories."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .base import InteractiveEnvironment, ResetResult


def replay_prefix(
    environment: InteractiveEnvironment,
    reset: ResetResult,
    trajectory: dict[str, Any],
    prefix_length: int,
) -> dict[str, Any]:
    """Replay a recorded action prefix and compare its public environment state."""
    steps = trajectory.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("trajectory must contain at least one recorded step")
    if prefix_length < 0 or prefix_length >= len(steps):
        raise ValueError("prefix_length must identify a nonterminal recorded state")

    expected_task = trajectory.get("task", {})
    expected_task_id = expected_task.get("task_id")
    task_id_equal = reset.task.task_id == expected_task_id
    current_observation = reset.observation
    current_actions = reset.valid_actions
    transitions: list[dict[str, Any]] = []
    reconstructed_rewards: list[float] = []
    reconstructed_done: list[bool] = []
    reconstructed_truncated: list[bool] = []

    for index in range(prefix_length):
        expected = steps[index]
        expected_actions = _actions(expected.get("valid_actions"))
        action = str(expected.get("action", ""))
        transition = environment.step(action)
        comparison = {
            "index": index,
            "action": action,
            "source_observation_equal": current_observation
            == expected.get("observation"),
            "source_admissible_order_equal": current_actions == expected_actions,
            "source_admissible_set_equal": _action_set(current_actions)
            == _action_set(expected_actions),
            "action_admissible": current_actions is not None
            and action in current_actions,
            "reward_equal": transition.reward == float(expected.get("reward", 0.0)),
            "done_equal": transition.done == bool(expected.get("done", False)),
            "truncated_equal": transition.truncated
            == bool(expected.get("metadata", {}).get("transition_truncated", False)),
        }
        comparison["exact"] = all(
            value for key, value in comparison.items() if key.endswith("_equal")
        ) and comparison["action_admissible"]
        transitions.append(comparison)
        reconstructed_rewards.append(transition.reward)
        reconstructed_done.append(transition.done)
        reconstructed_truncated.append(transition.truncated)
        current_observation = transition.observation
        current_actions = transition.valid_actions

    target = steps[prefix_length]
    target_actions = _actions(target.get("valid_actions"))
    original_rewards = [float(step.get("reward", 0.0)) for step in steps[:prefix_length]]
    original_done = [bool(step.get("done", False)) for step in steps[:prefix_length]]
    original_truncated = [
        bool(step.get("metadata", {}).get("transition_truncated", False))
        for step in steps[:prefix_length]
    ]
    target_comparison = {
        "step_index": prefix_length,
        "observation_equal": current_observation == target.get("observation"),
        "admissible_order_equal": current_actions == target_actions,
        "admissible_set_equal": _action_set(current_actions)
        == _action_set(target_actions),
        "observation": current_observation,
        "admissible_actions": list(current_actions) if current_actions else None,
    }
    histories = {
        "original_reward_history": original_rewards,
        "reconstructed_reward_history": reconstructed_rewards,
        "reward_history_equal": reconstructed_rewards == original_rewards,
        "original_done_history": original_done,
        "reconstructed_done_history": reconstructed_done,
        "done_history_equal": reconstructed_done == original_done,
        "original_truncated_history": original_truncated,
        "reconstructed_truncated_history": reconstructed_truncated,
        "truncated_history_equal": reconstructed_truncated == original_truncated,
    }
    exact = task_id_equal and all(item["exact"] for item in transitions) and all(
        target_comparison[key]
        for key in (
            "observation_equal",
            "admissible_order_equal",
            "admissible_set_equal",
        )
    ) and all(value for key, value in histories.items() if key.endswith("_equal"))
    return {
        "task_id": expected_task_id,
        "seed": trajectory.get("seed"),
        "prefix_length": prefix_length,
        "task_id_equal": task_id_equal,
        "original": {
            "task_id": expected_task_id,
            "step_index": prefix_length,
            "observation": target.get("observation"),
            "admissible_actions": list(target_actions) if target_actions else None,
            "environment_diagnostics": _recorded_diagnostics(
                trajectory, prefix_length
            ),
        },
        "transitions": transitions,
        "target": target_comparison,
        "histories": histories,
        "hidden_state": alfworld_hidden_state(environment),
        "exact_public_reconstruction": exact,
    }


def repeated_replay_equal(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, bool | None]:
    """Compare independently reconstructed target states."""
    first_target = first["target"]
    second_target = second["target"]
    first_hidden = first["hidden_state"]
    second_hidden = second["hidden_state"]
    return {
        "task_id_equal": first["task_id"] == second["task_id"],
        "observation_equal": first_target["observation"]
        == second_target["observation"],
        "admissible_order_equal": first_target["admissible_actions"]
        == second_target["admissible_actions"],
        "hidden_state_equal": (
            first_hidden == second_hidden
            if first_hidden is not None and second_hidden is not None
            else None
        ),
    }


def alfworld_hidden_state(environment: InteractiveEnvironment) -> dict[str, Any] | None:
    """Return stable read-only diagnostics from ALFWorld's active TextWorld state."""
    try:
        batch = getattr(environment, "_environment")
        active = batch.batch_env.envs[0]
        state = active.state
        facts = getattr(state, "facts", None)
        if facts is None:
            facts = getattr(state, "_facts")
        canonical_facts = sorted(str(fact) for fact in facts)
        payload = json.dumps(canonical_facts, separators=(",", ":"))
        result: dict[str, Any] = {
            "fact_count": len(canonical_facts),
            "facts_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        }
        for name in ("score", "moves", "won", "lost"):
            value = getattr(state, name, None)
            if isinstance(value, (bool, int, float, str)) or value is None:
                result[name] = value
        return result
    except (AttributeError, IndexError, TypeError):
        return None


def _actions(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("recorded valid_actions must be a list or null")
    return tuple(str(action) for action in value)


def _action_set(actions: tuple[str, ...] | None) -> frozenset[str] | None:
    return frozenset(actions) if actions is not None else None


def _recorded_diagnostics(
    trajectory: dict[str, Any], prefix_length: int
) -> dict[str, Any] | None:
    if prefix_length == 0:
        metadata = trajectory.get("metadata", {}).get("reset", {})
    else:
        metadata = trajectory["steps"][prefix_length - 1].get("metadata", {}).get(
            "environment", {}
        )
    info = metadata.get("alfworld_info") if isinstance(metadata, dict) else None
    if not isinstance(info, dict):
        return None
    return {
        key: info[key]
        for key in ("won", "extra.gamefile")
        if key in info
    }
