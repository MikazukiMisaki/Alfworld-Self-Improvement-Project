"""Matched free-form versus indexed action-interface regression utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


class RegressionComparisonError(ValueError):
    """Raised when two runs are not a valid matched comparison."""


@dataclass(frozen=True)
class TaskTargets:
    """Task object, transformation, and destination parsed from an ALFWorld ID."""

    target_object: str | None
    transformation: str | None
    destination: str | None


def validate_interface_config_equivalence(
    free_form_collection: dict[str, Any],
    indexed_collection: dict[str, Any],
    free_form_model: dict[str, Any],
    indexed_model: dict[str, Any],
    environment: dict[str, Any],
) -> None:
    """Require the two portable configs to differ only by action interface."""
    _require_mode(free_form_model, "free_form_v1", "free_form_validated")
    _require_mode(indexed_model, "indexed_v1", "indexed_admissible")
    free_collection = _without(free_form_collection, {"model_config"})
    indexed_collection_without_model = _without(indexed_collection, {"model_config"})
    if free_collection != indexed_collection_without_model:
        raise RegressionComparisonError("collection configs differ beyond model_config")
    free_model = _without(free_form_model, {"pipeline_version", "action_selection"})
    indexed_model_without_interface = _without(
        indexed_model, {"pipeline_version", "action_selection"}
    )
    if free_model != indexed_model_without_interface:
        raise RegressionComparisonError(
            "model configs differ beyond pipeline/action-selection fields"
        )
    if environment.get("split") not in {"train", "valid_seen", "valid_unseen"}:
        raise RegressionComparisonError("environment config has an unsupported split")


def validate_indexed_context_config_equivalence(
    h0_collection: dict[str, Any],
    hk_collection: dict[str, Any],
    h0_model: dict[str, Any],
    hk_model: dict[str, Any],
    environment: dict[str, Any],
    *,
    expected_window: int,
) -> None:
    """Require H0 and Hk to differ only by bounded-context provenance."""
    _require_mode(h0_model, "indexed_v1", "indexed_admissible")
    _require_mode(
        hk_model, "indexed_bounded_context_v1", "indexed_admissible"
    )
    if _without(h0_collection, {"model_config"}) != _without(
        hk_collection, {"model_config"}
    ):
        raise RegressionComparisonError("H0/Hk collection configs differ")
    if _without(h0_model, {"pipeline_version", "history_context"}) != _without(
        hk_model, {"pipeline_version", "history_context"}
    ):
        raise RegressionComparisonError(
            "H0/Hk model configs differ beyond pipeline/history context"
        )
    context = hk_model.get("history_context")
    if context != {"mode": "bounded_recent_state", "window": expected_window}:
        raise RegressionComparisonError(
            f"Hk must use bounded_recent_state with window {expected_window}"
        )
    if h0_model.get("history_context") is not None:
        raise RegressionComparisonError("H0 must retain its implicit full_raw context")
    if environment.get("split") not in {"train", "valid_seen", "valid_unseen"}:
        raise RegressionComparisonError("environment config has an unsupported split")


def compare_trajectory_sets(
    reference_trajectories: Sequence[dict[str, Any]],
    indexed_trajectories: Sequence[dict[str, Any]],
    *,
    reference_pipeline: str = "free_form_v1",
    candidate_pipeline: str = "indexed_v1",
) -> dict[str, Any]:
    """Compare ordered trajectories and fail if task/seed matching is not exact."""
    if reference_pipeline not in {"legacy_v1", "free_form_v1", "indexed_v1"}:
        raise RegressionComparisonError("unsupported reference pipeline")
    if candidate_pipeline not in {"indexed_v1", "indexed_bounded_context_v1"}:
        raise RegressionComparisonError("unsupported candidate pipeline")
    if len(reference_trajectories) != len(indexed_trajectories):
        raise RegressionComparisonError("run episode counts differ")
    comparisons: list[dict[str, Any]] = []
    for reference, indexed in zip(reference_trajectories, indexed_trajectories):
        free_key = _trajectory_key(reference)
        indexed_key = _trajectory_key(indexed)
        if free_key != indexed_key:
            raise RegressionComparisonError(
                f"task/seed mismatch: free_form={free_key!r}, indexed={indexed_key!r}"
            )
        comparisons.append(compare_trajectories(reference, indexed))
    return {
        "schema_version": 1,
        "comparison": f"{reference_pipeline}_vs_{candidate_pipeline}",
        "reference_pipeline": reference_pipeline,
        "candidate_pipeline": candidate_pipeline,
        "matched_episode_count": len(comparisons),
        "episodes": comparisons,
    }


def compare_trajectories(
    reference: dict[str, Any], indexed: dict[str, Any]
) -> dict[str, Any]:
    """Produce a step-level behavioral comparison for one matched task and seed."""
    if _trajectory_key(reference) != _trajectory_key(indexed):
        raise RegressionComparisonError("trajectories do not have matching task and seed")
    task = reference.get("task", {})
    task_id = str(task.get("task_id", "unknown"))
    targets = parse_task_targets(task_id)
    free_steps = reference.get("steps", [])
    indexed_steps = indexed.get("steps", [])
    max_length = max(len(free_steps), len(indexed_steps))
    rows: list[dict[str, Any]] = []
    earliest_divergence: int | None = None
    previous_free: str | None = None
    previous_indexed: str | None = None
    for index in range(max_length):
        free_step = free_steps[index] if index < len(free_steps) else None
        indexed_step = indexed_steps[index] if index < len(indexed_steps) else None
        free_action = _step_action(free_step)
        indexed_action = _step_action(indexed_step)
        same = free_action == indexed_action and free_step is not None and indexed_step is not None
        if earliest_divergence is None and not same:
            earliest_divergence = index
        rows.append(
            {
                "step": index,
                "reference_action": free_action,
                "indexed_action": indexed_action,
                "same": same,
                "reference_progress": progress_events(
                    free_step, targets, previous_action=previous_free
                ),
                "indexed_progress": progress_events(
                    indexed_step, targets, previous_action=previous_indexed
                ),
            }
        )
        previous_free = free_action
        previous_indexed = indexed_action
    return {
        "task_id": task_id,
        "task_text": task.get("text"),
        "split": task.get("split"),
        "seed": reference.get("seed"),
        "earliest_divergence_step": earliest_divergence,
        "reference": _outcome(reference),
        "indexed": _outcome(indexed),
        "steps": rows,
    }


def parse_task_targets(task_id: str) -> TaskTargets:
    """Parse standard ALFWorld task-folder metadata without guessing missing fields."""
    family = task_id.split("/", 1)[0]
    parts = family.split("-")
    if len(parts) < 4:
        return TaskTargets(None, None, None)
    prefix = parts[0]
    transformation = next(
        (name for name in ("cool", "heat", "clean") if f"_{name}_" in prefix),
        None,
    )
    return TaskTargets(parts[1].casefold(), transformation, parts[3].casefold())


def progress_events(
    step: dict[str, Any] | None,
    targets: TaskTargets,
    *,
    previous_action: str | None,
) -> list[str]:
    """Extract conservative, observable task-progress and loop signals."""
    if step is None:
        return ["missing_step"]
    action = str(step.get("action", "")).casefold()
    observation = str(step.get("observation", "")).casefold()
    events: list[str] = []
    target = targets.target_object
    destination = targets.destination
    if target and target in observation:
        events.append("target_observed")
    if target and action.startswith("take ") and target in action:
        events.append("target_pickup_attempt")
    if (
        target
        and targets.transformation
        and action.startswith(f"{targets.transformation} ")
        and target in action
    ):
        events.append("required_transformation_attempt")
    if destination and action.startswith("go to ") and destination in action:
        events.append("destination_reach_attempt")
    if (
        target
        and destination
        and action.startswith("move ")
        and target in action
        and destination in action
    ):
        events.append("placement_attempt")
    if previous_action is not None and action == previous_action.casefold():
        events.append("repeated_action")
    if step.get("action_valid") is False:
        events.append("invalid_action")
    return events


def render_comparison_markdown(report: dict[str, Any]) -> str:
    """Render side-by-side action and progress tables."""
    lines = ["# Action-Interface Regression", ""]
    for episode in report["episodes"]:
        lines.extend(
            [
                f"## {episode['task_id']} (seed {episode['seed']})",
                "",
                f"Earliest divergence: `{episode['earliest_divergence_step']}`",
                "",
                f"| Step | {report['reference_pipeline']} action | Indexed action | Same? | Reference progress | Indexed progress |",
                "|---:|---|---|---|---|---|",
            ]
        )
        for row in episode["steps"]:
            lines.append(
                f"| {row['step']} | `{row['reference_action'] or ''}` | "
                f"`{row['indexed_action'] or ''}` | {row['same']} | "
                f"{', '.join(row['reference_progress'])} | "
                f"{', '.join(row['indexed_progress'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def _require_mode(model: dict[str, Any], pipeline: str, mode: str) -> None:
    selection = model.get("action_selection")
    if model.get("pipeline_version") != pipeline or not isinstance(selection, dict):
        raise RegressionComparisonError(f"model config is not {pipeline}")
    if selection.get("mode") != mode:
        raise RegressionComparisonError(f"model config does not use {mode}")


def _without(mapping: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key not in keys}


def _trajectory_key(trajectory: dict[str, Any]) -> tuple[str | None, Any]:
    task = trajectory.get("task", {})
    task_id = task.get("task_id") if isinstance(task, dict) else None
    return task_id, trajectory.get("seed")


def _step_action(step: dict[str, Any] | None) -> str | None:
    if not isinstance(step, dict):
        return None
    action = step.get("action")
    return action if isinstance(action, str) else None


def _outcome(trajectory: dict[str, Any]) -> dict[str, Any]:
    steps = trajectory.get("steps", [])
    reward = sum(
        float(step.get("reward", 0.0)) for step in steps if isinstance(step, dict)
    )
    success = bool(
        steps
        and isinstance(steps[-1], dict)
        and steps[-1].get("done") is True
        and float(steps[-1].get("reward", 0.0)) > 0
    )
    metadata = trajectory.get("metadata", {})
    termination = metadata.get("termination_reason") if isinstance(metadata, dict) else None
    return {
        "success": success,
        "reward": reward,
        "steps": len(steps),
        "termination_reason": termination,
    }
