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
    parser_failure_rate: float
    inadmissible_candidate_rate: float
    invalid_action_rate: float
    selection_failure_rate: float
    malformed_id_rate: float
    out_of_range_id_rate: float
    mean_generated_tokens: float

    @classmethod
    def from_trajectories(cls, trajectories: Sequence[Trajectory]) -> "EvaluationMetrics":
        """Aggregate metrics, rejecting empty evaluation input."""
        if not trajectories:
            raise ValueError("at least one trajectory is required")
        steps = [step for trajectory in trajectories for step in trajectory.steps]
        free_form_steps = [
            step for step in steps if _selection_mode(step) == "free_form_validated"
        ]
        indexed_steps = [
            step for step in steps if _selection_mode(step) == "indexed_admissible"
        ]
        parser_failures = sum(
            step.parser_status in {"missing_action", "empty_action"}
            for step in free_form_steps
        )
        inadmissible_candidates = sum(
            step.parser_status == "not_admissible" for step in free_form_steps
        )
        invalid_actions = sum(step.action_valid is False for step in free_form_steps)
        selection_statuses = [_selection_status(step) for step in indexed_steps]
        selection_failures = sum(status != "selected" for status in selection_statuses)
        malformed_ids = sum(
            status in {"malformed_id", "ambiguous_id"}
            for status in selection_statuses
        )
        out_of_range_ids = sum(
            status == "out_of_range_id" for status in selection_statuses
        )
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
            parser_failure_rate=(
                parser_failures / len(free_form_steps) if free_form_steps else 0.0
            ),
            inadmissible_candidate_rate=(
                inadmissible_candidates / len(free_form_steps)
                if free_form_steps
                else 0.0
            ),
            invalid_action_rate=(
                invalid_actions / len(free_form_steps) if free_form_steps else 0.0
            ),
            selection_failure_rate=(
                selection_failures / len(indexed_steps) if indexed_steps else 0.0
            ),
            malformed_id_rate=(
                malformed_ids / len(indexed_steps) if indexed_steps else 0.0
            ),
            out_of_range_id_rate=(
                out_of_range_ids / len(indexed_steps) if indexed_steps else 0.0
            ),
            mean_generated_tokens=generated_tokens / len(trajectories),
        )


def _selection_mode(step: object) -> str:
    metadata = getattr(step, "metadata", {})
    if not isinstance(metadata, dict):
        return "free_form_validated"
    return str(metadata.get("action_selection_mode", "free_form_validated"))


def _selection_status(step: object) -> str:
    metadata = getattr(step, "metadata", {})
    if not isinstance(metadata, dict):
        return "missing"
    selection = metadata.get("action_selection", {})
    if not isinstance(selection, dict):
        return "missing"
    return str(selection.get("selection_status", "missing"))
