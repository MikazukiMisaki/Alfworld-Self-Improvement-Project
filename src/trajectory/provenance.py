"""Strict pipeline provenance contracts for trajectory consumers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PIPELINE_BY_ACTION_MODE = {
    "free_text": "legacy_v1",
    "free_form_validated": "free_form_v1",
    "indexed_admissible": "indexed_v1",
}
PIPELINE_BY_CONTEXT = {
    ("indexed_admissible", "bounded_recent_state"): "indexed_bounded_context_v1",
}


class ProvenanceError(ValueError):
    """Raised when a run cannot satisfy an explicit provenance requirement."""


@dataclass(frozen=True)
class ProvenanceRequirement:
    """Required lineage fields for one trajectory-loading operation."""

    pipeline_versions: frozenset[str]
    action_selection_modes: frozenset[str]
    splits: frozenset[str]

    @classmethod
    def one(
        cls,
        *,
        pipeline_version: str,
        action_selection_mode: str,
        split: str,
    ) -> "ProvenanceRequirement":
        """Build a requirement that admits exactly one pipeline and split."""
        return cls(
            frozenset({pipeline_version}),
            frozenset({action_selection_mode}),
            frozenset({split}),
        )


def pipeline_for_action_selection_mode(
    action_selection_mode: str, history_context_mode: str = "full_raw"
) -> str:
    """Return the canonical pipeline version for a supported action interface."""
    contextual = PIPELINE_BY_CONTEXT.get(
        (action_selection_mode, history_context_mode)
    )
    if contextual is not None:
        return contextual
    if history_context_mode != "full_raw":
        raise ProvenanceError(
            f"unsupported history context mode: {history_context_mode!r}"
        )
    try:
        return PIPELINE_BY_ACTION_MODE[action_selection_mode]
    except KeyError as error:
        raise ProvenanceError(
            f"unsupported action selection mode: {action_selection_mode!r}"
        ) from error


def validate_manifest_provenance(
    manifest: dict[str, Any],
    requirement: ProvenanceRequirement | None = None,
) -> None:
    """Require explicit, internally consistent run-level pipeline provenance."""
    pipeline_version = _required_string(manifest, "pipeline_version")
    action_selection_mode = _required_string(manifest, "action_selection_mode")
    split = _required_string(manifest, "split")
    resolved_config = manifest.get("resolved_config")
    if not isinstance(resolved_config, dict):
        raise ProvenanceError("manifest is missing resolved_config")
    model_config = resolved_config.get("model")
    environment_config = resolved_config.get("environment")
    if not isinstance(model_config, dict) or not isinstance(environment_config, dict):
        raise ProvenanceError("manifest resolved_config is incomplete")
    selection_config = model_config.get("action_selection")
    if not isinstance(selection_config, dict):
        raise ProvenanceError("manifest model config is missing action_selection")
    context_config = model_config.get("history_context", {})
    if not isinstance(context_config, dict):
        raise ProvenanceError("manifest model history_context must be a mapping")
    context_mode = str(context_config.get("mode", "full_raw"))
    expected_pipeline = pipeline_for_action_selection_mode(
        action_selection_mode, context_mode
    )
    if pipeline_version != expected_pipeline:
        raise ProvenanceError(
            "pipeline_version does not match action-selection/context provenance: "
            f"{pipeline_version!r} != {expected_pipeline!r}"
        )
    if selection_config.get("mode") != action_selection_mode:
        raise ProvenanceError("top-level and resolved action-selection modes disagree")
    if model_config.get("pipeline_version") != pipeline_version:
        raise ProvenanceError("top-level and resolved pipeline versions disagree")
    if environment_config.get("split") != split:
        raise ProvenanceError("top-level and resolved splits disagree")

    if requirement is None:
        return
    if pipeline_version not in requirement.pipeline_versions:
        raise ProvenanceError(f"pipeline {pipeline_version!r} is not allowed")
    if action_selection_mode not in requirement.action_selection_modes:
        raise ProvenanceError(
            f"action-selection mode {action_selection_mode!r} is not allowed"
        )
    if split not in requirement.splits:
        raise ProvenanceError(f"split {split!r} is not allowed")


def load_run_trajectories(
    run_directory: Path,
    *,
    requirement: ProvenanceRequirement,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Load one run only after its explicit provenance satisfies a requirement."""
    manifest_path = run_directory / "run_manifest.json"
    trajectory_path = run_directory / "trajectory.jsonl"
    manifest = _read_json_object(manifest_path)
    validate_manifest_provenance(manifest, requirement)
    trajectories = tuple(_read_json_lines(trajectory_path))
    if not trajectories:
        raise ProvenanceError(f"no trajectories in {trajectory_path}")
    return manifest, trajectories


def discover_run_directories(
    roots: Iterable[Path],
    *,
    requirement: ProvenanceRequirement,
) -> tuple[Path, ...]:
    """Discover only runs whose manifests pass explicit provenance filtering."""
    accepted: list[Path] = []
    for root in roots:
        for manifest_path in sorted(root.rglob("run_manifest.json")):
            manifest = _read_json_object(manifest_path)
            try:
                validate_manifest_provenance(manifest, requirement)
            except ProvenanceError:
                continue
            accepted.append(manifest_path.parent)
    return tuple(accepted)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProvenanceError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ProvenanceError(f"{path} must contain a JSON object")
    return value


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ProvenanceError(f"cannot read {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ProvenanceError(
                f"invalid JSON in {path} line {line_number}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise ProvenanceError(
                f"{path} line {line_number} must contain a JSON object"
            )
        records.append(value)
    return records


def _required_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ProvenanceError(f"manifest is missing explicit {key}")
    return value
