#!/usr/bin/env python3
"""Classify historical result runs using artifact contents, not names alone."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def audit_results(root: Path) -> dict[str, Any]:
    """Return a deterministic provenance report for run directories below root."""
    run_directories = sorted(
        {
            path.parent
            for artifact_name in ("run_manifest.json", "metrics.json", "trajectory.jsonl")
            for path in root.rglob(artifact_name)
        }
    )
    runs = [_classify_run(path, root) for path in run_directories]
    duplicate_groups: list[dict[str, Any]] = []
    by_trajectory_hash: dict[str, list[str]] = defaultdict(list)
    for run in runs:
        trajectory_hash = run["artifact_hashes"].get("trajectory.jsonl")
        if trajectory_hash:
            by_trajectory_hash[trajectory_hash].append(run["artifact_location"])
    for digest, locations in sorted(by_trajectory_hash.items()):
        if len(locations) > 1:
            duplicate_groups.append(
                {"sha256": digest, "artifact_locations": sorted(locations)}
            )

    pipeline_counts: dict[str, int] = defaultdict(int)
    artifact_type_counts: dict[str, int] = defaultdict(int)
    for run in runs:
        pipeline_counts[run["pipeline_version"]] += 1
        artifact_type_counts[run["artifact_type"]] += 1
    return {
        "schema_version": 1,
        "audit_scope": _portable_path(root),
        "classification_policy": (
            "Content evidence from manifests, resolved configs, trajectories, and "
            "metrics takes precedence over directory names. Missing evidence remains unknown."
        ),
        "environment_verification": (
            "Historical manifests lack a complete package, dataset, and hardware fingerprint; "
            "their original runtime environment is therefore not verified."
        ),
        "summary": {
            "run_count": len(runs),
            "pipeline_counts": dict(sorted(pipeline_counts.items())),
            "artifact_type_counts": dict(sorted(artifact_type_counts.items())),
            "verified_successful_legacy_cases": 0,
        },
        "duplicate_trajectory_groups": duplicate_groups,
        "runs": runs,
    }


def _classify_run(path: Path, root: Path) -> dict[str, Any]:
    manifest, manifest_error = _optional_json(path / "run_manifest.json")
    metrics, metrics_error = _optional_json(path / "metrics.json")
    trajectories, trajectory_error = _optional_jsonl(path / "trajectory.jsonl")
    evidence: list[str] = []
    notes: list[str] = []

    explicit_pipeline = manifest.get("pipeline_version") if manifest else None
    explicit_mode = manifest.get("action_selection_mode") if manifest else None
    resolved = manifest.get("resolved_config", {}) if manifest else {}
    model_config = resolved.get("model", {}) if isinstance(resolved, dict) else {}
    collection_config = (
        resolved.get("collection", {}) if isinstance(resolved, dict) else {}
    )
    environment_config = (
        resolved.get("environment", {}) if isinstance(resolved, dict) else {}
    )
    selection_config = (
        model_config.get("action_selection", {})
        if isinstance(model_config, dict)
        else {}
    )
    resolved_mode = (
        selection_config.get("mode") if isinstance(selection_config, dict) else None
    )
    model_config_path = (
        collection_config.get("model_config")
        if isinstance(collection_config, dict)
        else None
    )

    step_modes: set[str] = set()
    prompt_versions: set[str] = set()
    outputs: list[str] = []
    if trajectories:
        for trajectory in trajectories:
            for step in trajectory.get("steps", []):
                if not isinstance(step, dict):
                    continue
                metadata = step.get("metadata", {})
                if isinstance(metadata, dict):
                    mode = metadata.get("action_selection_mode")
                    if isinstance(mode, str):
                        step_modes.add(mode)
                    policy = metadata.get("policy", {})
                    if isinstance(policy, dict):
                        prompt_version = policy.get("prompt_version")
                        if isinstance(prompt_version, str):
                            prompt_versions.add(prompt_version)
                output = step.get("model_output")
                if isinstance(output, str):
                    outputs.append(output)

    pipeline_version = "unknown"
    action_selection_mode = explicit_mode or resolved_mode
    confidence = "low"
    if explicit_pipeline in {"legacy_v1", "free_form_v1", "indexed_v1"}:
        pipeline_version = explicit_pipeline
        confidence = "high"
        evidence.append("explicit top-level pipeline_version")
    elif (
        resolved_mode == "indexed_admissible"
        and step_modes <= {"indexed_admissible"}
        and ("indexed-admissible-action-v1" in prompt_versions or any(
            output.startswith("Action-ID:") for output in outputs
        ))
    ):
        pipeline_version = "indexed_v1"
        action_selection_mode = "indexed_admissible"
        confidence = "high"
        evidence.extend(
            ["resolved indexed_admissible mode", "indexed trajectory protocol evidence"]
        )
    elif manifest and isinstance(model_config, dict):
        pipeline_version = "free_form_v1"
        action_selection_mode = action_selection_mode or "free_form_validated"
        confidence = "medium"
        evidence.append("canonical Sprint 1 manifest and free-form model configuration")
        if outputs:
            evidence.append("trajectory outputs are not Action-ID selections")
    else:
        notes.append("insufficient content to identify a pipeline")

    complete = bool(manifest and metrics and trajectories)
    episode_count = len(trajectories or [])
    configured_episodes = (
        collection_config.get("episodes")
        if isinstance(collection_config, dict)
        else None
    )
    run_id = manifest.get("run_id") if manifest else path.name
    if not complete:
        artifact_type = "smoke_test"
        evidence.append("incomplete one-run artifact set")
    elif episode_count > 1 and "diagnostic" in str(run_id).casefold():
        artifact_type = "diagnostic"
        evidence.append("multi-episode artifact plus diagnostic run metadata")
    elif "smoke" in str(run_id).casefold() and configured_episodes == 1:
        artifact_type = "smoke_test"
        evidence.append("one configured episode plus smoke run metadata")
    else:
        artifact_type = "historical_run"

    success_rate = metrics.get("success_rate") if metrics else None
    successful_trajectories = _successful_trajectories(trajectories or [])
    if success_rate is None and trajectories:
        success_rate = successful_trajectories / len(trajectories)
    if manifest_error:
        notes.append(manifest_error)
    if metrics_error:
        notes.append(metrics_error)
    if trajectory_error:
        notes.append(trajectory_error)
    if manifest and not explicit_pipeline:
        notes.append("historical manifest lacks explicit pipeline_version")
    if manifest and not explicit_mode:
        notes.append("historical manifest lacks explicit action_selection_mode")
    notes.append("original runtime environment not fully verifiable from manifest")

    seeds = manifest.get("seed_schedule") if manifest else None
    split = manifest.get("split") if manifest else None
    if split is None and isinstance(environment_config, dict):
        split = environment_config.get("split")
    model = manifest.get("model_version") if manifest else None
    return {
        "run_id": run_id,
        "pipeline_version": pipeline_version,
        "action_selection_mode": action_selection_mode or "unknown",
        "artifact_type": artifact_type,
        "model": model,
        "model_config_path": model_config_path,
        "split": split,
        "seeds": seeds,
        "git_revision": manifest.get("git_revision") if manifest else None,
        "created_at": manifest.get("created_at") if manifest else None,
        "episodes": episode_count or configured_episodes,
        "success_rate": success_rate,
        "successful_trajectories": successful_trajectories,
        "artifact_location": _artifact_location(path, root),
        "classification_confidence": confidence,
        "classification_evidence": evidence,
        "environment_provenance": "partial" if manifest else "unknown",
        "artifact_hashes": _artifact_hashes(path),
        "notes": notes,
    }


def _successful_trajectories(trajectories: list[dict[str, Any]]) -> int:
    successes = 0
    for trajectory in trajectories:
        steps = trajectory.get("steps")
        if not isinstance(steps, list) or not steps:
            continue
        final = steps[-1]
        if (
            isinstance(final, dict)
            and final.get("done") is True
            and isinstance(final.get("reward"), (int, float))
            and final["reward"] > 0
        ):
            successes += 1
    return successes


def _artifact_hashes(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in ("run_manifest.json", "metrics.json", "trajectory.jsonl"):
        artifact = path / name
        if artifact.is_file():
            hashes[name] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    return hashes


def _optional_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, f"missing {path.name}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return None, f"unreadable {path.name}: {error}"
    return (value, None) if isinstance(value, dict) else (None, f"invalid {path.name}")


def _optional_jsonl(path: Path) -> tuple[list[dict[str, Any]] | None, str | None]:
    if not path.is_file():
        return None, f"missing {path.name}"
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    return None, f"invalid {path.name} record"
                records.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return None, f"unreadable {path.name}: {error}"
    return records, None


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact human-readable view of the machine report."""
    summary = report["summary"]
    lines = [
        "# Historical Result Provenance",
        "",
        "Classification uses artifact contents first. Directory names are only supplementary evidence.",
        "",
        f"- Runs: {summary['run_count']}",
        f"- Pipelines: `{json.dumps(summary['pipeline_counts'], sort_keys=True)}`",
        f"- Artifact types: `{json.dumps(summary['artifact_type_counts'], sort_keys=True)}`",
        "- Verified successful legacy cases: 0",
        "",
        "| Run | Pipeline | Mode | Type | Episodes | Success | Confidence |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for run in report["runs"]:
        rate = run["success_rate"]
        rendered_rate = "unknown" if rate is None else f"{float(rate):.3f}"
        lines.append(
            f"| `{run['run_id']}` | `{run['pipeline_version']}` | "
            f"`{run['action_selection_mode']}` | `{run['artifact_type']}` | "
            f"{run['episodes'] or 0} | {rendered_rate} | {run['classification_confidence']} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "The historical `AlfWorldLegacy` runner and its reported 210-episode/28-success corpus are not available in this checkout or in the audited server workspace. None of the available tracked trajectories succeeded, so no legacy-success regression case can be verified from current evidence.",
            "",
            "Historical manifests do not contain a complete dependency, dataset, and hardware fingerprint. Their runtime environments remain only partially attributable even when pipeline classification is strong.",
            "",
        ]
    )
    return "\n".join(lines)


def _portable_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return "external-audit-root"


def _artifact_location(path: Path, root: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return f"{_portable_path(root)}/{path.relative_to(root).as_posix()}"


def main() -> int:
    """Write machine- and human-readable provenance reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument(
        "--json-output", type=Path, default=PROJECT_ROOT / "reports/result_provenance.json"
    )
    parser.add_argument(
        "--markdown-output", type=Path, default=PROJECT_ROOT / "reports/result_provenance.md"
    )
    arguments = parser.parse_args()
    report = audit_results(arguments.root.resolve())
    arguments.json_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    arguments.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(f"classified {report['summary']['run_count']} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
