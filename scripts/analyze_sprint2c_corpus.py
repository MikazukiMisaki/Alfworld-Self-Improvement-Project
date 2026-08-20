#!/usr/bin/env python3
"""Validate Sprint 2C outcomes and freeze the qualitative review set."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from recovery.corpus import sha256_file, validate_prefix_manifest  # noqa: E402
from scripts.collect_sprint2c_paired_corpus import aggregate_pairs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    arguments = parser.parse_args()
    run = arguments.run.resolve()
    report_path = run / "report.json"
    manifest_path = run / "selected_prefix_manifest.json"
    report = _read_object(report_path)
    manifest = _read_object(manifest_path)
    validate_prefix_manifest(manifest)
    pairs = report.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 120:
        raise RuntimeError("Sprint 2C analysis requires exactly 120 pairs")
    recomputed_aggregate = aggregate_pairs(pairs)
    stored_aggregate_matches = report.get("aggregate") == recomputed_aggregate

    by_id = {item["prefix_id"]: item for item in pairs}
    if len(by_id) != 120:
        raise RuntimeError("duplicate paired prefix IDs")
    non_neutral = [
        prefix_id
        for prefix_id in manifest["selection_protocol"][
            "neutral_annotation_priority"
        ]
        if by_id[prefix_id]["classification"] != "neutral"
    ]
    neutral = [
        prefix_id
        for prefix_id in manifest["selection_protocol"][
            "neutral_annotation_priority"
        ]
        if by_id[prefix_id]["classification"] == "neutral"
    ][:20]
    annotation_ids = non_neutral + neutral
    template = {
        "schema_version": "sprint2c_qualitative_annotations_v1",
        "source_report_sha256": sha256_file(report_path),
        "source_prefix_manifest_sha256": sha256_file(manifest_path),
        "selection_rule": manifest["selection_protocol"][
            "neutral_annotation_rule"
        ],
        "all_beneficial_and_harmful_count": len(non_neutral),
        "neutral_sample_count": len(neutral),
        "annotations": [
            {
                "prefix_id": prefix_id,
                "classification": by_id[prefix_id]["classification"],
                "diagnosis_quality": None,
                "subgoal_quality": None,
                "subgoal_action_agreement": None,
                "downstream_effect": None,
                "notes": None,
            }
            for prefix_id in annotation_ids
        ],
    }
    template_path = run / "qualitative_annotations.json"
    if template_path.exists():
        raise RuntimeError(f"refusing to overwrite annotations: {template_path}")
    _write_json(template_path, template)

    beneficial = [item for item in pairs if item["classification"] == "beneficial"]
    summary = {
        "schema_version": "sprint2c_analysis_summary_v1",
        "source_report_sha256": sha256_file(report_path),
        "source_prefix_manifest_sha256": sha256_file(manifest_path),
        "aggregate": recomputed_aggregate,
        "stored_aggregate_matches_recomputed": stored_aggregate_matches,
        "stored_aggregate_correction": (
            None
            if stored_aggregate_matches
            else {
                "scope": "summary_only",
                "reason": (
                    "The original report counted fail-closed Stage-1 failures as "
                    "mapping failures even though Stage-2 mappings were valid."
                ),
                "raw_pair_records_modified": False,
            }
        ),
        "beneficial_episode_group_count": len(
            {item["episode_group_id"] for item in beneficial}
        ),
        "beneficial_task_family_count": len(
            {item["task_family"] for item in beneficial}
        ),
        "annotation_class_counts": dict(
            sorted(Counter(by_id[value]["classification"] for value in annotation_ids).items())
        ),
        "annotation_ids": annotation_ids,
        "gate": _gate(beneficial),
    }
    _write_json(run / "analysis_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _gate(beneficial: list[dict[str, Any]]) -> str:
    count = len(beneficial)
    if count <= 2:
        return "C_RECOVERY_OPPORTUNITY_TOO_SPARSE"
    if count <= 7:
        return "B_COLLECT_SECOND_PREREGISTERED_BATCH"
    groups = {item["episode_group_id"] for item in beneficial}
    families = {item["task_family"] for item in beneficial}
    if len(groups) >= 4 and len(families) >= 2:
        return "A_LABEL_DENSITY_SUFFICIENT"
    return "B_COLLECT_SECOND_PREREGISTERED_BATCH_INSUFFICIENT_DIVERSITY"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
