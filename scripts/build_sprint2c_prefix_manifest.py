#!/usr/bin/env python3
"""Build the deterministic 120-prefix Sprint 2C paired manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from recovery.corpus import (  # noqa: E402
    SELECTION_SEED,
    select_prefixes,
    sha256_file,
    state_fingerprint,
    validate_prefix_manifest,
)
from trajectory.provenance import (  # noqa: E402
    ProvenanceRequirement,
    load_run_trajectories,
)


PRIOR_MANIFESTS = (
    PROJECT_ROOT / "configs/experiments/sprint2b_revised_recovery_development.json",
    PROJECT_ROOT / "configs/experiments/r2_two_stage_development.json",
    PROJECT_ROOT / "configs/experiments/r2_two_stage_v2_development.json",
)
DEFAULT_OUTPUT = PROJECT_ROOT / "configs/corpus/sprint2c_paired_prefix_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if output.exists() or output.with_suffix(".sha256").exists():
        raise SystemExit(f"refusing to overwrite frozen prefix manifest: {output}")
    pool = arguments.pool.resolve()
    pool_manifest, trajectories_tuple = load_run_trajectories(
        pool,
        requirement=ProvenanceRequirement.one(
            pipeline_version="indexed_bounded_context_v1",
            action_selection_mode="indexed_admissible",
            split="valid_seen",
        ),
    )
    trajectories = list(trajectories_tuple)
    excluded = _prior_state_fingerprints()
    prefixes, selection_metadata = select_prefixes(
        trajectories, excluded_state_fingerprints=excluded
    )
    neutral_annotation_priority = sorted(
        (item["prefix_id"] for item in prefixes),
        key=lambda prefix_id: _stable_hash(
            SELECTION_SEED, "neutral-annotation-priority", prefix_id
        ),
    )
    manifest = {
        "schema_version": "sprint2c_paired_prefix_manifest_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_from_git_revision": _git_revision(),
        "frozen_recovery_protocol_commit": "1b84f922a39f58c3b6ed2ba31cb29d896486b0d0",
        "source": {
            "h4_pool": str(pool.relative_to(PROJECT_ROOT)),
            "h4_pool_git_revision": pool_manifest["git_revision"],
            "h4_trajectory_sha256": sha256_file(pool / "trajectory.jsonl"),
            "h4_run_manifest_sha256": sha256_file(pool / "run_manifest.json"),
            "h4_schedule_sha256": pool_manifest["metadata"]["schedule_sha256"],
            "prior_recovery_manifests": [
                {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "sha256": sha256_file(path),
                }
                for path in PRIOR_MANIFESTS
            ],
        },
        "selection_protocol": {
            **selection_metadata,
            "minimum_action_count": 1,
            "minimum_remaining_horizon": 8,
            "maximum_prefixes_per_episode": 2,
            "duplicate_state_hash_excludes_nominal_seed": True,
            "future_outcomes_are_stratification_only": True,
            "selector_features_path": "prefixes[].decision_features",
            "group_aware_split_key": "episode_group_id",
            "neutral_annotation_rule": (
                "annotate_all_beneficial_and_harmful_then_first_20_neutral_"
                "states_in_preregistered_priority_order"
            ),
            "neutral_annotation_priority": neutral_annotation_priority,
        },
        "prefixes": prefixes,
    }
    validate_prefix_manifest(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = sha256_file(output)
    output.with_suffix(".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps(selection_metadata, indent=2, sort_keys=True))
    print(digest)
    return 0


def _prior_state_fingerprints() -> set[str]:
    formal_runs: dict[Path, dict[int, dict]] = {}
    fingerprints = set()
    for manifest_path in PRIOR_MANIFESTS:
        config = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_path = PROJECT_ROOT / config["source"]["formal_run"]
        if run_path not in formal_runs:
            _, trajectories = load_run_trajectories(
                run_path,
                requirement=ProvenanceRequirement.one(
                    pipeline_version="indexed_bounded_context_v1",
                    action_selection_mode="indexed_admissible",
                    split="valid_seen",
                ),
            )
            formal_runs[run_path] = {int(item["seed"]): item for item in trajectories}
        by_seed = formal_runs[run_path]
        for prefix in config["prefixes"]:
            trajectory = by_seed[int(prefix["seed"])]
            fingerprints.add(
                state_fingerprint(trajectory, int(prefix["replayed_action_count"]))
            )
    return fingerprints


def _git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def _stable_hash(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
