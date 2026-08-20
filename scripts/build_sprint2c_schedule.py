#!/usr/bin/env python3
"""Materialize the exact Sprint 2C reset schedule without model inference."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from env.alfworld import AlfWorldTextEnvironment  # noqa: E402
from recovery.corpus import (  # noqa: E402
    EXTENSION_POOL_SIZE,
    INITIAL_POOL_SIZE,
    INITIAL_SEED,
    MINIMUM_INITIAL_SUCCESSES,
    POOL_BURN_IN_SEEDS,
    sha256_file,
    task_family_from_id,
    validate_pool_schedule,
)
from scripts.collect_baseline import _environment, _read_yaml  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "configs/corpus/sprint2c_h4_pool_schedule.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    checksum = output.with_suffix(".sha256")
    if output.exists() or checksum.exists():
        raise SystemExit(f"refusing to overwrite frozen schedule: {output}")

    collection = _read_yaml(PROJECT_ROOT / "configs/collection/baseline_indexed_h4.yaml")
    environment_config = _read_yaml(PROJECT_ROOT / collection["environment_config"])
    if environment_config.get("split") != "valid_seen":
        raise RuntimeError("Sprint 2C schedule requires valid_seen")
    environment = AlfWorldTextEnvironment(
        _environment(environment_config, PROJECT_ROOT)
    )
    burn_in_seeds = POOL_BURN_IN_SEEDS
    for seed in burn_in_seeds:
        environment.reset(seed=seed)

    episodes = []
    total = INITIAL_POOL_SIZE + EXTENSION_POOL_SIZE
    for index in range(total):
        seed = INITIAL_SEED + index
        reset = environment.reset(seed=seed)
        episodes.append(
            {
                "schedule_index": index,
                "reset_order_position": index + len(burn_in_seeds),
                "seed": seed,
                "pool_block": "initial" if index < INITIAL_POOL_SIZE else "extension",
                "split": "valid_seen",
                "task_id": reset.task.task_id,
                "task_family": task_family_from_id(reset.task.task_id),
            }
        )
    schedule = {
        "schema_version": "sprint2c_h4_pool_schedule_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_from_git_revision": _git_revision(),
        "frozen_h4_protocol_commit": "1b84f922a39f58c3b6ed2ba31cb29d896486b0d0",
        "split": "valid_seen",
        "burn_in_seeds": burn_in_seeds,
        "environment_burn_in_resets": len(burn_in_seeds),
        "reset_offset_rule": {
            "prior_formal_burn_in_resets": 3,
            "valid_seen_game_count": 140,
            "offset": INITIAL_SEED % 140,
            "total_burn_in_resets": len(burn_in_seeds),
        },
        "extension_gate": {
            "execute_extension_if_initial_success_count_below": MINIMUM_INITIAL_SUCCESSES,
            "initial_episode_count": INITIAL_POOL_SIZE,
            "extension_episode_count": EXTENSION_POOL_SIZE,
            "maximum_total_episodes": total,
            "adaptive_changes_after_initial_pool": False,
        },
        "episodes": episodes,
    }
    validate_pool_schedule(schedule)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(schedule, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = sha256_file(output)
    checksum.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(f"wrote {total} frozen reset positions: {output}")
    print(digest)
    return 0


def _git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


if __name__ == "__main__":
    raise SystemExit(main())
