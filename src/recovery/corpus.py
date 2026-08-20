"""Validation helpers for the frozen Sprint 2C paired corpus."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


TASK_FAMILIES = frozenset(
    {
        "pick_and_place_simple",
        "pick_clean_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_two_obj_and_place",
        "look_at_obj_in_light",
    }
)
INITIAL_POOL_SIZE = 200
EXTENSION_POOL_SIZE = 100
INITIAL_SEED = 2000
MINIMUM_INITIAL_SUCCESSES = 10
POOL_BURN_IN_SEEDS = list(range(1900, 1943))


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one immutable artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_family_from_id(task_id: str) -> str:
    """Recover the canonical ALFWorld task-family prefix."""
    family = task_id.split("-", 1)[0]
    if family not in TASK_FAMILIES:
        raise ValueError(f"unknown ALFWorld task family: {family}")
    return family


def validate_pool_schedule(schedule: dict[str, Any]) -> None:
    """Validate the initial and conditional extension reset schedule."""
    if schedule.get("schema_version") != "sprint2c_h4_pool_schedule_v1":
        raise ValueError("unexpected Sprint 2C schedule schema")
    if schedule.get("split") != "valid_seen":
        raise ValueError("Sprint 2C schedule must use valid_seen")
    if schedule.get("burn_in_seeds") != POOL_BURN_IN_SEEDS:
        raise ValueError("Sprint 2C burn-in reset schedule changed")
    if schedule.get("reset_offset_rule") != {
        "prior_formal_burn_in_resets": 3,
        "valid_seen_game_count": 140,
        "offset": INITIAL_SEED % 140,
        "total_burn_in_resets": len(POOL_BURN_IN_SEEDS),
    }:
        raise ValueError("Sprint 2C reset-offset rule changed")
    gate = schedule.get("extension_gate")
    if gate != {
        "execute_extension_if_initial_success_count_below": MINIMUM_INITIAL_SUCCESSES,
        "initial_episode_count": INITIAL_POOL_SIZE,
        "extension_episode_count": EXTENSION_POOL_SIZE,
        "maximum_total_episodes": INITIAL_POOL_SIZE + EXTENSION_POOL_SIZE,
        "adaptive_changes_after_initial_pool": False,
    }:
        raise ValueError("Sprint 2C extension gate changed")
    episodes = schedule.get("episodes")
    expected_count = INITIAL_POOL_SIZE + EXTENSION_POOL_SIZE
    if not isinstance(episodes, list) or len(episodes) != expected_count:
        raise ValueError("Sprint 2C schedule must preregister 300 positions")
    seen_tasks: set[tuple[int, str]] = set()
    for index, item in enumerate(episodes):
        expected_block = "initial" if index < INITIAL_POOL_SIZE else "extension"
        if item.get("schedule_index") != index:
            raise ValueError(f"schedule index mismatch at position {index}")
        if item.get("reset_order_position") != index + len(POOL_BURN_IN_SEEDS):
            raise ValueError(f"reset-order mismatch at position {index}")
        if item.get("seed") != INITIAL_SEED + index:
            raise ValueError(f"nominal seed mismatch at position {index}")
        if item.get("pool_block") != expected_block:
            raise ValueError(f"pool block mismatch at position {index}")
        if item.get("split") != "valid_seen":
            raise ValueError(f"split mismatch at position {index}")
        task_id = item.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"missing task identity at position {index}")
        if item.get("task_family") != task_family_from_id(task_id):
            raise ValueError(f"task-family mismatch at position {index}")
        key = (index, task_id)
        if key in seen_tasks:
            raise ValueError(f"duplicate schedule record at position {index}")
        seen_tasks.add(key)


def executed_schedule(
    schedule: dict[str, Any], initial_success_count: int
) -> list[dict[str, Any]]:
    """Apply only the preregistered extension gate."""
    validate_pool_schedule(schedule)
    count = (
        INITIAL_POOL_SIZE + EXTENSION_POOL_SIZE
        if initial_success_count < MINIMUM_INITIAL_SUCCESSES
        else INITIAL_POOL_SIZE
    )
    return list(schedule["episodes"][:count])
