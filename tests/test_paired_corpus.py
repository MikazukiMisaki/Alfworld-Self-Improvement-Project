from __future__ import annotations

import pytest

from scripts.collect_sprint2c_h4_pool import _validate_frozen_h4
from recovery.corpus import (
    POOL_BURN_IN_SEEDS,
    executed_schedule,
    task_family_from_id,
    validate_pool_schedule,
)


def schedule() -> dict:
    episodes = []
    families = [
        "pick_and_place_simple",
        "pick_clean_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_two_obj_and_place",
        "look_at_obj_in_light",
    ]
    for index in range(300):
        family = families[index % len(families)]
        episodes.append(
            {
                "schedule_index": index,
                "reset_order_position": index + len(POOL_BURN_IN_SEEDS),
                "seed": 2000 + index,
                "pool_block": "initial" if index < 200 else "extension",
                "split": "valid_seen",
                "task_id": f"{family}-Object-None-Receptacle-{index}/trial",
                "task_family": family,
            }
        )
    return {
        "schema_version": "sprint2c_h4_pool_schedule_v1",
        "split": "valid_seen",
        "burn_in_seeds": POOL_BURN_IN_SEEDS,
        "reset_offset_rule": {
            "prior_formal_burn_in_resets": 3,
            "valid_seen_game_count": 140,
            "offset": 40,
            "total_burn_in_resets": len(POOL_BURN_IN_SEEDS),
        },
        "extension_gate": {
            "execute_extension_if_initial_success_count_below": 10,
            "initial_episode_count": 200,
            "extension_episode_count": 100,
            "maximum_total_episodes": 300,
            "adaptive_changes_after_initial_pool": False,
        },
        "episodes": episodes,
    }


def test_pool_schedule_and_conditional_extension_are_fixed() -> None:
    value = schedule()
    validate_pool_schedule(value)

    assert len(executed_schedule(value, initial_success_count=10)) == 200
    assert len(executed_schedule(value, initial_success_count=9)) == 300


def test_pool_schedule_rejects_reset_order_or_task_family_changes() -> None:
    reset_changed = schedule()
    reset_changed["episodes"][7]["reset_order_position"] = 99
    with pytest.raises(ValueError, match="reset-order"):
        validate_pool_schedule(reset_changed)

    family_changed = schedule()
    family_changed["episodes"][7]["task_family"] = "unknown"
    with pytest.raises(ValueError, match="task-family"):
        validate_pool_schedule(family_changed)


def test_task_family_parser_rejects_unknown_families() -> None:
    assert (
        task_family_from_id("pick_and_place_simple-Book-None-Sofa-1/trial")
        == "pick_and_place_simple"
    )
    with pytest.raises(ValueError, match="unknown"):
        task_family_from_id("unsupported-Book-None-Sofa-1/trial")


def test_h4_pool_rejects_any_model_or_split_change() -> None:
    model = {
        "model_id": "Qwen/Qwen3-8B",
        "device": "auto",
        "dtype": "bfloat16",
        "trust_remote_code": False,
        "enable_thinking": False,
        "pipeline_version": "indexed_bounded_context_v1",
        "action_selection": {"mode": "indexed_admissible"},
        "history_context": {"mode": "bounded_recent_state", "window": 4},
        "generation": {
            "max_new_tokens": 32,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
        },
    }
    _validate_frozen_h4(model, {"split": "valid_seen"})

    changed = {**model, "enable_thinking": True}
    with pytest.raises(RuntimeError, match="configuration changed"):
        _validate_frozen_h4(changed, {"split": "valid_seen"})
    with pytest.raises(RuntimeError, match="valid_seen"):
        _validate_frozen_h4(model, {"split": "valid_unseen"})
