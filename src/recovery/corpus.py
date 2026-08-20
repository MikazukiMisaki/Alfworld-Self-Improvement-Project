"""Validation helpers for the frozen Sprint 2C paired corpus."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from recovery.fixed import loop_indicators
from recovery.pilot import prefix_hash


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
PREFIX_STRATUM_COUNTS = {
    "successful_continuation": 20,
    "past_loop_failed": 30,
    "high_uncertainty_non_loop_failed": 30,
    "random_non_loop_failed": 40,
}
SELECTION_SEED = "sprint2c-prefix-selection-v1"
DECISION_FEATURES = frozenset(
    {
        "task_family",
        "step_index",
        "normalized_step",
        "remaining_horizon",
        "normalized_remaining_horizon",
        "admissible_action_count",
        "h4_proposed_action_id",
        "h4_proposed_mapped_command",
        "h4_decision_token_entropy",
        "h4_selected_action_log_probability",
        "h4_input_token_count",
        "recent_adjacent_repeat_count",
        "recent_adjacent_repeat_indicator",
        "recent_two_cycle_count",
        "recent_two_cycle_indicator",
        "recent_unique_action_count",
        "current_observation_repeats_recent",
        "current_observation",
        "task_goal",
        "recent_transition_text",
        "entropy_quantile",
    }
)


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


def trajectory_succeeded(trajectory: dict[str, Any]) -> bool:
    """Read success from immutable trajectory outcomes."""
    steps = trajectory.get("steps", [])
    return bool(steps and steps[-1].get("done") and steps[-1].get("reward", 0) > 0)


def state_fingerprint(trajectory: dict[str, Any], action_count: int) -> str:
    """Hash a branch state without nominal seed so duplicate states collapse."""
    target = trajectory["steps"][action_count]
    payload = {
        "task_id": trajectory["task"]["task_id"],
        "action_count": action_count,
        "actions": [step["action"] for step in trajectory["steps"][:action_count]],
        "observation": target["observation"],
        "valid_actions": target["valid_actions"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def eligible_states(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract non-terminal, reconstructable decision states through step 42."""
    states = []
    steps = trajectory.get("steps", [])
    maximum_action_count = min(len(steps) - 1, 42)
    for action_count in range(1, maximum_action_count + 1):
        target = steps[action_count]
        valid_actions = target.get("valid_actions")
        selection = (
            target.get("metadata", {})
            .get("policy", {})
            .get("action_selection", {})
        )
        tokens = target.get("token_statistics") or {}
        if (
            not isinstance(valid_actions, list)
            or not valid_actions
            or selection.get("selection_status") != "selected"
            or selection.get("mapped_environment_command") != target.get("action")
            or tokens.get("mean_token_entropy") is None
        ):
            continue
        recent_steps = steps[max(0, action_count - 4) : action_count]
        recent_actions = [str(step["action"]) for step in recent_steps]
        loops = loop_indicators(recent_actions)
        recent_observations = [str(step["observation"]) for step in recent_steps]
        transition_lines = []
        for index in range(max(0, action_count - 4), action_count):
            transition_lines.append(
                f"Action: {steps[index]['action']}\n"
                f"Result: {steps[index + 1]['observation']}"
            )
        schedule_index = int(trajectory["metadata"]["schedule_index"])
        family = str(trajectory["metadata"]["task_family"])
        states.append(
            {
                "episode_group_id": str(trajectory["trajectory_id"]),
                "schedule_index": schedule_index,
                "seed": int(trajectory["seed"]),
                "task_id": str(trajectory["task"]["task_id"]),
                "task_family": family,
                "task_goal": str(trajectory["task"]["text"]),
                "source_continue_success": trajectory_succeeded(trajectory),
                "action_count": action_count,
                "prefix_step": action_count - 1,
                "remaining_horizon": 50 - action_count,
                "prefix_sha256": prefix_hash(trajectory, action_count),
                "state_fingerprint": state_fingerprint(trajectory, action_count),
                "stage_bucket": _stage_bucket(action_count),
                "past_loop": bool(
                    loops["has_adjacent_repeat"] or loops["has_two_cycle"]
                ),
                "decision_features": {
                    "task_family": family,
                    "step_index": action_count,
                    "normalized_step": action_count / 50,
                    "remaining_horizon": 50 - action_count,
                    "normalized_remaining_horizon": (50 - action_count) / 50,
                    "admissible_action_count": len(valid_actions),
                    "h4_proposed_action_id": selection.get("parsed_action_id"),
                    "h4_proposed_mapped_command": selection.get(
                        "mapped_environment_command"
                    ),
                    "h4_decision_token_entropy": tokens.get("mean_token_entropy"),
                    "h4_selected_action_log_probability": tokens.get(
                        "mean_token_log_probability"
                    ),
                    "h4_input_token_count": tokens.get("input_tokens"),
                    "recent_adjacent_repeat_count": loops[
                        "adjacent_repeat_events"
                    ],
                    "recent_adjacent_repeat_indicator": loops[
                        "has_adjacent_repeat"
                    ],
                    "recent_two_cycle_count": loops["two_cycle_events"],
                    "recent_two_cycle_indicator": loops["has_two_cycle"],
                    "recent_unique_action_count": len(set(recent_actions)),
                    "current_observation_repeats_recent": str(target["observation"])
                    in recent_observations,
                    "current_observation": str(target["observation"]),
                    "task_goal": str(trajectory["task"]["text"]),
                    "recent_transition_text": "\n\n".join(transition_lines),
                    "entropy_quantile": None,
                },
                "original_h4_continue_action": str(target["action"]),
                "original_h4_continue_action_id": selection.get("parsed_action_id"),
                "valid_actions": list(valid_actions),
            }
        )
    return states


def select_prefixes(
    trajectories: list[dict[str, Any]],
    *,
    excluded_state_fingerprints: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the fixed four-stratum deterministic sampling protocol."""
    candidates = [state for item in trajectories for state in eligible_states(item)]
    candidates = [
        state
        for state in candidates
        if state["state_fingerprint"] not in excluded_state_fingerprints
    ]
    non_loop_failed = [
        state
        for state in candidates
        if not state["source_continue_success"] and not state["past_loop"]
    ]
    entropies = sorted(
        float(state["decision_features"]["h4_decision_token_entropy"])
        for state in non_loop_failed
    )
    if not entropies:
        raise ValueError("no eligible non-loop failed states")
    thresholds = {
        "q25": _nearest_rank_quantile(entropies, 0.25),
        "q50": _nearest_rank_quantile(entropies, 0.50),
        "q75": _nearest_rank_quantile(entropies, 0.75),
    }
    entropy_degenerate = max(entropies) - min(entropies) < 1e-8
    for state in candidates:
        entropy = float(state["decision_features"]["h4_decision_token_entropy"])
        state["decision_features"]["entropy_quantile"] = _entropy_bucket(
            entropy, thresholds
        )

    selected: list[dict[str, Any]] = []
    selected_fingerprints: set[str] = set()
    episode_counts: Counter[str] = Counter()
    successful = [state for state in candidates if state["source_continue_success"]]
    selected.extend(
        _select_successful_states(
            successful,
            count=PREFIX_STRATUM_COUNTS["successful_continuation"],
            selected_fingerprints=selected_fingerprints,
            episode_counts=episode_counts,
        )
    )
    loop_failed = [
        state
        for state in candidates
        if not state["source_continue_success"] and state["past_loop"]
    ]
    selected.extend(
        _balanced_select(
            loop_failed,
            count=PREFIX_STRATUM_COUNTS["past_loop_failed"],
            stratum="past_loop_failed",
            selected_fingerprints=selected_fingerprints,
            episode_counts=episode_counts,
        )
    )
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in non_loop_failed:
        by_episode[state["episode_group_id"]].append(state)
    episode_q75 = {
        episode_id: _nearest_rank_quantile(
            sorted(
                float(state["decision_features"]["h4_decision_token_entropy"])
                for state in states
            ),
            0.75,
        )
        for episode_id, states in by_episode.items()
    }
    high_uncertainty = []
    if not entropy_degenerate:
        for state in non_loop_failed:
            threshold = episode_q75[state["episode_group_id"]]
            if (
                float(state["decision_features"]["h4_decision_token_entropy"])
                >= threshold
            ):
                state["selection_episode_entropy_q75"] = threshold
                high_uncertainty.append(state)
    uncertainty_stratum = "high_uncertainty_non_loop_failed"
    uncertainty_pool = high_uncertainty
    uncertainty_rule = "entropy_at_or_above_within_episode_non_loop_failed_q75"
    if not entropy_degenerate and len(high_uncertainty) < 30:
        raise ValueError("fewer than 30 states satisfy the preregistered q75 rule")
    if entropy_degenerate:
        uncertainty_pool = non_loop_failed
        uncertainty_rule = "preregistered_random_fallback_entropy_degenerate"
    selected.extend(
        _balanced_select(
            uncertainty_pool,
            count=PREFIX_STRATUM_COUNTS[uncertainty_stratum],
            stratum=uncertainty_stratum,
            selected_fingerprints=selected_fingerprints,
            episode_counts=episode_counts,
        )
    )
    selected.extend(
        _balanced_select(
            non_loop_failed,
            count=PREFIX_STRATUM_COUNTS["random_non_loop_failed"],
            stratum="random_non_loop_failed",
            selected_fingerprints=selected_fingerprints,
            episode_counts=episode_counts,
        )
    )
    if len(selected) != sum(PREFIX_STRATUM_COUNTS.values()):
        raise ValueError(f"selection produced {len(selected)} states instead of 120")
    for index, state in enumerate(selected):
        state["prefix_id"] = f"P{index:03d}"
        state["sampling_stratum"] = _selected_stratum(state)
        state.pop("source_continue_success", None)
        state.pop("past_loop", None)
    metadata = {
        "selection_seed": SELECTION_SEED,
        "stratum_counts": PREFIX_STRATUM_COUNTS,
        "entropy_grouping_reference_population": "all_eligible_non_loop_failed_states",
        "entropy_quantile_rule": "nearest_rank",
        "entropy_grouping_thresholds": thresholds,
        "uncertainty_reference_population": "eligible_non_loop_failed_states_within_each_episode",
        "entropy_numerically_degenerate": entropy_degenerate,
        "uncertainty_selection_rule": uncertainty_rule,
        "eligible_candidate_count": len(candidates),
        "eligible_successful_candidate_count": len(successful),
        "eligible_loop_failed_candidate_count": len(loop_failed),
        "eligible_non_loop_failed_candidate_count": len(non_loop_failed),
        "excluded_prior_state_count": len(excluded_state_fingerprints),
    }
    return selected, metadata


def validate_prefix_manifest(manifest: dict[str, Any]) -> None:
    """Reject leakage, duplicate states, or stratum/count drift."""
    if manifest.get("schema_version") != "sprint2c_paired_prefix_manifest_v1":
        raise ValueError("unexpected paired-prefix manifest schema")
    prefixes = manifest.get("prefixes")
    if not isinstance(prefixes, list) or len(prefixes) != 120:
        raise ValueError("paired-prefix manifest must contain exactly 120 states")
    counts = Counter(item.get("sampling_stratum") for item in prefixes)
    if dict(counts) != PREFIX_STRATUM_COUNTS:
        raise ValueError(f"paired-prefix strata changed: {dict(counts)}")
    fingerprints = [item.get("state_fingerprint") for item in prefixes]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("paired-prefix manifest contains duplicate states")
    groups = Counter(item.get("episode_group_id") for item in prefixes)
    if max(groups.values()) > 2:
        raise ValueError("more than two prefixes selected from one episode")
    annotation_priority = manifest.get("selection_protocol", {}).get(
        "neutral_annotation_priority"
    )
    expected_ids = {f"P{index:03d}" for index in range(120)}
    if (
        not isinstance(annotation_priority, list)
        or len(annotation_priority) != 120
        or set(annotation_priority) != expected_ids
    ):
        raise ValueError("neutral annotation priority must preregister all prefixes")
    forbidden_feature_fragments = {
        "hidden",
        "future",
        "continue_return",
        "recover_return",
        "qualitative",
    }
    for index, item in enumerate(prefixes):
        if item.get("prefix_id") != f"P{index:03d}":
            raise ValueError(f"prefix identifier mismatch at index {index}")
        if item.get("remaining_horizon", 0) < 8:
            raise ValueError(f"insufficient remaining horizon at index {index}")
        features = item.get("decision_features")
        if not isinstance(features, dict):
            raise ValueError(f"missing decision features at index {index}")
        if set(features) != DECISION_FEATURES:
            raise ValueError(f"decision-feature schema changed at index {index}")
        keys = " ".join(features).casefold()
        if any(fragment in keys for fragment in forbidden_feature_fragments):
            raise ValueError(f"future/hidden leakage in features at index {index}")
        action_id = features.get("h4_proposed_action_id")
        mapping = item.get("valid_actions")
        if (
            not isinstance(action_id, str)
            or len(action_id) != 4
            or action_id[0] != "A"
            or not action_id[1:].isdigit()
            or not isinstance(mapping, list)
            or int(action_id[1:]) >= len(mapping)
            or mapping[int(action_id[1:])]
            != features.get("h4_proposed_mapped_command")
        ):
            raise ValueError(f"H4 action mapping mismatch at index {index}")


def _selected_stratum(state: dict[str, Any]) -> str:
    value = state.get("_selected_stratum")
    if not isinstance(value, str):
        raise ValueError("selected state is missing its stratum")
    state.pop("_selected_stratum")
    return value


def _select_successful_states(
    candidates: list[dict[str, Any]],
    *,
    count: int,
    selected_fingerprints: set[str],
    episode_counts: Counter[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in candidates:
        grouped[state["episode_group_id"]].append(state)
    episode_order = _balanced_episode_order(grouped, "successful_continuation")
    chosen: list[dict[str, Any]] = []
    for pass_index, fraction in enumerate((1 / 3, 2 / 3)):
        for episode_id in episode_order:
            states = sorted(grouped[episode_id], key=lambda item: item["action_count"])
            position = round((len(states) - 1) * fraction)
            state = states[position]
            fingerprint = state["state_fingerprint"]
            if fingerprint in selected_fingerprints or episode_counts[episode_id] >= 2:
                continue
            state["_selected_stratum"] = "successful_continuation"
            state["selection_rule"] = (
                "family_balanced_episode_order_then_one_third_state"
                if pass_index == 0
                else "family_balanced_episode_order_then_two_thirds_state"
            )
            chosen.append(state)
            selected_fingerprints.add(fingerprint)
            episode_counts[episode_id] += 1
            if len(chosen) == count:
                return chosen
    raise ValueError(f"only {len(chosen)} successful-continuation states available")


def _balanced_episode_order(
    grouped: dict[str, list[dict[str, Any]]], stratum: str
) -> list[str]:
    by_family: dict[str, list[str]] = defaultdict(list)
    for episode_id, states in grouped.items():
        by_family[states[0]["task_family"]].append(episode_id)
    for family, episode_ids in by_family.items():
        episode_ids.sort(key=lambda value: _stable_rank(stratum, family, value))
    order = []
    while any(by_family.values()):
        for family in sorted(TASK_FAMILIES):
            if by_family[family]:
                order.append(by_family[family].pop(0))
    return order


def _balanced_select(
    candidates: list[dict[str, Any]],
    *,
    count: int,
    stratum: str,
    selected_fingerprints: set[str],
    episode_counts: Counter[str],
) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for state in candidates:
        cells[(state["task_family"], state["stage_bucket"])].append(state)
    for cell, states in cells.items():
        states.sort(
            key=lambda state: _stable_rank(
                stratum, cell[0], cell[1], state["state_fingerprint"]
            )
        )
    chosen = []
    cell_order = [
        (family, stage)
        for stage in ("early", "middle", "late")
        for family in sorted(TASK_FAMILIES)
    ]
    while len(chosen) < count:
        progress = False
        for cell in cell_order:
            queue = cells[cell]
            while queue:
                state = queue.pop(0)
                fingerprint = state["state_fingerprint"]
                episode_id = state["episode_group_id"]
                if (
                    fingerprint in selected_fingerprints
                    or episode_counts[episode_id] >= 2
                ):
                    continue
                state["_selected_stratum"] = stratum
                state["selection_rule"] = (
                    f"deterministic_family_stage_round_robin_sha256_{SELECTION_SEED}"
                )
                chosen.append(state)
                selected_fingerprints.add(fingerprint)
                episode_counts[episode_id] += 1
                progress = True
                break
            if len(chosen) == count:
                return chosen
        if not progress:
            raise ValueError(f"only {len(chosen)} states available for {stratum}")
    return chosen


def _stage_bucket(action_count: int) -> str:
    if action_count <= 16:
        return "early"
    if action_count <= 33:
        return "middle"
    return "late"


def _nearest_rank_quantile(values: list[float], probability: float) -> float:
    index = max(0, math.ceil(probability * len(values)) - 1)
    return float(values[index])


def _entropy_bucket(value: float, thresholds: dict[str, float]) -> str:
    if value < thresholds["q25"]:
        return "Q1"
    if value < thresholds["q50"]:
        return "Q2"
    if value < thresholds["q75"]:
        return "Q3"
    return "Q4"


def _stable_rank(*parts: str) -> str:
    payload = "\0".join((SELECTION_SEED, *parts)).encode()
    return hashlib.sha256(payload).hexdigest()
