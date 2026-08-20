from __future__ import annotations

from types import SimpleNamespace

import pytest

from env.base import Task
from models.policy import ActionRequest, GenerationOptions, TokenStatistics
from recovery.two_stage import (
    GenerationRecord,
    stage_one_prompt as v1_stage_one_prompt,
    stage_two_prompt as v1_stage_two_prompt,
)
from recovery.two_stage_v2 import (
    RECOVERY_OPERATOR_VERSION,
    STAGE_TWO_PROMPT_VERSION,
    TwoStageRecoveryV2Operator,
    parse_bare_stage_two_output,
    stage_one_prompt as v2_stage_one_prompt,
    stage_two_prompt,
)


def request() -> ActionRequest:
    return ActionRequest(
        task=Task("task-1", "heat the potato", "valid_seen"),
        observation="You are carrying potato 1.",
        history=(("At the counter.", "take potato 1 from countertop 1"),),
        valid_actions=("go to microwave 1", "heat potato 1 with microwave 1"),
    )


def generation(output: str, *, complete: bool = True) -> GenerationRecord:
    return GenerationRecord(
        prompt="prompt",
        raw_output=output,
        token_statistics=TokenStatistics(5, input_tokens=100),
        latency_seconds=0.1,
        output_complete=complete,
        token_cap_reached=not complete,
        max_new_tokens=16,
    )


def test_stage_one_is_exactly_the_unchanged_v1_prompt() -> None:
    assert v2_stage_one_prompt is v1_stage_one_prompt
    assert v2_stage_one_prompt(request()) == v1_stage_one_prompt(request())


def test_v2_stage_two_prompt_requires_only_one_bare_id() -> None:
    prompt = stage_two_prompt(
        request(),
        diagnosis="Potato still needs heating.",
        subgoal="Heat the held potato now.",
        stage_one_valid=True,
    )

    assert "Return exactly one bare action ID" in prompt
    assert "\nAxyz\n" in prompt
    assert "Action-ID: Axyz" not in prompt
    assert "[A001] heat potato 1 with microwave 1" in prompt
    assert "Heat the held potato now." in prompt


def test_v2_stage_two_decision_context_is_unchanged_from_v1() -> None:
    arguments = {
        "diagnosis": "Potato still needs heating.",
        "subgoal": "Heat the held potato now.",
        "stage_one_valid": True,
    }
    v1_prompt = v1_stage_two_prompt(request(), **arguments)
    v2_prompt = stage_two_prompt(request(), **arguments)

    assert v1_prompt.split("\n\n", 1)[1] == v2_prompt.split("\n\n", 1)[1]


def test_bare_id_parser_maps_exact_in_range_id() -> None:
    decision = parse_bare_stage_two_output(
        generation("A001"), request().valid_actions
    )

    assert decision.status == "selected"
    assert decision.action_id == "A001"
    assert decision.action == "heat potato 1 with microwave 1"


@pytest.mark.parametrize(
    "output",
    [
        "Action-ID: A001",
        "A1",
        "A001.",
        "A001\nUse the microwave.",
        " A001",
        "A001 ",
    ],
)
def test_bare_id_parser_rejects_every_non_exact_format(output: str) -> None:
    decision = parse_bare_stage_two_output(generation(output), request().valid_actions)

    assert decision.status == "malformed_id"
    assert decision.action == ""


def test_bare_id_parser_rejects_out_of_range_and_truncated_output() -> None:
    out_of_range = parse_bare_stage_two_output(
        generation("A999"), request().valid_actions
    )
    truncated = parse_bare_stage_two_output(
        generation("A001", complete=False), request().valid_actions
    )

    assert out_of_range.status == "out_of_range_id"
    assert out_of_range.action == ""
    assert truncated.status == "truncated_stage_two"
    assert truncated.action == ""


class FakePolicy:
    model_version = "Qwen/Qwen3-8B"

    def __init__(self) -> None:
        self._config = SimpleNamespace(
            model_id="Qwen/Qwen3-8B",
            action_selection_mode="indexed_admissible",
            history_context_mode="bounded_recent_state",
            history_window=4,
            enable_thinking=False,
            generation=GenerationOptions(max_new_tokens=32, do_sample=False),
        )


def test_v2_operator_keeps_two_calls_and_records_v2_provenance(monkeypatch) -> None:
    operator = TwoStageRecoveryV2Operator(FakePolicy())
    records = iter(
        [
            generation(
                "Diagnosis: Potato still needs heating.\n"
                "Subgoal: Heat the held potato now."
            ),
            generation("A001"),
        ]
    )
    calls: list[int] = []

    def fake_generate(prompt: str, max_new_tokens: int) -> GenerationRecord:
        calls.append(max_new_tokens)
        return next(records)

    monkeypatch.setattr(operator, "_generate", fake_generate)
    decision = operator.act(request())
    adapted = decision.as_action_decision("Qwen/Qwen3-8B")

    assert calls == [40, 16]
    assert decision.status == "selected"
    assert decision.action == "heat potato 1 with microwave 1"
    assert adapted.metadata["recovery_operator_version"] == RECOVERY_OPERATOR_VERSION
    assert adapted.metadata["prompt_version"] == STAGE_TWO_PROMPT_VERSION
