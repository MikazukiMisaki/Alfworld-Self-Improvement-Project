from __future__ import annotations

from types import SimpleNamespace

from env.base import Task
from models.policy import ActionRequest, GenerationOptions, TokenStatistics
from recovery.two_stage import (
    GenerationRecord,
    TwoStageRecoveryOperator,
    parse_stage_one_output,
    parse_stage_two_output,
    stage_one_prompt,
    stage_two_prompt,
)


def request() -> ActionRequest:
    return ActionRequest(
        task=Task("task-1", "heat the potato and put it on the counter", "valid_seen"),
        observation="You are carrying potato 1.",
        history=tuple(
            (f"observation-{index}", "look" if index > 1 else f"action-{index}")
            for index in range(6)
        ),
        valid_actions=("go to microwave 1", "heat potato 1 with microwave 1"),
    )


def generation(
    output: str,
    *,
    complete: bool = True,
    max_new_tokens: int = 40,
) -> GenerationRecord:
    return GenerationRecord(
        prompt="prompt",
        raw_output=output,
        token_statistics=TokenStatistics(8, input_tokens=100),
        latency_seconds=0.1,
        output_complete=complete,
        token_cap_reached=not complete,
        max_new_tokens=max_new_tokens,
    )


def test_stage_one_prompt_is_bounded_and_has_no_indexed_mapping() -> None:
    prompt = stage_one_prompt(request())

    assert "Task goal:\nheat the potato" in prompt
    assert "observation-0" not in prompt
    assert "observation-1" not in prompt
    assert "observation-2" not in prompt
    assert "observation-3" in prompt
    assert "Adjacent repeated actions: 3" in prompt
    assert "- go to microwave 1" in prompt
    assert "[A000]" not in prompt
    assert "Do not output an Action-ID" in prompt


def test_stage_one_parser_accepts_only_two_short_fields() -> None:
    valid = parse_stage_one_output(
        generation(
            "Diagnosis: Potato still needs heating.\n"
            "Subgoal: Heat the held potato now."
        )
    )
    malformed = parse_stage_one_output(
        generation(
            "Diagnosis: Potato is cold.\nSubgoal: Use A001 to heat it."
        )
    )
    too_long = parse_stage_one_output(
        generation(
            "Diagnosis: one two three four five six seven eight nine ten eleven twelve "
            "thirteen\nSubgoal: Heat potato."
        )
    )
    truncated = parse_stage_one_output(
        generation(
            "Diagnosis: Potato is cold.\nSubgoal: Heat potato.", complete=False
        )
    )

    assert valid.status == "valid"
    assert valid.diagnosis_word_count == 4
    assert valid.subgoal_word_count == 5
    assert malformed.status == "malformed_stage_one"
    assert too_long.status == "diagnosis_too_long"
    assert truncated.status == "truncated_stage_one"


def test_stage_two_prompt_freezes_subgoal_and_exact_mapping() -> None:
    prompt = stage_two_prompt(
        request(),
        diagnosis="Potato still needs heating.",
        subgoal="Heat the held potato now.",
        stage_one_valid=True,
    )

    assert "Frozen Stage-1 diagnosis:\nPotato still needs heating." in prompt
    assert "Frozen Stage-1 immediate subgoal:\nHeat the held potato now." in prompt
    assert "[A000] go to microwave 1" in prompt
    assert "[A001] heat potato 1 with microwave 1" in prompt
    assert "Return exactly one line" in prompt


def test_stage_two_parser_maps_exact_command_and_fails_closed() -> None:
    actions = ("go to microwave 1", "heat potato 1 with microwave 1")
    selected = parse_stage_two_output(
        generation("Action-ID: A001", max_new_tokens=16), actions
    )
    malformed = parse_stage_two_output(
        generation("A001", max_new_tokens=16), actions
    )
    out_of_range = parse_stage_two_output(
        generation("Action-ID: A999", max_new_tokens=16), actions
    )
    truncated = parse_stage_two_output(
        generation("Action-ID: A001", complete=False, max_new_tokens=16), actions
    )

    assert selected.status == "selected"
    assert selected.action_id == "A001"
    assert selected.action == "heat potato 1 with microwave 1"
    assert malformed.status == "malformed_id" and malformed.action == ""
    assert out_of_range.status == "out_of_range_id" and out_of_range.action == ""
    assert truncated.status == "truncated_stage_two" and truncated.action == ""


class FakePolicy:
    def __init__(self) -> None:
        self.model_version = "Qwen/Qwen3-8B"
        self._config = SimpleNamespace(
            model_id="Qwen/Qwen3-8B",
            action_selection_mode="indexed_admissible",
            history_context_mode="bounded_recent_state",
            history_window=4,
            enable_thinking=False,
            generation=GenerationOptions(max_new_tokens=32, do_sample=False),
        )


def test_operator_makes_exactly_two_calls_and_uses_stage_two_action(monkeypatch) -> None:
    operator = TwoStageRecoveryOperator(FakePolicy())
    records = iter(
        [
            generation(
                "Diagnosis: Potato still needs heating.\n"
                "Subgoal: Heat the held potato now."
            ),
            generation("Action-ID: A001", max_new_tokens=16),
        ]
    )
    calls: list[tuple[str, int]] = []

    def fake_generate(prompt: str, max_new_tokens: int) -> GenerationRecord:
        calls.append((prompt, max_new_tokens))
        return next(records)

    monkeypatch.setattr(operator, "_generate", fake_generate)
    decision = operator.act(request())

    assert [budget for _, budget in calls] == [40, 16]
    assert decision.model_call_count == 2
    assert decision.status == "selected"
    assert decision.action == "heat potato 1 with microwave 1"
    action_decision = decision.as_action_decision("Qwen/Qwen3-8B")
    assert action_decision.metadata["recovery_model_call_count"] == 2
    assert action_decision.metadata["action_selection"]["selected_index"] == 1


def test_operator_calls_stage_two_but_fails_closed_after_bad_stage_one(
    monkeypatch,
) -> None:
    operator = TwoStageRecoveryOperator(FakePolicy())
    records = iter(
        [
            generation("Diagnosis: Potato is cold."),
            generation("Action-ID: A001", max_new_tokens=16),
        ]
    )
    calls = []

    def fake_generate(prompt: str, max_new_tokens: int) -> GenerationRecord:
        calls.append((prompt, max_new_tokens))
        return next(records)

    monkeypatch.setattr(operator, "_generate", fake_generate)
    decision = operator.act(request())

    assert len(calls) == 2
    assert "(invalid Stage-1 output)" in calls[1][0]
    assert decision.status == "stage_one_failure"
    assert decision.action == ""
