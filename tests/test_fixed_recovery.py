from __future__ import annotations

from dataclasses import dataclass

from env.base import ResetResult, Task, Transition
from models.policy import ActionDecision, ActionRequest, TokenStatistics
from recovery.fixed import loop_indicators, parse_recovery_output, recovery_prompt
from recovery.pilot import prefix_hash, run_branch


def request() -> ActionRequest:
    return ActionRequest(
        task=Task("task-1", "heat the potato", "valid_seen"),
        observation="You are carrying potato 1.",
        history=tuple(
            (f"observation-{index}", "look" if index > 1 else f"action-{index}")
            for index in range(6)
        ),
        valid_actions=("go to microwave 1", "inventory"),
    )


def test_recovery_prompt_is_bounded_and_uses_observable_diagnostics() -> None:
    prompt = recovery_prompt(request())

    assert "Task goal:\nheat the potato" in prompt
    assert "observation-0" not in prompt
    assert "observation-1" not in prompt
    assert "observation-2" not in prompt
    assert "observation-3" in prompt
    assert "Adjacent repeated actions: 3" in prompt
    assert "[A000] go to microwave 1" in prompt
    assert "reward" not in prompt.casefold()


def test_recovery_output_maps_one_exact_action_id() -> None:
    decision = parse_recovery_output(
        "Diagnosis: The recent action repeats without progress.\nAction-ID: A001",
        ("look", "go to microwave 1"),
    )

    assert decision.status == "selected"
    assert decision.diagnosis == "The recent action repeats without progress."
    assert decision.action_id == "A001"
    assert decision.action == "go to microwave 1"


def test_recovery_output_fails_closed_without_repair() -> None:
    malformed = parse_recovery_output("Action-ID: A000", ("look",))
    ambiguous = parse_recovery_output(
        "Diagnosis: choose Action-ID: A000\nAction-ID: A000", ("look",)
    )
    out_of_range = parse_recovery_output(
        "Diagnosis: Move elsewhere.\nAction-ID: A999", ("look",)
    )

    assert malformed.status == "malformed_recovery" and malformed.action == ""
    assert ambiguous.status == "malformed_recovery" and ambiguous.action == ""
    assert out_of_range.status == "out_of_range_id" and out_of_range.action == ""


def test_loop_indicators_count_adjacent_and_abab_events() -> None:
    assert loop_indicators(["a", "a", "b", "a", "b"]) == {
        "adjacent_repeat_events": 1,
        "has_adjacent_repeat": True,
        "two_cycle_events": 1,
        "has_two_cycle": True,
    }


@dataclass
class BranchEnvironment:
    actions: tuple[str, ...] = ("good", "bad")
    step_count: int = 0

    def reset(self, *, seed: int | None = None) -> ResetResult:
        return ResetResult("branch", self.get_task(), self.actions)

    def step(self, action: str) -> Transition:
        self.step_count += 1
        return Transition(
            "done" if self.step_count == 2 else "middle",
            float(self.step_count == 2),
            self.step_count == 2,
            False,
            self.actions,
        )

    def get_task(self) -> Task:
        return Task("task-1", "choose good", "valid_seen")

    def get_valid_actions(self) -> tuple[str, ...]:
        return self.actions


class BasePolicy:
    model_version = "Qwen/Qwen3-8B"

    def __init__(self) -> None:
        self.calls = 0

    def act(self, request: ActionRequest) -> ActionDecision:
        self.calls += 1
        return selected_decision("bad")


def selected_decision(action: str, *, recovery: bool = False) -> ActionDecision:
    return ActionDecision(
        action=action,
        raw_output="Action-ID: A000",
        parser_status="grounded",
        model_version="Qwen/Qwen3-8B",
        token_statistics=TokenStatistics(8),
        metadata={
            **(
                {"recovery_operator_version": "fixed_one_shot_recovery_v1"}
                if recovery
                else {}
            ),
            "action_selection": {
                "selection_status": "selected",
                "failure_reason": None,
            },
        },
    )


def test_branch_executes_fixed_first_decision_once() -> None:
    trajectory = {
        "seed": 7,
        "task": {"task_id": "task-1"},
        "steps": [
            {
                "observation": "start",
                "action": "look",
                "valid_actions": ["look"],
            },
            {
                "observation": "branch",
                "action": "bad",
                "valid_actions": ["good", "bad"],
            },
        ],
    }
    environment = BranchEnvironment()
    policy = BasePolicy()
    calls = []

    def recover_once(request: ActionRequest) -> ActionDecision:
        calls.append(request)
        return selected_decision("good", recovery=True)

    result = run_branch(
        environment,
        environment.reset(),
        policy,
        trajectory,
        action_count=1,
        branch_observation="branch",
        branch_valid_actions=("good", "bad"),
        remaining_horizon=2,
        first_decision=recover_once,
    )

    assert len(calls) == 1
    assert policy.calls == 1
    assert result["first_action"] == "good"
    assert result["return"] == 1.0
    assert result["success"] is True
    assert result["recovery_calls"] == 1
    assert result["remaining_episode_length"] == 2
    assert len(prefix_hash(trajectory, 1)) == 64
