from __future__ import annotations

from dataclasses import dataclass

import pytest

from env.base import ResetResult, Task, Transition
from env.replay import replay_prefix, repeated_replay_equal


@dataclass
class FakeEnvironment:
    observation: str = "start"
    actions: tuple[str, ...] = ("open door", "wait")

    def reset(self, *, seed: int | None = None) -> ResetResult:
        self.observation = "start"
        self.actions = ("open door", "wait")
        return self.result()

    def step(self, action: str) -> Transition:
        assert action in self.actions
        if action == "open door":
            self.observation = "door open"
            self.actions = ("enter room", "wait")
        else:
            self.observation = "waited"
        return Transition(self.observation, 0.0, False, False, self.actions)

    def get_task(self) -> Task:
        return self.result().task

    def get_valid_actions(self) -> tuple[str, ...]:
        return self.actions

    def result(self) -> ResetResult:
        return ResetResult(
            self.observation,
            Task("task-1", "enter the room", "valid_seen"),
            self.actions,
        )


def trajectory() -> dict:
    return {
        "seed": 7,
        "task": {"task_id": "task-1"},
        "steps": [
            {
                "index": 0,
                "observation": "start",
                "valid_actions": ["open door", "wait"],
                "action": "open door",
                "reward": 0.0,
                "done": False,
                "metadata": {"transition_truncated": False},
            },
            {
                "index": 1,
                "observation": "door open",
                "valid_actions": ["enter room", "wait"],
                "action": "enter room",
                "reward": 1.0,
                "done": True,
                "metadata": {"transition_truncated": False},
            },
        ],
    }


def test_replay_prefix_exactly_reconstructs_public_state() -> None:
    environment = FakeEnvironment()
    result = replay_prefix(environment, environment.reset(seed=7), trajectory(), 1)

    assert result["exact_public_reconstruction"] is True
    assert result["target"]["observation"] == "door open"
    assert result["target"]["admissible_actions"] == ["enter room", "wait"]


def test_replay_prefix_reports_ordered_action_difference() -> None:
    recorded = trajectory()
    recorded["steps"][1]["valid_actions"] = ["wait", "enter room"]
    environment = FakeEnvironment()

    result = replay_prefix(environment, environment.reset(), recorded, 1)

    assert result["target"]["admissible_order_equal"] is False
    assert result["target"]["admissible_set_equal"] is True
    assert result["exact_public_reconstruction"] is False


def test_repeated_replay_compares_target_state() -> None:
    environment = FakeEnvironment()
    first = replay_prefix(environment, environment.reset(), trajectory(), 1)
    second = replay_prefix(environment, environment.reset(), trajectory(), 1)

    comparison = repeated_replay_equal(first, second)
    assert all(value for value in comparison.values() if value is not None)
    assert comparison["hidden_state_equal"] is None


@pytest.mark.parametrize("prefix", [-1, 2])
def test_replay_prefix_rejects_invalid_target(prefix: int) -> None:
    environment = FakeEnvironment()
    with pytest.raises(ValueError, match="nonterminal recorded state"):
        replay_prefix(environment, environment.reset(), trajectory(), prefix)
