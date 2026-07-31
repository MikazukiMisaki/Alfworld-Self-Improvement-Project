"""Run a dependency-free end-to-end smoke experiment."""

from __future__ import annotations

from alfworld_research.env.base import ResetResult, Task, Transition
from alfworld_research.evaluation.evaluator import evaluate
from alfworld_research.models.policy import ActionDecision, ActionRequest


class ToyEnvironment:
    """One-step environment illustrating the adapter contract."""

    def reset(self, *, seed: int | None = None) -> ResetResult:
        self.task = Task(f"toy-{seed}", "take the key", "toy", "toy")
        return ResetResult("A key is on the table.", self.task, ("take key",), {})

    def step(self, action: str) -> Transition:
        success = action == "take key"
        return Transition("Episode complete.", float(success), True, False, ("take key",), {"success": success})

    def get_task(self) -> Task:
        return self.task

    def get_valid_actions(self) -> tuple[str, ...]:
        return ("take key",)


class ToyPolicy:
    """Policy baseline that solves the toy task."""

    model_version = "toy-policy"

    def act(self, request: ActionRequest) -> ActionDecision:
        return ActionDecision("take key", "take key", "grounded", self.model_version)


def main() -> None:
    """Print metrics from an end-to-end collection and evaluation path."""
    report = evaluate(ToyEnvironment, ToyPolicy(), seeds=(0, 1, 2), max_steps=1)
    metrics = report.metrics
    print(f"episodes={metrics.episodes} success_rate={metrics.success_rate:.2f} mean_reward={metrics.mean_reward:.2f}")


if __name__ == "__main__":
    main()
