"""Run a dependency-free end-to-end smoke experiment."""

from __future__ import annotations

from typing import Any

from alfworld_research.evaluation.evaluator import evaluate


class ToyEnvironment:
    """One-step environment illustrating the adapter contract."""

    def reset(self, *, seed: int | None = None) -> tuple[str, dict[str, Any]]:
        return "A key is on the table.", {"task_id": f"toy-{seed}"}

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        success = action == "take key"
        return "Episode complete.", float(success), True, {"success": success}


class ToyPolicy:
    """Policy baseline that solves the toy task."""

    def act(self, observation: str, history: tuple[str, ...]) -> str:
        return "take key"


def main() -> None:
    """Print metrics from an end-to-end collection and evaluation path."""
    metrics = evaluate(ToyEnvironment, ToyPolicy(), episodes=3)
    print(f"episodes={metrics.episodes} success_rate={metrics.success_rate:.2f} mean_reward={metrics.mean_reward:.2f}")


if __name__ == "__main__":
    main()
