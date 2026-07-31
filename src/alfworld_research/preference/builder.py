"""Convert ranked trajectories into backend-neutral preference examples."""

from __future__ import annotations

from dataclasses import dataclass

from alfworld_research.trajectory.trajectory import Trajectory


@dataclass(frozen=True)
class PreferenceExample:
    """A prompt and preferred/dispreferred continuations for DPO-like trainers."""

    prompt: str
    chosen: str
    rejected: str


def _continuation(trajectory: Trajectory) -> str:
    return "\n".join(step.action for step in trajectory.steps)


def build_pair(first: Trajectory, second: Trajectory) -> PreferenceExample:
    """Rank two trajectories by reward and form a deterministic preference pair."""
    if first.initial_observation != second.initial_observation:
        raise ValueError("preference pairs require matching initial observations")
    chosen, rejected = (first, second) if first.total_reward >= second.total_reward else (second, first)
    return PreferenceExample(chosen.initial_observation, _continuation(chosen), _continuation(rejected))
