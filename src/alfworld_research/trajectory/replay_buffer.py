"""Append-only JSONL persistence for trajectory datasets."""

from __future__ import annotations

import json
from pathlib import Path

from .trajectory import Trajectory


class JsonlTrajectoryStore:
    """Store trajectories in an append-only, experiment-friendly JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, trajectory: Trajectory) -> None:
        """Append one serialized trajectory, creating its parent directory."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trajectory.to_dict(), sort_keys=True) + "\n")
