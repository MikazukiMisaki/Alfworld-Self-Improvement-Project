"""ALFWorld text-environment adapter.

The adapter imports ALFWorld only when it is instantiated, keeping schema and
analysis utilities usable on machines without the simulator installed.
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import ResetResult, Task, Transition

_TASK_PATTERN = re.compile(r"Your task is to:\s*(.+?)(?:\n|$)", re.IGNORECASE)


@dataclass(frozen=True)
class AlfWorldConfig:
    """Configuration required to create one ALFWorld text adapter."""

    config_path: Path
    split: str = "valid_seen"
    batch_size: int = 1
    data_path: Path | None = None


class AlfWorldTextEnvironment:
    """Normalize AlfredTWEnv to the project environment contract."""

    def __init__(self, config: AlfWorldConfig) -> None:
        if config.batch_size != 1:
            raise ValueError("the baseline adapter supports batch_size=1 only")
        self._config = config
        self._environment: Any | None = None
        self._task: Task | None = None
        self._valid_actions: tuple[str, ...] | None = None

    def reset(self, *, seed: int | None = None) -> ResetResult:
        """Reset ALFWorld and normalize its first observation and task metadata."""
        self._seed(seed)
        environment = self._get_environment()
        observations, info = environment.reset()
        observation = self._first(observations)
        info_dict = self._info_dict(info)
        self._valid_actions = self._valid_actions_from_info(info_dict)
        self._task = self._task_from(observation, info_dict)
        return ResetResult(
            observation=observation,
            task=self._task,
            valid_actions=self._valid_actions,
            metadata={"alfworld_info": info_dict},
        )

    def step(self, action: str) -> Transition:
        """Apply one text action to ALFWorld."""
        if self._environment is None:
            raise RuntimeError("reset() must be called before step()")
        observations, rewards, dones, info = self._environment.step([action])
        info_dict = self._info_dict(info)
        self._valid_actions = self._valid_actions_from_info(info_dict)
        return Transition(
            observation=self._first(observations),
            reward=float(self._first(rewards)),
            done=bool(self._first(dones)),
            truncated=False,
            valid_actions=self._valid_actions,
            metadata={"alfworld_info": info_dict},
        )

    def get_task(self) -> Task:
        """Return the active task after reset."""
        if self._task is None:
            raise RuntimeError("reset() must be called before get_task()")
        return self._task

    def get_valid_actions(self) -> tuple[str, ...] | None:
        """Return commands from the most recent ALFWorld info object."""
        return self._valid_actions

    def _get_environment(self) -> Any:
        if self._environment is not None:
            return self._environment
        try:
            import yaml
            from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
        except ImportError as error:
            raise RuntimeError(
                "ALFWorld collection requires PyYAML and the alfworld package. "
                "Install the ALFWorld runtime before running the script."
            ) from error
        if self._config.data_path is not None:
            os.environ["ALFWORLD_DATA"] = str(self._config.data_path)
        with self._config.config_path.open("r", encoding="utf-8") as handle:
            configuration = yaml.safe_load(handle)
        self._environment = AlfredTWEnv(configuration, train_eval=self._alfworld_split())
        self._environment = self._environment.init_env(batch_size=self._config.batch_size)
        return self._environment

    @staticmethod
    def _first(value: Any) -> Any:
        return value[0] if isinstance(value, (list, tuple)) else value

    @staticmethod
    def _info_dict(info: Any) -> dict[str, Any]:
        return info if isinstance(info, dict) else {"raw_info": repr(info)}

    def _alfworld_split(self) -> str:
        """Translate the public baseline split name to ALFWorld's API value."""
        splits = {
            "train": "train",
            "valid_seen": "eval_in_distribution",
            "valid_unseen": "eval_out_of_distribution",
        }
        try:
            return splits[self._config.split]
        except KeyError as error:
            raise ValueError(
                "split must be one of train, valid_seen, or valid_unseen"
            ) from error

    @classmethod
    def _valid_actions_from_info(cls, info: dict[str, Any]) -> tuple[str, ...] | None:
        actions = info.get("admissible_commands")
        if actions is None:
            return None
        first = cls._first(actions)
        if not isinstance(first, (list, tuple)):
            return None
        return tuple(str(action) for action in first)

    def _task_from(self, observation: str, info: dict[str, Any]) -> Task:
        match = _TASK_PATTERN.search(observation)
        task_text = match.group(1).strip().rstrip(".") if match else observation.strip()
        extra = info.get("extra_game_info", {})
        game_file = extra.get("game_file") if isinstance(extra, dict) else None
        task_id = Path(str(game_file)).stem if game_file else f"{self._config.split}:{task_text}"
        return Task(
            task_id=task_id,
            text=task_text,
            split=self._config.split,
            metadata={"game_file": game_file} if game_file else {},
        )

    @staticmethod
    def _seed(seed: int | None) -> None:
        if seed is None:
            return
        random.seed(seed)
        try:
            import numpy

            numpy.random.seed(seed)
        except ImportError:
            pass
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
