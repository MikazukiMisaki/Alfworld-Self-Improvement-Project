"""Policy contracts and generation records for interactive language agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from env.base import Task


@dataclass(frozen=True)
class GenerationOptions:
    """Decoding settings recorded with every baseline decision."""

    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None


@dataclass(frozen=True)
class TokenStatistics:
    """Optional token-level statistics returned by a model backend."""

    generated_tokens: int
    mean_token_log_probability: float | None = None
    mean_token_entropy: float | None = None
    input_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class ActionRequest:
    """All policy-visible information for one action decision."""

    task: Task
    observation: str
    history: tuple[tuple[str, str], ...]
    valid_actions: tuple[str, ...] | None


@dataclass(frozen=True)
class ActionDecision:
    """Parsed action plus raw generation and optional model diagnostics."""

    action: str
    raw_output: str
    parser_status: str
    model_version: str
    token_statistics: TokenStatistics | None = None
    reasoning: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ActionPolicy(Protocol):
    """Select one action without interacting with the environment or filesystem."""

    @property
    def model_version(self) -> str:
        """Return the immutable base model or checkpoint identifier."""

    def act(self, request: ActionRequest) -> ActionDecision:
        """Return a parsed action and its generation record."""
