"""Parsing and grounding helpers for textual environment actions."""

from __future__ import annotations

import re
from dataclasses import dataclass

_ACTION_PREFIX = re.compile(r"^\s*(?:action|command)\s*:\s*", re.IGNORECASE)
_THOUGHT_PREFIX = re.compile(r"^\s*(?:thought|reasoning)\s*:", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedAction:
    """Action-parser result retained in trajectory metadata."""

    action: str
    status: str
    reasoning: str | None = None


def parse_action(output: str, valid_actions: tuple[str, ...] | None = None) -> ParsedAction:
    """Extract one action and prefer an exact currently valid command."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    reasoning = next(
        (_THOUGHT_PREFIX.sub("", line).strip() for line in lines if _THOUGHT_PREFIX.match(line)),
        None,
    )
    candidates: list[str] = []
    for line in lines:
        if _THOUGHT_PREFIX.match(line):
            continue
        candidates.append(_ACTION_PREFIX.sub("", line).strip("'\" "))
    if valid_actions:
        normalized = {action.casefold().strip(): action for action in valid_actions}
        for candidate in candidates:
            if candidate.casefold() in normalized:
                return ParsedAction(normalized[candidate.casefold()], "grounded", reasoning)
        for action in valid_actions:
            if action.casefold() in output.casefold():
                return ParsedAction(action, "grounded_embedded", reasoning)
    if candidates:
        return ParsedAction(candidates[0], "ungrounded", reasoning)
    return ParsedAction("", "empty", reasoning)
