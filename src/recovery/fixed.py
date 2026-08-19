"""One-shot, fail-closed recovery using the frozen baseline model."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from models.action_parser import action_id_mapping, parse_action_id
from models.policy import ActionDecision, ActionRequest, TokenStatistics
from models.qwen import QwenPolicy


RECOVERY_OPERATOR_VERSION = "fixed_one_shot_recovery_v2"
RECOVERY_PROMPT_VERSION = "fixed-recovery-diagnosis-action-id-v2"
RECOVERY_INSTRUCTIONS = (
    "You are making one bounded recovery intervention for an ALFWorld household "
    "agent.\n"
    "1. Give one immediate diagnosis only.\n"
    "2. The diagnosis must contain at most 12 words.\n"
    "3. The selected action must directly address the diagnosis.\n"
    "4. Choose exactly one currently available Action-ID.\n"
    "5. Output exactly two lines.\n"
    "Return exactly two lines in this format:\n"
    "Diagnosis: <maximum 12 words>\n"
    "Action-ID: Axyz\n"
    "Do not output reasoning, commands, extra lines, or <think> tags."
)
_RECOVERY_OUTPUT = re.compile(
    r"\ADiagnosis: ([^\r\n]+)\r?\nAction-ID: (A\d{3})\Z"
)
_ACTION_ID_ANYWHERE = re.compile(r"Action-ID:\s*(A\d{3})")


@dataclass(frozen=True)
class RecoveryDecision:
    """Auditable result of exactly one fixed recovery generation."""

    diagnosis: str | None
    action_id: str | None
    action: str
    status: str
    failure_reason: str | None
    raw_output: str
    prompt: str
    token_statistics: TokenStatistics
    latency_seconds: float
    id_to_command: dict[str, str]
    diagnosis_word_count: int | None
    diagnosis_length_valid: bool
    output_complete: bool
    token_cap_reached: bool

    def as_action_decision(self, model_version: str) -> ActionDecision:
        """Adapt the recovery selection to the standard environment decision."""
        selected_index = int(self.action_id[1:]) if self.action_id else None
        return ActionDecision(
            action=self.action,
            raw_output=self.raw_output,
            parser_status="grounded" if self.status == "selected" else self.status,
            model_version=model_version,
            token_statistics=self.token_statistics,
            reasoning=None,
            metadata={
                "prompt": self.prompt,
                "prompt_version": RECOVERY_PROMPT_VERSION,
                "recovery_operator_version": RECOVERY_OPERATOR_VERSION,
                "recovery_diagnosis": self.diagnosis,
                "recovery_output_validation": {
                    "diagnosis_word_count": self.diagnosis_word_count,
                    "diagnosis_length_valid": self.diagnosis_length_valid,
                    "output_complete": self.output_complete,
                    "token_cap_reached": self.token_cap_reached,
                    "diagnosis_action_consistency": {
                        "annotation": None,
                        "used_as_execution_rule": False,
                    },
                },
                "action_selection_mode": "indexed_admissible",
                "action_selection": {
                    "action_selection_mode": "indexed_admissible",
                    "raw_model_output": self.raw_output,
                    "parsed_action_id": self.action_id,
                    "selected_index": selected_index,
                    "mapped_environment_command": self.action or None,
                    "id_to_command": self.id_to_command,
                    "selection_status": self.status,
                    "failure_reason": self.failure_reason,
                },
            },
        )


class FixedRecoveryOperator:
    """Invoke the frozen Qwen policy backend once with a recovery prompt."""

    def __init__(self, baseline_policy: QwenPolicy) -> None:
        self._policy = baseline_policy
        config = baseline_policy._config
        if (
            config.model_id != "Qwen/Qwen3-8B"
            or config.action_selection_mode != "indexed_admissible"
            or config.history_context_mode != "bounded_recent_state"
            or config.history_window != 4
            or config.enable_thinking
            or config.generation.do_sample
            or config.generation.max_new_tokens != 32
        ):
            raise ValueError("recovery requires the frozen Sprint 1.5 H4 policy")

    def act(self, request: ActionRequest) -> RecoveryDecision:
        """Generate one diagnosis/action pair without retry or repair."""
        tokenizer, model, torch = self._policy._load()
        prompt = recovery_prompt(request)
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            rendered = prompt
        inputs = tokenizer(rendered, return_tensors="pt")
        inputs = {name: value.to(model.device) for name, value in inputs.items()}
        started = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        latency = time.perf_counter() - started
        input_tokens = inputs["input_ids"].shape[1]
        generated_tokens = generated.sequences[0][input_tokens:]
        eos_token_ids = tokenizer.eos_token_id
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        eos_token_ids = set(eos_token_ids or [])
        ended_with_eos = bool(
            len(generated_tokens)
            and int(generated_tokens[-1].item()) in eos_token_ids
        )
        token_cap_reached = len(generated.scores or ()) >= 32
        output_complete = not token_cap_reached or ended_with_eos
        raw_output = tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()
        return parse_recovery_output(
            raw_output,
            request.valid_actions,
            prompt=prompt,
            token_statistics=self._policy._token_statistics(
                generated, torch, input_tokens=input_tokens
            ),
            latency_seconds=latency,
            output_complete=output_complete,
            token_cap_reached=token_cap_reached,
        )


def recovery_prompt(request: ActionRequest) -> str:
    """Render only bounded state and diagnostics observable at decision time."""
    mapping = action_id_mapping(request.valid_actions or ())
    actions = "\n".join(
        f"[{action_id}] {command}" for action_id, command in mapping.items()
    ) or "No admissible action list is available."
    history_window = request.history[-4:]
    transitions = []
    start = len(request.history) - len(history_window)
    for offset, (observation, action) in enumerate(history_window):
        absolute_index = start + offset
        result = (
            request.history[absolute_index + 1][0]
            if absolute_index + 1 < len(request.history)
            else request.observation
        )
        transitions.append(f"Action: {action}\nResult: {result}")
    recent_actions = [action for _, action in history_window]
    diagnostics = loop_indicators(recent_actions)
    inventory = (
        request.observation
        if request.history and request.history[-1][1].strip() == "inventory"
        else "(not available from the environment)"
    )
    transition_text = "\n\n".join(transitions) if transitions else "(none)"
    return (
        f"{RECOVERY_INSTRUCTIONS}\n\n"
        f"Task goal:\n{request.task.text}\n\n"
        f"Current observation:\n{request.observation}\n\n"
        f"Current inventory:\n{inventory}\n\n"
        "Recent action/result transitions (last 4):\n"
        f"{transition_text}\n\n"
        "Observable recent diagnostics:\n"
        f"Adjacent repeated actions: {diagnostics['adjacent_repeat_events']}\n"
        f"Two-action cycle events: {diagnostics['two_cycle_events']}\n\n"
        f"Indexed admissible actions:\n{actions}\n\n"
        "Your response:"
    )


def parse_recovery_output(
    output: str,
    valid_actions: tuple[str, ...] | None,
    *,
    prompt: str = "",
    token_statistics: TokenStatistics | None = None,
    latency_seconds: float = 0.0,
    output_complete: bool = True,
    token_cap_reached: bool = False,
) -> RecoveryDecision:
    """Validate the complete two-line contract and map one ID fail-closed."""
    mapping = action_id_mapping(valid_actions or ())
    stripped = output.strip()
    match = _RECOVERY_OUTPUT.fullmatch(stripped)
    embedded_ids = _ACTION_ID_ANYWHERE.findall(output)
    diagnosis = _diagnosis_candidate(stripped)
    word_count = len(diagnosis.split()) if diagnosis else None
    length_valid = word_count is not None and word_count <= 12
    common = {
        "raw_output": output,
        "prompt": prompt,
        "token_statistics": token_statistics or TokenStatistics(0),
        "latency_seconds": latency_seconds,
        "id_to_command": mapping,
        "diagnosis_word_count": word_count,
        "diagnosis_length_valid": length_valid,
        "output_complete": output_complete,
        "token_cap_reached": token_cap_reached,
    }
    if not output_complete:
        return RecoveryDecision(
            diagnosis,
            None,
            "",
            "truncated_recovery",
            "generation did not complete before the token limit",
            **common,
        )
    if match is None or len(embedded_ids) != 1:
        return RecoveryDecision(
            diagnosis,
            None,
            "",
            "malformed_recovery",
            "expected exactly one diagnosis line and one action-ID line",
            **common,
        )
    diagnosis, action_id = match.groups()
    if not length_valid:
        return RecoveryDecision(
            diagnosis.strip(),
            action_id,
            "",
            "diagnosis_too_long",
            "diagnosis exceeds 12 words",
            **common,
        )
    parsed = parse_action_id(f"Action-ID: {action_id}", valid_actions)
    return RecoveryDecision(
        diagnosis.strip(),
        parsed.action_id,
        parsed.action,
        parsed.status,
        parsed.invalid_reason,
        **common,
    )


def _diagnosis_candidate(output: str) -> str | None:
    lines = output.splitlines()
    if not lines or not lines[0].startswith("Diagnosis: "):
        return None
    candidate = lines[0].removeprefix("Diagnosis: ").strip()
    return candidate or None


def loop_indicators(actions: list[str]) -> dict[str, int | bool]:
    """Count observable adjacent repeats and strict A-B-A-B cycles."""
    adjacent = sum(left == right for left, right in zip(actions, actions[1:]))
    two_cycle = sum(
        actions[index] == actions[index - 2]
        and actions[index - 1] == actions[index - 3]
        for index in range(3, len(actions))
    )
    return {
        "adjacent_repeat_events": adjacent,
        "has_adjacent_repeat": adjacent > 0,
        "two_cycle_events": two_cycle,
        "has_two_cycle": two_cycle > 0,
    }
