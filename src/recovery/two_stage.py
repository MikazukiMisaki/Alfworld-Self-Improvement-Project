"""Two-stage, fail-closed recovery using the frozen H4 model backend."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from models.action_parser import action_id_mapping, parse_action_id
from models.policy import ActionDecision, ActionRequest, TokenStatistics
from models.qwen import QwenPolicy

from .fixed import loop_indicators


RECOVERY_OPERATOR_VERSION = "two_stage_bounded_recovery_v1"
STAGE_ONE_PROMPT_VERSION = "two-stage-diagnosis-subgoal-v1"
STAGE_TWO_PROMPT_VERSION = "two-stage-subgoal-action-id-v1"
STAGE_ONE_MAX_NEW_TOKENS = 40
STAGE_TWO_MAX_NEW_TOKENS = 16
STAGE_ONE_INSTRUCTIONS = (
    "You are diagnosing one immediate ALFWorld household-agent failure.\n"
    "Return exactly two lines in this format:\n"
    "Diagnosis: <single immediate problem, maximum 12 words>\n"
    "Subgoal: <single immediately executable objective, maximum 12 words>\n"
    "The subgoal must describe what should happen next, not a long plan.\n"
    "Do not output an Action-ID, command, explanation, extra line, or <think> tag."
)
STAGE_TWO_INSTRUCTIONS = (
    "Select the one currently available action that best executes the frozen "
    "immediate subgoal.\n"
    "Return exactly one line in this format:\n"
    "Action-ID: Axyz\n"
    "Choose one ID exactly as written in the indexed admissible-action mapping.\n"
    "Do not output a command, explanation, reasoning, extra line, or <think> tag."
)

_STAGE_ONE_OUTPUT = re.compile(
    r"\ADiagnosis: ([^\r\n]+)\r?\nSubgoal: ([^\r\n]+)\Z"
)
_ACTION_ID_TOKEN = re.compile(r"\bA\d{3}\b")


@dataclass(frozen=True)
class GenerationRecord:
    """Raw diagnostics for one bounded model generation."""

    prompt: str
    raw_output: str
    token_statistics: TokenStatistics
    latency_seconds: float
    output_complete: bool
    token_cap_reached: bool
    max_new_tokens: int


@dataclass(frozen=True)
class StageOneDecision:
    """Strictly parsed Stage-1 diagnosis and immediate subgoal."""

    diagnosis: str | None
    subgoal: str | None
    status: str
    failure_reason: str | None
    diagnosis_word_count: int | None
    subgoal_word_count: int | None
    generation: GenerationRecord


@dataclass(frozen=True)
class StageTwoDecision:
    """Strictly parsed and environment-grounded Stage-2 action selection."""

    action_id: str | None
    action: str
    status: str
    failure_reason: str | None
    id_to_command: dict[str, str]
    generation: GenerationRecord


@dataclass(frozen=True)
class TwoStageRecoveryDecision:
    """One bounded intervention composed of exactly two model calls."""

    stage_one: StageOneDecision
    stage_two: StageTwoDecision
    action: str
    status: str
    failure_reason: str | None
    model_call_count: int = 2

    @property
    def action_id(self) -> str | None:
        return self.stage_two.action_id

    def as_action_decision(self, model_version: str) -> ActionDecision:
        """Adapt the two-stage result to the standard branch runner contract."""
        selected_index = int(self.action_id[1:]) if self.action_id else None
        stage_one_tokens = self.stage_one.generation.token_statistics
        stage_two_tokens = self.stage_two.generation.token_statistics
        input_tokens = (
            stage_one_tokens.input_tokens + stage_two_tokens.input_tokens
            if stage_one_tokens.input_tokens is not None
            and stage_two_tokens.input_tokens is not None
            else None
        )
        combined_tokens = TokenStatistics(
            generated_tokens=(
                stage_one_tokens.generated_tokens
                + stage_two_tokens.generated_tokens
            ),
            input_tokens=input_tokens,
        )
        return ActionDecision(
            action=self.action,
            raw_output=self.stage_two.generation.raw_output,
            parser_status="grounded" if self.status == "selected" else self.status,
            model_version=model_version,
            token_statistics=combined_tokens,
            reasoning=None,
            metadata={
                "prompt": self.stage_two.generation.prompt,
                "prompt_version": STAGE_TWO_PROMPT_VERSION,
                "recovery_operator_version": RECOVERY_OPERATOR_VERSION,
                "recovery_model_call_count": self.model_call_count,
                "recovery_stage_one": {
                    "prompt_version": STAGE_ONE_PROMPT_VERSION,
                    "diagnosis": self.stage_one.diagnosis,
                    "subgoal": self.stage_one.subgoal,
                    "status": self.stage_one.status,
                    "failure_reason": self.stage_one.failure_reason,
                },
                "action_selection_mode": "indexed_admissible",
                "action_selection": {
                    "action_selection_mode": "indexed_admissible",
                    "raw_model_output": self.stage_two.generation.raw_output,
                    "parsed_action_id": self.action_id,
                    "selected_index": selected_index,
                    "mapped_environment_command": self.action or None,
                    "id_to_command": self.stage_two.id_to_command,
                    "selection_status": self.status,
                    "failure_reason": self.failure_reason,
                },
            },
        )


class TwoStageRecoveryOperator:
    """Invoke diagnosis and action selection once each, with no repair path."""

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

    def act(self, request: ActionRequest) -> TwoStageRecoveryDecision:
        """Make exactly two generations and return one fail-closed action."""
        stage_one = parse_stage_one_output(
            self._generate(stage_one_prompt(request), STAGE_ONE_MAX_NEW_TOKENS)
        )
        stage_two = parse_stage_two_output(
            self._generate(
                stage_two_prompt(
                    request,
                    diagnosis=stage_one.diagnosis,
                    subgoal=stage_one.subgoal,
                    stage_one_valid=stage_one.status == "valid",
                ),
                STAGE_TWO_MAX_NEW_TOKENS,
            ),
            request.valid_actions,
        )
        if stage_one.status != "valid":
            return TwoStageRecoveryDecision(
                stage_one,
                stage_two,
                "",
                "stage_one_failure",
                stage_one.failure_reason,
            )
        if stage_two.status != "selected":
            return TwoStageRecoveryDecision(
                stage_one,
                stage_two,
                "",
                stage_two.status,
                stage_two.failure_reason,
            )
        return TwoStageRecoveryDecision(
            stage_one, stage_two, stage_two.action, "selected", None
        )

    def _generate(self, prompt: str, max_new_tokens: int) -> GenerationRecord:
        tokenizer, model, torch = self._policy._load()
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
                max_new_tokens=max_new_tokens,
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
        ended_with_eos = bool(
            len(generated_tokens)
            and int(generated_tokens[-1].item()) in set(eos_token_ids or [])
        )
        token_cap_reached = len(generated.scores or ()) >= max_new_tokens
        return GenerationRecord(
            prompt=prompt,
            raw_output=tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip(),
            token_statistics=self._policy._token_statistics(
                generated, torch, input_tokens=input_tokens
            ),
            latency_seconds=latency,
            output_complete=not token_cap_reached or ended_with_eos,
            token_cap_reached=token_cap_reached,
            max_new_tokens=max_new_tokens,
        )


def stage_one_prompt(request: ActionRequest) -> str:
    """Render bounded observable state without exposing action IDs."""
    context = _bounded_context(request)
    commands = "\n".join(
        f"- {command}" for command in (request.valid_actions or ())
    ) or "No admissible action list is available."
    return (
        f"{STAGE_ONE_INSTRUCTIONS}\n\n"
        f"{context}\n\n"
        f"Currently executable commands (unindexed):\n{commands}\n\n"
        "Your response:"
    )


def stage_two_prompt(
    request: ActionRequest,
    *,
    diagnosis: str | None,
    subgoal: str | None,
    stage_one_valid: bool,
) -> str:
    """Render exact frozen Stage-1 fields and the current environment mapping."""
    mapping = action_id_mapping(request.valid_actions or ())
    actions = "\n".join(
        f"[{action_id}] {command}" for action_id, command in mapping.items()
    ) or "No admissible action list is available."
    frozen_diagnosis = diagnosis if stage_one_valid else "(invalid Stage-1 output)"
    frozen_subgoal = subgoal if stage_one_valid else "(invalid Stage-1 output)"
    return (
        f"{STAGE_TWO_INSTRUCTIONS}\n\n"
        f"{_bounded_context(request)}\n\n"
        "Frozen Stage-1 diagnosis:\n"
        f"{frozen_diagnosis}\n\n"
        "Frozen Stage-1 immediate subgoal:\n"
        f"{frozen_subgoal}\n\n"
        f"Indexed admissible-action mapping:\n{actions}\n\n"
        "Your response:"
    )


def parse_stage_one_output(generation: GenerationRecord) -> StageOneDecision:
    """Validate the complete two-line diagnosis/subgoal contract."""
    stripped = generation.raw_output.strip()
    match = _STAGE_ONE_OUTPUT.fullmatch(stripped)
    diagnosis, subgoal = _stage_one_candidates(stripped)
    diagnosis_words = len(diagnosis.split()) if diagnosis else None
    subgoal_words = len(subgoal.split()) if subgoal else None
    if not generation.output_complete:
        return StageOneDecision(
            diagnosis,
            subgoal,
            "truncated_stage_one",
            "Stage 1 did not complete before its token limit",
            diagnosis_words,
            subgoal_words,
            generation,
        )
    if match is None or _ACTION_ID_TOKEN.search(generation.raw_output):
        return StageOneDecision(
            diagnosis,
            subgoal,
            "malformed_stage_one",
            "expected exactly one diagnosis line and one subgoal line without an ID",
            diagnosis_words,
            subgoal_words,
            generation,
        )
    diagnosis, subgoal = (field.strip() for field in match.groups())
    diagnosis_words = len(diagnosis.split())
    subgoal_words = len(subgoal.split())
    if diagnosis_words > 12:
        return StageOneDecision(
            diagnosis,
            subgoal,
            "diagnosis_too_long",
            "diagnosis exceeds 12 words",
            diagnosis_words,
            subgoal_words,
            generation,
        )
    if subgoal_words > 12:
        return StageOneDecision(
            diagnosis,
            subgoal,
            "subgoal_too_long",
            "subgoal exceeds 12 words",
            diagnosis_words,
            subgoal_words,
            generation,
        )
    return StageOneDecision(
        diagnosis,
        subgoal,
        "valid",
        None,
        diagnosis_words,
        subgoal_words,
        generation,
    )


def parse_stage_two_output(
    generation: GenerationRecord, valid_actions: tuple[str, ...] | None
) -> StageTwoDecision:
    """Parse one Action-ID and recover only the environment-owned command."""
    mapping = action_id_mapping(valid_actions or ())
    if not generation.output_complete:
        return StageTwoDecision(
            None,
            "",
            "truncated_stage_two",
            "Stage 2 did not complete before its token limit",
            mapping,
            generation,
        )
    parsed = parse_action_id(generation.raw_output, valid_actions)
    return StageTwoDecision(
        parsed.action_id,
        parsed.action,
        parsed.status,
        parsed.invalid_reason,
        mapping,
        generation,
    )


def _bounded_context(request: ActionRequest) -> str:
    history_window = request.history[-4:]
    transitions = []
    start = len(request.history) - len(history_window)
    for offset, (_, action) in enumerate(history_window):
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
        f"Task goal:\n{request.task.text}\n\n"
        f"Current pre-action observation:\n{request.observation}\n\n"
        f"Current inventory:\n{inventory}\n\n"
        "Recent action/result transitions (last 4):\n"
        f"{transition_text}\n\n"
        "Observable recent diagnostics:\n"
        f"Adjacent repeated actions: {diagnostics['adjacent_repeat_events']}\n"
        f"Two-action cycle events: {diagnostics['two_cycle_events']}"
    )


def _stage_one_candidates(output: str) -> tuple[str | None, str | None]:
    diagnosis = None
    subgoal = None
    for line in output.splitlines():
        if line.startswith("Diagnosis: ") and diagnosis is None:
            diagnosis = line.removeprefix("Diagnosis: ").strip() or None
        elif line.startswith("Subgoal: ") and subgoal is None:
            subgoal = line.removeprefix("Subgoal: ").strip() or None
    return diagnosis, subgoal
