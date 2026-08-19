"""Lazy Qwen text-policy wrapper for the reproducible baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .action_parser import action_id_mapping, parse_action, parse_action_id
from .policy import ActionDecision, ActionPolicy, ActionRequest, GenerationOptions, TokenStatistics


@dataclass(frozen=True)
class QwenPolicyConfig:
    """Configuration for one Qwen3 text-generation policy."""

    model_id: str
    device: str = "auto"
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    enable_thinking: bool = False
    generation: GenerationOptions = GenerationOptions()
    action_selection_mode: str = "free_form_validated"
    history_context_mode: str = "full_raw"
    history_window: int | None = None

    def __post_init__(self) -> None:
        if self.action_selection_mode not in {
            "free_form_validated",
            "indexed_admissible",
        }:
            raise ValueError(
                "action_selection_mode must be free_form_validated or indexed_admissible"
            )
        if self.history_context_mode not in {"full_raw", "bounded_recent_state"}:
            raise ValueError(
                "history_context_mode must be full_raw or bounded_recent_state"
            )
        if self.history_context_mode == "bounded_recent_state":
            if self.action_selection_mode != "indexed_admissible":
                raise ValueError("bounded_recent_state requires indexed_admissible")
            if self.history_window is None or self.history_window < 1:
                raise ValueError("bounded_recent_state requires a positive history_window")
        elif self.history_window is not None:
            raise ValueError("history_window is only valid for bounded_recent_state")


class QwenPolicy(ActionPolicy):
    """Generate ALFWorld actions with a Qwen causal-language-model backend."""

    def __init__(self, config: QwenPolicyConfig) -> None:
        self._config = config
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    @property
    def model_version(self) -> str:
        """Return the configured model identifier."""
        return self._config.model_id

    @property
    def action_selection_mode(self) -> str:
        """Return the configured action-selection interface."""
        return self._config.action_selection_mode

    def act(self, request: ActionRequest) -> ActionDecision:
        """Generate, parse, and record one action without environment side effects."""
        tokenizer, model, torch = self._load()
        if self._config.action_selection_mode == "indexed_admissible":
            prompt = (
                self._indexed_bounded_prompt(request, self._config.history_window)
                if self._config.history_context_mode == "bounded_recent_state"
                else self._indexed_prompt(request)
            )
        else:
            prompt = self._prompt(request)
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self._config.enable_thinking,
            )
        else:
            rendered = prompt
        inputs = tokenizer(rendered, return_tensors="pt")
        inputs = {name: value.to(model.device) for name, value in inputs.items()}
        options = self._config.generation
        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": options.max_new_tokens,
            "do_sample": options.do_sample,
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        if options.do_sample and options.temperature is not None:
            generation_kwargs["temperature"] = options.temperature
        if options.do_sample and options.top_p is not None:
            generation_kwargs["top_p"] = options.top_p
        with torch.no_grad():
            generated = model.generate(**generation_kwargs)
        input_tokens = inputs["input_ids"].shape[1]
        generated_tokens = generated.sequences[0][input_tokens:]
        raw_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        if self._config.action_selection_mode == "indexed_admissible":
            return self._indexed_decision(
                request,
                prompt,
                raw_output,
                generated,
                torch,
                input_tokens=input_tokens,
            )
        parsed = parse_action(raw_output, request.valid_actions)
        return ActionDecision(
            action=parsed.action,
            raw_output=raw_output,
            parser_status=parsed.status,
            reasoning=parsed.reasoning,
            model_version=self.model_version,
            token_statistics=self._token_statistics(
                generated, torch, input_tokens=input_tokens
            ),
            metadata={
                "prompt": prompt,
                "prompt_version": "free-form-action-v1",
                "generation": self._generation_metadata(),
                "action_selection_mode": "free_form_validated",
                "action_selection": {
                    "action_selection_mode": "free_form_validated",
                    "raw_model_output": raw_output,
                    "parsed_action_id": None,
                    "selected_index": None,
                    "mapped_environment_command": parsed.action or None,
                    "id_to_command": {},
                    "selection_status": "not_applicable",
                    "failure_reason": None,
                },
                "parser": {
                    "candidate": parsed.candidate,
                    "invalid_reason": parsed.invalid_reason,
                },
            },
        )

    def _indexed_decision(
        self,
        request: ActionRequest,
        prompt: str,
        raw_output: str,
        generated: Any,
        torch: Any,
        *,
        input_tokens: int | None = None,
    ) -> ActionDecision:
        """Map one strict action ID to the exact environment-owned command."""
        parsed = parse_action_id(raw_output, request.valid_actions)
        mapping = action_id_mapping(request.valid_actions or ())
        return ActionDecision(
            action=parsed.action,
            raw_output=raw_output,
            parser_status="grounded" if parsed.status == "selected" else parsed.status,
            reasoning=None,
            model_version=self.model_version,
            token_statistics=self._token_statistics(
                generated, torch, input_tokens=input_tokens
            ),
            metadata={
                "prompt": prompt,
                "prompt_version": (
                    "indexed-admissible-bounded-context-v1"
                    if self._config.history_context_mode == "bounded_recent_state"
                    else "indexed-admissible-action-v1"
                ),
                "history_context": {
                    "mode": self._config.history_context_mode,
                    "window": self._config.history_window,
                },
                "generation": self._generation_metadata(),
                "action_selection_mode": "indexed_admissible",
                "action_selection": {
                    "action_selection_mode": "indexed_admissible",
                    "raw_model_output": raw_output,
                    "parsed_action_id": parsed.action_id,
                    "selected_index": parsed.selected_index,
                    "mapped_environment_command": parsed.action or None,
                    "id_to_command": mapping,
                    "selection_status": parsed.status,
                    "failure_reason": parsed.invalid_reason,
                },
                "parser": {
                    "candidate": parsed.action_id,
                    "invalid_reason": parsed.invalid_reason,
                },
            },
        )

    def _load(self) -> tuple[Any, Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            import torch

            return self._tokenizer, self._model, torch
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Qwen baseline collection requires torch and transformers. "
                "Install the model runtime before running the collection script."
            ) from error
        dtype = getattr(torch, self._config.dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._config.model_id,
            trust_remote_code=self._config.trust_remote_code,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        model_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "trust_remote_code": self._config.trust_remote_code,
        }
        if self._config.device == "auto":
            model_kwargs["device_map"] = "auto"
        self._model = AutoModelForCausalLM.from_pretrained(self._config.model_id, **model_kwargs)
        if self._config.device not in {"auto", "cpu"}:
            self._model = self._model.to(self._config.device)
        self._model.eval()
        return self._tokenizer, self._model, torch

    @staticmethod
    def _prompt(request: ActionRequest) -> str:
        valid_actions = (
            "\n".join(f"- {action}" for action in request.valid_actions)
            if request.valid_actions
            else "No admissible action list is available."
        )
        history = "\n".join(f"Observation: {observation}\nAction: {action}" for observation, action in request.history)
        return (
            "You are an ALFWorld household agent. Select exactly one next action.\n"
            "Return exactly one line in this format: Action: <command>.\n"
            "Copy <command> exactly from the valid-actions list. Do not explain, reason, "
            "or emit <think> tags.\n\n"
            "Decision rules:\n"
            "1. Keep pursuing the stated task until it is complete.\n"
            "2. Only take, move, heat, cool, or clean an object required by the task.\n"
            "3. When searching, inspect unexamined plausible locations before revisiting one.\n"
            "4. Do not alternate between locations or undo recent progress without a task-related reason.\n"
            "5. After taking a required object, use the task destination or required transformation.\n\n"
            f"Task: {request.task.text}\n"
            f"History:\n{history or '(none)'}\n\n"
            f"Current observation:\n{request.observation}\n\n"
            f"Valid actions:\n{valid_actions}\n\n"
            "Action:"
        )

    @staticmethod
    def _indexed_prompt(request: ActionRequest) -> str:
        mapping = action_id_mapping(request.valid_actions or ())
        valid_actions = (
            "\n".join(f"[{action_id}] {action}" for action_id, action in mapping.items())
            if mapping
            else "No admissible action list is available."
        )
        history = "\n".join(
            f"Observation: {observation}\nAction: {action}"
            for observation, action in request.history
        )
        return (
            "You are an ALFWorld household agent. Select exactly one next action ID.\n"
            "Return exactly one line in this format:\n"
            "Action-ID: Axyz\n"
            "Choose one ID exactly as written in the indexed valid-actions list. "
            "Do not output a command, explanation, reasoning, or <think> tags.\n\n"
            "Decision rules:\n"
            "1. Keep pursuing the stated task until it is complete.\n"
            "2. Only take, move, heat, cool, or clean an object required by the task.\n"
            "3. When searching, inspect unexamined plausible locations before revisiting one.\n"
            "4. Do not alternate between locations or undo recent progress without a task-related reason.\n"
            "5. After taking a required object, use the task destination or required transformation.\n\n"
            f"Task: {request.task.text}\n"
            f"History:\n{history or '(none)'}\n\n"
            f"Current observation:\n{request.observation}\n\n"
            f"Indexed valid actions:\n{valid_actions}\n\n"
            "Your response:"
        )

    @staticmethod
    def _indexed_bounded_prompt(
        request: ActionRequest, history_window: int | None
    ) -> str:
        """Render goal-preserving state with only the latest exact transitions."""
        if history_window is None or history_window < 1:
            raise ValueError("history_window must be positive")
        mapping = action_id_mapping(request.valid_actions or ())
        valid_actions = (
            "\n".join(
                f"[{action_id}] {action}" for action_id, action in mapping.items()
            )
            if mapping
            else "No admissible action list is available."
        )
        start = max(0, len(request.history) - history_window)
        transitions: list[str] = []
        for index in range(start, len(request.history)):
            action = request.history[index][1]
            result = (
                request.history[index + 1][0]
                if index + 1 < len(request.history)
                else request.observation
            )
            transitions.append(f"Action: {action}\nResult: {result}")
        inventory = (
            request.observation
            if request.history and request.history[-1][1].strip() == "inventory"
            else "(not available from the environment)"
        )
        transition_text = "\n\n".join(transitions) if transitions else "(none)"
        return (
            "You are an ALFWorld household agent. Select exactly one next action ID.\n"
            "Return exactly one line in this format:\n"
            "Action-ID: Axyz\n"
            "Choose one ID exactly as written in the indexed valid-actions list. "
            "Do not output a command, explanation, reasoning, or <think> tags.\n\n"
            "Decision rules:\n"
            "1. Keep pursuing the stated task until it is complete.\n"
            "2. Only take, move, heat, cool, or clean an object required by the task.\n"
            "3. When searching, inspect unexamined plausible locations before revisiting one.\n"
            "4. Do not alternate between locations or undo recent progress without a task-related reason.\n"
            "5. After taking a required object, use the task destination or required transformation.\n\n"
            f"Task goal:\n{request.task.text}\n\n"
            f"Current observation:\n{request.observation}\n\n"
            f"Current inventory:\n{inventory}\n\n"
            f"Recent action/result transitions (last {history_window}):\n"
            f"{transition_text}\n\n"
            f"Indexed valid actions:\n{valid_actions}\n\n"
            "Your response:"
        )

    def _generation_metadata(self) -> dict[str, Any]:
        options = self._config.generation
        return {
            "max_new_tokens": options.max_new_tokens,
            "do_sample": options.do_sample,
            "temperature": options.temperature,
            "top_p": options.top_p,
            "enable_thinking": self._config.enable_thinking,
            "action_selection_mode": self._config.action_selection_mode,
            "history_context_mode": self._config.history_context_mode,
            "history_window": self._config.history_window,
        }

    @staticmethod
    def _token_statistics(
        generated: Any, torch: Any, *, input_tokens: int | None = None
    ) -> TokenStatistics:
        scores = generated.scores or ()
        if not scores:
            return TokenStatistics(generated_tokens=0, input_tokens=input_tokens)
        log_probabilities: list[float] = []
        entropies: list[float] = []
        token_ids = generated.sequences[0][-len(scores) :]
        for score, token_id in zip(scores, token_ids):
            log_probs = torch.log_softmax(score[0], dim=-1)
            probs = torch.softmax(score[0], dim=-1)
            log_probabilities.append(float(log_probs[token_id].item()))
            entropies.append(float((-(probs * log_probs).sum()).item()))
        return TokenStatistics(
            generated_tokens=len(scores),
            mean_token_log_probability=sum(log_probabilities) / len(log_probabilities),
            mean_token_entropy=sum(entropies) / len(entropies),
            input_tokens=input_tokens,
        )
