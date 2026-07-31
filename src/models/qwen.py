"""Lazy Qwen text-policy wrapper for the reproducible baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .action_parser import parse_action
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

    def act(self, request: ActionRequest) -> ActionDecision:
        """Generate, parse, and record one action without environment side effects."""
        tokenizer, model, torch = self._load()
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
        parsed = parse_action(raw_output, request.valid_actions)
        return ActionDecision(
            action=parsed.action,
            raw_output=raw_output,
            parser_status=parsed.status,
            reasoning=parsed.reasoning,
            model_version=self.model_version,
            token_statistics=self._token_statistics(generated, torch),
            metadata={"prompt": prompt, "generation": self._generation_metadata()},
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

    def _generation_metadata(self) -> dict[str, Any]:
        options = self._config.generation
        return {
            "max_new_tokens": options.max_new_tokens,
            "do_sample": options.do_sample,
            "temperature": options.temperature,
            "top_p": options.top_p,
            "enable_thinking": self._config.enable_thinking,
        }

    @staticmethod
    def _token_statistics(generated: Any, torch: Any) -> TokenStatistics:
        scores = generated.scores or ()
        if not scores:
            return TokenStatistics(generated_tokens=0)
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
        )
