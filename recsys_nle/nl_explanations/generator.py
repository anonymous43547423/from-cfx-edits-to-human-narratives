"""Natural-language explanation generation pipeline."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

from .payloads import prepare_interaction_payload
from .prompts import (
    _build_explanation_messages,
    build_reasoning_messages,
    serialise_chat_messages,
    serialise_influential_interactions,
)

if TYPE_CHECKING:
    import pandas as pd

    from .llm import LLMClient

from .hf_json import parse_json_from_hf

MAX_EXPLANATION_WORDS = 18

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class UserExplanationInput:
    """Container for recommendation context consumed by the generator."""

    user_id: int
    cfx_interactions: pd.DataFrame
    non_cfx_interactions: pd.DataFrame


@dataclass(slots=True)
class GeneratedExplanation:
    """Structured outputs and confidences produced for a single user."""

    user_id: int
    reasoning: str
    explanation: str
    reasoning_prompt: str | None = None
    explanation_prompt: str | None = None
    explanation_conversation: list[dict[str, str]] | None = None
    explanation_confidence: float = float("nan")


class ExplanationGenerator:
    """Generate chain-of-thought reasoning and explanations for recommendations."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        n_cfx_interactions: int | None = 5,
        reasoning_tokens: int = 512,
        explanation_tokens: int = 96,
        use_reasoning: bool = True,
    ) -> None:
        """Configure generation limits and attach the language model client."""
        self._llm = llm_client
        self._n_cfx_interactions = n_cfx_interactions
        self._reasoning_tokens = reasoning_tokens
        self._explanation_tokens = explanation_tokens
        self._use_reasoning = use_reasoning

    def generate(self, user_input: UserExplanationInput) -> GeneratedExplanation:
        """Produce reasoning and explanation for a user."""
        results = self.generate_batch([user_input], batch_size=1)
        return results[user_input.user_id]

    def close(self) -> None:
        """Release LLM resources held by the generator."""
        close_fn = getattr(self._llm, "close", None)
        if callable(close_fn):
            close_fn()

    def generate_batch(
        self,
        inputs: Sequence[UserExplanationInput],
        *,
        batch_size: int | None = None,
    ) -> dict[int, GeneratedExplanation]:
        """Generate reasoning and explanations for multiple users at once."""
        if not inputs:
            return {}

        prepared = [
            _PreparedInput.from_user_input(
                user_input,
                n_cfx_interactions=self._n_cfx_interactions,
            )
            for user_input in inputs
        ]

        reasoning_prompts, reasoning_outputs = self._build_reasoning_batch(
            prepared=prepared,
            batch_size=batch_size,
        )

        explanation_messages = self._build_explanation_messages_batch(
            prepared=prepared,
            reasoning_outputs=reasoning_outputs,
        )

        explanation_prompts = [serialise_chat_messages(messages) for messages in explanation_messages]
        explanation_payloads = self._invoke_llm_batch(
            explanation_messages,
            max_new_tokens=self._explanation_tokens,
            temperature=0.2,
            top_p=0.7,
            batch_size=batch_size,
        )

        results: dict[int, GeneratedExplanation] = {}
        for index, item in enumerate(prepared):
            explanation_payload = explanation_payloads[index]
            explanation, explanation_confidence = _parse_explanation(explanation_payload)

            explanation_conversation = list(explanation_messages[index])
            if explanation_payload:
                explanation_conversation.append({"role": "assistant", "content": explanation_payload})

            results[item.user_input.user_id] = GeneratedExplanation(
                user_id=item.user_input.user_id,
                reasoning=reasoning_outputs[index],
                explanation=explanation,
                reasoning_prompt=reasoning_prompts[index],
                explanation_prompt=explanation_prompts[index],
                explanation_conversation=explanation_conversation,
                explanation_confidence=explanation_confidence,
            )
        return results

    def _invoke_llm(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """Call the language model and return a cleaned response."""
        return self._llm.generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=("<|eot_id|>",),
        )

    def _invoke_llm_batch(
        self,
        messages_batch: Sequence[Sequence[Mapping[str, str]]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        batch_size: int | None = None,
    ) -> list[str]:
        """Call the language model for multiple prompts using batch execution when available."""
        outputs = self._llm.generate_batch(
            messages_batch,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=("<|eot_id|>",),
            batch_size=batch_size,
        )
        return [str(output or "").strip() for output in outputs]

    def _build_reasoning_batch(
        self,
        *,
        prepared: Sequence[_PreparedInput],
        batch_size: int | None,
    ) -> tuple[list[str | None], list[str]]:
        """Build reasoning prompts and outputs for a prepared batch."""
        if not self._use_reasoning:
            outputs: list[str] = ["" for _ in prepared]
            prompts: list[str | None] = [None for _ in prepared]
            return prompts, outputs

        reasoning_messages = [
            build_reasoning_messages(
                cfx_interactions_text=item.cfx_interactions_text,
                non_cfx_interactions_text=item.non_cfx_interactions_text,
            )
            for item in prepared
        ]

        prompts = [serialise_chat_messages(messages) for messages in reasoning_messages]
        outputs = self._invoke_llm_batch(
            reasoning_messages,
            max_new_tokens=self._reasoning_tokens,
            temperature=0.3,
            top_p=0.8,
            batch_size=batch_size,
        )
        return prompts, outputs

    def _build_explanation_messages_batch(
        self,
        *,
        prepared: Sequence[_PreparedInput],
        reasoning_outputs: Sequence[str],
    ) -> list[list[dict[str, str]]]:
        """Build explanation messages for a prepared batch."""
        include_reasoning = self._use_reasoning
        return [
            _build_explanation_messages(
                reasoning_text=reasoning_outputs[index] if include_reasoning else None,
                cfx_interactions_text=item.cfx_interactions_text,
                non_cfx_interactions_text=item.non_cfx_interactions_text,
                include_reasoning=include_reasoning,
            )
            for index, item in enumerate(prepared)
        ]


@dataclass(slots=True)
class _PreparedInput:
    """Intermediate representation that stores prompt text for batch execution."""

    user_input: UserExplanationInput
    cfx_interactions_text: str
    non_cfx_interactions_text: str

    @classmethod
    def from_user_input(
        cls,
        user_input: UserExplanationInput,
        *,
        n_cfx_interactions: int | None,
    ) -> _PreparedInput:
        """Create a prepared prompt payload from raw user input."""
        # Limit CFX interactions for prompts (but evaluators use all)
        cfx_payload = prepare_interaction_payload(
            user_input.cfx_interactions,
            max_items=n_cfx_interactions,
        )
        # Non-CFX is already limited to N during computation
        non_cfx_payload = prepare_interaction_payload(
            user_input.non_cfx_interactions,
            max_items=None,
        )
        return cls(
            user_input=user_input,
            cfx_interactions_text=serialise_influential_interactions(cfx_payload),
            non_cfx_interactions_text=serialise_influential_interactions(non_cfx_payload),
        )


def _normalise_llm_output_text(text: str) -> str:
    """Normalise arbitrary LLM output text into a cleaned string."""
    cleaned = text.replace("\n", " ").strip()
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")
    return cleaned


def _lowercase_first_char(text: str) -> str:
    """Convert the first character of text to lowercase."""
    if not text:
        return text
    return text[0].lower() + text[1:]


def _parse_explanation(text: str) -> tuple[str, float]:
    """Decode an explanation payload into verb phrase, trait, and confidence."""
    parsed = parse_json_from_hf(text)
    if not isinstance(parsed, Mapping):
        LOGGER.warning(
            "Output could not be parsed as JSON; Raw output: %s",
            text,
        )
        return "", float("nan")
    pattern_raw = parsed["pattern"]
    pattern = _lowercase_first_char(_normalise_llm_output_text(pattern_raw))
    confidence = _parse_confidence(parsed["confidence"])
    return pattern, confidence


def _parse_confidence(value: object | None) -> float:
    """Parse a confidence value into a bounded 0-1 float or NaN when invalid."""
    if value is None or isinstance(value, bool):
        return float("nan")
    if not isinstance(value, (int, float, str)):
        return float("nan")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        return float("nan")
    return float(parsed)
