"""Tests for ExplanationGenerator reasoning configuration."""

from __future__ import annotations

# ruff: noqa: S101, I001

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from recsys_nle.nl_explanations.generator import (
    ExplanationGenerator,
    UserExplanationInput,
)

if TYPE_CHECKING:
    from recsys_nle.nl_explanations.llm import ChatMessage, LLMClient


@dataclass(slots=True)
class _StubLLMClient:
    """Stub LLM client capturing batch calls and returning canned payloads."""

    batches: list[Sequence[Sequence[ChatMessage]]] = field(default_factory=list)

    def generate(  # pragma: no cover - guarded by generate_batch in tests
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop_sequences: Sequence[str] | None = None,
    ) -> str:
        """Fallback single-prompt generation (should not be used in these tests)."""
        del messages, max_new_tokens, temperature, top_p, stop_sequences
        message = "generate() should not be invoked for _StubLLMClient"
        raise AssertionError(message)

    def generate_batch(
        self,
        messages_batch: Sequence[Sequence[ChatMessage]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop_sequences: Sequence[str] | None = None,
        batch_size: int | None = None,
    ) -> Sequence[str]:
        """Record batch invocations and return deterministic JSON payloads."""
        del max_new_tokens, temperature, top_p, stop_sequences, batch_size
        self.batches.append(messages_batch)
        # Determine which kind of prompt this batch represents based on the text content.
        first_messages = messages_batch[0] if messages_batch else []
        first_text = " ".join(str(message.get("content", "")) for message in first_messages)

        if "Return a JSON object with exactly two keys" in first_text:
            # Explanation JSON payloads.
            return ['{"pattern": "classic 80s sci-fi action", "confidence": 0.9}' for _ in messages_batch]

        # For reasoning or any other prompts, return a simple deterministic text payload.
        return ["Reasoning output" for _ in messages_batch]

    def close(self) -> None:
        """No-op close for protocol compatibility."""


def _make_user_input(user_id: int = 1) -> UserExplanationInput:
    """Construct a minimal user input frame."""
    cfx_interactions = pd.DataFrame(
        [
            {"movie_id": 5, "rating": 4.0, "weight": 0.4, "importance": 0.4},
        ]
    )
    non_cfx_interactions = pd.DataFrame(
        [
            {"movie_id": 10, "rating": 3.0},
        ]
    )
    return UserExplanationInput(
        user_id=user_id,
        cfx_interactions=cfx_interactions,
        non_cfx_interactions=non_cfx_interactions,
    )


def test_generate_batch_without_reasoning_skips_reasoning_stage() -> None:
    """It generates explanations without invoking a reasoning batch."""
    llm_client: LLMClient = _StubLLMClient()
    generator = ExplanationGenerator(
        llm_client=llm_client,
        n_cfx_interactions=3,
        use_reasoning=False,
    )

    inputs = [_make_user_input(user_id=1)]
    results = generator.generate_batch(inputs)

    assert isinstance(results, Mapping)
    assert 1 in results

    generated = results[1]
    # Reasoning should be empty when disabled.
    assert generated.reasoning == ""

    # The stub LLM should have been called exactly once: for explanations.
    assert isinstance(llm_client, _StubLLMClient)
    expected_batches = 1
    assert len(llm_client.batches) == expected_batches

    # Prompts must not reference prior chain-of-thought reasoning.
    for batch in llm_client.batches:
        for messages in batch:
            for message in messages:
                content = message.get("content", "")
                assert "Chain-of-thought reasoning" not in content
                assert "chain-of-thought analysis" not in content
