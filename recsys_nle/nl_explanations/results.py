"""Result containers for explanation generation and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from recsys_nle.nl_explanations.evaluation import EvaluationResult
    from recsys_nle.nl_explanations.generator import GeneratedExplanation


@dataclass(slots=True)
class NaturalLanguageExplanationResult:
    """Aggregate generation outputs with plausibility, cfx_match, non_cfx_match, and faithfulness scores."""

    user_id: int
    reasoning: str
    explanation: str
    explanation_plausibility: EvaluationResult
    explanation_cfx_match: EvaluationResult
    explanation_readability: EvaluationResult | None = None
    explanation_non_cfx_match: EvaluationResult | None = None
    faithfulness_removal: EvaluationResult | None = None
    faithfulness_removal_baseline: EvaluationResult | None = None
    faithfulness_replacement: EvaluationResult | None = None
    faithfulness_replacement_baseline: EvaluationResult | None = None
    reasoning_prompt: str | None = None
    explanation_prompt: str | None = None
    explanation_conversation: list[dict[str, str]] | None = None
    explanation_confidence: float = float("nan")

    @classmethod
    def from_components(
        cls,
        generated: GeneratedExplanation,
        *,
        explanation_plausibility: EvaluationResult,
        explanation_cfx_match: EvaluationResult,
        explanation_readability: EvaluationResult | None = None,
        explanation_non_cfx_match: EvaluationResult | None = None,
        faithfulness_removal: EvaluationResult | None = None,
        faithfulness_removal_baseline: EvaluationResult | None = None,
        faithfulness_replacement: EvaluationResult | None = None,
        faithfulness_replacement_baseline: EvaluationResult | None = None,
    ) -> NaturalLanguageExplanationResult:
        """Create a result bundle from generated content and evaluation outputs."""
        return cls(
            user_id=generated.user_id,
            reasoning=generated.reasoning,
            explanation=generated.explanation,
            explanation_plausibility=explanation_plausibility,
            explanation_cfx_match=explanation_cfx_match,
            explanation_readability=explanation_readability,
            explanation_non_cfx_match=explanation_non_cfx_match,
            faithfulness_removal=faithfulness_removal,
            faithfulness_removal_baseline=faithfulness_removal_baseline,
            faithfulness_replacement=faithfulness_replacement,
            faithfulness_replacement_baseline=faithfulness_replacement_baseline,
            reasoning_prompt=generated.reasoning_prompt,
            explanation_prompt=generated.explanation_prompt,
            explanation_conversation=generated.explanation_conversation,
            explanation_confidence=generated.explanation_confidence,
        )
