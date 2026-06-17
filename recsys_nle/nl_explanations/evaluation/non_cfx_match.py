"""LLM-based non-CFX match evaluation for explanations."""

from __future__ import annotations

from recsys_nle.nl_explanations.evaluation.interaction_scoring import InteractionScoringEvaluator


class NonCFXMatchEvaluator(InteractionScoringEvaluator):
    """Measure non-CFX match of explanations against non-CFX interactions."""

    def __init__(self) -> None:
        """Configure non-CFX match evaluation parameters."""
        super().__init__(
            metric_name="Non-CFX match",
        )
