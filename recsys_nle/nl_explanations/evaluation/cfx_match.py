"""LLM-based CFX match evaluation for explanations."""

from __future__ import annotations

from recsys_nle.nl_explanations.evaluation.interaction_scoring import InteractionScoringEvaluator


class CFXMatchEvaluator(InteractionScoringEvaluator):
    """Measure CFX match of explanations to CFX attributions."""

    def __init__(self) -> None:
        """Configure CFX match evaluation parameters."""
        super().__init__(
            metric_name="CFX match",
        )
