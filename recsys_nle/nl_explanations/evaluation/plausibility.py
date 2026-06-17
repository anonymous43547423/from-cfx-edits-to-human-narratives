"""LLM-based evaluation of explanation plausibility."""

from __future__ import annotations

from typing import TYPE_CHECKING

from recsys_nle.nl_explanations.evaluation.base import BaseEvaluator

if TYPE_CHECKING:
    import pandas as pd


class PlausibilityEvaluator(BaseEvaluator):
    """Evaluate explanation plausibility using LLMs and recommendation evidence."""

    def build_prompt(
        self,
        *,
        explanation: str,
        interactions: pd.DataFrame | None = None,
    ) -> list[dict[str, str]]:
        """Build chat messages for evaluating explanation plausibility."""
        del interactions
        system_prompt = (
            "You are evaluating natural language explanations for recommendation lists. "
            "Judge whether the explanation is factually plausible and logically coherent with the recommendations."
        )
        user_prompt = (
            "You will be provided with a candidate explanation. "
            "Assess whether the explanation can be perceived as valid given the recommended items. "
            "Focus exclusively on factual plausibility and logical coherence with the provided data. "
            "The explanation is intentionally concise and phrased as a Netflix-style 'Because you ...' clause; "
            "do NOT penalise brevity or the lack of peripheral detail if the core claim is supported.\n\n"
            "Candidate explanation:\n"
            f"{explanation.strip()}\n\n"
            "Respond with a JSON object containing exactly two fields:\n"
            '{"judgment": "<short textual assessment>", "score": <numeric score between 0 and 1>}.\n'
            "Keep the 'judgment' text concise: no more than three sentences and under 80 words in total. "
            "Reference the explanation and at least two recommended items when discussing factual accuracy "
            "and logical support. Highlight what is correct, questionable, or missing, including any uncertainties. "
            "Assign higher scores when the explanation accurately reflects real attributes present in the "
            "recommendations, and lower scores when it is vague, incorrect, or unsupported."
        )
        return [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]
