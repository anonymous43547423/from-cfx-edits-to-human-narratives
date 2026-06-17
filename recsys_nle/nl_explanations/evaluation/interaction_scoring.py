"""Shared scoring logic for interaction-based explanation evaluation metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence

from recsys_nle.nl_explanations.evaluation.base import (
    BaseEvaluator,
    EvaluationResult,
    aggregate_scores,
    normalise_score,
)
from recsys_nle.nl_explanations.hf_json import parse_json_from_hf
from recsys_nle.nl_explanations.payloads import prepare_interaction_payload
from recsys_nle.nl_explanations.prompts import format_interaction_prompt_attributes

if TYPE_CHECKING:
    import pandas as pd


class InteractionScoringEvaluator(BaseEvaluator):
    """Base evaluator for scoring interactions against explanation text patterns."""

    def __init__(
        self,
        *,
        metric_name: str = "interaction_scoring",
    ) -> None:
        """Configure interaction scoring evaluation parameters."""
        self._metric_name = metric_name

    def build_prompt(
        self,
        *,
        explanation: str,
        interactions: pd.DataFrame | None = None,
    ) -> list[dict[str, str]]:
        """Build a single prompt - not used directly for interaction scoring."""
        msg = "InteractionScoringEvaluator uses build_all_prompts() instead of build_prompt()"
        raise NotImplementedError(msg)

    def build_all_prompts(
        self,
        *,
        explanation: str,
        interactions: pd.DataFrame,
    ) -> list[tuple[str, list[dict[str, str]]]]:
        """Build prompts for all interactions, returning (interaction_desc, messages) tuples."""
        cleaned_explanation = explanation.strip()
        if interactions.empty or not cleaned_explanation:
            return []

        payload = prepare_interaction_payload(interactions, max_items=None)
        if not payload:
            return []

        interaction_descriptions = build_interaction_descriptions(payload)
        prompts: list[tuple[str, list[dict[str, str]]]] = []
        for description in interaction_descriptions:
            messages = build_single_interaction_scoring_messages(
                interaction_description=description,
                target_text=cleaned_explanation,
            )
            prompts.append((description, messages))
        return prompts

    def build_all_prompts_with_ids(
        self,
        *,
        explanation: str,
        interactions: pd.DataFrame,
    ) -> list[tuple[int, str, list[dict[str, str]]]]:
        """Build prompts with item IDs, returning (movie_id, interaction_desc, messages) tuples."""
        cleaned_explanation = explanation.strip()
        if interactions.empty or not cleaned_explanation:
            return []

        payload = prepare_interaction_payload(interactions, max_items=None)
        if not payload:
            return []

        interaction_descriptions = build_interaction_descriptions(payload)
        prompts: list[tuple[int, str, list[dict[str, str]]]] = []
        for item_record, description in zip(payload, interaction_descriptions, strict=True):
            raw_movie_id = item_record.get("movie_id", -1)
            movie_id = int(raw_movie_id) if isinstance(raw_movie_id, (int, float, str)) else -1
            messages = build_single_interaction_scoring_messages(
                interaction_description=description,
                target_text=cleaned_explanation,
            )
            prompts.append((movie_id, description, messages))
        return prompts

    def parse_single_interaction_result(
        self,
        raw_output: str,
        interaction_description: str,
    ) -> Mapping[str, object]:
        """Parse a single interaction scoring response into structured detail."""
        decoded = parse_json_from_hf(raw_output)
        data: Mapping[str, object] = decoded if isinstance(decoded, dict) else {}

        judgment = str(data.get("judgment") or data.get("error") or "").strip()
        return {
            "interaction": interaction_description,
            "score": normalise_score(data.get("score", float("nan"))),
            **({"judgment": judgment} if judgment else {}),
        }

    def aggregate_results(
        self,
        per_interaction_scores: Sequence[Mapping[str, object]],
    ) -> EvaluationResult:
        """Aggregate per-interaction results into a final evaluation result."""
        if not per_interaction_scores:
            return EvaluationResult(
                judgment=f"No interactions could be evaluated for {self._metric_name}.",
                score=float("nan"),
                details={"per_interaction_scores": []},
            )

        mean_score, warnings = aggregate_scores(
            per_interaction_scores,
            empty_warning=(
                f"All per-interaction {self._metric_name} scores are NaN; "
                f"aggregate {self._metric_name} score set to NaN."
            ),
        )
        judgment = (
            f"Average {self._metric_name} score {mean_score:.2f} across {len(per_interaction_scores)} interactions."
        )
        all_details: dict[str, object] = {"per_interaction_scores": list(per_interaction_scores)}
        if warnings:
            all_details["warnings"] = warnings
        return EvaluationResult(judgment=judgment, score=mean_score, details=all_details)

    def build_empty_result(self, *, reason: str) -> EvaluationResult:
        """Return an empty evaluation result with the given reason."""
        return EvaluationResult(
            judgment=reason,
            score=float("nan"),
            details={"per_interaction_scores": []},
        )


def build_interaction_descriptions(
    items: Sequence[Mapping[str, object]],
) -> list[str]:
    """Serialise interaction payload entries into concise attribute lines."""
    return [format_interaction_prompt_attributes(item) for item in items]


def build_single_interaction_scoring_messages(
    *,
    interaction_description: str,
    target_text: str,
) -> list[dict[str, str]]:
    """Compose prompts that assess a single interaction against a pattern."""
    system_prompt = (
        "You are evaluating how well a proposed pattern, which describes what a user's "
        "past interactions have in common, matches a specific interaction."
    )
    user_prompt = (
        "You are given:\n"
        "1. A single past interaction.\n"
        "2. A proposed pattern that aims to describe what the user's interactions have in common.\n\n"
        "Interaction to evaluate:\n"
        f"{interaction_description}\n\n"
        "Proposed pattern:\n"
        f"{target_text}\n\n"
        "Write a short judgment (brief bullet points, at most 20 words total) "
        "explaining, in a chain-of-thought manner, whether or not this interaction "
        "matches the proposed pattern. "
        "Then assign a discrete score using the following scale:\n"
        "1. The interaction fits the pattern (score 1.0).\n"
        "2. The interaction fits the pattern, but there are some clear minor differences (score 0.66).\n"
        "3. The interaction is only somewhat related to the pattern; it obviously "
        "differs in a major way (score 0.33).\n"
        "4. The interaction is unrelated or only very marginally related to the pattern (score 0.0).\n\n"
        "Respond ONLY with a JSON object of the form:\n"
        '{"judgment": "<short explanation>", "score": <one of 1.0, 0.66, 0.33, 0.0>}.\n'
        "Do not include any additional keys, text, or commentary outside this JSON object. "
        "The pattern does not need to describe the interaction in detail - do not penalize "
        "missing information in the pattern."
    )
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]
