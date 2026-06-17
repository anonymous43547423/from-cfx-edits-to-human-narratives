"""Faithfulness evaluation metrics for natural-language explanations."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

import torch  # noqa: TC002

from recsys_nle.nl_explanations.evaluation.base import EvaluationResult
from recsys_nle.nl_explanations.evaluation.interaction_scoring import (
    InteractionScoringEvaluator,
)

if TYPE_CHECKING:
    import pandas as pd

    from recsys_nle.core.recommender_wrapper import RecommenderWrapper


@dataclass(slots=True)
class FaithfulnessConfig:
    """Configuration for faithfulness evaluation."""

    n_sampled_faithfulness_interactions: int
    match_threshold: float
    n_interactions_min_limit: int
    n_faithfulness_trials: int
    n_faithfulness_samples: int


@dataclass(slots=True)
class ScoredItem:
    """An item with its LLM-computed similarity score."""

    item_id: int
    score: float
    interaction_description: str


def _create_cfx_excluded_history(
    user_history: torch.Tensor,
    cfx_item_ids: Sequence[int],
) -> torch.Tensor:
    """Create user history with CFX item indices zeroed out."""
    modified = user_history.clone()
    for item_id in cfx_item_ids:
        if 0 <= item_id < modified.numel():
            modified[item_id] = 0.0
    return modified


class BaseFaithfulnessEvaluator(InteractionScoringEvaluator):
    """Base evaluator for faithfulness metrics with two-phase evaluation."""

    def __init__(
        self,
        *,
        metric_name: str = "faithfulness",
    ) -> None:
        """Configure faithfulness evaluation parameters."""
        super().__init__(metric_name=metric_name)

    def compute_results_from_scores(
        self,
        *,
        scored_items: Sequence[ScoredItem],
        user_history: torch.Tensor,
        target_item: int,
        recommender: RecommenderWrapper,
        config: FaithfulnessConfig,
        cfx_item_ids: Sequence[int] = (),
    ) -> tuple[EvaluationResult, EvaluationResult]:
        """Run trials using pre-computed scores. Returns (regular, baseline)."""
        raise NotImplementedError

    def _split_by_similarity(
        self,
        scored_items: Sequence[ScoredItem],
        threshold: float,
    ) -> tuple[list[ScoredItem], list[ScoredItem], list[ScoredItem]]:
        """Split items into similar, dissimilar, and NaN-score buckets."""
        similar: list[ScoredItem] = []
        dissimilar: list[ScoredItem] = []
        nan_items: list[ScoredItem] = []
        for item in scored_items:
            if math.isnan(item.score):
                nan_items.append(item)
                continue
            if item.score >= threshold:
                similar.append(item)
            else:
                dissimilar.append(item)
        return similar, dissimilar, nan_items

    def _build_nan_result(self, reason: str) -> EvaluationResult:
        """Build a NaN result with the given reason."""
        return EvaluationResult(
            judgment=reason,
            score=float("nan"),
            details={"reason": reason},
        )

    def _build_success_result(
        self,
        median_score: float,
        n_candidates: int,
        n_trials: int,
        n_evaluated: int,
        n_samples_per_trial: int,
        metric_name: str,
    ) -> EvaluationResult:
        """Build a result from median target-item scores."""
        judgment = f"{metric_name} median score {median_score:.4f} from {n_evaluated} of {n_trials} trials."
        return EvaluationResult(
            judgment=judgment,
            score=median_score,
            details={
                "n_candidates": n_candidates,
                "n_trials": n_trials,
                "n_evaluated": n_evaluated,
                "n_samples_per_trial": n_samples_per_trial,
                "median_score": median_score,
            },
        )

    def _run_trials(
        self,
        *,
        candidates: Sequence[ScoredItem],
        user_history: torch.Tensor,
        target_item: int,
        recommender: RecommenderWrapper,
        config: FaithfulnessConfig,
        metric_name: str,
        candidates_label: str,
        update_value: float,
    ) -> tuple[EvaluationResult, dict[int, float]]:
        """Run evaluations with the given candidate set."""
        min_limit = max(0, config.n_interactions_min_limit)

        if not candidates:
            return (
                self._build_nan_result(f"No {candidates_label} items available for {metric_name}."),
                {},
            )

        if len(candidates) < min_limit:
            return (
                self._build_nan_result(
                    f"Insufficient {candidates_label} items ({len(candidates)} < {min_limit}) for {metric_name}."
                ),
                {},
            )

        trial_scores = self._compute_trial_scores(
            candidates=candidates,
            user_history=user_history,
            target_item=target_item,
            recommender=recommender,
            update_value=update_value,
            n_trials=config.n_faithfulness_trials,
            n_samples=config.n_faithfulness_samples,
            rng_seed=self._build_rng_seed(
                target_item=target_item,
                n_candidates=len(candidates),
                update_value=update_value,
            ),
        )

        scores = [score for score in trial_scores if math.isfinite(score)]
        if not scores:
            return self._build_nan_result(f"No valid scores computed for {metric_name}."), {}

        median_score = statistics.median(scores)
        result = self._build_success_result(
            median_score=median_score,
            n_candidates=len(candidates),
            n_trials=config.n_faithfulness_trials,
            n_evaluated=len(scores),
            n_samples_per_trial=config.n_faithfulness_samples,
            metric_name=metric_name,
        )
        details: dict[str, object] = dict(result.details or {})
        details["trial_scores"] = list(trial_scores)
        result.details = details
        return result, {}

    def _evaluate_with_update(
        self,
        *,
        scored_items: Sequence[ScoredItem],
        user_history: torch.Tensor,
        target_item: int,
        recommender: RecommenderWrapper,
        config: FaithfulnessConfig,
        cfx_item_ids: Sequence[int],
        metric_prefix: str,
        update_value: float,
    ) -> tuple[EvaluationResult, EvaluationResult]:
        """Compute regular and baseline results for a given update value."""
        del cfx_item_ids
        similar, dissimilar, nan_items = self._split_by_similarity(scored_items, config.match_threshold)

        regular_result, _ = self._run_trials(
            candidates=similar,
            user_history=user_history,
            target_item=target_item,
            recommender=recommender,
            config=config,
            metric_name=f"{metric_prefix}_regular",
            candidates_label="similar",
            update_value=update_value,
        )
        baseline_result, _ = self._run_trials(
            candidates=dissimilar,
            user_history=user_history,
            target_item=target_item,
            recommender=recommender,
            config=config,
            metric_name=f"{metric_prefix}_baseline",
            candidates_label="dissimilar",
            update_value=update_value,
        )

        self._attach_per_interaction_scores(
            regular_result,
            self._build_per_interaction_scores(similar),
        )
        self._attach_per_interaction_scores(
            baseline_result,
            self._build_per_interaction_scores([*dissimilar, *nan_items]),
        )
        return regular_result, baseline_result

    @staticmethod
    def _build_per_interaction_scores(
        items: Sequence[ScoredItem],
    ) -> list[dict[str, object]]:
        """Serialise scored items into per-interaction score dictionaries."""
        return [
            {
                "interaction": item.interaction_description,
                "item_id": item.item_id,
                "match_score": float(item.score),
            }
            for item in items
        ]

    @staticmethod
    def _attach_per_interaction_scores(
        result: EvaluationResult,
        per_interaction_scores: Sequence[Mapping[str, object]],
    ) -> None:
        """Attach per-interaction scores to an EvaluationResult details payload."""
        details: dict[str, object] = dict(result.details or {})
        details["per_interaction_scores"] = list(per_interaction_scores)
        result.details = details

    @staticmethod
    def _compute_trial_scores(
        *,
        candidates: Sequence[ScoredItem],
        user_history: torch.Tensor,
        target_item: int,
        recommender: RecommenderWrapper,
        update_value: float,
        n_trials: int,
        n_samples: int,
        rng_seed: int,
    ) -> list[float]:
        """Compute recommender scores for trial-based history updates."""
        history_size = int(user_history.numel())
        valid_item_ids = [item.item_id for item in candidates if 0 <= item.item_id < history_size]
        if not valid_item_ids or n_trials <= 0:
            return []

        sample_size = max(0, n_samples)
        if sample_size >= len(valid_item_ids):
            sampled_sets = [list(valid_item_ids) for _ in range(n_trials)]
        else:
            rng = random.Random(rng_seed)  # noqa: S311
            sampled_sets = [rng.sample(valid_item_ids, sample_size) for _ in range(n_trials)]

        scores: list[float] = []
        for sampled_ids in sampled_sets:
            modified_history = user_history.clone()
            for item_id in sampled_ids:
                modified_history[item_id] = update_value
            new_score = recommender.get_item_score(modified_history, target_item)
            scores.append(float(new_score) if math.isfinite(new_score) else float("nan"))
        return scores

    @staticmethod
    def _build_rng_seed(*, target_item: int, n_candidates: int, update_value: float) -> int:
        """Build a deterministic seed for trial sampling."""
        return int(target_item + n_candidates * 1000 + update_value * 10)


class FaithfulnessRemovalEvaluator(BaseFaithfulnessEvaluator):
    """Evaluator for faithfulness_removal and faithfulness_removal_baseline metrics."""

    def __init__(self) -> None:
        """Configure faithfulness removal evaluation parameters."""
        super().__init__(metric_name="faithfulness_removal")

    def compute_results_from_scores(
        self,
        *,
        scored_items: Sequence[ScoredItem],
        user_history: torch.Tensor,
        target_item: int,
        recommender: RecommenderWrapper,
        config: FaithfulnessConfig,
        cfx_item_ids: Sequence[int] = (),
    ) -> tuple[EvaluationResult, EvaluationResult]:
        """Run removal trials using pre-computed scores. Returns (regular, baseline)."""
        return self._evaluate_with_update(
            scored_items=scored_items,
            user_history=user_history,
            target_item=target_item,
            recommender=recommender,
            config=config,
            cfx_item_ids=cfx_item_ids,
            metric_prefix="faithfulness_removal",
            update_value=0.0,
        )


class FaithfulnessReplacementEvaluator(BaseFaithfulnessEvaluator):
    """Evaluator for faithfulness_replacement and faithfulness_replacement_baseline metrics."""

    def __init__(self) -> None:
        """Configure faithfulness replacement evaluation parameters."""
        super().__init__(metric_name="faithfulness_replacement")

    def compute_results_from_scores(
        self,
        *,
        scored_items: Sequence[ScoredItem],
        user_history: torch.Tensor,
        target_item: int,
        recommender: RecommenderWrapper,
        config: FaithfulnessConfig,
        cfx_item_ids: Sequence[int] = (),
    ) -> tuple[EvaluationResult, EvaluationResult]:
        """Run replacement trials using pre-computed scores. Returns (regular, baseline)."""
        return self._evaluate_with_update(
            scored_items=scored_items,
            user_history=user_history,
            target_item=target_item,
            recommender=recommender,
            config=config,
            cfx_item_ids=cfx_item_ids,
            metric_prefix="faithfulness_replacement",
            update_value=1.0,
        )


def build_scored_items_from_results(
    interactions: pd.DataFrame,
    parsed_results: Sequence[Mapping[str, object]],
) -> list[ScoredItem]:
    """Build ScoredItem list from interaction DataFrame and parsed LLM results."""
    scored_items: list[ScoredItem] = []

    if "movie_id" not in interactions.columns:
        return scored_items

    movie_ids = interactions["movie_id"].tolist()

    for idx, result in enumerate(parsed_results):
        if idx >= len(movie_ids):
            break

        item_id = int(movie_ids[idx])
        raw_score = result.get("score", float("nan"))
        score = float(raw_score) if isinstance(raw_score, (int, float)) else float("nan")
        interaction_desc = str(result.get("interaction", ""))

        scored_items.append(
            ScoredItem(
                item_id=item_id,
                score=score,
                interaction_description=interaction_desc,
            )
        )

    return scored_items
