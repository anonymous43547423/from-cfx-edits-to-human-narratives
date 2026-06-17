"""Helpers for exporting pipeline artefacts as user-level datasets."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence, cast

import pandas as pd

from recsys_nle.nl_explanations.evaluation.readability import (
    READABILITY_SUBSCORE_KEYS,
)
from recsys_nle.nl_explanations.evaluation.readability import (
    extract_subscore as _readability_subscore,
)
from recsys_nle.nl_explanations.payloads import _movie_metadata_index
from recsys_nle.pipeline.distance_metrics import _DISTANCE_METRIC_KEYS
from recsys_nle.pipeline.metrics import safe_score
from recsys_nle.pipeline.run_summary import _compute_faithfulness_pvalue_complement

if TYPE_CHECKING:
    from recsys_nle.nl_explanations.results import NaturalLanguageExplanationResult
    from recsys_nle.pipeline.config import OutputConfig
    from recsys_nle.pipeline.workflow import PipelineResult


_INTERACTION_ID_PATTERN = re.compile(r"(?:id|movie_id)=(\d+)")


def _extract_movie_id_from_interaction_label(text: str) -> int | None:
    """Parse a movie identifier from an interaction description label when present."""
    match = _INTERACTION_ID_PATTERN.search(text)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _coerce_int(value: object) -> int | None:
    """Convert an input to int when possible."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


@dataclass(slots=True)
class DatasetsExporter:
    """Persist pipeline results as filtered user-level datasets."""

    output_config: OutputConfig

    def export(self, pipeline_result: PipelineResult) -> Path | None:
        """Write user, item, interaction, and faithfulness datasets to disk."""
        output_base = self.output_config.output_datasets_path
        if output_base is None:
            # No output directory requested; nothing to do.
            return None

        base_path = Path(output_base)
        base_path.mkdir(parents=True, exist_ok=True)

        if self.output_config.create_output_datasets_subdirectory:
            timestamp = datetime.now(UTC).isoformat(timespec="seconds")
            run_dir = base_path / timestamp
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = base_path

        sampled_user_ids = sorted(set(self._resolve_sampled_users(pipeline_result)))

        users = self._build_users_frame(sampled_user_ids)
        recommendations = self._build_recommendations_frame(pipeline_result.recommendations, sampled_user_ids)
        interactions = self._build_interactions_frame(
            pipeline_result.all_interactions,
            sampled_user_ids,
        )
        items, item_genres = self._build_item_frames(recommendations, interactions)
        generation = self._build_generation_frame(pipeline_result.explanations, sampled_user_ids)
        evaluation = self._build_evaluation_frame(
            pipeline_result.explanations,
            sampled_user_ids,
            pipeline_result.distance_metrics_by_user,
        )
        cfx_match_details = self._build_cfx_match_details_frame(
            pipeline_result.explanations,
            sampled_user_ids,
            interactions,
        )
        non_cfx_match_details = self._build_non_cfx_match_details_frame(
            pipeline_result.explanations,
            sampled_user_ids,
            interactions,
        )
        faithfulness_removal = self._build_faithfulness_details_frame(
            pipeline_result.explanations,
            sampled_user_ids,
            interactions,
            explanation_attr="faithfulness_removal",
        )
        faithfulness_removal_baseline = self._build_faithfulness_details_frame(
            pipeline_result.explanations,
            sampled_user_ids,
            interactions,
            explanation_attr="faithfulness_removal_baseline",
        )
        faithfulness_replacement = self._build_faithfulness_details_frame(
            pipeline_result.explanations,
            sampled_user_ids,
            interactions,
            explanation_attr="faithfulness_replacement",
        )
        faithfulness_replacement_baseline = self._build_faithfulness_details_frame(
            pipeline_result.explanations,
            sampled_user_ids,
            interactions,
            explanation_attr="faithfulness_replacement_baseline",
        )
        faithfulness_removal_trials = self._build_faithfulness_trials_frame(
            pipeline_result.explanations,
            sampled_user_ids,
            explanation_attr="faithfulness_removal",
        )
        faithfulness_removal_baseline_trials = self._build_faithfulness_trials_frame(
            pipeline_result.explanations,
            sampled_user_ids,
            explanation_attr="faithfulness_removal_baseline",
        )
        faithfulness_replacement_trials = self._build_faithfulness_trials_frame(
            pipeline_result.explanations,
            sampled_user_ids,
            explanation_attr="faithfulness_replacement",
        )
        faithfulness_replacement_baseline_trials = self._build_faithfulness_trials_frame(
            pipeline_result.explanations,
            sampled_user_ids,
            explanation_attr="faithfulness_replacement_baseline",
        )

        users.to_feather(run_dir / "users.feather")
        items.to_feather(run_dir / "items.feather")
        item_genres.to_feather(run_dir / "item_genres.feather")
        interactions.to_feather(run_dir / "interactions.feather")
        recommendations.to_feather(run_dir / "recommendations.feather")
        generation.to_feather(run_dir / "generation.feather")
        evaluation.to_feather(run_dir / "evaluation.feather")
        cfx_match_details.to_feather(run_dir / "cfx_match_details.feather")
        non_cfx_match_details.to_feather(run_dir / "non_cfx_match_details.feather")
        faithfulness_removal.to_feather(run_dir / "faithfulness_removal.feather")
        faithfulness_removal_baseline.to_feather(run_dir / "faithfulness_removal_baseline.feather")
        faithfulness_replacement.to_feather(run_dir / "faithfulness_replacement.feather")
        faithfulness_replacement_baseline.to_feather(run_dir / "faithfulness_replacement_baseline.feather")
        faithfulness_removal_trials.to_feather(run_dir / "faithfulness_removal_trials.feather")
        faithfulness_removal_baseline_trials.to_feather(run_dir / "faithfulness_removal_baseline_trials.feather")
        faithfulness_replacement_trials.to_feather(run_dir / "faithfulness_replacement_trials.feather")
        faithfulness_replacement_baseline_trials.to_feather(
            run_dir / "faithfulness_replacement_baseline_trials.feather"
        )

        return run_dir

    def _resolve_sampled_users(self, pipeline_result: PipelineResult) -> list[int]:
        """Infer which users the pipeline has been run for."""
        if pipeline_result.sampled_user_ids:
            return list(pipeline_result.sampled_user_ids)
        explanations = pipeline_result.explanations
        return [int(record["user_id"]) for record in explanations.dataset]

    def _build_users_frame(self, user_ids: Sequence[int]) -> pd.DataFrame:
        """Create a users dataframe with user ids."""
        return pd.DataFrame(user_ids, columns=["user_id"])

    def _build_recommendations_frame(
        self,
        recommendations: pd.DataFrame,
        user_ids: Sequence[int],
    ) -> pd.DataFrame:
        """Create a recommendations dataframe filtered to sampled users."""
        if recommendations.empty:
            return pd.DataFrame(columns=["user_id", "item_id", "rank", "score"])

        frame = recommendations.copy()
        user_ids_set = set(user_ids)
        frame = frame[frame["user_id"].isin(user_ids_set)]
        if frame.empty:
            return pd.DataFrame(columns=["user_id", "item_id", "rank", "score"])

        if "movie_id" in frame.columns:
            frame = frame.rename(columns={"movie_id": "item_id"})

        # Ensure ranks are available and stable per user.
        if "rank" not in frame.columns:
            frame = frame.sort_values(by=["user_id", "score"], ascending=[True, False])
            frame["rank"] = frame.groupby("user_id").cumcount() + 1

        ordered = frame[["user_id", "item_id", "rank", "score"]].copy()
        return ordered.reset_index(drop=True)

    def _build_interactions_frame(
        self,
        all_interactions: pd.DataFrame,
        user_ids: Sequence[int],
    ) -> pd.DataFrame:
        """Create interaction dataset for sampled users."""
        columns = ["interaction_id", "user_id", "item_id", "rating", "attribution_score", "is_counterfactual"]
        if all_interactions.empty:
            return pd.DataFrame(columns=columns)

        frame = all_interactions.copy()
        if "user_id" not in frame.columns:
            return pd.DataFrame(columns=columns)

        user_ids_set = set(user_ids)
        frame = frame[frame["user_id"].isin(user_ids_set)]
        if frame.empty:
            return pd.DataFrame(columns=columns)

        frame = frame.reset_index(drop=True)
        # Assign new interaction_id
        frame["interaction_id"] = frame.index.astype(int)

        return frame[columns].copy()

    def _build_item_frames(  # noqa: C901, PLR0912
        self,
        recommendations: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create item and item-genre datasets restricted to referenced items."""
        item_ids: set[int] = set()
        for frame in (recommendations, interactions):
            if "item_id" in frame.columns:
                for value in frame["item_id"].dropna().tolist():
                    try:
                        item_ids.add(int(value))
                    except (TypeError, ValueError):
                        continue

        if not item_ids:
            empty_items = pd.DataFrame(columns=["item_id", "title", "year"])
            empty_genres = pd.DataFrame(columns=["item_id", "genre"])
            return empty_items, empty_genres

        metadata_index: Mapping[int, Mapping[str, object]] = _movie_metadata_index()

        item_rows: list[dict[str, object]] = []
        genre_rows: list[dict[str, object]] = []
        for item_id in sorted(item_ids):
            meta = metadata_index.get(item_id, {})
            title = str(meta.get("movie_title") or meta.get("title") or "")
            year_value = meta.get("year")
            year: int | None
            if isinstance(year_value, (int, float, str)):
                try:
                    year = int(year_value)
                except (TypeError, ValueError):
                    year = None
            else:
                year = None

            item_rows.append({"item_id": item_id, "title": title, "year": year})

            raw_genres = meta.get("genres")
            if isinstance(raw_genres, Sequence) and not isinstance(raw_genres, str):
                genres_iterable = raw_genres
            elif isinstance(raw_genres, str):
                genres_iterable = [part.strip() for part in raw_genres.split("|") if part.strip()]
            else:
                genres_iterable = []

            for genre in genres_iterable:
                text = str(genre).strip()
                if not text:
                    continue
                genre_rows.append({"item_id": item_id, "genre": text})

        items = pd.DataFrame(item_rows, columns=["item_id", "title", "year"])
        item_genres = pd.DataFrame(genre_rows, columns=["item_id", "genre"])
        return items, item_genres

    def _build_generation_frame(
        self,
        explanations: object,
        user_ids: Sequence[int],
    ) -> pd.DataFrame:
        """Summarise explanation generation outputs per user."""
        columns = [
            "user_id",
            "reasoning_enabled",
            "reasoning_text",
            "explanation_text",
            "explanation_confidence",
            "explanation_conversation",
        ]
        if explanations is None or not hasattr(explanations, "results_by_user"):
            return pd.DataFrame(columns=columns)

        results_by_user: Mapping[int, NaturalLanguageExplanationResult] = explanations.results_by_user
        user_ids_set = set(user_ids)

        rows: list[dict[str, object]] = []
        for user_id, result in results_by_user.items():
            if user_id not in user_ids_set:
                continue

            reasoning_enabled = bool(result.reasoning)
            reasoning_text: str | None = result.reasoning if reasoning_enabled else None

            explanation_conversation_json = ""
            if result.explanation_conversation:
                explanation_conversation_json = json.dumps(result.explanation_conversation)

            rows.append(
                {
                    "user_id": user_id,
                    "reasoning_enabled": reasoning_enabled,
                    "reasoning_text": reasoning_text,
                    "explanation_text": result.explanation,
                    "explanation_confidence": result.explanation_confidence,
                    "explanation_conversation": explanation_conversation_json,
                }
            )

        if not rows:
            return pd.DataFrame(columns=columns)

        return pd.DataFrame(rows, columns=columns)

    def _build_evaluation_frame(
        self,
        explanations: object,
        user_ids: Sequence[int],
        distance_metrics_by_user: Mapping[int, Mapping[str, float]],
    ) -> pd.DataFrame:
        """Summarise overall evaluation scores per user."""
        distance_metric_columns = list(_DISTANCE_METRIC_KEYS)
        readability_mean_columns = [f"readability_{key}_mean" for key in READABILITY_SUBSCORE_KEYS]
        columns = [
            "user_id",
            "explanation_plausibility",
            *readability_mean_columns,
            "readability_overall_mean",
            "explanation_cfx_pattern_match_mean",
            "explanation_cfx_pattern_match_success_rate",
            "explanation_non_cfx_pattern_match_mean",
            "explanation_non_cfx_pattern_match_success_rate",
            "overall_faithfulness_removal_score",
            "overall_faithfulness_removal_baseline_score",
            "overall_faithfulness_replacement_score",
            "overall_faithfulness_replacement_baseline_score",
            "faithfulness_removal_pvalue_complement",
            "faithfulness_replacement_pvalue_complement",
            *distance_metric_columns,
        ]
        if explanations is None or not hasattr(explanations, "results_by_user"):
            return pd.DataFrame(columns=columns)

        results_by_user: Mapping[int, NaturalLanguageExplanationResult] = explanations.results_by_user
        user_ids_set = set(user_ids)
        default_distance_metrics = dict.fromkeys(distance_metric_columns, math.nan)

        rows: list[dict[str, object]] = []
        for user_id, result in results_by_user.items():
            if user_id not in user_ids_set:
                continue

            cfx_score = safe_score(result.explanation_cfx_match)
            non_cfx_score = safe_score(result.explanation_non_cfx_match)
            distance_metrics = distance_metrics_by_user.get(user_id, default_distance_metrics)
            row_eval: dict[str, object] = {
                "user_id": user_id,
                "explanation_plausibility": safe_score(result.explanation_plausibility),
            }
            for key in READABILITY_SUBSCORE_KEYS:
                row_eval[f"readability_{key}_mean"] = _readability_subscore(
                    result.explanation_readability,
                    key,
                )
            row_eval["readability_overall_mean"] = safe_score(result.explanation_readability)
            rows.append(
                {
                    **row_eval,
                    "explanation_cfx_pattern_match_mean": cfx_score,
                    "explanation_cfx_pattern_match_success_rate": 1.0 if math.isfinite(cfx_score) else 0.0,
                    "explanation_non_cfx_pattern_match_mean": non_cfx_score,
                    "explanation_non_cfx_pattern_match_success_rate": 1.0 if math.isfinite(non_cfx_score) else 0.0,
                    "overall_faithfulness_removal_score": safe_score(result.faithfulness_removal),
                    "overall_faithfulness_removal_baseline_score": safe_score(result.faithfulness_removal_baseline),
                    "overall_faithfulness_replacement_score": safe_score(result.faithfulness_replacement),
                    "overall_faithfulness_replacement_baseline_score": safe_score(
                        result.faithfulness_replacement_baseline
                    ),
                    "faithfulness_removal_pvalue_complement": _compute_faithfulness_pvalue_complement(
                        result.faithfulness_removal,
                        result.faithfulness_removal_baseline,
                        alternative="less",
                    ),
                    "faithfulness_replacement_pvalue_complement": _compute_faithfulness_pvalue_complement(
                        result.faithfulness_replacement,
                        result.faithfulness_replacement_baseline,
                        alternative="greater",
                    ),
                    **distance_metrics,
                }
            )

        if not rows:
            return pd.DataFrame(columns=columns)

        return pd.DataFrame(rows, columns=columns)

    def _build_cfx_match_details_frame(
        self,
        explanations: object,
        user_ids: Sequence[int],
        interactions: pd.DataFrame,
    ) -> pd.DataFrame:
        """Construct per-attribution CFX match scores joined to interaction identifiers."""
        return self._build_match_details_frame(
            explanations,
            user_ids,
            interactions,
            explanation_attr="explanation_cfx_match",
        )

    def _build_non_cfx_match_details_frame(
        self,
        explanations: object,
        user_ids: Sequence[int],
        interactions: pd.DataFrame,
    ) -> pd.DataFrame:
        """Construct per-interaction non-CFX match scores."""
        return self._build_match_details_frame(
            explanations,
            user_ids,
            interactions,
            explanation_attr="explanation_non_cfx_match",
        )

    def _build_match_details_frame(  # noqa: C901, PLR0912
        self,
        explanations: object,
        user_ids: Sequence[int],
        interactions: pd.DataFrame,
        *,
        explanation_attr: str,
    ) -> pd.DataFrame:
        """Construct per-interaction match scores for specified evaluation attributes."""
        columns = ["user_id", "interaction_id", "score", "judgment"]
        if explanations is None or not hasattr(explanations, "results_by_user"):
            return pd.DataFrame(columns=columns)

        if interactions.empty:
            return pd.DataFrame(columns=columns)

        results_by_user: Mapping[int, NaturalLanguageExplanationResult] = explanations.results_by_user
        user_ids_set = set(user_ids)

        # Build a lookup from (user_id, item_id) to interaction row identifier.
        interaction_key_to_id: dict[tuple[int, int], int] = {}
        for row in interactions.itertuples(index=False):
            if isinstance(row.user_id, int) and isinstance(row.item_id, int):
                interaction_key_to_id[(row.user_id, row.item_id)] = cast("int", row.interaction_id)

        rows: list[dict[str, object]] = []
        for user_id, result in results_by_user.items():
            if user_id not in user_ids_set:
                continue

            evaluation = getattr(result, explanation_attr, None)
            if evaluation is None or not hasattr(evaluation, "details"):
                continue

            details = cast("Any", evaluation).details or {}
            raw_entries = details.get("per_interaction_scores")
            if not isinstance(raw_entries, Sequence):
                continue

            judgment_overall = getattr(evaluation, "judgment", "")

            for entry in raw_entries:
                if not isinstance(entry, Mapping):
                    continue
                interaction_label = str(entry.get("interaction", "")).strip()
                if not interaction_label:
                    continue

                score_raw = entry.get("score")
                try:
                    score = float(score_raw) if score_raw is not None else float("nan")
                except (TypeError, ValueError):
                    score = float("nan")

                judgment_raw = entry.get("judgment", "")
                judgment = str(judgment_raw).strip() if isinstance(judgment_raw, str) else ""
                if not judgment:
                    judgment = judgment_overall

                movie_id = _coerce_int(entry.get("item_id"))
                if movie_id is None:
                    movie_id = _extract_movie_id_from_interaction_label(interaction_label)
                interaction_id = interaction_key_to_id.get((user_id, movie_id)) if movie_id is not None else None

                rows.append(
                    {
                        "user_id": user_id,
                        "interaction_id": interaction_id,
                        "score": score,
                        "judgment": judgment,
                    }
                )

        if not rows:
            return pd.DataFrame(columns=columns)

        return pd.DataFrame(rows, columns=columns)

    def _build_faithfulness_details_frame(
        self,
        explanations: object,
        user_ids: Sequence[int],
        interactions: pd.DataFrame,
        *,
        explanation_attr: str,
    ) -> pd.DataFrame:
        """Construct per-interaction faithfulness scores."""
        columns = ["user_id", "interaction_id", "item_id", "match_score"]
        if explanations is None or not hasattr(explanations, "results_by_user"):
            return pd.DataFrame(columns=columns)

        if interactions.empty:
            return pd.DataFrame(columns=columns)

        results_by_user: Mapping[int, NaturalLanguageExplanationResult] = explanations.results_by_user
        user_ids_set = set(user_ids)

        interaction_key_to_id = self._build_interaction_id_lookup(interactions)
        rows = self._collect_faithfulness_rows(
            results_by_user,
            user_ids_set,
            interaction_key_to_id,
            explanation_attr,
        )
        if not rows:
            return pd.DataFrame(columns=columns)

        return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def _build_interaction_id_lookup(interactions: pd.DataFrame) -> dict[tuple[int, int], int]:
        """Build a lookup from (user_id, item_id) to interaction_id."""
        interaction_key_to_id: dict[tuple[int, int], int] = {}
        for row in interactions.itertuples(index=False):
            if isinstance(row.user_id, int) and isinstance(row.item_id, int):
                interaction_key_to_id[(row.user_id, row.item_id)] = cast("int", row.interaction_id)
        return interaction_key_to_id

    def _collect_faithfulness_rows(
        self,
        results_by_user: Mapping[int, NaturalLanguageExplanationResult],
        user_ids_set: set[int],
        interaction_key_to_id: Mapping[tuple[int, int], int],
        explanation_attr: str,
    ) -> list[dict[str, object]]:
        """Collect faithfulness rows for the requested evaluation attribute."""
        rows: list[dict[str, object]] = []
        for user_id, result in results_by_user.items():
            if user_id not in user_ids_set:
                continue

            evaluation = getattr(result, explanation_attr, None)
            entries = self._extract_per_interaction_entries(evaluation)
            if not entries:
                continue

            for entry in entries:
                row = self._build_faithfulness_row(user_id, entry, interaction_key_to_id)
                if row is not None:
                    rows.append(row)
        return rows

    @staticmethod
    def _extract_per_interaction_entries(
        evaluation: object,
    ) -> list[Mapping[str, object]]:
        """Extract per-interaction score entries from an evaluation result."""
        if evaluation is None or not hasattr(evaluation, "details"):
            return []
        details = cast("Any", evaluation).details or {}
        raw_entries = details.get("per_interaction_scores")
        if not isinstance(raw_entries, Sequence):
            return []
        return [entry for entry in raw_entries if isinstance(entry, Mapping)]

    @staticmethod
    def _build_faithfulness_row(
        user_id: int,
        entry: Mapping[str, object],
        interaction_key_to_id: Mapping[tuple[int, int], int],
    ) -> dict[str, object] | None:
        """Build a faithfulness row from a per-interaction entry."""
        interaction_label = str(entry.get("interaction", "")).strip()
        if not interaction_label:
            return None

        item_id = _coerce_int(entry.get("item_id"))
        if item_id is None:
            item_id = _extract_movie_id_from_interaction_label(interaction_label)

        match_score_raw = entry.get("match_score")
        match_score = float(cast("Any", match_score_raw)) if match_score_raw is not None else float("nan")

        interaction_id = interaction_key_to_id.get((user_id, item_id)) if item_id is not None else None

        return {
            "user_id": user_id,
            "interaction_id": interaction_id,
            "item_id": item_id,
            "match_score": match_score,
        }

    def _build_faithfulness_trials_frame(
        self,
        explanations: object,
        user_ids: Sequence[int],
        *,
        explanation_attr: str,
    ) -> pd.DataFrame:
        """Construct per-trial faithfulness scores."""
        columns = ["user_id", "trial_no", "score"]
        if explanations is None or not hasattr(explanations, "results_by_user"):
            return pd.DataFrame(columns=columns)

        results_by_user: Mapping[int, NaturalLanguageExplanationResult] = explanations.results_by_user
        user_ids_set = set(user_ids)

        rows: list[dict[str, object]] = []
        for user_id, result in results_by_user.items():
            if user_id not in user_ids_set:
                continue

            evaluation = getattr(result, explanation_attr, None)
            if evaluation is None or not hasattr(evaluation, "details"):
                continue

            details = cast("Any", evaluation).details or {}
            trial_scores = details.get("trial_scores")
            if not isinstance(trial_scores, Sequence):
                continue

            for idx, score_raw in enumerate(trial_scores):
                try:
                    score = float(score_raw) if score_raw is not None else float("nan")
                except (TypeError, ValueError):
                    score = float("nan")
                rows.append({"user_id": user_id, "trial_no": idx, "score": score})

        if not rows:
            return pd.DataFrame(columns=columns)

        return pd.DataFrame(rows, columns=columns)
