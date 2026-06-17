"""Logging helpers for presenting pipeline outputs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Mapping, Sequence

import pandas as pd

from recsys_nle.core.logging_utils import log_frames
from recsys_nle.pipeline.workflow import load_movies_metadata

if TYPE_CHECKING:
    from pandas import DataFrame

    from recsys_nle.core.attribution import UserAttribution
    from recsys_nle.core.movielens import MovielensArtifacts
    from recsys_nle.nl_explanations.evaluation.base import EvaluationResult
    from recsys_nle.nl_explanations.results import NaturalLanguageExplanationResult
    from recsys_nle.pipeline.config import PipelineConfig
    from recsys_nle.pipeline.workflow import PipelineResult


class PipelineReporter:
    """High-level logger for pipeline artefacts."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialise the reporter with an optional logger."""
        self._logger = logger or logging.getLogger(__name__)
        self._artifacts: MovielensArtifacts | None = None

    def render(
        self,
        *,
        pipeline_result: PipelineResult,
        config: PipelineConfig,
    ) -> None:
        """Emit a representative summary of pipeline results."""
        # Sample a subset of users for concise logging.
        all_user_ids = pipeline_result.recommendations["user_id"].unique().tolist()
        sampled_users: list[int]
        if not all_user_ids or config.sample_user_count <= 0:
            sampled_users = []
        else:
            sampled_users = [int(uid) for uid in all_user_ids[: config.sample_user_count]]

        self.log_recommendation_samples(
            sampled_users=sampled_users,
            recommendations=pipeline_result.recommendations,
            top_k=min(5, config.recommendation.top_k),
            user_attributions=pipeline_result.user_attributions,
            user_nle_results=(pipeline_result.explanations.results_by_user if pipeline_result.explanations else None),
            show_prompts=config.show_prompts,
        )

    def log_recommendation_samples(
        self,
        *,
        sampled_users: Sequence[int],
        recommendations: DataFrame,
        top_k: int,
        user_attributions: Mapping[int, UserAttribution] | None,
        user_nle_results: Mapping[int, NaturalLanguageExplanationResult] | None,
        show_prompts: bool = False,
    ) -> None:
        """Log recommendation, attribution, and explanation samples for users."""
        target_users: list[int] = []
        seen_users: set[int] = set()

        detail_candidates: list[int] = []
        if user_attributions:
            detail_candidates.extend(int(user_id) for user_id in user_attributions)
        if user_nle_results:
            detail_candidates.extend(int(user_id) for user_id in user_nle_results)

        for user_id in detail_candidates:
            if user_id in seen_users:
                continue
            seen_users.add(user_id)
            target_users.append(user_id)

        for user_id in sampled_users:
            if user_id in seen_users:
                continue
            seen_users.add(user_id)
            target_users.append(user_id)
        if not target_users:
            self._logger.warning(
                "No users available to display recommendation samples",
            )
            return

        target_users = target_users[:5]

        for user_id in target_users:
            self._log_user_snapshot(
                user_id=user_id,
                recommendations=recommendations,
                top_k=top_k,
                user_attributions=user_attributions,
                user_nle_results=user_nle_results,
                show_prompts=show_prompts,
            )

    def _log_user_snapshot(
        self,
        *,
        user_id: int,
        recommendations: DataFrame,
        top_k: int,
        user_attributions: Mapping[int, UserAttribution] | None,
        user_nle_results: Mapping[int, NaturalLanguageExplanationResult] | None,
        show_prompts: bool,
    ) -> None:
        """Log recommendations, attributions, and explanations for a single user."""
        self._logger.info("%s", "═" * 80)

        display_label = f"User {user_id}"

        user_recommendations = _collect_user_recommendations(
            recommendations,
            user_id=user_id,
            top_k=top_k,
        )

        artifacts = _ensure_artifacts(self)
        if artifacts is not None and not user_recommendations.empty:
            user_recommendations = _with_titles(artifacts, user_recommendations)

        frames: list[tuple[str, DataFrame, list[str]]] = [
            (
                f"{display_label} top-{top_k} recommendations",
                user_recommendations,
                _select_recommendation_columns(user_recommendations),
            ),
        ]

        if user_attributions is not None:
            attribution = user_attributions.get(user_id)
            if attribution is not None:
                interactions = attribution.cfx_interactions
                if not interactions.empty:
                    if artifacts is not None:
                        interactions = _with_titles(artifacts, interactions)
                    frames.append(
                        (
                            f"{display_label} influential interactions",
                            interactions,
                            _select_interaction_columns(interactions),
                        )
                    )

        log_frames(self._logger, frames)

        if user_nle_results is None:
            return

        result = user_nle_results.get(user_id)
        if result is None:
            return
        self._log_nle_details(header=display_label, result=result, show_prompts=show_prompts)

    def _log_nle_details(
        self,
        *,
        header: str,
        result: NaturalLanguageExplanationResult,
        show_prompts: bool,
    ) -> None:
        """Log detailed reasoning, explanations, and evaluation summaries."""
        summary_frame, summary_warnings = _build_evaluation_summary_frame(
            result,
            include_prompt=show_prompts,
        )
        claims_frame, claim_warnings = _build_correctness_claims_frame(
            result,
            include_prompt=show_prompts,
        )
        extraction_frame = _build_correctness_extraction_frame(result) if show_prompts else pd.DataFrame()
        faithfulness_frame, faithfulness_warnings = _build_nle_cf_faithfulness_frame(
            result,
            include_prompt=show_prompts,
        )
        specificity_frame, specificity_warnings = _build_nle_cf_specificity_frame(
            result,
            include_prompt=show_prompts,
        )
        warnings: list[str] = [*summary_warnings, *claim_warnings, *faithfulness_warnings, *specificity_warnings]

        text_frame, text_columns = _build_text_frame(
            result,
            include_prompt=show_prompts,
        )

        interaction_columns = (
            ["prompt", "interaction", "judgment", "score"] if show_prompts else ["interaction", "judgment", "score"]
        )

        frames: list[tuple[str, DataFrame, list[str]]] = [
            (f"{header} reasoning and explanation", text_frame, text_columns),
            (
                f"{header} explanation scores",
                summary_frame,
                ["metric", "prompt", "score", "judgment"] if show_prompts else ["metric", "score", "judgment"],
            ),
        ]
        if not claims_frame.empty:
            frames.append(
                (
                    f"{header} correctness claims",
                    claims_frame,
                    ["prompt", "claim", "judgment", "score"] if show_prompts else ["claim", "judgment", "score"],
                )
            )
        if show_prompts and not extraction_frame.empty:
            frames.append(
                (
                    f"{header} extracted correctness claims",
                    extraction_frame,
                    ["prompt", "claims"],
                )
            )
        if not faithfulness_frame.empty:
            frames.append(
                (
                    f"{header} NLE-CF faithfulness results",
                    faithfulness_frame,
                    interaction_columns,
                )
            )
        if not specificity_frame.empty:
            frames.append(
                (
                    f"{header} NLE-CF specificity results",
                    specificity_frame,
                    interaction_columns,
                )
            )

        if warnings:
            for message in warnings:
                self._logger.warning("%s", message)

        log_frames(self._logger, frames)


def _collect_user_recommendations(
    recommendations: DataFrame,
    *,
    user_id: int,
    top_k: int | None,
) -> DataFrame:
    """Select and order recommendations for a target user."""
    frame = recommendations[recommendations["user_id"] == user_id]
    if frame.empty:
        return frame.copy()

    ordered = frame.copy()
    if "rank" in ordered.columns:
        ordered = ordered.sort_values(by="rank", ascending=True)
    elif "score" in ordered.columns:
        ordered = ordered.sort_values(by="score", ascending=False)
    else:
        ordered = ordered.sort_values(by="movie_id", ascending=True)
    if top_k is not None:
        ordered = ordered.head(top_k)
    return ordered.reset_index(drop=True)


def _select_title_column(frame: DataFrame) -> str | None:
    """Choose the best available title-like column for display."""
    for candidate in ("movie_title", "title"):
        if candidate in frame.columns:
            return candidate
    return None


def _select_recommendation_columns(frame: DataFrame) -> list[str]:
    """Choose a compact set of columns for recommendation display."""
    columns: list[str] = []
    title_column = _select_title_column(frame)
    if title_column is not None:
        columns.append(title_column)
    elif "movie_id" in frame.columns:
        columns.append("movie_id")
    columns.extend(name for name in ("score", "rank") if name in frame.columns)
    return columns or frame.columns.tolist()


def _select_interaction_columns(frame: DataFrame) -> list[str]:
    """Choose a compact set of columns for interaction display."""
    columns: list[str] = []
    title_column = _select_title_column(frame)
    if title_column is not None:
        columns.append(title_column)
    elif "movie_id" in frame.columns:
        columns.append("movie_id")
    columns.extend(name for name in ("rating", "weight", "importance") if name in frame.columns)
    return columns or frame.columns.tolist()


def _with_titles(
    artifacts: MovielensArtifacts,
    frame: DataFrame,
) -> DataFrame:
    """Attach human-readable titles to a ratings frame."""
    return artifacts.with_titles(frame)


def _ensure_artifacts(self: PipelineReporter) -> MovielensArtifacts | None:
    """Lazily load Movielens artifacts for attaching titles."""
    if self._artifacts is not None:
        return self._artifacts
    self._artifacts = load_movies_metadata()
    return self._artifacts


def _build_evaluation_summary_frame(
    result: NaturalLanguageExplanationResult,
    *,
    include_prompt: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a compact table summarising explanation scores."""
    metrics: list[tuple[str, EvaluationResult | None]] = [
        ("explanation_plausibility", result.explanation_plausibility),
        ("explanation_cfx_match", result.explanation_cfx_match),
        ("explanation_non_cfx_match", result.explanation_non_cfx_match),
    ]
    rows: list[dict[str, object]] = []
    for name, evaluation in metrics:
        if evaluation is None:
            continue
        row: dict[str, object] = {
            "metric": name,
            "score": f"{evaluation.score:.2f}",
            "judgment": evaluation.judgment,
        }
        if include_prompt:
            row["prompt"] = evaluation.prompt or ""
        rows.append(row)
    warnings: list[str] = []
    for name, evaluation in metrics:
        if evaluation is None:
            continue
        details = evaluation.details or {}
        raw_warnings = details.get("warnings")
        if isinstance(raw_warnings, list):
            for message in raw_warnings:
                text = str(message).strip()
                if text:
                    warnings.append(f"{name}: {text}")

    columns = ["metric", "prompt", "score", "judgment"] if include_prompt else ["metric", "score", "judgment"]
    return pd.DataFrame(rows, columns=columns), warnings


def _build_text_frame(
    result: NaturalLanguageExplanationResult,
    *,
    include_prompt: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """Construct a reasoning and explanation table with confidence values and parse errors."""
    text_rows: list[dict[str, object]] = [
        {"field": "reasoning", "value": result.reasoning, "confidence": float("nan")},
        {
            "field": "explanation",
            "value": result.explanation,
            "confidence": result.explanation_confidence,
        },
    ]

    if include_prompt:
        prompt_lookup = {
            "reasoning": result.reasoning_prompt or "",
            "explanation": result.explanation_prompt or "",
        }
        for row in text_rows:
            field_name = str(row.get("field", ""))
            row["prompt"] = prompt_lookup.get(field_name, "")
        text_frame = pd.DataFrame(text_rows, columns=["field", "prompt", "value", "confidence"])
        columns = ["field", "prompt", "value", "confidence"]
    else:
        text_frame = pd.DataFrame(text_rows, columns=["field", "value", "confidence"])
        columns = ["field", "value", "confidence"]

    return text_frame, columns


def _build_correctness_claims_frame(
    result: NaturalLanguageExplanationResult,
    *,
    include_prompt: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """Construct a table of per-claim correctness scores for explanations."""
    warnings: list[str] = []
    rows: list[dict[str, object]] = []
    evaluation = result.explanation_cfx_match
    aggregate_judgment = evaluation.judgment.strip()
    details = evaluation.details or {}
    raw_entries = details.get("per_claim_scores")
    if isinstance(raw_entries, list):
        for entry in raw_entries:
            row = _build_correctness_row(
                aggregate_judgment=aggregate_judgment,
                entry=entry,
                warnings=warnings,
            )
            if row is not None:
                if include_prompt:
                    row["prompt"] = evaluation.prompt or ""
                rows.append(row)

    if not rows:
        empty_columns = ["prompt", "claim", "judgment", "score"] if include_prompt else ["claim", "judgment", "score"]
        return pd.DataFrame(columns=empty_columns), warnings
    columns = ["prompt", "claim", "judgment", "score"] if include_prompt else ["claim", "judgment", "score"]
    return pd.DataFrame(rows, columns=columns), warnings


def _build_correctness_extraction_frame(
    result: NaturalLanguageExplanationResult,
) -> pd.DataFrame:
    """Construct a single-row summary of extracted correctness claims and prompts."""
    aggregated_claims: list[str] = []

    evaluation = result.explanation_cfx_match
    details = evaluation.details or {}
    extraction = details.get("claim_extraction")
    if not isinstance(extraction, Mapping):
        return pd.DataFrame(columns=["prompt", "claims"])

    prompt_value = str(extraction.get("prompt", "")).strip()

    raw_claims = extraction.get("claims")
    if isinstance(raw_claims, Sequence) and not isinstance(raw_claims, str):
        claim_iterable = raw_claims
    else:
        claim_iterable = [raw_claims]

    for claim in claim_iterable:
        claim_text = str(claim).strip()
        if claim_text:
            aggregated_claims.append(claim_text)

    unique_claims = list(dict.fromkeys(aggregated_claims))
    claims_value = ", ".join(unique_claims)

    row = {
        "prompt": prompt_value,
        "claims": claims_value,
    }
    return pd.DataFrame([row], columns=["prompt", "claims"])


def _build_nle_cf_faithfulness_frame(
    result: NaturalLanguageExplanationResult,
    *,
    include_prompt: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """Construct a table of per-attribution CFX match scores for explanations."""
    return _build_interaction_scoring_frame(
        evaluation=result.explanation_cfx_match,
        metric_name="CFX match",
        include_prompt=include_prompt,
    )


def _build_nle_cf_specificity_frame(
    result: NaturalLanguageExplanationResult,
    *,
    include_prompt: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """Construct a table of per-interaction non-CFX match scores for explanations."""
    return _build_interaction_scoring_frame(
        evaluation=result.explanation_non_cfx_match,
        metric_name="Non-CFX match",
        include_prompt=include_prompt,
    )


def _build_interaction_scoring_frame(
    *,
    evaluation: EvaluationResult | None,
    metric_name: str,
    include_prompt: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """Construct a table of per-interaction scores for a given metric."""
    warnings: list[str] = []
    rows: list[dict[str, object]] = []
    if evaluation is not None:
        aggregate_judgment = evaluation.judgment.strip()
        details = evaluation.details or {}
        raw_entries = details.get("per_interaction_scores")
        if isinstance(raw_entries, list):
            for entry in raw_entries:
                row = _build_interaction_scoring_row(
                    aggregate_judgment=aggregate_judgment,
                    entry=entry,
                    metric_name=metric_name,
                    warnings=warnings,
                )
                if row is not None:
                    if include_prompt:
                        row["prompt"] = evaluation.prompt or ""
                    rows.append(row)

    if not rows:
        empty_columns = (
            ["prompt", "interaction", "judgment", "score"] if include_prompt else ["interaction", "judgment", "score"]
        )
        return pd.DataFrame(columns=empty_columns), warnings
    columns = ["prompt", "interaction", "judgment", "score"] if include_prompt else ["interaction", "judgment", "score"]
    return pd.DataFrame(rows, columns=columns), warnings


def _build_interaction_scoring_row(
    *,
    aggregate_judgment: str,
    entry: object,
    metric_name: str,
    warnings: list[str],
) -> dict[str, object] | None:
    """Create a single interaction scoring table row from a raw entry."""
    if isinstance(entry, Mapping):
        interaction_raw = entry.get("interaction") or entry.get("claim", "")
        score_raw = entry.get("score")
        judgment_raw = entry.get("judgment", "")
    else:
        interaction_raw = entry
        score_raw = None
        judgment_raw = ""

    interaction = str(interaction_raw).strip()
    if not interaction:
        return None

    judgment = str(judgment_raw).strip() if isinstance(judgment_raw, str) else ""
    if not judgment and aggregate_judgment:
        # For interaction scoring we only require numeric scores; silently fall back
        # to the aggregate judgment text when per-interaction judgments are absent.
        judgment = aggregate_judgment

    score_value = None
    if isinstance(score_raw, (int, float, str)):
        try:
            score_value = float(score_raw)
        except ValueError:
            score_value = None

    if score_value is None:
        warnings.append(
            f"{metric_name}: LLM output missing or invalid 'score' for interaction "
            f"'{interaction}'; setting score to NaN.",
        )
        score = float("nan")
    else:
        score = max(0.0, min(1.0, float(score_value)))

    return {
        "interaction": interaction,
        "judgment": judgment,
        "score": score,
    }


def _build_correctness_row(
    *,
    aggregate_judgment: str,
    entry: object,
    warnings: list[str],
) -> dict[str, object] | None:
    """Create a single correctness table row from a raw entry."""
    if isinstance(entry, Mapping):
        claim_raw = entry.get("claim", "")
        score_raw = entry.get("score")
        judgment_raw = entry.get("judgment", "")
    else:
        claim_raw = entry
        score_raw = None
        judgment_raw = ""

    claim = str(claim_raw).strip()
    if not claim:
        return None

    judgment = str(judgment_raw).strip() if isinstance(judgment_raw, str) else ""
    if not judgment:
        message = (
            f"Correctness: LLM output missing 'judgment' for claim '{claim}'. "
            "Using aggregate correctness judgment where available."
        )
        warnings.append(message)
        if aggregate_judgment:
            judgment = f"{aggregate_judgment} (WARNING: missing per-claim judgment)"
        else:
            judgment = "WARNING: missing per-claim judgment from LLM"

    score_value = None
    if isinstance(score_raw, (int, float, str)):
        try:
            score_value = float(score_raw)
        except ValueError:
            score_value = None

    if score_value is None:
        warnings.append(
            f"Correctness: LLM output missing or invalid 'score' for claim '{claim}'; setting score to NaN.",
        )
        score = float("nan")
    else:
        score = max(0.0, min(1.0, float(score_value)))

    return {
        "claim": claim,
        "judgment": judgment,
        "score": score,
    }
