"""Shared primitives for evaluating natural-language explanations."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

from recsys_nle.nl_explanations.hf_json import parse_json_from_hf

if TYPE_CHECKING:
    import pandas as pd


@dataclass(slots=True)
class EvaluationResult:
    """Container for an evaluation technique's textual assessment and numeric score."""

    judgment: str
    score: float
    details: Mapping[str, object] | None = None
    prompt: str | None = None


class BaseEvaluator(ABC):
    """Shared helpers for explanation evaluators using batch-first HuggingFace inference."""

    @abstractmethod
    def build_prompt(
        self,
        *,
        explanation: str,
        interactions: pd.DataFrame | None = None,
    ) -> list[dict[str, str]]:
        """Build chat messages for a single evaluation."""
        raise NotImplementedError

    def parse_result(self, raw_output: str, *, prompt: str | None = None) -> EvaluationResult:
        """Parse raw LLM output into a structured evaluation result."""
        issues: list[str] = []

        parsed, parse_error = self._normalise_payload(raw_output)

        if parse_error:
            issues.append(
                "LLM output for evaluation could not be parsed as JSON; using raw text and NaN score.",
            )

        judgment = str(parsed.get("judgment", "")).strip()
        if not judgment:
            issues.append("LLM output for evaluation is missing a 'judgment' field.")

        score_value = parsed.get("score")
        if isinstance(score_value, (int, float, str)):
            try:
                score = float(score_value)
            except ValueError:
                score = float("nan")
        else:
            score = float("nan")

        if not math.isnan(score):
            score = max(0.0, min(1.0, score))

        details: dict[str, object] = {}
        if issues:
            details["warnings"] = issues

        return EvaluationResult(judgment=judgment, score=score, details=details or None, prompt=prompt)

    def _normalise_payload(self, raw_result: object) -> tuple[Mapping[str, object], bool]:
        """Convert raw model output into a mapping and indicate parse success."""
        if isinstance(raw_result, Mapping) and raw_result.get("__evaluation_error__"):
            message = str(raw_result.get("error", "Evaluation failed."))
            return {"judgment": message, "score": float("nan")}, True

        if isinstance(raw_result, Mapping):
            return {str(key): value for key, value in raw_result.items()}, False

        # Use the HuggingFace JSON parser for string outputs
        text = str(raw_result).strip()
        decoded = parse_json_from_hf(text)

        if isinstance(decoded, dict):
            return {str(key): value for key, value in decoded.items()}, False

        return {"judgment": text, "score": float("nan")}, True


def normalise_score(raw_score: object) -> float:
    """Convert an arbitrary raw score into the [0.0, 1.0] interval."""
    if not isinstance(raw_score, (int, float, str)):
        return float("nan")
    try:
        numeric = float(raw_score)
    except ValueError:
        return float("nan")
    if math.isnan(numeric):
        return numeric
    return max(0.0, min(1.0, numeric))


def aggregate_scores(
    entries: Sequence[Mapping[str, object]],
    *,
    empty_warning: str,
) -> tuple[float, list[str]]:
    """Aggregate per-item scores into a mean value and optional warnings."""
    scores: list[float] = []
    for entry in entries:
        raw_score = entry.get("score") if isinstance(entry, Mapping) else None
        scores.append(normalise_score(raw_score))

    valid_scores = [value for value in scores if not math.isnan(value)]
    if not valid_scores:
        return float("nan"), [empty_warning]

    mean_score = float(sum(valid_scores) / len(valid_scores))
    mean_score = max(0.0, min(1.0, mean_score))
    return mean_score, []
