"""Evaluate LLM judges against human-labeled readability and interaction-match CSV datasets."""

# ruff: noqa: E402

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from recsys_nle.nl_explanations.evaluation.interaction_scoring import (
    InteractionScoringEvaluator,
    build_single_interaction_scoring_messages,
)
from recsys_nle.nl_explanations.evaluation.readability import (
    READABILITY_SUBSCORE_KEYS,
    ReadabilityEvaluator,
)
from recsys_nle.pipeline.workflow import _build_evaluation_llm

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from recsys_nle.nl_explanations.evaluation.base import EvaluationResult
    from recsys_nle.nl_explanations.llm import LLMClient

LOGGER = logging.getLogger(__name__)

_READABILITY_REQUIRED_COLUMNS = ("explanation_text", "overall")
_INTERACTION_REQUIRED_COLUMNS = ("explanation_text", "interaction_description", "score")
_EVAL_MAX_NEW_TOKENS = 256
_EVAL_TEMPERATURE = 0.0
_EXPLANATION_COLUMN_WIDTH = 72
_ELLIPSIS_RESERVE = 3


class JudgeMode(StrEnum):
    """Supported human-labeled dataset evaluation modes."""

    READABILITY = "readability"
    INTERACTION_MATCH = "interaction_match"


@dataclass(frozen=True, slots=True)
class ThresholdAccuracyResult:
    """Optimal-threshold binary classification accuracy against human labels."""

    best_threshold: float
    accuracy: float
    n_scored: int


@dataclass(frozen=True, slots=True)
class ValidLabeledRow:
    """One human-labeled row retained for evaluation."""

    row_index: int
    explanation_text: str
    label: int
    source: dict[str, object]
    interaction_description: str | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for LLM judge evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Run an LLM judge on a human-labeled CSV and report optimal-threshold "
            "accuracies against human binary labels."
        ),
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--readability-human-dataset-path",
        type=Path,
        help="Path to readability-human-labeled.csv (must include explanation_text and overall).",
    )
    mode_group.add_argument(
        "--interaction-match-human-dataset-path",
        type=Path,
        help=(
            "Path to interaction-match-human-labeled.csv "
            "(must include explanation_text, interaction_description, and score)."
        ),
    )
    parser.add_argument(
        "--model-id-evaluation",
        type=str,
        required=True,
        help=(
            "Model id for LLM judge evaluation: Hugging Face id, or EINFRA/<api_model_id> for "
            "e-INFRA OpenAI-compatible chat completions."
        ),
    )
    parser.add_argument(
        "--evaluation-llm-batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation LLM inference (default: 4).",
    )
    parser.add_argument(
        "--output-json-path",
        type=Path,
        default=None,
        help="Optional path to write the JSON report (default: print to stdout).",
    )
    parser.add_argument(
        "--log-level",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        default="INFO",
        help="Logging verbosity level (default: INFO).",
    )
    return parser.parse_args(argv)


def resolve_judge_mode(args: argparse.Namespace) -> tuple[JudgeMode, Path]:
    """Return the selected evaluation mode and its dataset path."""
    if args.readability_human_dataset_path is not None:
        return JudgeMode.READABILITY, args.readability_human_dataset_path
    if args.interaction_match_human_dataset_path is not None:
        return JudgeMode.INTERACTION_MATCH, args.interaction_match_human_dataset_path
    msg = "Exactly one dataset mode must be selected."
    raise ValueError(msg)


def load_human_dataset(path: Path, *, mode: JudgeMode) -> pd.DataFrame:
    """Load and validate a human-labeled CSV for the given evaluation mode."""
    if not path.is_file():
        msg = f"Dataset path does not exist or is not a file: {path}"
        raise ValueError(msg)

    required_columns = _READABILITY_REQUIRED_COLUMNS if mode is JudgeMode.READABILITY else _INTERACTION_REQUIRED_COLUMNS
    df = pd.read_csv(path)
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        msg = f"Dataset missing required columns: {missing}"
        raise ValueError(msg)
    return df


def _coerce_binary_label(value: object) -> int | None:
    """Return 0 or 1 for a human binary label, or None when missing/invalid."""
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            numeric = float(value)
        except ValueError:
            return None
    elif isinstance(value, bool):
        numeric = float(int(value))
    elif isinstance(value, (int, float)):
        numeric = float(value)
    else:
        return None
    if math.isnan(numeric) or numeric not in (0.0, 1.0):
        return None
    return int(numeric)


def _coerce_text(value: object) -> str:
    """Return stripped text from a CSV cell, or empty string when missing."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def extract_valid_labeled_row_records(
    df: pd.DataFrame,
    *,
    mode: JudgeMode,
) -> list[ValidLabeledRow]:
    """Return labeled rows with original CSV context for rows with valid binary labels."""
    label_column = "overall" if mode is JudgeMode.READABILITY else "score"
    records: list[ValidLabeledRow] = []
    for position, (_, row) in enumerate(df.iterrows()):
        label = _coerce_binary_label(row[label_column])
        if label is None:
            continue
        interaction_description = None
        if mode is JudgeMode.INTERACTION_MATCH:
            interaction_description = _coerce_text(row["interaction_description"])
        source = {str(key): value for key, value in row.to_dict().items()}
        records.append(
            ValidLabeledRow(
                row_index=position,
                explanation_text=_coerce_text(row["explanation_text"]),
                label=label,
                source=source,
                interaction_description=interaction_description,
            ),
        )
    return records


def extract_valid_labeled_rows(
    df: pd.DataFrame,
    *,
    mode: JudgeMode,
) -> tuple[list[str], list[int]]:
    """Return explanation texts and binary labels for rows with valid labels."""
    records = extract_valid_labeled_row_records(df, mode=mode)
    return [record.explanation_text for record in records], [record.label for record in records]


def compute_min_overall_score(evaluation: EvaluationResult) -> float:
    """Return the minimum readability subscore, or NaN when any subscore is missing."""
    details = evaluation.details or {}
    values: list[float] = []
    for key in READABILITY_SUBSCORE_KEYS:
        raw = details.get(key)
        if raw is None or not isinstance(raw, (int, float)) or math.isnan(float(raw)):
            return float("nan")
        values.append(float(raw))
    if len(values) < len(READABILITY_SUBSCORE_KEYS):
        return float("nan")
    return float(min(values))


def _threshold_candidates(scores: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Build candidate thresholds from finite scores plus edge sentinels."""
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return np.array([0.0, 1.0], dtype=np.float64)
    unique = np.unique(finite)
    below_min = float(unique.min()) - 1.0
    above_max = float(unique.max()) + 1.0
    combined = np.concatenate([unique, np.array([below_min, above_max], dtype=np.float64)])
    result: np.ndarray[Any, np.dtype[np.float64]] = np.unique(combined)
    return result


def best_threshold_accuracy(
    y_true: Sequence[int],
    y_score: Sequence[float],
) -> ThresholdAccuracyResult:
    """Find the threshold maximizing accuracy; tie-break toward the smallest threshold."""
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(y_score, dtype=np.float64)
    if labels.shape[0] != scores.shape[0]:
        msg = "y_true and y_score must have the same length"
        raise ValueError(msg)

    valid_mask = np.isfinite(scores)
    labels = labels[valid_mask]
    scores = scores[valid_mask]
    n_scored = int(labels.shape[0])
    if n_scored == 0:
        return ThresholdAccuracyResult(best_threshold=float("nan"), accuracy=float("nan"), n_scored=0)

    best_threshold = float("nan")
    best_accuracy = -1.0
    for threshold in _threshold_candidates(scores):
        predictions = (scores >= threshold).astype(np.int64)
        accuracy = float(np.mean(predictions == labels))
        if accuracy > best_accuracy or (accuracy == best_accuracy and threshold < best_threshold):
            best_accuracy = accuracy
            best_threshold = float(threshold)

    return ThresholdAccuracyResult(
        best_threshold=best_threshold,
        accuracy=best_accuracy,
        n_scored=n_scored,
    )


def _build_classified_row(
    record: ValidLabeledRow,
    *,
    judge_score: float,
    prediction: int,
    label_column: str,
) -> dict[str, object]:
    """Merge source CSV fields with evaluation outputs for one scored row."""
    row: dict[str, object] = dict(record.source)
    row["row_index"] = record.row_index
    row["explanation_text"] = record.explanation_text
    row[label_column] = record.label
    row["judge_score"] = judge_score
    row["prediction"] = prediction
    if record.interaction_description is not None:
        row["interaction_description"] = record.interaction_description
    return row


def compute_threshold_fp_fn(
    records: Sequence[ValidLabeledRow],
    judge_scores: Sequence[float],
    threshold: float,
    *,
    label_column: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return false-positive and false-negative rows at the given score threshold."""
    false_positives: list[dict[str, object]] = []
    false_negatives: list[dict[str, object]] = []
    if not math.isfinite(threshold):
        return false_positives, false_negatives

    for record, raw_score in zip(records, judge_scores, strict=True):
        if not isinstance(raw_score, (int, float)) or math.isnan(float(raw_score)):
            continue
        judge_score = float(raw_score)
        prediction = 1 if judge_score >= threshold else 0
        if record.label == 0 and prediction == 1:
            false_positives.append(
                _build_classified_row(
                    record,
                    judge_score=judge_score,
                    prediction=prediction,
                    label_column=label_column,
                ),
            )
        elif record.label == 1 and prediction == 0:
            false_negatives.append(
                _build_classified_row(
                    record,
                    judge_score=judge_score,
                    prediction=prediction,
                    label_column=label_column,
                ),
            )
    return false_positives, false_negatives


def _truncate_text(value: object, *, max_width: int) -> str:
    """Truncate text for fixed-width table cells."""
    text = "" if value is None else str(value)
    if len(text) <= max_width:
        return text
    if max_width <= _ELLIPSIS_RESERVE:
        return text[:max_width]
    return text[: max_width - _ELLIPSIS_RESERVE] + "..."


def _table_cell(value: object, *, width: int) -> str:
    """Format one table cell with fixed width."""
    if isinstance(value, float):
        text = "nan" if math.isnan(value) else f"{value:.4f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    text = _truncate_text(text, max_width=width)
    return text.ljust(width)


def format_classification_table(
    rows: Sequence[Mapping[str, object]],
    *,
    columns: Sequence[str],
) -> str:
    """Render classified rows as a fixed-width ASCII table."""
    if not rows:
        return "(none)"

    widths: dict[str, int] = {}
    for column in columns:
        if column in {"explanation_text", "interaction_description"}:
            widths[column] = _EXPLANATION_COLUMN_WIDTH
            continue
        header_width = len(column)
        data_width = max((len(_table_cell(row.get(column), width=header_width)) for row in rows), default=0)
        widths[column] = max(header_width, data_width)

    header = " | ".join(column.ljust(widths[column]) for column in columns)
    separator = "-+-".join("-" * widths[column] for column in columns)
    body_lines = [" | ".join(_table_cell(row.get(column), width=widths[column]) for column in columns) for row in rows]
    return "\n".join([header, separator, *body_lines])


def _empty_threshold_metric(*, include_fp_fn: bool) -> dict[str, object]:
    """Return an empty threshold metric block for reports with no valid rows."""
    metric: dict[str, object] = {
        "best_threshold": None,
        "accuracy": None,
        "n_scored": 0,
    }
    if include_fp_fn:
        metric["false_positives"] = []
        metric["false_negatives"] = []
    return metric


def _build_threshold_metric(
    records: Sequence[ValidLabeledRow],
    judge_scores: Sequence[float],
    labels: Sequence[int],
    *,
    label_column: str,
) -> dict[str, object]:
    """Compute threshold accuracy and FP/FN rows for one judge score stream."""
    result = best_threshold_accuracy(labels, judge_scores)
    false_positives, false_negatives = compute_threshold_fp_fn(
        records,
        judge_scores,
        result.best_threshold,
        label_column=label_column,
    )
    return {
        "best_threshold": result.best_threshold,
        "accuracy": result.accuracy,
        "n_scored": result.n_scored,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def run_readability_judge_batch(
    explanations: Sequence[str],
    *,
    model_id: str,
    batch_size: int,
) -> list[EvaluationResult]:
    """Run ReadabilityEvaluator over explanations using batched LLM inference."""
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    evaluator = ReadabilityEvaluator()
    messages_batch = [evaluator.build_prompt(explanation=text) for text in explanations]

    llm: LLMClient = _build_evaluation_llm(model_id)
    try:
        raw_outputs = list(
            llm.generate_batch(
                messages_batch,
                max_new_tokens=_EVAL_MAX_NEW_TOKENS,
                temperature=_EVAL_TEMPERATURE,
                batch_size=batch_size,
            ),
        )
    finally:
        llm.close()

    if len(raw_outputs) != len(explanations):
        msg = f"Expected {len(explanations)} LLM outputs, got {len(raw_outputs)}"
        raise RuntimeError(msg)

    return [evaluator.parse_result(raw) for raw in raw_outputs]


def run_interaction_match_judge_batch(
    records: Sequence[ValidLabeledRow],
    *,
    model_id: str,
    batch_size: int,
) -> list[float]:
    """Run interaction scoring prompts over labeled rows using batched LLM inference."""
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    evaluator = InteractionScoringEvaluator()
    messages_batch = [
        build_single_interaction_scoring_messages(
            interaction_description=record.interaction_description or "",
            target_text=record.explanation_text,
        )
        for record in records
    ]

    llm: LLMClient = _build_evaluation_llm(model_id)
    try:
        raw_outputs = list(
            llm.generate_batch(
                messages_batch,
                max_new_tokens=_EVAL_MAX_NEW_TOKENS,
                temperature=_EVAL_TEMPERATURE,
                batch_size=batch_size,
            ),
        )
    finally:
        llm.close()

    if len(raw_outputs) != len(records):
        msg = f"Expected {len(records)} LLM outputs, got {len(raw_outputs)}"
        raise RuntimeError(msg)

    scores: list[float] = []
    for record, raw_output in zip(records, raw_outputs, strict=True):
        parsed = evaluator.parse_single_interaction_result(
            raw_output,
            record.interaction_description or "",
        )
        raw_score = parsed.get("score")
        if isinstance(raw_score, (int, float)):
            scores.append(float(raw_score))
        else:
            scores.append(float("nan"))
    return scores


def evaluate_readability_judge(
    df: pd.DataFrame,
    *,
    model_id: str,
    batch_size: int,
) -> dict[str, object]:
    """Score explanations and compute mean/min overall optimal-threshold accuracies."""
    records = extract_valid_labeled_row_records(df, mode=JudgeMode.READABILITY)
    explanations = [record.explanation_text for record in records]
    labels = [record.label for record in records]
    n_rows = len(df)
    n_valid_rows = len(labels)

    if n_valid_rows == 0:
        return {
            "mode": JudgeMode.READABILITY.value,
            "n_rows": n_rows,
            "n_valid_rows": n_valid_rows,
            "mean_overall": _empty_threshold_metric(include_fp_fn=True),
            "min_overall": _empty_threshold_metric(include_fp_fn=False),
        }

    evaluations = run_readability_judge_batch(
        explanations,
        model_id=model_id,
        batch_size=batch_size,
    )
    mean_scores = [result.score for result in evaluations]
    min_scores = [compute_min_overall_score(result) for result in evaluations]

    mean_metric = _build_threshold_metric(
        records,
        mean_scores,
        labels,
        label_column="overall",
    )
    min_result = best_threshold_accuracy(labels, min_scores)

    return {
        "mode": JudgeMode.READABILITY.value,
        "n_rows": n_rows,
        "n_valid_rows": n_valid_rows,
        "mean_overall": mean_metric,
        "min_overall": {
            "best_threshold": min_result.best_threshold,
            "accuracy": min_result.accuracy,
            "n_scored": min_result.n_scored,
        },
    }


def evaluate_interaction_match_judge(
    df: pd.DataFrame,
    *,
    model_id: str,
    batch_size: int,
) -> dict[str, object]:
    """Score interaction rows and compute optimal-threshold accuracy against human score labels."""
    records = extract_valid_labeled_row_records(df, mode=JudgeMode.INTERACTION_MATCH)
    labels = [record.label for record in records]
    n_rows = len(df)
    n_valid_rows = len(labels)

    if n_valid_rows == 0:
        return {
            "mode": JudgeMode.INTERACTION_MATCH.value,
            "n_rows": n_rows,
            "n_valid_rows": n_valid_rows,
            "score": _empty_threshold_metric(include_fp_fn=True),
        }

    judge_scores = run_interaction_match_judge_batch(
        records,
        model_id=model_id,
        batch_size=batch_size,
    )
    score_metric = _build_threshold_metric(
        records,
        judge_scores,
        labels,
        label_column="score",
    )

    return {
        "mode": JudgeMode.INTERACTION_MATCH.value,
        "n_rows": n_rows,
        "n_valid_rows": n_valid_rows,
        "score": score_metric,
    }


def write_report(report: dict[str, object], output_path: Path | None) -> None:
    """Write or print the evaluation report as JSON."""
    payload = json.dumps(report, indent=2)
    if output_path is None:
        print(payload)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload + "\n", encoding="utf-8")


def print_threshold_fp_fn(
    report: dict[str, object],
    *,
    metric_key: str,
    label: str,
    table_columns: Sequence[str],
) -> None:
    """Print false positives and false negatives as ASCII tables for one metric."""
    metric = report.get(metric_key)
    if not isinstance(metric, dict):
        return

    threshold = metric.get("best_threshold")
    false_positives = metric.get("false_positives", [])
    false_negatives = metric.get("false_negatives", [])
    if not isinstance(false_positives, list) or not isinstance(false_negatives, list):
        return

    print(f"\nFalse positives ({label}, threshold={threshold}, n={len(false_positives)})")
    print(format_classification_table(false_positives, columns=table_columns))
    print(f"\nFalse negatives ({label}, threshold={threshold}, n={len(false_negatives)})")
    print(format_classification_table(false_negatives, columns=table_columns))


def print_report_tables(report: dict[str, object]) -> None:
    """Print FP/FN tables appropriate for the evaluation mode."""
    mode = report.get("mode")
    if mode == JudgeMode.READABILITY.value:
        print_threshold_fp_fn(
            report,
            metric_key="mean_overall",
            label="mean_overall",
            table_columns=("row_index", "overall", "prediction", "judge_score", "explanation_text"),
        )
        return
    if mode == JudgeMode.INTERACTION_MATCH.value:
        print_threshold_fp_fn(
            report,
            metric_key="score",
            label="score",
            table_columns=(
                "row_index",
                "score",
                "prediction",
                "judge_score",
                "explanation_text",
                "interaction_description",
            ),
        )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for LLM judge evaluation."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    if args.evaluation_llm_batch_size <= 0:
        msg = f"--evaluation-llm-batch-size must be positive, got {args.evaluation_llm_batch_size}"
        raise ValueError(msg)

    try:
        mode, dataset_path = resolve_judge_mode(args)
        df = load_human_dataset(dataset_path.resolve(), mode=mode)
        if mode is JudgeMode.READABILITY:
            report = evaluate_readability_judge(
                df,
                model_id=args.model_id_evaluation,
                batch_size=args.evaluation_llm_batch_size,
            )
        else:
            report = evaluate_interaction_match_judge(
                df,
                model_id=args.model_id_evaluation,
                batch_size=args.evaluation_llm_batch_size,
            )
    except ValueError as exc:
        LOGGER.error("%s", exc)  # noqa: TRY400
        return 1

    output_path = args.output_json_path.resolve() if args.output_json_path is not None else None
    write_report(report, output_path)
    print_report_tables(report)
    if output_path is not None:
        LOGGER.info("Wrote report to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
