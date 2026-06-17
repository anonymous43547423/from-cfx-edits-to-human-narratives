"""Generate a human-calibrated LaTeX results table from pipeline experiment outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_RUN_PREFIX = "run_pipeline_"
_DPO_RUN_PREFIX = "run_eval_eval_dpo_eval_"

_KNOWN_METHODS = ("jaccard", "cosine", "lime", "shap", "accent", "lxr", "spinrec")

_METHOD_DISPLAY: dict[str, str] = {
    "jaccard": "Jaccard",
    "cosine": "Cosine",
    "lime": "LIME-RS",
    "shap": "SHAP",
    "accent": "ACCENT",
    "lxr": "LXR",
    "spinrec": "SPINRec",
}

_MODEL_DISPLAY: dict[str, str] = {
    "Ministral-8B-Instruct-2410": "Ministral 8B",
    "gemma-3-12b-it": "Gemma 3 12B",
    # "Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    # "Phi-4-mini-instruct": "Phi-4-Mini",
    "Qwen3-8B": "Qwen3 8B",
}

_METHOD_ORDER = list(_KNOWN_METHODS)
_MODEL_ORDER = list(_MODEL_DISPLAY.keys())

# Number of decimal places for formatted metric values.
_DECIMAL_PLACES = 2
_MIN_CFX_SIZE_SAMPLE = 2
_HMF_REWARD_COMPOSITE_KEY = "reward_composite_human_feedback_model"


@dataclass(slots=True)
class RunIdentity:
    """Parsed identity of a single pipeline run directory."""

    method: str
    model: str
    is_dpo: bool


@dataclass(slots=True)
class MetricValue:
    """A single metric consisting of a mean and optional standard deviation."""

    mean: float
    std: float


@dataclass(slots=True)
class RunVariant:
    """Loaded summary results and on-disk run directory for one pipeline variant."""

    results: dict[str, object]
    run_dir: Path
    hmf_results: dict[str, object] | None = None


@dataclass(slots=True)
class SweepTrialCandidate:
    """One sweep trial candidate discovered under an experiment timestamp."""

    run_id: str
    timestamp_dir: Path
    run_dir: Path
    results: dict[str, object]
    hmf_validation: dict[str, object] | None
    hmf_test: dict[str, object] | None


@dataclass(slots=True)
class RowMetrics:
    """All metric values for one (method, model) table row."""

    cfx_size: MetricValue
    cfx_success_rate: float
    cfx_simple_rate: float
    correctness_vanilla: MetricValue
    correctness_dpo: MetricValue
    informativeness_vanilla: MetricValue
    informativeness_dpo: MetricValue
    readability_vanilla: MetricValue
    readability_dpo: MetricValue
    correctness_cal_vanilla: MetricValue
    correctness_cal_dpo: MetricValue
    informativeness_cal_vanilla: MetricValue
    informativeness_cal_dpo: MetricValue
    readability_cal_vanilla: MetricValue
    readability_cal_dpo: MetricValue


def parse_run_directory_name(name: str) -> RunIdentity | None:
    """Extract method, model, and DPO flag from a run directory name."""
    if name.startswith(_DPO_RUN_PREFIX):
        suffix = name[len(_DPO_RUN_PREFIX) :]
    elif name.startswith(_RUN_PREFIX):
        suffix = name[len(_RUN_PREFIX) :]
    else:
        return None
    if not suffix:
        return None

    segments = suffix.split("_")
    is_dpo = "dpo" in segments
    filtered = [seg for seg in segments if seg != "dpo"]

    method: str | None = None
    method_index: int | None = None
    for idx, seg in enumerate(filtered):
        if seg in _KNOWN_METHODS:
            method = seg
            method_index = idx
            break

    if method is None or method_index is None:
        return None

    model_segments = filtered[:method_index] + filtered[method_index + 1 :]
    if not model_segments:
        return None
    model = "_".join(model_segments)

    return RunIdentity(method=method, model=model, is_dpo=is_dpo)


def _load_summary_payload(run_dir: Path, file_name: str) -> dict[str, object] | None:
    """Load a summary JSON file into a dict payload."""
    summary_path = run_dir / file_name
    if not summary_path.is_file():
        return None
    try:
        with summary_path.open() as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _list_timestamp_dirs(experiment_dir: Path) -> list[Path]:
    """Return sorted timestamp directories under one experiment directory."""
    if not experiment_dir.is_dir():
        return []
    return sorted((child for child in experiment_dir.iterdir() if child.is_dir()), key=lambda path: path.name)


def load_run_results(run_dir: Path) -> dict[str, object] | None:
    """Load the results dict from run_summary.json in a timestamped directory."""
    return _load_results_from_summary_file(run_dir, "run_summary.json")


def load_hmf_results(run_dir: Path) -> dict[str, object] | None:
    """Load the results dict from run_human_model_feedback_summary.json in a run directory."""
    return _load_results_from_summary_file(run_dir, "run_human_model_feedback_summary.json")


def load_hmf_split_results(run_dir: Path, split: str) -> dict[str, object] | None:
    """Load one nested split results object from run_human_model_feedback_summary.json."""
    all_results = load_hmf_results(run_dir)
    if all_results is None:
        return None
    split_results = all_results.get(split)
    if not isinstance(split_results, dict):
        return None
    return split_results


def _load_results_from_summary_file(run_dir: Path, file_name: str) -> dict[str, object] | None:
    """Load the ``results`` payload from a summary JSON file under ``run_dir``."""
    summary_path = run_dir / file_name
    if not summary_path.is_file():
        return None
    try:
        with summary_path.open() as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    results = data.get("results")
    if not isinstance(results, dict):
        return None
    return results


def resolve_trial_disk_path(sweep_path: Path, trial: dict[str, object]) -> Path:
    """Resolve sweep trial folder on disk, mirroring the UI Outputs browser helpers."""
    raw = trial.get("trial_dir")
    if isinstance(raw, str):
        candidate = Path(raw)
        try:
            if candidate.is_dir():
                return candidate.resolve()
        except OSError:
            pass

    run_id = trial.get("run_id")
    if isinstance(run_id, (str, int)):
        nested = sweep_path / str(run_id)
        try:
            if nested.is_dir():
                return nested.resolve()
        except OSError:
            pass

    if isinstance(raw, str):
        return Path(raw)

    # Fall back inside sweep folder (may not exist; caller checks load_run_results).
    return sweep_path


def _read_trials_payload(sweep_path: Path) -> list[dict[str, object]]:
    """Load list of trial objects from sweep trials.json."""
    trials_json = sweep_path / "trials.json"
    if not trials_json.is_file():
        return []
    try:
        with trials_json.open() as fh:
            payload = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(payload, dict):
        return []
    raw_trials = payload.get("trials")
    if not isinstance(raw_trials, list):
        return []
    return [trial for trial in raw_trials if isinstance(trial, dict)]


def _collect_sweep_trial_candidates(experiment_dir: Path) -> list[SweepTrialCandidate]:
    """Collect sweep trial candidates across all timestamps for one experiment."""
    candidates: list[SweepTrialCandidate] = []
    for timestamp_dir in _list_timestamp_dirs(experiment_dir):
        sweep_path = timestamp_dir / "sweep"
        if not sweep_path.is_dir():
            continue
        for trial in _read_trials_payload(sweep_path):
            run_id_obj = trial.get("run_id")
            if run_id_obj is None:
                continue
            trial_dir = resolve_trial_disk_path(sweep_path, trial)
            loaded = load_run_results(trial_dir)
            if loaded is None:
                continue
            candidates.append(
                SweepTrialCandidate(
                    run_id=str(run_id_obj),
                    timestamp_dir=timestamp_dir,
                    run_dir=trial_dir,
                    results=loaded,
                    hmf_validation=load_hmf_split_results(trial_dir, "validation"),
                    hmf_test=load_hmf_split_results(trial_dir, "test"),
                ),
            )
    return candidates


def _best_validation_run_id(candidates: list[SweepTrialCandidate]) -> str | None:
    """Return best run_id by validation calibrated HMF composite."""
    best_score_by_run: dict[str, float] = {}
    for candidate in candidates:
        if candidate.hmf_validation is None:
            continue
        raw_score = candidate.hmf_validation.get(_HMF_REWARD_COMPOSITE_KEY)
        if raw_score is None or isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            continue
        score = float(raw_score)
        if not math.isfinite(score):
            continue
        previous = best_score_by_run.get(candidate.run_id)
        if previous is None or score > previous:
            best_score_by_run[candidate.run_id] = score
    if not best_score_by_run:
        return None
    ranked = sorted(
        ((run_id, score) for run_id, score in best_score_by_run.items()), key=lambda item: (-item[1], item[0])
    )
    # Deterministic tie-break: highest score, then lexicographically smallest run_id.
    return ranked[0][0]


def _latest_run_variant(experiment_dir: Path, *, hmf_split: str = "test") -> RunVariant | None:
    """Load latest non-sweep run variant under one experiment directory."""
    latest_split_results: dict[str, object] | None = None
    for split_timestamp_dir in reversed(_list_timestamp_dirs(experiment_dir)):
        if (split_timestamp_dir / "sweep").is_dir():
            continue
        split_results = load_hmf_split_results(split_timestamp_dir, hmf_split)
        if split_results is not None:
            latest_split_results = split_results
            break
    for timestamp_dir in reversed(_list_timestamp_dirs(experiment_dir)):
        if (timestamp_dir / "sweep").is_dir():
            continue
        loaded = load_run_results(timestamp_dir)
        if loaded is None:
            continue
        return RunVariant(
            results=loaded,
            run_dir=timestamp_dir,
            hmf_results=latest_split_results,
        )
    return None


def load_dpo_run_from_split_timestamps(experiment_dir: Path) -> RunVariant | None:
    """Load DPO variant by validation-best run_id with test-display metrics."""
    candidates = _collect_sweep_trial_candidates(experiment_dir)
    if not candidates:
        return _latest_run_variant(experiment_dir)
    best_id = _best_validation_run_id(candidates)
    if best_id is None:
        return None
    latest_timestamp_name = max(candidate.timestamp_dir.name for candidate in candidates)
    latest_candidates = [candidate for candidate in candidates if candidate.timestamp_dir.name == latest_timestamp_name]
    matching = [candidate for candidate in latest_candidates if candidate.run_id == best_id]
    if not matching:
        return None
    with_test = [candidate for candidate in matching if candidate.hmf_test is not None]
    display_candidates = with_test or matching
    pick = max(display_candidates, key=lambda candidate: str(candidate.run_dir))
    return RunVariant(
        results=pick.results,
        run_dir=pick.run_dir,
        hmf_results=pick.hmf_test,
    )


def load_run_variant(run_dir: Path, *, hmf_split: str = "test") -> RunVariant | None:
    """Load summary results and retain the on-disk run directory."""
    results = load_run_results(run_dir)
    if results is None:
        return None
    return RunVariant(results=results, run_dir=run_dir, hmf_results=load_hmf_split_results(run_dir, hmf_split))


def _safe_float(value: object) -> float:
    """Coerce a value to float, returning NaN on failure."""
    if value is None:
        return float("nan")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return result


def _extract_metric(results: dict[str, object], mean_key: str, std_key: str) -> MetricValue:
    """Extract a MetricValue from a results dict."""
    return MetricValue(
        mean=_safe_float(results.get(mean_key)),
        std=_safe_float(results.get(std_key)),
    )


def discover_runs(outputs_dir: Path) -> dict[tuple[str, str], dict[str, RunVariant]]:
    """Discover all run directories and load their results.

    Returns a mapping of ``(method, model) -> {"vanilla": RunVariant, "dpo": RunVariant}``.
    """
    collected: dict[tuple[str, str], dict[str, RunVariant]] = {}
    if not outputs_dir.is_dir():
        return collected

    for child in sorted(outputs_dir.iterdir()):
        if not child.is_dir():
            continue
        identity = parse_run_directory_name(child.name)
        if identity is None:
            continue
        loaded = load_dpo_run_from_split_timestamps(child) if identity.is_dpo else _latest_run_variant(child)
        if loaded is None:
            continue

        key = (identity.method, identity.model)
        variant = "dpo" if identity.is_dpo else "vanilla"
        collected.setdefault(key, {})[variant] = loaded

    return collected


_NAN_METRIC = MetricValue(mean=float("nan"), std=float("nan"))


def _is_counterfactual_cell(value: object) -> bool:
    """Return True when a feather ``is_counterfactual`` cell marks a CFX interaction row."""
    if value is True or value == 1:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value != 0
    if isinstance(value, float) and math.isfinite(value):
        return value != 0.0
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"1", "true"}
    return False


def _to_user_id(value: object) -> int | None:
    """Coerce a feather ``user_id`` cell to an integer user id."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) else None
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return int(parsed) if math.isfinite(parsed) else None


def cfx_interaction_count_metric_from_dataframe(df: pd.DataFrame) -> MetricValue | None:
    """Compute mean and sample std of per-user counterfactual interaction counts."""
    if "user_id" not in df.columns or "is_counterfactual" not in df.columns:
        return None

    per_user: dict[int, int] = {}
    for user_id_raw, is_cfx_raw in zip(df["user_id"], df["is_counterfactual"], strict=True):
        user_id = _to_user_id(user_id_raw)
        if user_id is None or not _is_counterfactual_cell(is_cfx_raw):
            continue
        per_user[user_id] = per_user.get(user_id, 0) + 1

    if not per_user:
        return None

    counts = list(per_user.values())
    count_n = len(counts)
    mean = sum(counts) / count_n
    if count_n < _MIN_CFX_SIZE_SAMPLE:
        return MetricValue(mean=mean, std=float("nan"))

    variance = sum((count - mean) ** 2 for count in counts) / (count_n - 1)
    return MetricValue(mean=mean, std=math.sqrt(variance))


def cfx_interaction_count_metric_for_run_dir(run_dir: Path | None) -> MetricValue:
    """Load ``interactions.feather`` under ``run_dir`` and return CFX size stats."""
    if run_dir is None:
        return _NAN_METRIC

    feather_path = run_dir / "interactions.feather"
    if not feather_path.is_file():
        return _NAN_METRIC

    try:
        interactions = pd.read_feather(feather_path)
    except (OSError, ValueError):
        return _NAN_METRIC

    metric = cfx_interaction_count_metric_from_dataframe(interactions)
    if metric is None:
        return _NAN_METRIC
    return metric


def build_row_metrics(
    vanilla: RunVariant | None,
    dpo: RunVariant | None,
) -> RowMetrics:
    """Assemble a RowMetrics from vanilla and DPO loaded runs."""
    v = vanilla.results if vanilla is not None else {}
    d = dpo.results if dpo is not None else {}
    vh = vanilla.hmf_results if vanilla is not None and vanilla.hmf_results is not None else {}
    dh = dpo.hmf_results if dpo is not None and dpo.hmf_results is not None else {}
    vanilla_run_dir = vanilla.run_dir if vanilla is not None else None
    return RowMetrics(
        cfx_size=cfx_interaction_count_metric_for_run_dir(vanilla_run_dir),
        cfx_success_rate=_safe_float(v.get("cfx_success_rate")),
        cfx_simple_rate=_safe_float(v.get("cfx_simple_rate")),
        correctness_vanilla=_extract_metric(
            v, "explanation_cfx_pattern_match_mean", "explanation_cfx_pattern_match_std"
        ),
        correctness_dpo=_extract_metric(d, "explanation_cfx_pattern_match_mean", "explanation_cfx_pattern_match_std"),
        informativeness_vanilla=_extract_metric(
            v, "explanation_pattern_contrast_mean", "explanation_pattern_contrast_std"
        ),
        informativeness_dpo=_extract_metric(d, "explanation_pattern_contrast_mean", "explanation_pattern_contrast_std"),
        readability_vanilla=_extract_metric(v, "readability_overall_mean", "readability_overall_std"),
        readability_dpo=_extract_metric(d, "readability_overall_mean", "readability_overall_std"),
        correctness_cal_vanilla=_extract_metric(
            vh,
            "explanation_cfx_pattern_human_feedback_model_match_mean",
            "explanation_cfx_pattern_human_feedback_model_match_std",
        ),
        correctness_cal_dpo=_extract_metric(
            dh,
            "explanation_cfx_pattern_human_feedback_model_match_mean",
            "explanation_cfx_pattern_human_feedback_model_match_std",
        ),
        informativeness_cal_vanilla=_extract_metric(
            vh,
            "explanation_pattern_human_feedback_model_contrast_mean",
            "explanation_pattern_human_feedback_model_contrast_std",
        ),
        informativeness_cal_dpo=_extract_metric(
            dh,
            "explanation_pattern_human_feedback_model_contrast_mean",
            "explanation_pattern_human_feedback_model_contrast_std",
        ),
        readability_cal_vanilla=_extract_metric(
            vh,
            "readability_human_feedback_model_score_mean",
            "readability_human_feedback_model_score_std",
        ),
        readability_cal_dpo=_extract_metric(
            dh,
            "readability_human_feedback_model_score_mean",
            "readability_human_feedback_model_score_std",
        ),
    )


_RANK_BEST = 1
_RANK_SECOND = 2


def rank_column(values: list[float], *, higher_is_better: bool = True) -> list[int | None]:
    """Assign rank 1 (best) and 2 (second-best) values in a column.

    Non-finite values receive ``None``. Ties share the same rank.
    """
    ranks: list[int | None] = [None] * len(values)
    indexed = [(val, idx) for idx, val in enumerate(values) if math.isfinite(val)]
    if not indexed:
        return ranks

    indexed.sort(key=lambda pair: pair[0], reverse=higher_is_better)
    best_val = indexed[0][0]
    for val, idx in indexed:
        if val == best_val:
            ranks[idx] = _RANK_BEST

    if higher_is_better:
        second_candidates = [(val, idx) for val, idx in indexed if val < best_val]
    else:
        second_candidates = [(val, idx) for val, idx in indexed if val > best_val]
    if second_candidates:
        second_val = second_candidates[0][0]
        for val, idx in second_candidates:
            if val == second_val:
                ranks[idx] = _RANK_SECOND

    return ranks


def _fmt_number(value: float) -> str:
    """Format a float to the standard decimal precision."""
    return f"{value:.{_DECIMAL_PLACES}f}"


def format_mean_std(metric: MetricValue) -> str:
    r"""Format a MetricValue as ``mean $\pm$ std`` or blank if missing."""
    if not math.isfinite(metric.mean):
        return ""
    text = _fmt_number(metric.mean)
    if math.isfinite(metric.std):
        text += f" $\\pm$ {_fmt_number(metric.std)}"
    return text


def format_single(value: float) -> str:
    """Format a single float value or blank if missing."""
    if not math.isfinite(value):
        return ""
    return _fmt_number(value)


def _wrap_highlight(text: str, rank: int | None) -> str:
    """Wrap cell text in bold or underline LaTeX commands based on rank."""
    if not text or rank is None:
        return text
    if rank == _RANK_BEST:
        return f"\\textbf{{{text}}}"
    if rank == _RANK_SECOND:
        return f"\\underline{{{text}}}"
    return text


def _build_table_rows(
    all_rows: list[tuple[str, str, RowMetrics]],
) -> list[list[str]]:
    """Build formatted and highlighted cell strings for every table row.

    Each inner list has 9 cell strings corresponding to the value columns:
    cfx_success, cfx_size, cfx_simple, corr_cal_v, corr_cal_d, info_cal_v,
    info_cal_d, readability_cal_v, readability_cal_d.
    """
    col_cfx_size = [row.cfx_size.mean for _, _, row in all_rows]
    col_cfx = [row.cfx_success_rate for _, _, row in all_rows]
    col_cfx_simple = [row.cfx_simple_rate for _, _, row in all_rows]
    col_corr_cal_v = [row.correctness_cal_vanilla.mean for _, _, row in all_rows]
    col_corr_cal_d = [row.correctness_cal_dpo.mean for _, _, row in all_rows]
    col_info_cal_v = [row.informativeness_cal_vanilla.mean for _, _, row in all_rows]
    col_info_cal_d = [row.informativeness_cal_dpo.mean for _, _, row in all_rows]
    col_read_cal_v = [row.readability_cal_vanilla.mean for _, _, row in all_rows]
    col_read_cal_d = [row.readability_cal_dpo.mean for _, _, row in all_rows]

    ranks: list[list[int | None]] = [
        rank_column(col_cfx),
        rank_column(col_cfx_size, higher_is_better=False),
        rank_column(col_cfx_simple),
        rank_column(col_corr_cal_v),
        rank_column(col_corr_cal_d),
        rank_column(col_info_cal_v),
        rank_column(col_info_cal_d),
        rank_column(col_read_cal_v),
        rank_column(col_read_cal_d),
    ]

    formatted: list[list[str]] = []
    for i, (_, _, row) in enumerate(all_rows):
        cells = [
            format_single(row.cfx_success_rate),
            format_mean_std(row.cfx_size),
            format_single(row.cfx_simple_rate),
            format_mean_std(row.correctness_cal_vanilla),
            format_mean_std(row.correctness_cal_dpo),
            format_mean_std(row.informativeness_cal_vanilla),
            format_mean_std(row.informativeness_cal_dpo),
            format_mean_std(row.readability_cal_vanilla),
            format_mean_std(row.readability_cal_dpo),
        ]
        highlighted = [_wrap_highlight(cell, ranks[col_idx][i]) for col_idx, cell in enumerate(cells)]
        formatted.append(highlighted)
    return formatted


def _build_ordered_rows(
    runs: dict[tuple[str, str], dict[str, RunVariant]],
) -> list[tuple[str, str, RowMetrics]]:
    """Build the ordered list of (method, model, metrics) rows for the table."""
    all_rows: list[tuple[str, str, RowMetrics]] = []
    for method in _METHOD_ORDER:
        for model in _MODEL_ORDER:
            variants = runs.get((method, model), {})
            metrics = build_row_metrics(variants.get("vanilla"), variants.get("dpo"))
            all_rows.append((method, model, metrics))
    return all_rows


def _emit_header(lines: list[str]) -> None:
    """Append the LaTeX table preamble and column header rows."""
    lines.extend(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Comparison of performances of counterfactual (CFX) generators and corresponding LLM mediators,",
            r"averaged across users (mean $\pm$ std).",
            r"\textit{Correctness} measures semantic alignment between the generated explanation",
            r"and the counterfactual edits,",
            r"\textit{Informativeness} measures how well the explanation distinguishes",
            r"counterfactual interactions from the rest of the user's history,",
            r"while \textit{Linguistic Quality} measures human preference for the explanation's",
            r"fluency, clarity, and naturalness.",
            r"Correctness, Informativeness, and Linguistic Quality report human-calibrated scores,",
            r"with separate columns for the Vanilla LLM and DPO-optimized variants.",
            r"Higher values are better ($\uparrow$).",
            r"Best results are in \textbf{bold}; second-best are \underline{underlined}.}",
            r"\label{tab:main_results_new}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{4pt}",
            "",
            r"\begin{tabular}{lccc l cc cc cc}",
            r"\toprule",
            "",
            # Header row 1: top-level groups.
            r"\multirow{2}{*}{Method} &",
            r"\multirow{2}{*}{\shortstack{CFX Success\\Rate $\uparrow$}} &",
            r"\multirow{2}{*}{\shortstack{CFX\\Size $\downarrow$}} &",
            r"\multirow{2}{*}{\shortstack{Simple CFX\\Rate $\uparrow$}} &",
            r"\multirow{2}{*}{LLM} &",
            r"\multicolumn{2}{c}{Correctness $\uparrow$} &",
            r"\multicolumn{2}{c}{Informativeness $\uparrow$} &",
            r"\multicolumn{2}{c}{Linguistic Quality $\uparrow$} \\",
            "",
            r"\cmidrule(lr){6-7}",
            r"\cmidrule(lr){8-9}",
            r"\cmidrule(lr){10-11}",
            "",
            # Header row 2: Vanilla/DPO labels under each metric pair.
            r"& & & & &",
            r"Vanilla LLM & DPO &",
            r"Vanilla LLM & DPO &",
            r"Vanilla LLM & DPO \\",
            "",
            r"\midrule",
        ]
    )


def _emit_body(lines: list[str], highlighted: list[list[str]]) -> None:
    """Append method/model data rows grouped by method with midrules."""
    row_idx = 0
    for method_idx, method in enumerate(_METHOD_ORDER):
        model_count = len(_MODEL_ORDER)
        method_label = _METHOD_DISPLAY.get(method, method)

        if method_idx > 0:
            lines.append("")
            lines.append(r"\midrule")

        lines.append("")
        lines.append(f"\\multirow{{{model_count}}}{{*}}{{{method_label}}}")

        for model_offset, model in enumerate(_MODEL_ORDER):
            model_label = _MODEL_DISPLAY.get(model, model)
            cells = highlighted[row_idx]
            # cells order: cfx_success, cfx_size, cfx_simple, corr_cal_v,
            #              corr_cal_d, info_cal_v, info_cal_d,
            #              readability_cal_v, readability_cal_d
            if model_offset == 0:
                # Method-level columns use multirow for vertical centering.
                mr = f"\\multirow{{{model_count}}}{{*}}"
                cfx_s = f"{mr}{{{cells[0]}}}" if cells[0] else ""
                cfx_size = f"{mr}{{{cells[1]}}}" if cells[1] else ""
                cfx_sim = f"{mr}{{{cells[2]}}}" if cells[2] else ""
            else:
                cfx_s = ""
                cfx_size = ""
                cfx_sim = ""
            llm_cells = " & ".join(cells[3:])
            lines.append(f"& {cfx_s} & {cfx_size} & {cfx_sim} & {model_label} & {llm_cells} \\\\")
            row_idx += 1


def generate_latex(outputs_dir: Path) -> str:
    """Generate the full LaTeX table string from discovered experiment outputs."""
    runs = discover_runs(outputs_dir)
    all_rows = _build_ordered_rows(runs)
    highlighted = _build_table_rows(all_rows)

    lines: list[str] = []
    _emit_header(lines)
    _emit_body(lines, highlighted)
    lines.append("")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a human-calibrated LaTeX results table from pipeline experiment outputs.",
    )
    default_outputs = Path(__file__).resolve().parents[1] / "runs"
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=default_outputs,
        help=f"Root directory containing run_pipeline_* and run_eval_eval_dpo_eval_* folders (default: {default_outputs}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the LaTeX table generator."""
    args = parse_args(argv)
    outputs_dir: Path = args.outputs_dir
    if not outputs_dir.is_dir():
        print(f"Error: outputs directory does not exist: {outputs_dir}", file=sys.stderr)
        return 1
    latex = generate_latex(outputs_dir)
    print(latex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
