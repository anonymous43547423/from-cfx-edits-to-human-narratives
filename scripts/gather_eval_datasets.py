"""Build paired evaluation CSV datasets (readability + interaction-match) from pipeline outputs."""

from __future__ import annotations

import argparse
import json
import logging
import math
import numbers
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

from recsys_nle.nl_explanations.evaluation.interaction_scoring import build_interaction_descriptions
from recsys_nle.nl_explanations.evaluation.readability import READABILITY_SUBSCORE_KEYS
from recsys_nle.nl_explanations.payloads import prepare_interaction_payload

LOGGER = logging.getLogger(__name__)

_READABILITY_OUTPUT_COLUMNS = [
    "id",
    "generation_feather_path",
    "user_id",
    "explanation_text",
    *READABILITY_SUBSCORE_KEYS,
    "overall",
]

_READABILITY_METRIC_COLUMNS = [*READABILITY_SUBSCORE_KEYS, "overall"]

_INTERACTION_MATCH_AI_COLUMNS = [
    "id",
    "match_details_feather_path",
    "user_id",
    "interaction_id",
    "explanation_text",
    "interaction_description",
    "score",
    "judgment",
]

_INTERACTION_MATCH_HUMAN_COLUMNS = [
    "id",
    "match_details_feather_path",
    "user_id",
    "interaction_id",
    "explanation_text",
    "interaction_description",
    "score",
]

_EVAL_OUTPUT_CSV_FILENAMES = (
    "readability-ai-labeled.csv",
    "readability-human-labeled.csv",
    "interaction-match-ai-labeled.csv",
    "interaction-match-human-labeled.csv",
)

# Populated only on candidate rows before sampling; removed before final CSV dicts.
_INTERACTION_DESC_SOURCE_KEY = "_interaction_desc_source"
_DPO_RUN_PREFIX = "run_eval_eval_dpo_eval_"
_HMF_REWARD_COMPOSITE_KEY = "reward_composite_human_feedback_model"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for gathering evaluation datasets."""
    parser = argparse.ArgumentParser(
        description=(
            "Sample rows from pipeline output directories and write readability and "
            "interaction-match CSV pairs for human labeling."
        ),
    )
    parser.add_argument(
        "--data-source-dir",
        type=Path,
        required=True,
        help=(
            "Root directory containing top-level pipeline output folders "
            '(each with exactly one timestamp subdirectory; e.g. repo "outputs").'
        ),
    )
    parser.add_argument(
        "--n-readability-samples",
        type=int,
        required=True,
        help="Total readability explanation rows in each readability output CSV.",
    )
    parser.add_argument(
        "--n-interaction-match-samples",
        type=int,
        required=True,
        help="Total interaction-match rows in each interaction-match output CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output CSV files (default: current working directory).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for reproducible sampling and shuffling (default: 42).",
    )
    parser.add_argument(
        "--best-runs-only",
        type=_parse_bool_flag,
        required=True,
        help="When true, sample only table-equivalent best runs; when false, keep current sampling behavior.",
    )
    parser.add_argument(
        "--log-level",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        default="INFO",
        help="Logging verbosity level (default: INFO).",
    )
    return parser.parse_args(argv)


def _parse_bool_flag(value: str) -> bool:
    """Parse strict CLI boolean strings ``true``/``false``."""
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    msg = f"Expected 'true' or 'false', got {value!r}"
    raise argparse.ArgumentTypeError(msg)


def _readability_mean_columns() -> tuple[str, ...]:
    """Return evaluation.feather column names for readability subscores."""
    return tuple(f"readability_{key}_mean" for key in READABILITY_SUBSCORE_KEYS)


def _scalar_user_id(value: object) -> object:
    """Normalise a Feather ``user_id`` cell for CSV export."""
    if value is None or value is pd.NA:
        out: object = ""
    elif isinstance(value, float) and math.isnan(value):
        out = ""
    elif isinstance(value, bool):
        out = value
    elif isinstance(value, numbers.Integral):
        out = int(value)
    elif hasattr(value, "item"):
        try:
            out = _scalar_user_id(value.item())
        except (AttributeError, ValueError):
            out = value
    else:
        out = value
    return out


def _scalar_interaction_id(value: object) -> object:
    """Normalise ``interaction_id`` for CSV export."""
    if value is None or value is pd.NA:
        out: object = ""
    elif isinstance(value, float) and math.isnan(value):
        out = ""
    elif isinstance(value, numbers.Integral):
        out = int(value)
    elif hasattr(value, "item"):
        try:
            out = _scalar_interaction_id(value.item())
        except (AttributeError, ValueError):
            out = value
    else:
        out = value
    return out


def _explanation_text_from_cell(explanation: object) -> str:
    """Coalesce explanation text for CSV export."""
    if explanation is None or (isinstance(explanation, float) and math.isnan(explanation)):
        return ""
    return str(explanation)


def _score_for_csv(value: object) -> object:
    """Format a score for CSV (blank when missing/NaN)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return value


def _list_timestamp_roots(data_source_dir: Path) -> list[tuple[Path, Path]]:
    """Return sorted (top_level_dir, single_timestamp_subdir) pairs.

    Raises:
        ValueError: If there are no top-level directories or a folder has != 1 subdir.

    """
    if not data_source_dir.is_dir():
        msg = f"Data source is not a directory: {data_source_dir}"
        raise ValueError(msg)

    tops = sorted(p for p in data_source_dir.iterdir() if p.is_dir())
    if not tops:
        msg = f"No top-level directories under {data_source_dir}"
        raise ValueError(msg)

    out: list[tuple[Path, Path]] = []
    for top in tops:
        subdirs = sorted(p for p in top.iterdir() if p.is_dir())
        if len(subdirs) != 1:
            msg = f"Expected exactly one timestamp subdirectory under {top}, found {len(subdirs)}"
            raise ValueError(msg)
        out.append((top, subdirs[0]))
    return out


def _load_readability_rows(run_leaf: Path) -> list[dict[str, object]]:
    """Load aligned explanation + readability rows from one run leaf directory."""
    gen_path = run_leaf / "generation.feather"
    eval_path = run_leaf / "evaluation.feather"
    if not gen_path.is_file() or not eval_path.is_file():
        return []

    gen = pd.read_feather(gen_path)
    ev = pd.read_feather(eval_path)

    if len(gen) != len(ev):
        msg = f"generation/evaluation row count mismatch in {run_leaf}: {len(gen)} vs {len(ev)}"
        raise ValueError(msg)
    if "user_id" not in gen.columns or "user_id" not in ev.columns:
        msg = f"Missing user_id column in {run_leaf} feather exports"
        raise ValueError(msg)

    gen_uid = gen["user_id"].reset_index(drop=True)
    ev_uid = ev["user_id"].reset_index(drop=True)
    if not gen_uid.equals(ev_uid):
        msg = f"user_id alignment mismatch between generation and evaluation in {run_leaf}"
        raise ValueError(msg)

    mean_cols = _readability_mean_columns()
    overall_col = "readability_overall_mean"
    missing = [c for c in (*mean_cols, overall_col) if c not in ev.columns]
    if missing:
        msg = f"evaluation.feather in {run_leaf} missing columns: {missing}"
        raise ValueError(msg)

    rows: list[dict[str, object]] = []
    gen_path_str = str(gen_path.resolve())
    for i in range(len(gen)):
        user_id = _scalar_user_id(gen_uid.iloc[i])
        explanation = gen["explanation_text"].iloc[i] if "explanation_text" in gen.columns else ""
        explanation_text = _explanation_text_from_cell(explanation)

        row_out: dict[str, object] = {
            "generation_feather_path": gen_path_str,
            "user_id": user_id,
            "explanation_text": explanation_text,
        }
        for key, col in zip(READABILITY_SUBSCORE_KEYS, mean_cols, strict=True):
            val = ev[col].iloc[i]
            if val is None or (isinstance(val, float) and math.isnan(val)):
                row_out[key] = ""
            else:
                row_out[key] = val
        overall_val = ev[overall_col].iloc[i]
        if overall_val is None or (isinstance(overall_val, float) and math.isnan(overall_val)):
            row_out["overall"] = ""
        else:
            row_out["overall"] = overall_val
        rows.append(row_out)

    return rows


def _scalar_is_na(value: object) -> bool:
    """Return True if ``value`` is a missing scalar for CSV/feather cells."""
    if value is None or value is pd.NA:
        return True
    return bool(pd.isna(value))  # type: ignore[call-overload]


def _try_int(value: object) -> int | None:
    """Coerce a scalar cell to ``int`` or return ``None``."""
    if _scalar_is_na(value):
        out: int | None = None
    elif isinstance(value, numbers.Integral):
        out = int(value)
    elif isinstance(value, float):
        out = None if math.isnan(value) else int(value)
    elif hasattr(value, "item"):
        try:
            out = _try_int(value.item())
        except (AttributeError, ValueError):
            out = None
    else:
        try:
            out = int(value)  # type: ignore[call-overload]
        except (TypeError, ValueError):
            out = None
    return out


def _judgment_as_str(judgment_raw: object) -> str:
    """Normalise judgment text for CSV export."""
    if _scalar_is_na(judgment_raw):
        return ""
    if isinstance(judgment_raw, str):
        return judgment_raw.strip()
    return str(judgment_raw).strip()


def _explanations_by_user(gen: pd.DataFrame) -> dict[int, str]:
    """Map ``user_id`` to explanation text from ``generation.feather``."""
    explanations: dict[int, str] = {}
    for _, g_row in gen.iterrows():
        uid = _try_int(g_row["user_id"])
        if uid is None:
            continue
        explanations[uid] = _explanation_text_from_cell(g_row["explanation_text"])
    return explanations


def _interaction_row_index(interactions: pd.DataFrame) -> dict[tuple[int, int], pd.Series]:
    """Map ``(user_id, interaction_id)`` to the interactions row."""
    int_index: dict[tuple[int, int], pd.Series] = {}
    for _, irow in interactions.iterrows():
        u_i = _try_int(irow["user_id"])
        i_i = _try_int(irow["interaction_id"])
        if u_i is None or i_i is None:
            continue
        int_index[(u_i, i_i)] = irow
    return int_index


def interaction_row_to_evaluator_frame(row: Mapping[str, object]) -> pd.DataFrame:
    """Build a one-row frame compatible with ``prepare_interaction_payload``."""
    item_id = row.get("item_id")
    movie_id: int | None = None
    if isinstance(item_id, numbers.Integral) or (isinstance(item_id, float) and not math.isnan(item_id)):
        movie_id = int(item_id)

    if movie_id is None:
        return pd.DataFrame()

    out_row: dict[str, object] = {"movie_id": movie_id}
    rating = row.get("rating")
    if isinstance(rating, numbers.Real) and not (isinstance(rating, float) and math.isnan(rating)):
        out_row["rating"] = float(rating)
    att = row.get("attribution_score")
    if isinstance(att, numbers.Real) and not (isinstance(att, float) and math.isnan(att)):
        w = float(att)
        out_row["weight"] = w
        out_row["importance"] = abs(w)
    return pd.DataFrame([out_row])


def interaction_description_from_exported_interaction_row(row: Mapping[str, object]) -> str:
    """Return the same interaction line used inside interaction-scoring prompts."""
    frame = interaction_row_to_evaluator_frame(row)
    if frame.empty:
        return "{}"
    payload = prepare_interaction_payload(frame, max_items=None)
    if not payload:
        return "{}"
    descs = build_interaction_descriptions(payload)
    return descs[0] if descs else "{}"


def _rows_from_match_details(
    det_path: Path,
    *,
    int_index: dict[tuple[int, int], pd.Series],
    explanations: dict[int, str],
) -> list[dict[str, object]]:
    """Expand one ``*_match_details.feather`` file into gatherable rows (description filled after sampling)."""
    det = pd.read_feather(det_path)
    need = {"user_id", "interaction_id", "score", "judgment"}
    if det.empty or not need.issubset(det.columns):
        return []
    det_path_str = str(det_path.resolve())
    out: list[dict[str, object]] = []
    for _, drow in det.iterrows():
        uid = _try_int(drow["user_id"])
        inter_id = _try_int(drow["interaction_id"])
        if uid is None or inter_id is None:
            continue
        irow = int_index.get((uid, inter_id))
        if irow is None:
            continue
        row_for_desc: dict[str, object] = {str(k): v for k, v in irow.to_dict().items()}
        out.append(
            {
                "match_details_feather_path": det_path_str,
                "user_id": _scalar_user_id(uid),
                "interaction_id": _scalar_interaction_id(inter_id),
                "explanation_text": explanations.get(uid, ""),
                _INTERACTION_DESC_SOURCE_KEY: row_for_desc,
                "score": drow["score"],
                "judgment": _judgment_as_str(drow["judgment"]),
            },
        )
    return out


def _load_interaction_match_rows(run_leaf: Path) -> list[dict[str, object]]:
    """Load candidate interaction-match rows from cfx and non-cfx detail feathers."""
    gen_path = run_leaf / "generation.feather"
    int_path = run_leaf / "interactions.feather"
    if not gen_path.is_file() or not int_path.is_file():
        return []

    gen = pd.read_feather(gen_path)
    interactions = pd.read_feather(int_path)
    if "user_id" not in gen.columns or "explanation_text" not in gen.columns:
        return []
    required_cols = {"user_id", "interaction_id", "item_id"}
    if interactions.empty or not required_cols.issubset(interactions.columns):
        return []

    explanations = _explanations_by_user(gen)
    int_index = _interaction_row_index(interactions)
    rows: list[dict[str, object]] = []
    for rel_name in ("cfx_match_details.feather", "non_cfx_match_details.feather"):
        det_path = run_leaf / rel_name
        if det_path.is_file():
            rows.extend(_rows_from_match_details(det_path, int_index=int_index, explanations=explanations))
    return rows


@dataclass(frozen=True)
class RunLeafPool:
    """Candidate rows loaded from one pipeline run leaf directory."""

    path: Path
    rows: list[dict[str, object]]


@dataclass(frozen=True)
class TopSource:
    """One top-level experiment and its per-trial row pools."""

    top_dir: Path
    is_sweep: bool
    run_leaves: list[RunLeafPool]


@dataclass(frozen=True)
class SweepTrialCandidate:
    """One sweep trial candidate discovered under a top-level experiment."""

    run_id: str
    timestamp_dir: Path
    trial_dir: Path
    validation_score: float | None
    has_test_hmf: bool


def _dedup_key_value(value: object) -> object:
    """Normalise a row cell to a hashable value for duplicate detection."""
    if value is None or value is pd.NA:
        out: object = None
    elif isinstance(value, float) and math.isnan(value):
        out = None
    elif isinstance(value, bool):
        out = value
    elif isinstance(value, numbers.Integral):
        out = int(value)
    elif isinstance(value, numbers.Real):
        out = float(value)
    elif isinstance(value, str):
        out = value
    elif hasattr(value, "item"):
        try:
            out = _dedup_key_value(value.item())
        except (AttributeError, ValueError):
            out = str(value)
    else:
        out = str(value)
    return out


def _row_dedup_key(row: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    """Return a hashable key from full row content, excluding internal-only fields."""
    return tuple((key, _dedup_key_value(row[key])) for key in sorted(row) if key != _INTERACTION_DESC_SOURCE_KEY)


def _run_leaves_for_timestamp(
    timestamp_root: Path,
    load_run_leaf: Callable[[Path], list[dict[str, object]]],
) -> list[RunLeafPool]:
    """List run-leaf pools under one timestamp directory (sweep trials or a single leaf)."""
    sweep_dir = timestamp_root / "sweep"
    if sweep_dir.is_dir():
        return [
            RunLeafPool(path=trial, rows=load_run_leaf(trial))
            for trial in sorted(p for p in sweep_dir.iterdir() if p.is_dir())
        ]
    return [RunLeafPool(path=timestamp_root, rows=load_run_leaf(timestamp_root))]


def _load_summary_payload(run_dir: Path, file_name: str) -> dict[str, object] | None:
    """Load one summary JSON payload as a plain mapping."""
    summary_path = run_dir / file_name
    if not summary_path.is_file():
        return None
    try:
        with summary_path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _timestamp_dirs_for_top_dir(top_dir: Path) -> list[Path]:
    """Return sorted timestamp directories under one top-level experiment directory."""
    if not top_dir.is_dir():
        return []
    return sorted((path for path in top_dir.iterdir() if path.is_dir()), key=lambda path: path.name)


def _load_hmf_split_results(run_leaf: Path, split: str) -> dict[str, object] | None:
    """Load one split from run_human_model_feedback_summary results."""
    payload = _load_summary_payload(run_leaf, "run_human_model_feedback_summary.json")
    if payload is None:
        return None
    results_obj = payload.get("results")
    if not isinstance(results_obj, dict):
        return None
    split_obj = results_obj.get(split)
    if not isinstance(split_obj, dict):
        return None
    return split_obj


def _read_trials_payload(sweep_path: Path) -> list[dict[str, object]]:
    """Load trials list from sweep/trials.json."""
    payload = _load_summary_payload(sweep_path, "trials.json")
    if payload is None:
        return []
    trials_obj = payload.get("trials")
    if not isinstance(trials_obj, list):
        return []
    return [trial for trial in trials_obj if isinstance(trial, dict)]


def _resolve_trial_disk_path(sweep_path: Path, trial: dict[str, object]) -> Path:
    """Resolve an on-disk sweep trial folder from metadata."""
    trial_dir_obj = trial.get("trial_dir")
    if isinstance(trial_dir_obj, str):
        candidate = Path(trial_dir_obj)
        try:
            if candidate.is_dir():
                return candidate.resolve()
        except OSError:
            pass
    run_id_obj = trial.get("run_id")
    if isinstance(run_id_obj, (str, int)):
        nested = sweep_path / str(run_id_obj)
        try:
            if nested.is_dir():
                return nested.resolve()
        except OSError:
            pass
    if isinstance(trial_dir_obj, str):
        return Path(trial_dir_obj)
    return sweep_path


def _is_dir(path: Path) -> bool:
    """Return ``True`` when a path is an existing directory."""
    try:
        return path.is_dir()
    except OSError:
        return False


def _parse_validation_score(run_leaf: Path) -> float | None:
    """Return finite validation HMF composite score for one run leaf, else ``None``."""
    validation_results = _load_hmf_split_results(run_leaf, "validation")
    if validation_results is None:
        return None
    raw_score = validation_results.get(_HMF_REWARD_COMPOSITE_KEY)
    if raw_score is None or isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        return None
    score = float(raw_score)
    if not math.isfinite(score):
        return None
    return score


def _collect_sweep_trial_candidates(top_dir: Path) -> list[SweepTrialCandidate]:
    """Collect sweep trial candidates across all timestamps for one top-level directory."""
    candidates: list[SweepTrialCandidate] = []
    for timestamp_dir in _timestamp_dirs_for_top_dir(top_dir):
        sweep_path = timestamp_dir / "sweep"
        if not _is_dir(sweep_path):
            continue
        for trial in _read_trials_payload(sweep_path):
            run_id_obj = trial.get("run_id")
            if run_id_obj is None:
                continue
            trial_dir = _resolve_trial_disk_path(sweep_path, trial)
            if not _is_dir(trial_dir):
                continue
            candidates.append(
                SweepTrialCandidate(
                    run_id=str(run_id_obj),
                    timestamp_dir=timestamp_dir,
                    trial_dir=trial_dir,
                    validation_score=_parse_validation_score(trial_dir),
                    has_test_hmf=_load_hmf_split_results(trial_dir, "test") is not None,
                ),
            )
    return candidates


def _best_validation_run_id_from_candidates(candidates: list[SweepTrialCandidate], *, top_dir: Path) -> str:
    """Return best run_id by validation HMF score from candidate trials."""
    best_score_by_run_id: dict[str, float] = {}
    for candidate in candidates:
        if candidate.validation_score is None:
            continue
        previous = best_score_by_run_id.get(candidate.run_id)
        if previous is None or candidate.validation_score > previous:
            best_score_by_run_id[candidate.run_id] = candidate.validation_score
    if not best_score_by_run_id:
        msg = f"Validation HMF calibrated composite not available for best-run selection: {top_dir}"
        raise ValueError(msg)
    ranked = sorted(
        ((run_id, score) for run_id, score in best_score_by_run_id.items()),
        key=lambda pair: (-pair[1], pair[0]),
    )
    # Deterministic tie-break: highest score, then lexicographically smallest run_id.
    return ranked[0][0]


def _select_latest_leaf_for_run_id(candidates: list[SweepTrialCandidate], run_id: str, *, top_dir: Path) -> Path:
    """Select a run leaf for one run_id from the latest sweep timestamp."""
    latest_timestamp = max(candidate.timestamp_dir.name for candidate in candidates)
    latest_candidates = [candidate for candidate in candidates if candidate.timestamp_dir.name == latest_timestamp]
    matching = [candidate for candidate in latest_candidates if candidate.run_id == run_id]
    if not matching:
        msg = f"Best validation run id {run_id!r} not present in latest sweep timestamp for {top_dir}"
        raise ValueError(msg)
    with_test = [candidate for candidate in matching if candidate.has_test_hmf]
    pool = with_test or matching
    selected = max(pool, key=lambda candidate: str(candidate.trial_dir))
    return selected.trial_dir


def _latest_non_sweep_run_leaf(top_dir: Path) -> Path | None:
    """Return latest non-sweep timestamp directory containing run_summary.json."""
    for timestamp_dir in reversed(_timestamp_dirs_for_top_dir(top_dir)):
        if _is_dir(timestamp_dir / "sweep"):
            continue
        if (timestamp_dir / "run_summary.json").is_file():
            return timestamp_dir
    return None


def _best_run_leaves_for_top_dir(
    top_dir: Path,
    load_run_leaf: Callable[[Path], list[dict[str, object]]],
) -> tuple[bool, list[RunLeafPool]]:
    """Return run leaves for best-runs-only mode under one top-level run directory."""
    candidates = _collect_sweep_trial_candidates(top_dir)
    if candidates:
        if not top_dir.name.startswith(_DPO_RUN_PREFIX):
            msg = f"Expected DPO run prefix for sweep in best-runs-only mode: {top_dir}"
            raise ValueError(msg)
        best_id = _best_validation_run_id_from_candidates(candidates, top_dir=top_dir)
        test_leaf = _select_latest_leaf_for_run_id(candidates, best_id, top_dir=top_dir)
        return True, [RunLeafPool(path=test_leaf, rows=load_run_leaf(test_leaf))]
    latest_leaf = _latest_non_sweep_run_leaf(top_dir)
    if latest_leaf is None:
        msg = f"Missing latest non-sweep run leaf with run_summary.json under {top_dir}"
        raise ValueError(msg)
    return False, [RunLeafPool(path=latest_leaf, rows=load_run_leaf(latest_leaf))]


def _build_sampling_index(
    data_source_dir: Path,
    load_run_leaf: Callable[[Path], list[dict[str, object]]],
    *,
    best_runs_only: bool,
) -> list[TopSource]:
    """Build sampling index entries for top-level dirs that have at least one candidate row."""
    index: list[TopSource] = []
    if best_runs_only:
        if not data_source_dir.is_dir():
            msg = f"Data source is not a directory: {data_source_dir}"
            raise ValueError(msg)
        top_dirs = sorted(path for path in data_source_dir.iterdir() if path.is_dir())
        if not top_dirs:
            msg = f"No top-level directories under {data_source_dir}"
            raise ValueError(msg)
        for top_dir in top_dirs:
            is_sweep, run_leaves = _best_run_leaves_for_top_dir(top_dir, load_run_leaf)
            if any(leaf.rows for leaf in run_leaves):
                index.append(
                    TopSource(
                        top_dir=top_dir,
                        is_sweep=is_sweep,
                        run_leaves=run_leaves,
                    ),
                )
        return index
    for top_dir, ts_path in _list_timestamp_roots(data_source_dir):
        run_leaves = _run_leaves_for_timestamp(ts_path, load_run_leaf)
        if any(leaf.rows for leaf in run_leaves):
            index.append(
                TopSource(
                    top_dir=top_dir,
                    is_sweep=(ts_path / "sweep").is_dir(),
                    run_leaves=run_leaves,
                ),
            )
    return index


def _count_unique_rows(index: list[TopSource]) -> int:
    """Count globally unique rows across all indexed run leaves."""
    seen: set[tuple[tuple[str, object], ...]] = set()
    for top in index:
        for leaf in top.run_leaves:
            for row in leaf.rows:
                seen.add(_row_dedup_key(row))
    return len(seen)


def _gather_sampled_rows(
    *,
    data_source_dir: Path,
    n_samples: int,
    random_seed: int,
    load_run_leaf: Callable[[Path], list[dict[str, object]]],
    not_enough_item_name: str,
    best_runs_only: bool = False,
) -> list[dict[str, object]]:
    """Sample ``n_samples`` unique rows via round-by-round random top/trial/row selection."""
    if n_samples < 0:
        msg = f"n_samples must be non-negative, got {n_samples}"
        raise ValueError(msg)
    if n_samples == 0:
        return []

    index = _build_sampling_index(data_source_dir, load_run_leaf, best_runs_only=best_runs_only)
    if not index:
        msg = f"No {not_enough_item_name} found under {data_source_dir}"
        raise ValueError(msg)

    unique_count = _count_unique_rows(index)
    if n_samples > unique_count:
        msg = f"Not enough unique {not_enough_item_name}: need {n_samples}, have {unique_count}"
        raise ValueError(msg)

    rng = random.Random(random_seed)  # noqa: S311 - dataset sampling, not cryptographic use
    sampled: list[dict[str, object]] = []
    seen: set[tuple[tuple[str, object], ...]] = set()
    consecutive_dupes = 0

    while len(sampled) < n_samples:
        top = rng.choice(index)
        leaves_with_rows = [leaf for leaf in top.run_leaves if leaf.rows]
        if not leaves_with_rows:
            continue

        leaf = rng.choice(leaves_with_rows) if top.is_sweep else leaves_with_rows[0]
        row = dict(rng.choice(leaf.rows))
        key = _row_dedup_key(row)

        if key in seen:
            consecutive_dupes += 1
            if consecutive_dupes % 100 == 0:
                LOGGER.warning(
                    "Skipped %d consecutive duplicate rows while sampling",
                    consecutive_dupes,
                )
            continue

        consecutive_dupes = 0
        seen.add(key)
        sampled.append(row)

    rng.shuffle(sampled)
    return sampled


def _rows_with_sequential_ids(
    rows: list[dict[str, object]],
    columns: list[str],
) -> list[dict[str, object]]:
    """Return rows with a first-column ``id`` field numbered from 1 to N."""
    numbered_rows: list[dict[str, object]] = []
    for row_index, row in enumerate(rows, start=1):
        numbered_row: dict[str, object] = {}
        for column in columns:
            if column == "id":
                numbered_row["id"] = row_index
            else:
                numbered_row[column] = row.get(column, "")
        numbered_rows.append(numbered_row)
    return numbered_rows


def gather_readability_rows(
    *,
    data_source_dir: Path,
    n_samples: int,
    random_seed: int,
    best_runs_only: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Sample rows and return (ai_rows, human_rows) with identical order; human metrics blank."""
    sampled = _gather_sampled_rows(
        data_source_dir=data_source_dir,
        n_samples=n_samples,
        random_seed=random_seed,
        load_run_leaf=_load_readability_rows,
        not_enough_item_name="explanations",
        best_runs_only=best_runs_only,
    )

    ai_rows: list[dict[str, object]] = []
    human_rows: list[dict[str, object]] = []
    for row in sampled:
        ai_rows.append({k: row.get(k, "") for k in _READABILITY_OUTPUT_COLUMNS})
        human_row = {k: row.get(k, "") for k in _READABILITY_OUTPUT_COLUMNS}
        for m in _READABILITY_METRIC_COLUMNS:
            human_row[m] = ""
        human_rows.append(human_row)

    return ai_rows, human_rows


def gather_interaction_match_rows(
    *,
    data_source_dir: Path,
    n_samples: int,
    random_seed: int,
    best_runs_only: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Sample interaction-match rows; human rows omit judgment and blank score."""
    sampled = _gather_sampled_rows(
        data_source_dir=data_source_dir,
        n_samples=n_samples,
        random_seed=random_seed,
        load_run_leaf=_load_interaction_match_rows,
        not_enough_item_name="interaction-match rows",
        best_runs_only=best_runs_only,
    )

    ai_rows: list[dict[str, object]] = []
    human_rows: list[dict[str, object]] = []
    for row in sampled:
        desc_source_raw = row.pop(_INTERACTION_DESC_SOURCE_KEY)
        if not isinstance(desc_source_raw, dict):
            msg = f"Expected mapping for {_INTERACTION_DESC_SOURCE_KEY!r}"
            raise TypeError(msg)
        desc_source: dict[str, object] = {str(k): v for k, v in desc_source_raw.items()}
        interaction_description = interaction_description_from_exported_interaction_row(desc_source)
        score_val = _score_for_csv(row.get("score"))
        judgment = str(row.get("judgment", "") or "")
        ai_rows.append(
            {
                "match_details_feather_path": row.get("match_details_feather_path", ""),
                "user_id": row.get("user_id", ""),
                "interaction_id": row.get("interaction_id", ""),
                "explanation_text": row.get("explanation_text", ""),
                "interaction_description": interaction_description,
                "score": score_val,
                "judgment": judgment,
            },
        )
        human_rows.append(
            {
                "match_details_feather_path": row.get("match_details_feather_path", ""),
                "user_id": row.get("user_id", ""),
                "interaction_id": row.get("interaction_id", ""),
                "explanation_text": row.get("explanation_text", ""),
                "interaction_description": interaction_description,
                "score": "",
            },
        )

    return ai_rows, human_rows


def write_readability_csvs(
    *,
    output_dir: Path,
    ai_rows: list[dict[str, object]],
    human_rows: list[dict[str, object]],
) -> tuple[Path, Path]:
    """Write paired readability CSV files; ``human_rows`` must mirror ``ai_rows`` with blank metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ai_path = output_dir / "readability-ai-labeled.csv"
    human_path = output_dir / "readability-human-labeled.csv"

    ai_df = pd.DataFrame(
        _rows_with_sequential_ids(ai_rows, _READABILITY_OUTPUT_COLUMNS), columns=_READABILITY_OUTPUT_COLUMNS
    )
    human_df = pd.DataFrame(
        _rows_with_sequential_ids(human_rows, _READABILITY_OUTPUT_COLUMNS),
        columns=_READABILITY_OUTPUT_COLUMNS,
    )
    ai_df.to_csv(ai_path, index=False)
    human_df.to_csv(human_path, index=False)
    return ai_path, human_path


def write_interaction_match_csvs(
    *,
    output_dir: Path,
    ai_rows: list[dict[str, object]],
    human_rows: list[dict[str, object]],
) -> tuple[Path, Path]:
    """Write paired interaction-match CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ai_path = output_dir / "interaction-match-ai-labeled.csv"
    human_path = output_dir / "interaction-match-human-labeled.csv"

    ai_df = pd.DataFrame(
        _rows_with_sequential_ids(ai_rows, _INTERACTION_MATCH_AI_COLUMNS), columns=_INTERACTION_MATCH_AI_COLUMNS
    )
    human_df = pd.DataFrame(
        _rows_with_sequential_ids(human_rows, _INTERACTION_MATCH_HUMAN_COLUMNS),
        columns=_INTERACTION_MATCH_HUMAN_COLUMNS,
    )
    ai_df.to_csv(ai_path, index=False)
    human_df.to_csv(human_path, index=False)
    return ai_path, human_path


def raise_if_eval_outputs_exist(output_dir: Path) -> None:
    """Raise ``ValueError`` when any standard evaluation CSV already exists under ``output_dir``."""
    existing = [name for name in _EVAL_OUTPUT_CSV_FILENAMES if (output_dir / name).is_file()]
    if existing:
        listed = ", ".join(sorted(existing))
        msg = f"Refusing to overwrite existing output file(s) under {output_dir}: {listed}"
        raise ValueError(msg)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    out_dir = args.output_dir.resolve() if args.output_dir is not None else Path.cwd()

    try:
        raise_if_eval_outputs_exist(out_dir)
    except ValueError as exc:
        LOGGER.error("%s", exc)  # noqa: TRY400
        return 1

    # Decorrelate interaction-match RNG from readability when seed is shared.
    interaction_seed = args.random_seed + 1_000_003

    try:
        r_ai, r_human = gather_readability_rows(
            data_source_dir=args.data_source_dir.resolve(),
            n_samples=args.n_readability_samples,
            random_seed=args.random_seed,
            best_runs_only=args.best_runs_only,
        )
        i_ai, i_human = gather_interaction_match_rows(
            data_source_dir=args.data_source_dir.resolve(),
            n_samples=args.n_interaction_match_samples,
            random_seed=interaction_seed,
            best_runs_only=args.best_runs_only,
        )
    except ValueError as exc:
        LOGGER.error("%s", exc)  # noqa: TRY400
        return 1

    r_ai_path, r_human_path = write_readability_csvs(
        output_dir=out_dir,
        ai_rows=r_ai,
        human_rows=r_human,
    )
    i_ai_path, i_human_path = write_interaction_match_csvs(
        output_dir=out_dir,
        ai_rows=i_ai,
        human_rows=i_human,
    )
    LOGGER.info("Wrote %s", r_ai_path)
    LOGGER.info("Wrote %s", r_human_path)
    LOGGER.info("Wrote %s", i_ai_path)
    LOGGER.info("Wrote %s", i_human_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
