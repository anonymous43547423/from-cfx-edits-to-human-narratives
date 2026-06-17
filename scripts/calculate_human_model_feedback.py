"""Score pipeline run outputs with trained human-feedback ModernBERT classifiers."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from recsys_nle.pipeline.metrics import flatten_summary, summarise_metric
from recsys_nle.pipeline.reward import REWARD_TERMS, RewardType, compute_reward

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.gather_eval_datasets import (
    _explanation_text_from_cell,
    _explanations_by_user,
    _interaction_row_index,
    _try_int,
    interaction_description_from_exported_interaction_row,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from transformers import PreTrainedTokenizerBase

LOGGER = logging.getLogger(__name__)

_READABILITY_OUTPUT_NAME = "evaluation_human_feedback_model.feather"
_CFX_OUTPUT_NAME = "cfx_match_details_human_feedback_model.feather"
_NON_CFX_OUTPUT_NAME = "non_cfx_match_details_human_feedback_model.feather"
_SUMMARY_OUTPUT_NAME = "run_human_model_feedback_summary.json"
_MATCH_DETAIL_SOURCES = (
    ("cfx_match_details.feather", _CFX_OUTPUT_NAME),
    ("non_cfx_match_details.feather", _NON_CFX_OUTPUT_NAME),
)


@dataclass(slots=True)
class HumanModelFeedbackRunConfig:
    """Configuration recorded in each human-feedback run summary."""

    readability_human_feedback_model_path: Path
    interaction_human_feedback_model_path: Path
    batch_size: int
    hmf_split: str

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> HumanModelFeedbackRunConfig:
        """Build run configuration from parsed CLI arguments."""
        return cls(
            readability_human_feedback_model_path=args.readability_human_feedback_model_path,
            interaction_human_feedback_model_path=args.interaction_human_feedback_model_path,
            batch_size=args.batch_size,
            hmf_split=args.hmf_split,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable configuration mapping."""
        return {
            "readability_human_feedback_model_path": str(self.readability_human_feedback_model_path),
            "interaction_human_feedback_model_path": str(self.interaction_human_feedback_model_path),
            "batch_size": self.batch_size,
            "hmf_split": self.hmf_split,
        }


@dataclass(slots=True)
class RunLeafMetadata:
    """Derived metadata for one discovered run leaf."""

    run_leaf: Path
    experiment_dir: Path
    timestamp_dir: Path
    run_id: str | None
    target_set: str | None
    user_pool: str | None
    reward_metric_name: str | None


@dataclass(slots=True)
class RunSelection:
    """One selected run leaf plus optional validation payload to merge."""

    run_leaf: Path
    validation_results_for_merge: dict[str, object] | None = None


_CALIBRATED_REWARD_KEY_MAP: dict[str, str] = {
    "explanation_cfx_pattern_match_mean": "explanation_cfx_pattern_human_feedback_model_match_mean",
    "explanation_non_cfx_pattern_match_mean": "explanation_non_cfx_pattern_human_feedback_model_match_mean",
    "readability_overall_mean": "readability_human_feedback_model_score_mean",
}
_HMF_REWARD_COMPOSITE_KEY = "reward_composite_human_feedback_model"
_HMF_REWARD_METRIC_NAME_KEY = "reward_metric_name_human_feedback_model"
_HMF_REWARD_MISSING_COLUMNS_KEY = "reward_composite_human_feedback_model_missing_columns"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for human-feedback model inference."""
    parser = argparse.ArgumentParser(
        description=(
            "Apply trained readability and interaction-match classifiers to pipeline "
            "output run directories and write human-feedback score feathers."
        ),
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        required=True,
        help="Root directory containing pipeline output experiment folders.",
    )
    parser.add_argument(
        "--readability-human-feedback-model-path",
        type=Path,
        required=True,
        help="Directory with the saved readability classifier and tokenizer.",
    )
    parser.add_argument(
        "--interaction-human-feedback-model-path",
        type=Path,
        required=True,
        help="Directory with the saved interaction-match classifier and tokenizer.",
    )
    parser.add_argument(
        "--hmf-split",
        choices=("validation", "test"),
        required=True,
        help="Split mode to calculate and write (`validation` or `test`).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size (default: 32).",
    )
    parser.add_argument(
        "--log-level",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        default="INFO",
        help="Logging verbosity level (default: INFO).",
    )
    return parser.parse_args(argv)


def is_run_leaf(path: Path) -> bool:
    """Return True when ``path`` contains a pipeline generation export."""
    return (path / "generation.feather").is_file()


def discover_run_leaves(outputs_dir: Path) -> list[Path]:
    """Return sorted run-leaf directories under ``outputs_dir`` (including sweep trials)."""
    if not outputs_dir.is_dir():
        msg = f"Outputs directory does not exist or is not a directory: {outputs_dir}"
        raise ValueError(msg)

    leaves: list[Path] = []
    for top in sorted(p for p in outputs_dir.iterdir() if p.is_dir()):
        timestamp_dirs = sorted(p for p in top.iterdir() if p.is_dir())
        if not timestamp_dirs:
            LOGGER.debug("Skipping top-level directory without timestamp subdirs: %s", top)
            continue
        for timestamp_root in timestamp_dirs:
            sweep_dir = timestamp_root / "sweep"
            if sweep_dir.is_dir():
                for trial in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
                    if is_run_leaf(trial):
                        leaves.append(trial)
                    else:
                        LOGGER.debug("Skipping sweep trial without generation.feather: %s", trial)
            elif is_run_leaf(timestamp_root):
                leaves.append(timestamp_root)
            else:
                LOGGER.debug("Skipping timestamp directory without generation.feather: %s", timestamp_root)

    return leaves


def _safe_dict(value: object) -> dict[str, object] | None:
    """Return ``value`` when it is a plain JSON object."""
    if isinstance(value, dict):
        return value
    return None


def _load_run_summary_payload(run_leaf: Path) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Load `(config, results)` from sibling ``run_summary.json`` under ``run_leaf``."""
    summary_path = run_leaf / "run_summary.json"
    if not summary_path.is_file():
        return None, None
    try:
        with summary_path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    return _safe_dict(payload.get("config")), _safe_dict(payload.get("results"))


def _build_run_leaf_metadata(run_leaf: Path) -> RunLeafMetadata:
    """Build run-leaf metadata from path structure and ``run_summary.json``."""
    config, results = _load_run_summary_payload(run_leaf)
    target_set_obj = config.get("target_set") if config is not None else None
    user_pool_obj = config.get("user_pool") if config is not None else None
    reward_metric_obj = results.get("reward_metric_name") if results is not None else None
    timestamp_dir = run_leaf.parent.parent if run_leaf.parent.name == "sweep" else run_leaf
    return RunLeafMetadata(
        run_leaf=run_leaf,
        experiment_dir=timestamp_dir.parent,
        timestamp_dir=timestamp_dir,
        run_id=run_leaf.name if run_leaf.parent.name == "sweep" else None,
        target_set=target_set_obj if isinstance(target_set_obj, str) else None,
        user_pool=user_pool_obj if isinstance(user_pool_obj, str) else None,
        reward_metric_name=reward_metric_obj if isinstance(reward_metric_obj, str) else None,
    )


def _load_hmf_split_results(run_leaf: Path, split: str) -> dict[str, object] | None:
    """Load one split object from ``run_human_model_feedback_summary.json``."""
    summary_path = run_leaf / _SUMMARY_OUTPUT_NAME
    if not summary_path.is_file():
        return None
    try:
        with summary_path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    results = _safe_dict(payload.get("results"))
    if results is None:
        return None
    split_obj = _safe_dict(results.get(split))
    if split_obj is None:
        return None
    return split_obj


def _load_existing_split_results(run_leaf: Path) -> dict[str, dict[str, object]]:
    """Load existing nested split results from prior HMF summary, if present."""
    summary_path = run_leaf / _SUMMARY_OUTPUT_NAME
    if not summary_path.is_file():
        return {}
    try:
        with summary_path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    results = _safe_dict(payload.get("results"))
    if results is None:
        return {}
    nested: dict[str, dict[str, object]] = {}
    for split_name in ("validation", "test"):
        split_obj = _safe_dict(results.get(split_name))
        if split_obj is not None:
            nested[split_name] = dict(split_obj)
    return nested


def _calibrated_reward_composite(
    split_results: dict[str, object],
    reward_metric_name: str | None,
) -> tuple[float, list[str]]:
    """Return calibrated reward composite and missing calibrated columns."""
    if reward_metric_name is None:
        return float("nan"), sorted(_CALIBRATED_REWARD_KEY_MAP)
    try:
        reward_type = RewardType(reward_metric_name)
    except ValueError:
        return float("nan"), sorted(_CALIBRATED_REWARD_KEY_MAP)
    calibrated_values: dict[str, float] = {}
    missing: list[str] = []
    for raw_key in REWARD_TERMS[reward_type]:
        mapped_key = _CALIBRATED_REWARD_KEY_MAP.get(raw_key)
        if mapped_key is None:
            missing.append(raw_key)
            continue
        raw_value = split_results.get(mapped_key)
        if raw_value is None:
            missing.append(raw_key)
            continue
        try:
            parsed = float(raw_value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            missing.append(raw_key)
            continue
        if not math.isfinite(parsed):
            missing.append(raw_key)
            continue
        calibrated_values[raw_key] = parsed
    if missing:
        return float("nan"), sorted(missing)
    return compute_reward(calibrated_values, reward_type), []


def _with_calibrated_composite(
    split_results: dict[str, float],
    reward_metric_name: str | None,
) -> dict[str, object]:
    """Add calibrated composite metadata to one split result payload."""
    merged: dict[str, object] = dict(split_results)
    score, missing = _calibrated_reward_composite(merged, reward_metric_name)
    merged[_HMF_REWARD_METRIC_NAME_KEY] = reward_metric_name
    merged[_HMF_REWARD_COMPOSITE_KEY] = score
    merged[_HMF_REWARD_MISSING_COLUMNS_KEY] = missing
    return merged


def _best_validation_run_id_for_experiment(
    experiment_leaves: list[RunLeafMetadata],
) -> tuple[str, dict[str, object]]:
    """Resolve best validation sweep trial by calibrated composite score."""
    validation_trials = [meta for meta in experiment_leaves if meta.run_id is not None and meta.timestamp_dir.is_dir()]
    if not validation_trials:
        msg = "Missing validation sweep runs required for test split selection."
        raise ValueError(msg)
    validation_by_run: list[tuple[str, float, dict[str, object]]] = []
    for meta in validation_trials:
        split_results = _load_hmf_split_results(meta.run_leaf, "validation")
        if split_results is None:
            continue
        raw_score = split_results.get(_HMF_REWARD_COMPOSITE_KEY)
        if raw_score is None or isinstance(raw_score, bool):
            continue
        if not isinstance(raw_score, (int, float)):
            continue
        score = float(raw_score)
        if not math.isfinite(score):
            continue
        if meta.run_id is None:
            continue
        validation_by_run.append((meta.run_id, score, split_results))
    if not validation_by_run:
        msg = "Validation HMF calibrated composite not available for best-run selection."
        raise ValueError(msg)
    validation_by_run.sort(key=lambda item: (-item[1], item[0]))
    # Deterministic tie-break: highest score, then lexicographically smallest run_id.
    best_id, _, validation_results = validation_by_run[0]
    return best_id, dict(validation_results)


def _select_test_run_leaves_for_experiment(
    experiment_dir: Path,
    experiment_leaves: list[RunLeafMetadata],
) -> list[RunSelection]:
    """Select test runs for one experiment (vanilla + best-validation DPO trial)."""
    if not experiment_leaves:
        return []
    sweep_test = [meta for meta in experiment_leaves if meta.run_id is not None]
    nonsweep_test = [meta for meta in experiment_leaves if meta.run_id is None]
    selections: list[RunSelection] = []
    if sweep_test:
        best_run_id, validation_results = _best_validation_run_id_for_experiment(experiment_leaves)
        matching = [meta for meta in sweep_test if meta.run_id == best_run_id]
        if not matching:
            msg = f"Best validation run id {best_run_id!r} not present in test sweep for {experiment_dir}"
            raise ValueError(msg)
        if len(matching) != 1:
            msg = f"Duplicate test sweep run id {best_run_id!r} found for {experiment_dir}"
            raise ValueError(msg)
        selections.append(
            RunSelection(run_leaf=matching[0].run_leaf, validation_results_for_merge=validation_results),
        )
    selections.extend(RunSelection(run_leaf=meta.run_leaf) for meta in sorted(nonsweep_test, key=lambda m: m.run_leaf))
    return selections


def _select_run_leaves(metadata: list[RunLeafMetadata], hmf_split: str) -> list[RunSelection]:
    """Select run leaves for the requested split mode."""
    if hmf_split == "validation":
        return [RunSelection(run_leaf=meta.run_leaf) for meta in metadata]
    grouped: dict[Path, list[RunLeafMetadata]] = {}
    for meta in metadata:
        grouped.setdefault(meta.experiment_dir, []).append(meta)
    selections: list[RunSelection] = []
    for experiment_dir in sorted(grouped):
        selections.extend(_select_test_run_leaves_for_experiment(experiment_dir, grouped[experiment_dir]))
    return selections


def load_readability_inputs(run_leaf: Path) -> pd.DataFrame | None:
    """Load user-level readability model inputs from ``generation.feather``."""
    gen_path = run_leaf / "generation.feather"
    if not gen_path.is_file():
        return None

    gen = pd.read_feather(gen_path)
    if gen.empty or "user_id" not in gen.columns:
        return None

    rows: list[dict[str, object]] = []
    for _, row in gen.iterrows():
        uid = _try_int(row["user_id"])
        if uid is None:
            continue
        explanation = row["explanation_text"] if "explanation_text" in gen.columns else ""
        rows.append(
            {
                "user_id": uid,
                "explanation_text": _explanation_text_from_cell(explanation),
            },
        )
    if not rows:
        return None
    return pd.DataFrame(rows)


def load_match_details_inputs(run_leaf: Path, match_details_name: str) -> pd.DataFrame | None:
    """Load interaction-match model inputs from one match-details feather file."""
    gen_path = run_leaf / "generation.feather"
    int_path = run_leaf / "interactions.feather"
    det_path = run_leaf / match_details_name
    required_paths = (gen_path, int_path, det_path)
    if not all(path.is_file() for path in required_paths):
        return None

    gen = pd.read_feather(gen_path)
    interactions = pd.read_feather(int_path)
    det = pd.read_feather(det_path)
    need = {"user_id", "interaction_id"}
    required_cols = {"user_id", "interaction_id", "item_id"}
    if (
        det.empty
        or not need.issubset(det.columns)
        or interactions.empty
        or not required_cols.issubset(interactions.columns)
        or "user_id" not in gen.columns
        or "explanation_text" not in gen.columns
    ):
        return None

    explanations = _explanations_by_user(gen)
    int_index = _interaction_row_index(interactions)
    rows: list[dict[str, object]] = []
    for _, drow in det.iterrows():
        uid = _try_int(drow["user_id"])
        inter_id = _try_int(drow["interaction_id"])
        if uid is None or inter_id is None:
            continue
        irow = int_index.get((uid, inter_id))
        if irow is None:
            continue
        row_for_desc = {str(k): v for k, v in irow.to_dict().items()}
        rows.append(
            {
                "user_id": uid,
                "interaction_id": inter_id,
                "explanation_text": explanations.get(uid, ""),
                "interaction_description": interaction_description_from_exported_interaction_row(
                    row_for_desc,
                ),
            },
        )
    if not rows:
        return None
    return pd.DataFrame(rows)


def predict_labels(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    texts1: Sequence[str],
    texts2: Sequence[str] | None = None,
    *,
    batch_size: int,
    device: torch.device,
    use_fp16: bool,
) -> list[int]:
    """Run batched inference and return hard argmax labels (0 or 1)."""
    if texts2 is not None and len(texts1) != len(texts2):
        msg = "texts1 and texts2 must have the same length"
        raise ValueError(msg)
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    model.eval()
    out: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(texts1), batch_size):
            batch1 = list(texts1[start : start + batch_size])
            if texts2 is None:
                enc = tokenizer(
                    batch1,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                batch2 = list(texts2[start : start + batch_size])
                enc = tokenizer(
                    batch1,
                    batch2,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
            enc = {k: v.to(device) for k, v in enc.items()}
            if use_fp16:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    logits = model(**enc).logits
            else:
                logits = model(**enc).logits
            out.extend(logits.argmax(dim=-1).cpu().tolist())
    return out


def write_readability_scores(
    run_leaf: Path,
    inputs: pd.DataFrame,
    scores: list[int],
) -> None:
    """Write ``evaluation_human_feedback_model.feather`` for one run leaf."""
    out = pd.DataFrame(
        {
            "user_id": inputs["user_id"].tolist(),
            "readability_human_feedback_model_score": scores,
        },
    )
    out.to_feather(run_leaf / _READABILITY_OUTPUT_NAME)


def write_match_scores(
    run_leaf: Path,
    output_name: str,
    inputs: pd.DataFrame,
    scores: list[int],
) -> None:
    """Write one interaction-match human-feedback feather for a run leaf."""
    out = pd.DataFrame(
        {
            "interaction_id": inputs["interaction_id"].tolist(),
            "human_feedback_model_score": scores,
            "user_id": inputs["user_id"].tolist(),
        },
    )
    out.to_feather(run_leaf / output_name)


def _load_user_ids(run_leaf: Path) -> list[int]:
    """Return unique user ids from ``generation.feather`` in encounter order."""
    gen_path = run_leaf / "generation.feather"
    if not gen_path.is_file():
        return []

    gen = pd.read_feather(gen_path)
    if gen.empty or "user_id" not in gen.columns:
        return []

    user_ids: list[int] = []
    seen: set[int] = set()
    for _, row in gen.iterrows():
        uid = _try_int(row["user_id"])
        if uid is not None and uid not in seen:
            seen.add(uid)
            user_ids.append(uid)
    return user_ids


def _user_mean_from_match_df(df: pd.DataFrame) -> dict[int, float]:
    """Return per-user mean ``human_feedback_model_score`` values."""
    grouped = df.groupby("user_id")["human_feedback_model_score"].mean()
    by_user: dict[int, float] = {}
    for uid, score in grouped.items():
        user_id = _try_int(uid)
        if user_id is not None:
            by_user[user_id] = float(score)
    return by_user


def _scores_for_users(user_ids: list[int], by_user: dict[int, float]) -> list[float]:
    """Map user ids to scores, using NaN when a user has no score."""
    return [by_user.get(uid, float("nan")) for uid in user_ids]


def _resolve_user_ids(
    run_leaf: Path,
    *,
    readability_df: pd.DataFrame | None,
    cfx_match_df: pd.DataFrame | None,
    non_cfx_match_df: pd.DataFrame | None,
) -> list[int]:
    """Return the user universe for summary aggregation."""
    from_generation = _load_user_ids(run_leaf)
    if from_generation:
        return from_generation

    users: set[int] = set()
    for df in (readability_df, cfx_match_df, non_cfx_match_df):
        if df is not None and not df.empty and "user_id" in df.columns:
            users.update(int(uid) for uid in df["user_id"].unique())
    return sorted(users)


def compute_human_model_feedback_results(
    run_leaf: Path,
    *,
    readability_df: pd.DataFrame | None,
    cfx_match_df: pd.DataFrame | None,
    non_cfx_match_df: pd.DataFrame | None,
    scored_cfx_match: bool,
    scored_non_cfx_match: bool,
) -> dict[str, float]:
    """Aggregate human-feedback scores into run-level summary metrics."""
    user_ids = _resolve_user_ids(
        run_leaf,
        readability_df=readability_df,
        cfx_match_df=cfx_match_df,
        non_cfx_match_df=non_cfx_match_df,
    )
    if not user_ids:
        return {}

    total = len(user_ids)
    results: dict[str, float] = {}

    if readability_df is not None and not readability_df.empty:
        readability_by_user = {
            int(uid): float(score)
            for uid, score in zip(
                readability_df["user_id"],
                readability_df["readability_human_feedback_model_score"],
                strict=True,
            )
        }
        readability_scores = _scores_for_users(user_ids, readability_by_user)
        results.update(
            flatten_summary(
                summarise_metric(readability_scores, total),
                "readability_human_feedback_model_score",
            ),
        )

    cfx_by_user: dict[int, float] = {}
    if cfx_match_df is not None and not cfx_match_df.empty:
        cfx_by_user = _user_mean_from_match_df(cfx_match_df)
        cfx_scores = _scores_for_users(user_ids, cfx_by_user)
        results.update(
            flatten_summary(
                summarise_metric(cfx_scores, total),
                "explanation_cfx_pattern_human_feedback_model_match",
            ),
        )

    non_cfx_by_user: dict[int, float] = {}
    if non_cfx_match_df is not None and not non_cfx_match_df.empty:
        non_cfx_by_user = _user_mean_from_match_df(non_cfx_match_df)
        non_cfx_scores = _scores_for_users(user_ids, non_cfx_by_user)
        results.update(
            flatten_summary(
                summarise_metric(non_cfx_scores, total),
                "explanation_non_cfx_pattern_human_feedback_model_match",
            ),
        )

    if scored_cfx_match and scored_non_cfx_match:
        contrast_scores = [
            cfx_by_user.get(uid, float("nan")) - non_cfx_by_user.get(uid, float("nan")) for uid in user_ids
        ]
        results.update(
            flatten_summary(
                summarise_metric(contrast_scores, total),
                "explanation_pattern_human_feedback_model_contrast",
            ),
        )

    return results


def build_human_model_feedback_summary(
    config: HumanModelFeedbackRunConfig,
    split_results: dict[str, dict[str, object]],
    *,
    target_set: str | None,
    user_pool: str | None,
) -> dict[str, object]:
    """Build the full run summary payload with config and results."""
    config_payload = config.to_dict()
    config_payload["target_set"] = target_set
    config_payload["user_pool"] = user_pool
    return {
        "config": config_payload,
        "results": split_results,
    }


def _nan_to_none(obj: object) -> object:
    """Recursively replace float NaN with None for JSON-safe serialization."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def write_human_model_feedback_summary(run_leaf: Path, summary: dict[str, object]) -> None:
    """Write ``run_human_model_feedback_summary.json`` for one run leaf."""
    (run_leaf / _SUMMARY_OUTPUT_NAME).write_text(
        json.dumps(_nan_to_none(summary), indent=2),
        encoding="utf-8",
    )


def process_run_leaf(  # noqa: PLR0915
    run_leaf: Path,
    *,
    readability_model: torch.nn.Module,
    readability_tokenizer: PreTrainedTokenizerBase,
    interaction_model: torch.nn.Module,
    interaction_tokenizer: PreTrainedTokenizerBase,
    config: HumanModelFeedbackRunConfig,
    device: torch.device,
    use_fp16: bool,
    hmf_split: str,
    validation_results_for_merge: dict[str, object] | None = None,
) -> None:
    """Score one run leaf and write human-feedback feather exports."""
    readability_df: pd.DataFrame | None = None
    cfx_match_df: pd.DataFrame | None = None
    non_cfx_match_df: pd.DataFrame | None = None
    scored_cfx_match = False
    scored_non_cfx_match = False
    produced_scores = False

    readability_inputs = load_readability_inputs(run_leaf)
    if readability_inputs is not None:
        readability_texts = readability_inputs["explanation_text"].fillna("").astype(str).tolist()
        readability_scores = predict_labels(
            readability_model,
            readability_tokenizer,
            readability_texts,
            batch_size=config.batch_size,
            device=device,
            use_fp16=use_fp16,
        )
        write_readability_scores(run_leaf, readability_inputs, readability_scores)
        readability_df = pd.DataFrame(
            {
                "user_id": readability_inputs["user_id"].tolist(),
                "readability_human_feedback_model_score": readability_scores,
            },
        )
        produced_scores = True
        LOGGER.info(
            "Wrote %s (%d rows) -> %s",
            _READABILITY_OUTPUT_NAME,
            len(readability_scores),
            run_leaf,
        )
    else:
        LOGGER.warning("Skipping readability scoring for %s (missing generation.feather rows)", run_leaf)

    for source_name, output_name in _MATCH_DETAIL_SOURCES:
        match_inputs = load_match_details_inputs(run_leaf, source_name)
        if match_inputs is None:
            LOGGER.debug("Skipping %s for %s (missing or empty inputs)", output_name, run_leaf)
            continue
        explanation_texts = match_inputs["explanation_text"].fillna("").astype(str).tolist()
        interaction_descriptions = match_inputs["interaction_description"].fillna("").astype(str).tolist()
        match_scores = predict_labels(
            interaction_model,
            interaction_tokenizer,
            explanation_texts,
            interaction_descriptions,
            batch_size=config.batch_size,
            device=device,
            use_fp16=use_fp16,
        )
        write_match_scores(run_leaf, output_name, match_inputs, match_scores)
        match_df = pd.DataFrame(
            {
                "interaction_id": match_inputs["interaction_id"].tolist(),
                "human_feedback_model_score": match_scores,
                "user_id": match_inputs["user_id"].tolist(),
            },
        )
        if output_name == _CFX_OUTPUT_NAME:
            cfx_match_df = match_df
            scored_cfx_match = True
        else:
            non_cfx_match_df = match_df
            scored_non_cfx_match = True
        produced_scores = True
        LOGGER.info(
            "Wrote %s (%d rows) -> %s",
            output_name,
            len(match_scores),
            run_leaf,
        )

    if not produced_scores:
        LOGGER.debug("Skipping %s for %s (no scores produced)", _SUMMARY_OUTPUT_NAME, run_leaf)
        return

    results = compute_human_model_feedback_results(
        run_leaf,
        readability_df=readability_df,
        cfx_match_df=cfx_match_df,
        non_cfx_match_df=non_cfx_match_df,
        scored_cfx_match=scored_cfx_match,
        scored_non_cfx_match=scored_non_cfx_match,
    )
    if not results:
        LOGGER.debug("Skipping %s for %s (empty summary results)", _SUMMARY_OUTPUT_NAME, run_leaf)
        return

    run_config, run_results = _load_run_summary_payload(run_leaf)
    reward_metric_name_obj = run_results.get("reward_metric_name") if run_results is not None else None
    reward_metric_name = reward_metric_name_obj if isinstance(reward_metric_name_obj, str) else None
    split_payload = _with_calibrated_composite(results, reward_metric_name)
    split_payload["target_set"] = hmf_split
    user_pool_obj = run_config.get("user_pool") if run_config is not None else None
    split_payload["user_pool"] = user_pool_obj if isinstance(user_pool_obj, str) else None

    merged_results = _load_existing_split_results(run_leaf)
    if validation_results_for_merge is not None:
        merged_results["validation"] = dict(validation_results_for_merge)
    merged_results[hmf_split] = split_payload
    summary = build_human_model_feedback_summary(
        config,
        merged_results,
        target_set=hmf_split,
        user_pool=split_payload["user_pool"] if isinstance(split_payload["user_pool"], str) else None,
    )
    write_human_model_feedback_summary(run_leaf, summary)
    LOGGER.info("Wrote %s -> %s", _SUMMARY_OUTPUT_NAME, run_leaf)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for human-feedback model inference over pipeline outputs."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    if args.batch_size <= 0:
        msg = f"--batch-size must be positive, got {args.batch_size}"
        raise ValueError(msg)

    run_leaves = discover_run_leaves(args.outputs_dir)
    if not run_leaves:
        LOGGER.warning("No run leaves found under %s", args.outputs_dir)
        return 0
    metadata = [_build_run_leaf_metadata(run_leaf) for run_leaf in run_leaves]
    selected = _select_run_leaves(metadata, args.hmf_split)
    if not selected:
        LOGGER.warning("No run leaves selected for split=%s under %s", args.hmf_split, args.outputs_dir)
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda"
    LOGGER.info("Using device=%s fp16=%s batch_size=%d", device, use_fp16, args.batch_size)

    readability_tokenizer = AutoTokenizer.from_pretrained(args.readability_human_feedback_model_path)
    interaction_tokenizer = AutoTokenizer.from_pretrained(args.interaction_human_feedback_model_path)
    readability_model = AutoModelForSequenceClassification.from_pretrained(
        args.readability_human_feedback_model_path,
    ).to(device)
    interaction_model = AutoModelForSequenceClassification.from_pretrained(
        args.interaction_human_feedback_model_path,
    ).to(device)

    config = HumanModelFeedbackRunConfig.from_args(args)
    for selection in selected:
        LOGGER.info("Scoring run leaf (split=%s): %s", args.hmf_split, selection.run_leaf)
        process_run_leaf(
            selection.run_leaf,
            readability_model=readability_model,
            readability_tokenizer=readability_tokenizer,
            interaction_model=interaction_model,
            interaction_tokenizer=interaction_tokenizer,
            config=config,
            device=device,
            use_fp16=use_fp16,
            hmf_split=args.hmf_split,
            validation_results_for_merge=selection.validation_results_for_merge,
        )

    LOGGER.info("Finished scoring %d run leaf(ves) for split=%s", len(selected), args.hmf_split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
