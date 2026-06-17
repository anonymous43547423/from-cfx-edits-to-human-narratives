"""Utilities for computing interaction distance metrics."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

_DISTANCE_METRIC_KEYS = (
    "user_based_mean_cfx_distance",
    "user_based_median_cfx_distance",
    "user_based_mean_non_cfx_distance",
    "user_based_median_non_cfx_distance",
    "user_based_mean_cfx_non_cfx_distance",
    "user_based_median_cfx_non_cfx_distance",
    "user_based_mean_separation",
    "user_based_median_separation",
    "item_based_mean_cfx_distance",
    "item_based_median_cfx_distance",
    "item_based_mean_non_cfx_distance",
    "item_based_median_non_cfx_distance",
    "item_based_mean_cfx_non_cfx_distance",
    "item_based_median_cfx_non_cfx_distance",
    "item_based_mean_separation",
    "item_based_median_separation",
)
_MIN_DISTINCT_ITEMS = 2
FloatArray = NDArray[np.floating[Any]]


@dataclass(slots=True)
class DistanceContext:
    """Pre-computed matrices for distance calculations."""

    user_item_matrix: FloatArray
    item_feature_matrix: FloatArray
    user_id_to_row: dict[int, int]
    item_ids: list[int]
    item_id_to_index: dict[int, int]


def build_distance_context(
    interactions: pd.DataFrame,
    metadata_index: Mapping[int, Mapping[str, object]],
    sampled_user_ids: Sequence[int],
) -> DistanceContext:
    """Build pre-computed matrices from sampled test-set interactions."""
    item_ids = _collect_item_ids(interactions, sampled_user_ids)
    item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    user_item_matrix, user_id_to_row = _build_user_item_matrix(
        interactions,
        sampled_user_ids,
        item_id_to_index,
    )
    item_feature_matrix = _build_item_feature_matrix(metadata_index, item_ids)
    return DistanceContext(
        user_item_matrix=user_item_matrix,
        item_feature_matrix=item_feature_matrix,
        user_id_to_row=user_id_to_row,
        item_ids=item_ids,
        item_id_to_index=item_id_to_index,
    )


def compute_user_distance_metrics(
    cfx_item_ids: Sequence[int],
    non_cfx_item_ids: Sequence[int],
    context: DistanceContext,
    n_pairs: int,
    rng: random.Random,
) -> dict[str, float]:
    """Compute all 6 user-based distance metrics for one user."""
    item_vectors = context.user_item_matrix.T
    return _compute_distance_metrics(
        cfx_item_ids=cfx_item_ids,
        non_cfx_item_ids=non_cfx_item_ids,
        item_vectors=item_vectors,
        n_pairs=n_pairs,
        rng=rng,
        prefix="user_based",
    )


def compute_item_distance_metrics(
    cfx_item_ids: Sequence[int],
    non_cfx_item_ids: Sequence[int],
    context: DistanceContext,
    n_pairs: int,
    rng: random.Random,
) -> dict[str, float]:
    """Compute all 6 item-based distance metrics for one user."""
    return _compute_distance_metrics(
        cfx_item_ids=cfx_item_ids,
        non_cfx_item_ids=non_cfx_item_ids,
        item_vectors=context.item_feature_matrix,
        n_pairs=n_pairs,
        rng=rng,
        prefix="item_based",
    )


def compute_all_distance_metrics_for_user(
    user_id: int,
    interactions: pd.DataFrame,
    context: DistanceContext,
    n_pairs: int,
    random_seed: int,
) -> dict[str, float]:
    """Compute all 12 distance metrics for a single user."""
    if interactions.empty:
        return _default_metrics()
    cfx_item_ids = _map_item_ids(
        _extract_item_ids(interactions, user_id, is_counterfactual=True),
        context.item_id_to_index,
    )
    non_cfx_item_ids = _map_item_ids(
        _extract_item_ids(interactions, user_id, is_counterfactual=False),
        context.item_id_to_index,
    )
    rng = random.Random(random_seed)  # noqa: S311
    metrics: dict[str, float] = {}
    metrics.update(
        compute_user_distance_metrics(
            cfx_item_ids=cfx_item_ids,
            non_cfx_item_ids=non_cfx_item_ids,
            context=context,
            n_pairs=n_pairs,
            rng=rng,
        )
    )
    metrics.update(
        compute_item_distance_metrics(
            cfx_item_ids=cfx_item_ids,
            non_cfx_item_ids=non_cfx_item_ids,
            context=context,
            n_pairs=n_pairs,
            rng=rng,
        )
    )
    return metrics


def _build_user_item_matrix(
    interactions: pd.DataFrame,
    sampled_user_ids: Sequence[int],
    item_id_to_index: Mapping[int, int],
) -> tuple[FloatArray, dict[int, int]]:
    """Build a user-item matrix limited to the sampled users."""
    user_ids = _unique_in_order(sampled_user_ids)
    user_id_to_row = {user_id: idx for idx, user_id in enumerate(user_ids)}
    matrix = np.zeros((len(user_ids), len(item_id_to_index)), dtype=np.float32)
    if interactions.empty or not item_id_to_index:
        return matrix, user_id_to_row
    if "user_id" not in interactions.columns or "item_id" not in interactions.columns:
        return matrix, user_id_to_row
    for row in interactions.itertuples(index=False):
        raw_user_id = getattr(row, "user_id", None)
        raw_item_id = getattr(row, "item_id", None)
        user_id = _coerce_int(raw_user_id)
        if user_id is None or user_id not in user_id_to_row:
            continue
        item_id = _coerce_int(raw_item_id)
        if item_id is None or item_id not in item_id_to_index:
            continue
        matrix[user_id_to_row[user_id], item_id_to_index[item_id]] = 1.0
    return matrix, user_id_to_row


def _build_item_feature_matrix(
    metadata_index: Mapping[int, Mapping[str, object]],
    item_ids: Sequence[int],
) -> FloatArray:
    """Build item feature vectors using genres and year bins."""
    genre_list = _collect_genres(metadata_index, item_ids)
    genre_index = {genre: idx for idx, genre in enumerate(genre_list)}
    year_map = _collect_years(metadata_index, item_ids)
    feature_count = len(genre_list) + 5
    matrix = np.zeros((len(item_ids), feature_count), dtype=np.float32)
    item_id_to_row = {item_id: idx for idx, item_id in enumerate(item_ids)}
    _populate_genre_features(matrix, metadata_index, genre_index, item_ids)
    year_bins = _build_year_bins(year_map)
    if year_bins is not None:
        _populate_year_features(matrix, year_bins, item_id_to_row, len(genre_list))
    return matrix


def _collect_genres(
    metadata_index: Mapping[int, Mapping[str, object]],
    item_ids: Sequence[int],
) -> list[str]:
    """Collect the unique genres across items."""
    genre_set: set[str] = set()
    for item_id in item_ids:
        meta = metadata_index.get(item_id, {})
        genres = meta.get("genres", [])
        if isinstance(genres, str):
            genres = [part.strip() for part in genres.split("|") if part.strip()]
        if isinstance(genres, Sequence):
            for genre in genres:
                text = str(genre).strip()
                if text:
                    genre_set.add(text)
    return sorted(genre_set)


def _collect_years(
    metadata_index: Mapping[int, Mapping[str, object]],
    item_ids: Sequence[int],
) -> dict[int, int]:
    """Collect item years as integers."""
    year_map: dict[int, int] = {}
    for item_id in item_ids:
        meta = metadata_index.get(item_id, {})
        year = meta.get("year")
        if isinstance(year, (int, np.integer)):
            year_map[item_id] = int(year)
        elif isinstance(year, str):
            try:
                year_map[item_id] = int(year)
            except ValueError:
                continue
    return year_map


def _populate_genre_features(
    matrix: FloatArray,
    metadata_index: Mapping[int, Mapping[str, object]],
    genre_index: Mapping[str, int],
    item_ids: Sequence[int],
) -> None:
    """Populate genre feature columns in the matrix."""
    for row_idx, item_id in enumerate(item_ids):
        meta = metadata_index.get(item_id, {})
        genres = meta.get("genres", [])
        if isinstance(genres, str):
            genres = [part.strip() for part in genres.split("|") if part.strip()]
        if isinstance(genres, Sequence):
            for genre in genres:
                idx = genre_index.get(str(genre).strip())
                if idx is not None:
                    matrix[row_idx, idx] = 1.0


def _populate_year_features(
    matrix: FloatArray,
    year_bins: Mapping[int, int | None],
    item_id_to_row: Mapping[int, int],
    offset: int,
) -> None:
    """Populate year-bin feature columns in the matrix."""
    for item_id, bin_code in year_bins.items():
        if bin_code is None:
            continue
        row_idx = item_id_to_row.get(item_id)
        if row_idx is None:
            continue
        matrix[row_idx, offset + bin_code] = 1.0


def _build_year_bins(year_map: Mapping[int, int]) -> dict[int, int | None] | None:
    """Assign year bins (0-4) using equal-height quantiles."""
    if not year_map:
        return None
    series = pd.Series(year_map, dtype="float64")
    try:
        bin_codes = pd.qcut(series, q=5, labels=False, duplicates="drop")
    except ValueError:
        return None
    bins: dict[int, int | None] = {}
    for item_id, code in bin_codes.items():
        item_id_int = int(cast("int", item_id))
        if pd.isna(code):
            bins[item_id_int] = None
        else:
            bins[item_id_int] = int(code)
    return bins


def _extract_item_ids(
    interactions: pd.DataFrame,
    user_id: int,
    is_counterfactual: bool,
) -> list[int]:
    """Collect unique item identifiers for a user and counterfactual flag."""
    if interactions.empty:
        return []
    if "user_id" not in interactions.columns or "item_id" not in interactions.columns:
        return []
    if "is_counterfactual" not in interactions.columns:
        return []
    subset = interactions[
        (interactions["user_id"] == user_id) & (interactions["is_counterfactual"] == is_counterfactual)
    ]
    if subset.empty:
        return []
    items: list[int] = []
    for value in subset["item_id"].dropna().tolist():
        try:
            item_id = int(value)
        except (TypeError, ValueError):
            continue
        if item_id not in items:
            items.append(item_id)
    return items


def _collect_item_ids(interactions: pd.DataFrame, sampled_user_ids: Sequence[int]) -> list[int]:
    """Collect unique item identifiers for the sampled users."""
    if interactions.empty:
        return []
    if "user_id" not in interactions.columns or "item_id" not in interactions.columns:
        return []
    sampled_set = set(sampled_user_ids)
    items: list[int] = []
    seen: set[int] = set()
    for row in interactions.itertuples(index=False):
        user_id = _coerce_int(getattr(row, "user_id", None))
        if user_id is None or user_id not in sampled_set:
            continue
        item_id = _coerce_int(getattr(row, "item_id", None))
        if item_id is None or item_id in seen:
            continue
        seen.add(item_id)
        items.append(item_id)
    return items


def _map_item_ids(item_ids: Sequence[int], item_id_to_index: Mapping[int, int]) -> list[int]:
    """Map item identifiers to compact indices."""
    mapped: list[int] = []
    for item_id in item_ids:
        idx = item_id_to_index.get(item_id)
        if idx is not None:
            mapped.append(idx)
    return mapped


def _compute_distance_metrics(
    *,
    cfx_item_ids: Sequence[int],
    non_cfx_item_ids: Sequence[int],
    item_vectors: FloatArray,
    n_pairs: int,
    rng: random.Random,
    prefix: str,
) -> dict[str, float]:
    """Compute distance metrics for cfx/cfx, non/non, and cfx/non pairs."""
    cfx_mean, cfx_median = _sample_pairs_and_compute_stats(
        item_ids_a=cfx_item_ids,
        item_ids_b=cfx_item_ids,
        item_vectors=item_vectors,
        n_pairs=n_pairs,
        rng=rng,
        require_distinct=True,
    )
    non_mean, non_median = _sample_pairs_and_compute_stats(
        item_ids_a=non_cfx_item_ids,
        item_ids_b=non_cfx_item_ids,
        item_vectors=item_vectors,
        n_pairs=n_pairs,
        rng=rng,
        require_distinct=True,
    )
    cross_mean, cross_median = _sample_pairs_and_compute_stats(
        item_ids_a=cfx_item_ids,
        item_ids_b=non_cfx_item_ids,
        item_vectors=item_vectors,
        n_pairs=n_pairs,
        rng=rng,
        require_distinct=False,
    )
    mean_separation = _safe_ratio(cross_mean, cfx_mean)
    median_separation = _safe_ratio(cross_median, cfx_median)
    return {
        f"{prefix}_mean_cfx_distance": cfx_mean,
        f"{prefix}_median_cfx_distance": cfx_median,
        f"{prefix}_mean_non_cfx_distance": non_mean,
        f"{prefix}_median_non_cfx_distance": non_median,
        f"{prefix}_mean_cfx_non_cfx_distance": cross_mean,
        f"{prefix}_median_cfx_non_cfx_distance": cross_median,
        f"{prefix}_mean_separation": mean_separation,
        f"{prefix}_median_separation": median_separation,
    }


def _sample_pairs_and_compute_stats(
    *,
    item_ids_a: Sequence[int],
    item_ids_b: Sequence[int],
    item_vectors: FloatArray,
    n_pairs: int,
    rng: random.Random,
    require_distinct: bool,
) -> tuple[float, float]:
    """Sample item pairs and return mean/median cosine distances."""
    if n_pairs <= 0:
        return float("nan"), float("nan")
    if not item_ids_a or not item_ids_b:
        return float("nan"), float("nan")
    if require_distinct and len(item_ids_a) < _MIN_DISTINCT_ITEMS:
        return float("nan"), float("nan")
    distances: list[float] = []
    for _ in range(n_pairs):
        if require_distinct:
            first_id, second_id = rng.sample(list(item_ids_a), 2)
        else:
            first_id = rng.choice(list(item_ids_a))
            second_id = rng.choice(list(item_ids_b))
        if first_id < 0 or first_id >= item_vectors.shape[0]:
            continue
        if second_id < 0 or second_id >= item_vectors.shape[0]:
            continue
        distance = _compute_cosine_distance(item_vectors[first_id], item_vectors[second_id])
        if math.isfinite(distance):
            distances.append(distance)
    if not distances:
        return float("nan"), float("nan")
    distances_array = np.array(distances, dtype=np.float64)
    return float(distances_array.mean()), float(np.median(distances_array))


def _compute_cosine_distance(vec1: FloatArray, vec2: FloatArray) -> float:
    """Compute cosine distance as 1 - cosine similarity."""
    denom = float(np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom == 0.0:
        return float("nan")
    similarity = float(np.dot(vec1, vec2) / denom)
    return 1.0 - similarity


def _unique_in_order(values: Sequence[int]) -> list[int]:
    """Return unique values while preserving order."""
    seen: set[int] = set()
    result: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(int(value))
    return result


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Compute a finite ratio or return NaN."""
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return float("nan")
    if denominator == 0.0:
        return float("nan")
    return float(numerator / denominator)


def _default_metrics() -> dict[str, float]:
    """Return a default metric dictionary filled with NaNs."""
    return {key: float("nan") for key in _DISTANCE_METRIC_KEYS}


def _coerce_int(value: object) -> int | None:
    """Coerce an input value into an integer when possible."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None
