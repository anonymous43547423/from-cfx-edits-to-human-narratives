"""Tests for distance metric helpers."""

# ruff: noqa: S101, PLR2004

from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd
import pytest

from recsys_nle.pipeline.distance_metrics import (
    _build_item_feature_matrix,
    _compute_distance_metrics,
    build_distance_context,
    compute_all_distance_metrics_for_user,
)


def test_item_feature_matrix_includes_genres_and_year_bins() -> None:
    """It encodes genres and year bins into a fixed-length feature vector."""
    metadata_index = {
        0: {"genres": ["Action", "Comedy"], "year": 2000},
        1: {"genres": ["Drama"], "year": 2005},
    }
    matrix = _build_item_feature_matrix(metadata_index, [0, 1])
    assert matrix.shape[0] == 2
    assert matrix.shape[1] == 3 + 5
    assert matrix[0, 0:3].sum() == 2
    assert matrix[1, 0:3].sum() == 1
    assert matrix[0, 3:].sum() == 1
    assert matrix[1, 3:].sum() == 1


def test_compute_all_distance_metrics_for_user_handles_small_sets() -> None:
    """It returns NaN for metrics requiring at least two items."""
    interactions = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 0, "is_counterfactual": True},
            {"user_id": 1, "item_id": 1, "is_counterfactual": False},
        ]
    )
    metadata_index = {
        0: {"genres": ["Action"], "year": 2000},
        1: {"genres": ["Drama"], "year": 2005},
    }
    context = build_distance_context(interactions, metadata_index, [1])
    metrics = compute_all_distance_metrics_for_user(
        1,
        interactions,
        context,
        n_pairs=5,
        random_seed=1,
    )
    assert math.isnan(metrics["user_based_mean_cfx_distance"])
    assert math.isnan(metrics["user_based_mean_non_cfx_distance"])
    assert math.isfinite(metrics["user_based_mean_cfx_non_cfx_distance"])
    assert math.isfinite(metrics["item_based_mean_cfx_non_cfx_distance"])
    assert 0.0 <= metrics["item_based_mean_cfx_non_cfx_distance"] <= 1.0


def test_build_distance_context_limits_to_sampled_users() -> None:
    """It builds the user-item matrix using only sampled users."""
    interactions = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 0, "is_counterfactual": True},
            {"user_id": 2, "item_id": 1, "is_counterfactual": False},
        ]
    )
    metadata_index = {0: {"genres": ["Action"], "year": 2000}, 1: {"genres": ["Drama"], "year": 2005}}
    context = build_distance_context(interactions, metadata_index, [1])
    assert context.user_item_matrix.shape == (1, 1)
    assert context.user_id_to_row == {1: 0}
    assert context.item_ids == [0]
    assert np.all(context.user_item_matrix[0] == np.array([1.0], dtype=np.float32))


def test_compute_distance_metrics_adds_separation_values() -> None:
    """It computes separation ratios for cross/cfx distances."""
    item_vectors = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rng = random.Random(42)  # noqa: S311

    metrics = _compute_distance_metrics(
        cfx_item_ids=[0, 1],
        non_cfx_item_ids=[2],
        item_vectors=item_vectors,
        n_pairs=5,
        rng=rng,
        prefix="user_based",
    )

    assert metrics["user_based_mean_cfx_distance"] == pytest.approx(2.0)
    assert metrics["user_based_mean_cfx_non_cfx_distance"] == pytest.approx(1.0)
    assert metrics["user_based_mean_separation"] == pytest.approx(0.5)
    assert metrics["user_based_median_separation"] == pytest.approx(0.5)
