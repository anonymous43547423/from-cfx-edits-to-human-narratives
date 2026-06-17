# ruff: noqa: S101, PLR2004
"""Tests for building NL explanation datasets."""

from __future__ import annotations

import pandas as pd

from recsys_nle.core.attribution import UserAttribution
from recsys_nle.nl_explanations.dataset_builder import (
    build_explanation_dataset,
    build_explanation_record,
)


def test_build_explanation_record_returns_payload() -> None:
    """It builds serialisable records when data is available."""
    cfx_interactions = pd.DataFrame(
        [
            {"movie_id": 1, "rating": 5.0, "weight": 0.4, "importance": 0.4},
            {"movie_id": 3, "rating": 4.5, "weight": 0.3, "importance": 0.3},
        ]
    )
    non_cfx_interactions = pd.DataFrame(
        [
            {"movie_id": 5, "rating": 4.0},
        ]
    )

    record = build_explanation_record(
        user_id=42,
        cfx_interactions=cfx_interactions,
        non_cfx_interactions=non_cfx_interactions,
    )

    assert record is not None
    assert isinstance(record, dict)
    assert record["user_id"] == 42

    cfx_payload = record["cfx_interactions"]
    assert isinstance(cfx_payload, list)
    assert len(cfx_payload) == 2
    first_cfx = cfx_payload[0]
    assert isinstance(first_cfx, dict)
    assert first_cfx["movie_id"] == 1

    non_cfx_payload = record["non_cfx_interactions"]
    assert isinstance(non_cfx_payload, list)
    assert len(non_cfx_payload) == 1


def test_build_explanation_record_rejects_empty_inputs() -> None:
    """It returns None when CFX interactions are empty."""
    cfx_interactions = pd.DataFrame(columns=["movie_id", "rating", "weight", "importance"])
    non_cfx_interactions = pd.DataFrame(columns=["movie_id", "rating"])

    record = build_explanation_record(
        user_id=123,
        cfx_interactions=cfx_interactions,
        non_cfx_interactions=non_cfx_interactions,
    )

    assert record is None


def test_build_explanation_dataset_returns_hf_dataset() -> None:
    """It returns a Hugging Face dataset populated with records."""
    cfx_interactions = pd.DataFrame(
        [
            {"movie_id": 101, "rating": 5.0, "weight": 0.4, "importance": 0.4},
            {"movie_id": 102, "rating": 4.5, "weight": 0.3, "importance": 0.3},
        ]
    )
    non_cfx_interactions = pd.DataFrame(
        [
            {"movie_id": 201, "rating": 3.0},
        ]
    )
    attributions: dict[int, UserAttribution] = {
        1: UserAttribution(
            user_id=1,
            cfx_interactions=cfx_interactions,
            non_cfx_interactions=non_cfx_interactions,
        ),
    }

    dataset = build_explanation_dataset(
        attributions=attributions,
        user_ids=[1],
    )

    assert dataset is not None
    assert len(dataset) == 1
    row = dataset[0]
    assert row["user_id"] == 1
    # All CFX interactions should be in the dataset (limiting happens in generator)
    assert len(row["cfx_interactions"]) == 2


def test_build_explanation_dataset_skips_missing_attributions() -> None:
    """It skips users without matching attribution results."""
    cfx_interactions = pd.DataFrame(
        [
            {"movie_id": 101, "rating": 5.0, "weight": 0.4, "importance": 0.4},
        ]
    )
    non_cfx_interactions = pd.DataFrame(columns=["movie_id", "rating"])
    attributions: dict[int, UserAttribution] = {
        1: UserAttribution(
            user_id=1,
            cfx_interactions=cfx_interactions,
            non_cfx_interactions=non_cfx_interactions,
        ),
    }

    dataset = build_explanation_dataset(
        attributions=attributions,
        user_ids=[1, 2],
    )

    assert dataset is not None
    assert len(dataset) == 1
    assert dataset[0]["user_id"] == 1
