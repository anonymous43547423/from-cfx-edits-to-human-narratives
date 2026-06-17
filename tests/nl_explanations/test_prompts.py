"""Tests covering prompt serialisation helpers."""

# ruff: noqa: S101

from __future__ import annotations

from recsys_nle.nl_explanations.prompts import (
    serialise_influential_interactions,
)


def test_serialise_influential_interactions_includes_extended_metadata() -> None:
    """Influential interactions should include enriched descriptive fields."""
    items = [
        {
            "movie_id": 236,
            "movie_title": "Forget Paris (1995)",
            "rating": 1.0,
            "weight": 0.93,
            "importance": 0.92,
            "genres": "Comedy|Romance",
            "tags": ["sports", "romance"],
        },
    ]

    text = serialise_influential_interactions(items)

    assert text == '1. {genres="Comedy, Romance", keywords="sports, romance"}'
