"""Payload creation helpers for LLM prompt construction."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import pandas as pd


def prepare_recommendation_payload(
    frame: pd.DataFrame,
    *,
    max_items: int | None = 10,
) -> list[Mapping[str, object]]:
    """Serialise recommendation rows for downstream prompting."""
    return _prepare_payload(
        frame,
        max_items=max_items,
        sort_by=(("rank", True), ("score", False), ("movie_id", True)),
    )


def prepare_interaction_payload(
    frame: pd.DataFrame,
    *,
    max_items: int | None = 5,
) -> list[Mapping[str, object]]:
    """Serialise influential interaction rows for downstream prompting."""
    return _prepare_payload(
        frame,
        max_items=max_items,
        sort_by=(("importance", False), ("weight", False)),
    )


def _prepare_payload(
    frame: pd.DataFrame,
    *,
    max_items: int | None,
    sort_by: tuple[tuple[str, bool], ...] | None = None,
) -> list[Mapping[str, object]]:
    """Convert a frame into a sorted, normalised list of records."""
    if frame.empty:
        return []

    ordered = frame.copy()
    if sort_by:
        for column, ascending in sort_by:
            if column in ordered.columns:
                ordered = ordered.sort_values(by=column, ascending=ascending)
                break

    payload = ordered.head(max_items) if max_items else ordered
    raw_records = payload.to_dict(orient="records")
    typed_records = [{str(key): value for key, value in record.items()} for record in raw_records]
    return [_normalise_record(record) for record in typed_records]


def _normalise_record(record: Mapping[str, object]) -> Mapping[str, object]:
    """Normalise timestamps, attach metadata, and harmonise title aliases."""
    clean_record: dict[str, object] = {}
    for key, value in record.items():
        if isinstance(value, pd.Timestamp):
            clean_record[key] = value.isoformat()
        else:
            clean_record[key] = value
    if "movie_title" not in clean_record and "title" in clean_record:
        clean_record.setdefault("movie_title", clean_record["title"])

    movie_metadata = _lookup_movie_metadata(clean_record.get("movie_id"))
    if movie_metadata:
        clean_record.setdefault("movie_title", movie_metadata["movie_title"])
        clean_record.setdefault("genres", movie_metadata["genres"])
        if movie_metadata.get("year") is not None:
            clean_record.setdefault("year", movie_metadata["year"])

    # Ensure genres are stored as a list of strings.
    genres_value = clean_record.get("genres")
    if isinstance(genres_value, str):
        genres = [part.strip() for part in genres_value.split("|") if part.strip()]
        clean_record["genres"] = genres

    return clean_record


def _lookup_movie_metadata(movie_id: object) -> dict[str, object] | None:
    """Retrieve metadata for a given movie identifier."""
    if movie_id is None:
        return None
    try:
        numeric_id = int(movie_id)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None
    return _movie_metadata_index().get(numeric_id)


@lru_cache(maxsize=1)
def _movie_metadata_index() -> dict[int, dict[str, object]]:
    """Load and cache movie-level metadata keyed by identifier."""
    movies_path = Path(__file__).resolve().parents[2] / "datasets" / "ML1M" / "data files" / "movies.dat"
    if not movies_path.exists():
        return {}

    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        header=None,
        names=["MovieID", "MovieName", "Genre"],
        encoding="latin-1",
    )
    movies["movie_id"] = movies["MovieID"].astype(int) - 1
    movies["movie_title"] = movies["MovieName"].astype(str)
    movies["genres"] = (
        movies["Genre"]
        .fillna("")
        .apply(
            lambda value: [part.strip() for part in str(value).split("|") if part.strip()] if value else [],
        )
    )
    movies["year"] = movies["MovieName"].apply(_extract_year)

    index: dict[int, dict[str, object]] = {}
    for _, row in movies[["movie_id", "movie_title", "genres", "year"]].iterrows():
        movie_id = int(row["movie_id"])
        title = str(row["movie_title"] or "")
        raw_genres = row["genres"] or []
        genres = [str(genre).strip() for genre in raw_genres if str(genre).strip()]
        year_value = row["year"]
        year = int(year_value) if not pd.isna(year_value) else None
        index[movie_id] = {"movie_title": title, "genres": genres, "year": year}
    return index


def _extract_year(title: str) -> int | None:
    """Extract a four-digit year from a movie title when present."""
    match = re.search(r"\((\d{4})\)\s*$", title)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None
