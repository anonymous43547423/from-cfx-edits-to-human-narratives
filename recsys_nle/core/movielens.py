"""Lightweight Movielens artifact container reused across components."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(slots=True)
class MovielensArtifacts:
    """Container with the core pieces required by the recommendation pipeline."""

    ratings: pd.DataFrame
    items: pd.DataFrame
    metadata: pd.DataFrame
    _title_lookup: pd.Series = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Construct derived lookup structures after initialisation."""
        self._title_lookup = self._build_title_lookup()

    def _build_title_lookup(self) -> pd.Series:
        """Construct a mapping from movie identifiers to preferred titles."""
        metadata = self.metadata
        if "movie_id" not in metadata.columns:
            return pd.Series(dtype=object, name="movie_title")

        metadata_indexed = metadata.drop_duplicates(subset="movie_id", keep="first").set_index("movie_id")
        preferred_title_columns = ("title", "metadata_title")
        title_sources: list[pd.Series] = [
            metadata_indexed[column].astype(object)
            for column in preferred_title_columns
            if column in metadata_indexed.columns
        ]

        if not title_sources and "title" in self.items.columns:
            items_indexed = self.items.drop_duplicates(subset="movie_id", keep="first").set_index("movie_id")
            title_sources.append(items_indexed["title"].astype(object))

        if not title_sources:
            return pd.Series(dtype=object, name="movie_title")

        title_lookup = title_sources[0].copy()
        for candidate in title_sources[1:]:
            title_lookup = title_lookup.combine_first(candidate)
        return title_lookup.rename("movie_title")

    @property
    def title_lookup(self) -> pd.Series:
        """Return a copy of the internal movie title lookup."""
        return self._title_lookup.copy()

    def with_titles(
        self,
        frame: pd.DataFrame,
        *,
        output_column: str = "movie_title",
        fill_value: str | None = "Unknown Title",
    ) -> pd.DataFrame:
        """Attach movie titles to a frame of movie identifiers."""
        if "movie_id" not in frame.columns:
            return frame.copy()

        result = frame.copy()
        titles = result["movie_id"].map(self._title_lookup)
        if output_column in result.columns:
            existing = result[output_column].astype(object)
            titles = titles.combine_first(existing)
        if fill_value is not None:
            titles = titles.fillna(fill_value)
        result[output_column] = titles
        return result
