# ruff: noqa: S101, SLF001, PLR2004
"""Tests for the create_enriched_movie_metadata script."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pandas as pd

from scripts import create_enriched_movie_metadata

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# _parse_keyword_names
# ---------------------------------------------------------------------------


def test_parse_keyword_names_extracts_names() -> None:
    """Extract keyword names from a well-formed Python-literal string."""
    raw = "[{'id': 1, 'name': 'jealousy'}, {'id': 2, 'name': 'toy'}]"
    assert create_enriched_movie_metadata._parse_keyword_names(raw) == "jealousy|toy"


def test_parse_keyword_names_empty_list() -> None:
    """Return empty string for an empty list literal."""
    assert create_enriched_movie_metadata._parse_keyword_names("[]") == ""


def test_parse_keyword_names_non_string() -> None:
    """Return empty string for non-string input."""
    assert create_enriched_movie_metadata._parse_keyword_names(None) == ""
    assert create_enriched_movie_metadata._parse_keyword_names(42) == ""


def test_parse_keyword_names_malformed() -> None:
    """Return empty string for unparseable input."""
    assert create_enriched_movie_metadata._parse_keyword_names("not a list") == ""


def test_parse_keyword_names_skips_bad_entries() -> None:
    """Skip entries that are not dicts or lack a 'name' key."""
    raw = "[{'id': 1, 'name': 'ok'}, 'bad', {'id': 2}]"
    assert create_enriched_movie_metadata._parse_keyword_names(raw) == "ok"


# ---------------------------------------------------------------------------
# build_enriched_movies
# ---------------------------------------------------------------------------


def _sample_movies() -> pd.DataFrame:
    """Return a tiny movies DataFrame mimicking movies.dat."""
    return pd.DataFrame(
        {
            "MovieID": [1, 2, 3],
            "MovieName": ["Toy Story (1995)", "Jumanji (1995)", "Unknown (1999)"],
            "Genre": ["Animation|Comedy", "Adventure|Fantasy", "Drama"],
        }
    )


def _sample_links() -> pd.DataFrame:
    """Return a tiny links DataFrame."""
    return pd.DataFrame(
        {
            "movieId": [1, 2],
            "imdbId": ["0114709", "0113497"],
            "tmdbId": [862, 8844],
        }
    )


def _sample_keywords() -> pd.DataFrame:
    """Return a tiny keywords DataFrame."""
    return pd.DataFrame(
        {
            "id": [862, 8844],
            "keywords": [
                "[{'id': 1, 'name': 'jealousy'}, {'id': 2, 'name': 'toy'}]",
                "[{'id': 3, 'name': 'board game'}]",
            ],
        }
    )


def test_build_enriched_movies_joins_correctly() -> None:
    """Keywords are joined via links and pipe-delimited."""
    result = create_enriched_movie_metadata.build_enriched_movies(
        _sample_movies(),
        _sample_links(),
        _sample_keywords(),
    )

    assert list(result.columns) == ["MovieID", "MovieName", "Genre", "keyword_names"]
    assert len(result) == 3

    row_toy = result[result["MovieID"] == 1].iloc[0]
    assert row_toy["keyword_names"] == "jealousy|toy"

    row_jumanji = result[result["MovieID"] == 2].iloc[0]
    assert row_jumanji["keyword_names"] == "board game"

    row_unknown = result[result["MovieID"] == 3].iloc[0]
    assert row_unknown["keyword_names"] == ""


def test_build_enriched_movies_preserves_order() -> None:
    """Output rows are in the same order as the input movies."""
    result = create_enriched_movie_metadata.build_enriched_movies(
        _sample_movies(),
        _sample_links(),
        _sample_keywords(),
    )
    assert list(result["MovieID"]) == [1, 2, 3]


# ---------------------------------------------------------------------------
# main (end-to-end with tmp files)
# ---------------------------------------------------------------------------


def test_main_writes_enriched_dat(tmp_path: Path) -> None:
    """Verify main() reads inputs, joins, and writes the output file."""
    movies_path = tmp_path / "movies.dat"
    movies_path.write_text(
        "1::Toy Story (1995)::Animation|Comedy\n2::Jumanji (1995)::Adventure|Fantasy\n",
        encoding="latin-1",
    )

    links_path = tmp_path / "links.csv"
    links_path.write_text("movieId,imdbId,tmdbId\n1,0114709,862\n2,0113497,8844\n")

    keywords_path = tmp_path / "keywords.csv"
    keywords_path.write_text(
        "id,keywords\n"
        "862,\"[{'id': 1, 'name': 'jealousy'}, {'id': 2, 'name': 'toy'}]\"\n"
        "8844,\"[{'id': 3, 'name': 'board game'}]\"\n"
    )

    output_path = tmp_path / "movies_enriched.dat"

    with patch.object(
        create_enriched_movie_metadata,
        "parse_args",
        return_value=Mock(
            movies_dat_path=movies_path,
            links_csv_path=links_path,
            keywords_csv_path=keywords_path,
            output_path=output_path,
            log_level="INFO",
        ),
    ):
        exit_code = create_enriched_movie_metadata.main()

    assert exit_code == 0
    assert output_path.exists()

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert lines[0] == "1::Toy Story (1995)::Animation|Comedy::jealousy|toy"
    assert lines[1] == "2::Jumanji (1995)::Adventure|Fantasy::board game"
