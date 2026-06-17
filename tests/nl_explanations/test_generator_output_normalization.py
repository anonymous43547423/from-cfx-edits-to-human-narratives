"""Tests for ExplanationGenerator output normalization helpers."""

from __future__ import annotations

from recsys_nle.nl_explanations.generator import _lowercase_first_char, _parse_explanation


def test_lowercase_first_char_handles_empty_string() -> None:
    """It returns empty strings unchanged."""
    assert _lowercase_first_char("") == ""  # noqa: S101


def test_parse_explanation_lowercases_after_strip() -> None:
    """It lowercases the first character after whitespace normalization."""
    payload = '{"pattern": "  Hello world", "confidence": 0.5}'

    pattern, confidence = _parse_explanation(payload)

    assert pattern == "hello world"  # noqa: S101
    assert confidence == 0.5  # noqa: PLR2004, S101
