"""Tests for parsing JSON fragments from language-model outputs."""

from __future__ import annotations

from recsys_nle.nl_explanations.hf_json import parse_json_from_hf


def test_parse_plain_json_object() -> None:
    """It parses a plain JSON object without extra decoration."""
    payload = '{"verb_phrase": "savor deeply", "trait": "classic 80s sci-fi", "confidence": 0.94}'

    parsed = parse_json_from_hf(payload)

    assert isinstance(parsed, dict)  # noqa: S101
    assert parsed["verb_phrase"] == "savor deeply"  # noqa: S101
    assert parsed["trait"] == "classic 80s sci-fi"  # noqa: S101
    assert parsed["confidence"] == 0.94  # noqa: PLR2004, S101


def test_parse_json_inside_code_fence() -> None:
    """It extracts and parses JSON embedded in markdown code fences."""
    payload = """```json
{"pattern": "Descriptor One", "confidence": 0.8}
```"""

    parsed = parse_json_from_hf(payload)

    assert isinstance(parsed, dict)  # noqa: S101
    assert parsed["pattern"] == "Descriptor One"  # noqa: S101
    assert parsed["confidence"] == 0.8  # noqa: PLR2004, S101


def test_parse_json_with_leading_and_trailing_text() -> None:
    """It recovers JSON fragments surrounded by explanatory text."""
    payload = """
Here is your JSON answer:
{"pattern": "A", "confidence": 0.7}
Thanks!
"""

    parsed = parse_json_from_hf(payload)

    assert isinstance(parsed, dict)  # noqa: S101
    assert parsed["pattern"] == "A"  # noqa: S101
    assert parsed["confidence"] == 0.7  # noqa: PLR2004, S101


def test_returns_none_when_no_json_can_be_parsed() -> None:
    """It returns None if the text does not contain a valid JSON fragment."""
    parsed = parse_json_from_hf("This is not JSON.")

    assert parsed is None  # noqa: S101
