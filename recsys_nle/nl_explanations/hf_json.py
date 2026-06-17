"""Helpers for parsing JSON fragments from language-model outputs."""

from __future__ import annotations

import json
import re
from typing import Any


def _normalise_text(text: object) -> str:
    """Return a stripped string representation of the input."""
    if text is None:
        return ""
    return str(text).strip()


def parse_json_from_hf(text: object) -> Any | None:
    """Best-effort parser for JSON-like payloads returned by chat models."""
    cleaned = _normalise_text(text)
    if not cleaned:
        return None

    candidates: list[str] = []

    # Prefer content inside Markdown-style code fences when present, e.g.:
    # ```json
    # { "foo": "bar" }
    # ```
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.IGNORECASE | re.DOTALL)
    if fence_match:
        inner = fence_match.group(1).strip()
        if inner:
            candidates.append(inner)

    # Try the full string as-is.
    candidates.append(cleaned)

    # Fallback: extract the substring starting from the first JSON-looking
    # character up to the last matching closing brace/bracket.
    brace_match = re.search(r"[{\[]", cleaned)
    if brace_match:
        start = brace_match.start()
        opener = cleaned[start]
        closer = "}" if opener == "{" else "]"
        end = cleaned.rfind(closer)
        if end > start:
            fragment = cleaned[start : end + 1].strip()
            if fragment:
                candidates.append(fragment)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None
