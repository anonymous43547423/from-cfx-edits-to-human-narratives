"""Prompt construction helpers for explanation generation."""

from __future__ import annotations

import re
from typing import Mapping, Sequence


def _format_metadata_value(value: object) -> str:
    """Format metadata fields into clean text segments."""
    if value is None:
        return ""
    return str(value).strip()


def _normalise_string_sequence(value: object) -> list[str]:
    """Convert metadata values into a deduplicated list of descriptive strings."""
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, str):
        raw_items = [str(item).strip() for item in value if str(item).strip()]
    else:
        text = _format_metadata_value(value)
        if not text:
            return []
        raw_items = [part.strip() for part in re.split(r"[|,]", text) if part.strip()]
    unique: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique.append(item)
    return unique


def _collect_descriptors(
    item: Mapping[str, object],
    keys: Sequence[str],
) -> list[str]:
    """Gather the first non-empty descriptor list from a set of candidate keys."""
    for key in keys:
        if key not in item:
            continue
        descriptors = _normalise_string_sequence(item[key])
        if descriptors:
            return descriptors
    return []


def _build_item_heading(
    *,
    label: str,
    item_id: object | None,
    title: str | None,
) -> str:
    """Compose a heading that combines identifier, title, and optional year."""
    title_text = _format_metadata_value(title)
    identifier = f"{label} {item_id}" if item_id is not None else ""
    if identifier:
        return identifier
    if title_text:
        return title_text
    return label


def _quote_value(value: str) -> str:
    """Escape double quotes for inclusion inside quoted fields."""
    return value.replace('"', '\\"')


def format_interaction_prompt_attributes(item: Mapping[str, object]) -> str:
    """Format year, genres, and keywords into a brace-delimited attribute string."""
    year_val = item.get("year") or item.get("release_year") or item.get("movie_year")

    genres = _collect_descriptors(
        item,
        ("genres", "metadata_genres", "genre"),
    )
    keywords = _collect_descriptors(
        item,
        ("keywords", "movie_keywords", "metadata_keywords", "tags"),
    )

    fields: list[str] = []

    if year_val is not None:
        fields.append(f"year={year_val}")

    if genres:
        genres_text = ", ".join(genres)
        fields.append(f'genres="{_quote_value(genres_text)}"')

    if keywords:
        keywords_text = ", ".join(keywords)
        fields.append(f'keywords="{_quote_value(keywords_text)}"')

    return "{" + ", ".join(fields) + "}"


def _format_item_line(item: Mapping[str, object], index: int) -> str:
    """Create a numbered line summarising a recommendation or interaction."""
    attributes = format_interaction_prompt_attributes(item)
    return f"{index}. {attributes}"


def serialise_influential_interactions(items: Sequence[Mapping[str, object]]) -> str:
    """Render the influential interactions list for prompts."""
    if not items:
        return "No influential interactions were identified."
    lines = [_format_item_line(item, index + 1) for index, item in enumerate(items)]
    return "\n".join(lines)


def _build_chat_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    """Produce chat-formatted message dictionaries."""
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]


def serialise_chat_messages(messages: Sequence[Mapping[str, object]]) -> str:
    """Serialise chat-style messages into a human-readable prompt string."""
    serialised: list[str] = []
    for message in messages:
        role = str(message.get("role", "")).strip() if isinstance(message, Mapping) else ""
        content = ""
        if isinstance(message, Mapping):
            raw_content = message.get("content", "")
            content = str(raw_content).strip()
        entry = f"{role}: {content}" if role else content
        serialised.append(entry.strip())
    return "\n\n".join(part for part in serialised if part)


def build_reasoning_messages(
    *,
    cfx_interactions_text: str,
    non_cfx_interactions_text: str,
) -> list[dict[str, str]]:
    """Compose prompts that elicit reasoning about contrastive patterns in interactions."""
    system_prompt = (
        "You are an expert analyst for a recommender system. "
        "Your task is to find a pattern that describes what the CFX interactions have in common, "
        "while NOT matching the non-CFX interactions. Focus on simple, discriminating attributes."
    )
    user_prompt = (
        "Find a pattern that matches as many CFX interactions as possible, "
        "while matching as few non-CFX interactions as possible.\n\n"
        "CFX interactions (the pattern SHOULD match these):\n"
        f"{cfx_interactions_text}\n\n"
        "Non-CFX interactions (the pattern should NOT match these):\n"
        f"{non_cfx_interactions_text}\n\n"
        "Write a concise chain-of-thought reasoning (under 100 words) explaining what pattern "
        "best distinguishes the CFX interactions from the non-CFX interactions."
    )
    return _build_chat_messages(system_prompt, user_prompt)


def _build_explanation_messages(
    *,
    reasoning_text: str | None,
    cfx_interactions_text: str,
    non_cfx_interactions_text: str,
    include_reasoning: bool,
) -> list[dict[str, str]]:
    """Build explanation-generation prompts describing a contrastive pattern."""
    if include_reasoning:
        system_mid = (
            "You convert chain-of-thought reasoning about a contrastive pattern into a compact JSON description. "
        )
        user_intro = (
            "You are given reasoning about a pattern that distinguishes CFX interactions from non-CFX interactions.\n\n"
            "CFX interactions (the pattern SHOULD match these):\n"
            f"{cfx_interactions_text}\n\n"
            "Non-CFX interactions (the pattern should NOT match these):\n"
            f"{non_cfx_interactions_text}\n\n"
            "Summarise this distinguishing pattern as a short descriptor.\n\n"
        )
    else:
        system_mid = (
            "You identify a contrastive pattern from two sets of interactions and emit a compact JSON description. "
        )
        user_intro = (
            "Find a pattern that matches as many CFX interactions as possible, "
            "while matching as few non-CFX interactions as possible.\n\n"
            "CFX interactions (the pattern SHOULD match these):\n"
            f"{cfx_interactions_text}\n\n"
            "Non-CFX interactions (the pattern should NOT match these):\n"
            f"{non_cfx_interactions_text}\n\n"
        )

    system_prompt = (
        "You are a pattern-detection component inside a recommender system. "
        f"{system_mid}"
        "You only emit machine-readable JSON."
    )
    user_core = (
        "Return a JSON object with exactly two keys and no surrounding text:\n"
        "{\n"
        '  "pattern": "<short description (at most 8 words) of the pattern that matches CFX interactions '
        'but not non-CFX interactions>",\n'
        '  "confidence": <number between 0 and 1 indicating how confident you are in this pattern>\n'
        "}\n\n"
        "Requirements for pattern:\n"
        "• The pattern must be phrased so that it could be used to fluently finish the sentence "
        '"Because you watched ..." (good example: "drama films from pre-1990s", bad example: '
        '"single or two genres, 1938-1996") \n'
        '• The pattern must be purely representative (good example: "fantasy films from 1970s and 1980s"), '
        'without providing negative information (bad example: "films except of comedies").\n'
        "• The pattern should match as many CFX interactions as possible.\n"
        "• The pattern should match as few non-CFX interactions as possible.\n"
        "• Use at most 8 words.\n\n"
        "Requirements for confidence:\n"
        "• Use a numeric value between 0 and 1 inclusive.\n"
        "• Higher values when the pattern clearly distinguishes CFX from non-CFX.\n"
        "• Lower values when the distinction is weak or ambiguous.\n\n"
        "Good JSON example:\n"
        '{"pattern": "comedies and dramas from 1995", "confidence": 0.94}\n\n'
        "Respond with the JSON object only—no explanations, comments, or extra text."
    )
    if include_reasoning and reasoning_text is not None:
        user_prompt = f"{user_intro}{user_core}\n\nChain-of-thought reasoning:\n{reasoning_text}"
    else:
        user_prompt = f"{user_intro}{user_core}"
    return _build_chat_messages(system_prompt, user_prompt)
