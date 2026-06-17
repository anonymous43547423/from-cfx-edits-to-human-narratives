"""Miscellaneous helpers for logging tabular recommendation artefacts."""

from __future__ import annotations

import shutil
import textwrap
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:
    import logging

    import pandas as pd


ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_CYAN = "\033[36m"
ANSI_YELLOW = "\033[33m"


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, object]]:
    """Convert a DataFrame to a JSON-friendly list of records with string keys."""
    return [{str(key): value for key, value in row.items()} for row in df.to_dict(orient="records")]


def log_frames(
    logger: logging.Logger,
    frames: Iterable[tuple[str, pd.DataFrame, Sequence[str]]],
) -> None:
    """Log multiple named DataFrames using the provided logger."""
    for raw_title, frame, columns in frames:
        title = str(raw_title)
        column_list = list(columns)
        display = frame.reindex(columns=column_list)

        table = _format_ascii_table(title=title, frame=display)
        logger.info("%s", table)


def _format_ascii_table(*, title: str, frame: pd.DataFrame) -> str:
    """Render a single DataFrame as a coloured ASCII table with a decorated header."""
    emoji = _emoji_for_title(title)
    header_text = f"{ANSI_BOLD}{ANSI_CYAN}{emoji} {title}{ANSI_RESET}"

    if frame.empty:
        empty_line = f"{ANSI_DIM}└── no data to display ──┘{ANSI_RESET}"
        return f"{header_text}\n{empty_line}"

    headers = [str(col) for col in frame.columns]
    rows = [[("" if value is None else str(value)) for value in row] for _, row in frame.iterrows()]

    widths = _compute_column_widths(headers=headers, rows=rows)

    top_border = _build_border(widths, left="┌", mid="┬", right="┐")
    mid_border = _build_border(widths, left="├", mid="┼", right="┤")
    bottom_border = _build_border(widths, left="└", mid="┴", right="┘")

    # Render header cells with colour while keeping alignment based on plain text widths.
    header_cells: list[str] = []
    for name, width in zip(headers, widths, strict=True):
        text = name if len(name) <= width else name[: width - 1] + "…"
        padded = text.ljust(width)
        header_cells.append(f"{ANSI_BOLD}{ANSI_YELLOW}{padded}{ANSI_RESET}")
    header_row = f"│{'│'.join(f' {cell} ' for cell in header_cells)}│"

    body_rows: list[str] = []
    for value_row in rows:
        wrapped_columns = [textwrap.wrap(cell, width) or [""] for cell, width in zip(value_row, widths, strict=True)]
        max_lines = max(len(lines) for lines in wrapped_columns)
        for line_index in range(max_lines):
            line_values = [
                column_lines[line_index] if line_index < len(column_lines) else "" for column_lines in wrapped_columns
            ]
            body_rows.append(_format_row(line_values, widths))

    lines = [header_text, top_border, header_row, mid_border, *body_rows, bottom_border]
    return "\n".join(lines)


def _compute_column_widths(headers: list[str], rows: list[list[str]]) -> list[int]:
    """Derive column widths that respect the current terminal width."""
    term_width = shutil.get_terminal_size(fallback=(120, 24)).columns
    min_col_width = 8

    raw_widths: list[int] = []
    for idx, header in enumerate(headers):
        column_values = [row[idx] for row in rows]
        longest = max(len(header), *(len(value) for value in column_values)) if column_values else len(header)
        raw_widths.append(max(min_col_width, longest))

    padding_width = 3 * len(headers) + 1
    available_width = max(term_width - padding_width, min_col_width * len(headers))
    total_raw_width = sum(raw_widths)

    if total_raw_width <= available_width:
        return raw_widths

    scale = available_width / total_raw_width if total_raw_width else 1.0
    widths = [max(min_col_width, int(width * scale)) for width in raw_widths]
    while sum(widths) > available_width:
        for index, current in enumerate(widths):
            if current > min_col_width and sum(widths) > available_width:
                widths[index] = current - 1
    return widths


def _build_border(widths: Sequence[int], *, left: str, mid: str, right: str) -> str:
    """Construct a border line for an ASCII table."""
    segments = ["─" * (width + 2) for width in widths]
    return f"{left}{mid.join(segments)}{right}"


def _format_row(values: Sequence[str], widths: Sequence[int]) -> str:
    """Format a single table row with padding."""
    cells = []
    for value, width in zip(values, widths, strict=True):
        text = value if len(value) <= width else value[: width - 1] + "…"
        cells.append(f" {text.ljust(width)} ")
    return f"│{'│'.join(cells)}│"


def _emoji_for_title(title: str) -> str:
    """Choose a suitable emoji based on the semantic content of the title."""
    lowered = title.lower()
    if "held-out" in lowered:
        return "🎯"
    if "top-" in lowered or "recommendation" in lowered:
        return "⭐"
    if "influential interaction" in lowered or "attribution" in lowered:
        return "🧠"
    if "debug user" in lowered:
        return "🧪"
    if "explanation" in lowered:
        return "📝"
    return "📊"
