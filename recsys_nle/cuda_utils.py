"""CUDA memory allocator configuration helpers."""

from __future__ import annotations

import os

_ALLOC_CONF_KEY = "PYTORCH_CUDA_ALLOC_CONF"
_EXPANDABLE_SEGMENTS_SETTING = "expandable_segments:True"


def enable_expandable_segments() -> None:
    """Append ``expandable_segments:True`` to ``PYTORCH_CUDA_ALLOC_CONF`` if not already present."""
    current = os.environ.get(_ALLOC_CONF_KEY, "")
    if _EXPANDABLE_SEGMENTS_SETTING.lower() in current.lower():
        return
    if current.strip():
        os.environ[_ALLOC_CONF_KEY] = f"{current},{_EXPANDABLE_SEGMENTS_SETTING}"
    else:
        os.environ[_ALLOC_CONF_KEY] = _EXPANDABLE_SEGMENTS_SETTING
