# ruff: noqa: S101
"""Unit tests for :mod:`recsys_nle.cuda_utils`."""

from __future__ import annotations

import os

import pytest

from recsys_nle.cuda_utils import enable_expandable_segments

_KEY = "PYTORCH_CUDA_ALLOC_CONF"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the allocator env var before each test."""
    monkeypatch.delenv(_KEY, raising=False)


def test_sets_expandable_segments_when_unset() -> None:
    """When the env var is absent, it is created with the expected value."""
    enable_expandable_segments()
    assert os.environ[_KEY] == "expandable_segments:True"


def test_appends_to_existing_value() -> None:
    """When the env var already has other settings, the new setting is appended."""
    os.environ[_KEY] = "max_split_size_mb:512"
    enable_expandable_segments()
    assert os.environ[_KEY] == "max_split_size_mb:512,expandable_segments:True"


def test_idempotent_when_already_present() -> None:
    """Calling twice does not duplicate the setting."""
    enable_expandable_segments()
    first = os.environ[_KEY]
    enable_expandable_segments()
    assert os.environ[_KEY] == first


def test_case_insensitive_detection() -> None:
    """Detects existing setting regardless of casing."""
    os.environ[_KEY] = "expandable_segments:true"
    enable_expandable_segments()
    assert os.environ[_KEY] == "expandable_segments:true"
