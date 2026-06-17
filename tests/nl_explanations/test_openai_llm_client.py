# ruff: noqa: S101
"""Tests for OpenAI-compatible LLM client (e-INFRA gateway)."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from recsys_nle.nl_explanations.llm import (
    EINFRA_QWEN_DISABLE_THINKING_MODEL,
    OpenAIChatLLMClient,
    einfra_api_model_id,
)


def test_einfra_api_model_id_strips_prefix() -> None:
    """Strip ``EINFRA/`` and return the remote model id."""
    assert einfra_api_model_id("EINFRA/qwen2.5-coder:32b") == "qwen2.5-coder:32b"


def test_einfra_api_model_id_empty_suffix_is_stripped_empty() -> None:
    """No characters after the prefix (or only whitespace) yields an empty id."""
    assert einfra_api_model_id("EINFRA/") == ""
    assert einfra_api_model_id("EINFRA/  \t") == ""


def test_einfra_api_model_id_strips_surrounding_whitespace() -> None:
    """Leading and trailing spaces around the remote model id are removed."""
    assert einfra_api_model_id("EINFRA/  qwen2.5-coder:32b  ") == "qwen2.5-coder:32b"


def test_einfra_api_model_id_requires_prefix() -> None:
    """Reject values without the expected prefix."""
    with pytest.raises(ValueError, match="EINFRA"):
        einfra_api_model_id("foo/bar")


def test_openai_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing ``OPENAI_API_KEY`` fails fast."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIChatLLMClient(model="m")


def test_generate_sets_disable_thinking_extra_body_only_for_qwen35_122b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``qwen3.5-122b`` sends ``chat_template_kwargs.enable_thinking=false`` via ``extra_body``."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    captured: dict[str, object] = {}

    def fake_create(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="ok"))]
        return response

    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm.OpenAI",
        lambda **_kwargs: MagicMock(chat=MagicMock(completions=MagicMock(create=fake_create))),
    )

    client = OpenAIChatLLMClient(model=EINFRA_QWEN_DISABLE_THINKING_MODEL, base_url="http://example.invalid/v1")
    assert client.generate([{"role": "user", "content": "hi"}]) == "ok"
    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_generate_omits_extra_body_for_non_qwen35_122b(monkeypatch: pytest.MonkeyPatch) -> None:
    """Models other than ``qwen3.5-122b`` do not set ``extra_body``."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    captured: dict[str, object] = {}

    def fake_create(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="x"))]
        return response

    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm.OpenAI",
        lambda **_kwargs: MagicMock(chat=MagicMock(completions=MagicMock(create=fake_create))),
    )

    client = OpenAIChatLLMClient(model="remote-model", base_url="http://example.invalid/v1")
    assert client.generate([{"role": "user", "content": "?"}]) == "x"
    assert "extra_body" not in captured


def test_generate_batch_parallelism_and_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Many prompts run with bounded concurrency and outputs stay aligned to inputs."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    batch_len = 5
    in_flight = 0
    max_in_flight = 0
    lock = threading.Lock()

    def fake_create(**kwargs: object) -> MagicMock:
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        time.sleep(0.04)
        with lock:
            in_flight -= 1
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        token = str(messages[-1]["content"])
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content=token))]
        return response

    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm.OpenAI",
        lambda **_kwargs: MagicMock(chat=MagicMock(completions=MagicMock(create=fake_create))),
    )

    client = OpenAIChatLLMClient(model="remote-model", base_url="http://example.invalid/v1")
    letters = ["a", "b", "c", "d", "e"]
    batch = [[{"role": "user", "content": letter}] for letter in letters]
    out = list(
        client.generate_batch(
            batch,
            max_new_tokens=8,
            temperature=0.0,
            batch_size=batch_len,
        ),
    )
    assert out == letters
    assert max_in_flight >= batch_len
