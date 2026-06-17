# ruff: noqa: S101
"""Tests for HuggingFace text-generation LLM client initialization."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Mapping, Sequence
from unittest.mock import MagicMock, call

import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

from recsys_nle.nl_explanations.llm import (
    _HF_MODEL_GPT_OSS_OPENAI,
    HuggingFaceLLMClient,
    _build_chat_prompt,
    _build_generation_kwargs,
    _hf_chat_template_kwargs,
    _is_mistral_model,
    _mistral_tokenizer_cached,
)

_MISTRAL_SMALL_HUB_ID = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"


@pytest.fixture(autouse=True)
def _clear_mistral_tokenizer_cache() -> Generator[None, None, None]:
    """Avoid cross-test pollution from ``@lru_cache`` on ``_mistral_tokenizer_cached``."""
    _mistral_tokenizer_cached.cache_clear()
    yield
    _mistral_tokenizer_cached.cache_clear()


def test_is_mistral_model_true_for_hub_mistral_prefix() -> None:
    """Hub ids under ``mistralai/`` are treated as Mistral-family without reading config."""
    assert _is_mistral_model("mistralai/Ministral-8B-Instruct-2410")


def test_is_mistral_model_true_for_local_ministral_config(tmp_path: Path) -> None:
    """Local checkpoints with ``model_type: ministral`` are detected from ``config.json``."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "ministral"}), encoding="utf-8")
    assert _is_mistral_model(str(tmp_path))


def test_is_mistral_model_true_for_local_mistral_config(tmp_path: Path) -> None:
    """Local checkpoints with ``model_type: mistral`` are detected from ``config.json``."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "mistral"}), encoding="utf-8")
    assert _is_mistral_model(str(tmp_path))


def test_is_mistral_model_false_for_local_llama(tmp_path: Path) -> None:
    """Non-Mistral ``model_type`` values do not match."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    assert not _is_mistral_model(str(tmp_path))


def test_is_mistral_model_false_without_config_json(tmp_path: Path) -> None:
    """Directories without ``config.json`` or ``adapter_config.json`` are not Mistral."""
    assert not _is_mistral_model(str(tmp_path))


def test_is_mistral_model_true_for_peft_adapter_mistral_base(tmp_path: Path) -> None:
    """PEFT checkpoints expose ``base_model_name_or_path`` instead of base ``config.json``."""
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "mistralai/Ministral-8B-Instruct-2410"}),
        encoding="utf-8",
    )
    assert _is_mistral_model(str(tmp_path))


def test_is_mistral_model_false_for_peft_adapter_non_mistral_base(tmp_path: Path) -> None:
    """LoRA on non-Mistral bases does not use the Mistral tokenizer path."""
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "meta-llama/Llama-3-8B-Instruct"}),
        encoding="utf-8",
    )
    assert not _is_mistral_model(str(tmp_path))


def test_is_mistral_model_false_for_invalid_config_json(tmp_path: Path) -> None:
    """Invalid JSON in ``config.json`` yields False (treat as non-Mistral)."""
    (tmp_path / "config.json").write_text("{ not json", encoding="utf-8")
    assert not _is_mistral_model(str(tmp_path))


def test_mistral_tokenizer_cached_local_dir_uses_from_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Local model directories load Tekken via ``MistralTokenizer.from_file``."""
    tekken = tmp_path / "tekken.json"
    tekken.write_text("{}", encoding="utf-8")
    fake_tok = object()
    captured: dict[str, str] = {}

    def fake_from_file(path: str, **_k: object) -> object:
        captured["path"] = path
        return fake_tok

    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm.MistralTokenizer.from_file",
        fake_from_file,
    )
    assert _mistral_tokenizer_cached(str(tmp_path)) is fake_tok
    assert captured["path"] == str(tekken)


def test_mistral_tokenizer_cached_local_dir_missing_tekken_raises(tmp_path: Path) -> None:
    """Local directories without ``tekken*.json`` raise ``FileNotFoundError``."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "ministral"}), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="tekken"):
        _mistral_tokenizer_cached(str(tmp_path))


def test_mistral_tokenizer_cached_local_dir_uses_lexicographically_first_tekken_glob(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When multiple ``tekken*.json`` exist, the first sorted name is passed to ``from_file``."""
    (tmp_path / "tekken_b.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tekken_a.json").write_text("{}", encoding="utf-8")
    chosen: list[str] = []

    def fake_from_file(path: str, **_k: object) -> object:
        chosen.append(path)
        return MagicMock()

    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm.MistralTokenizer.from_file",
        fake_from_file,
    )
    _mistral_tokenizer_cached(str(tmp_path))
    assert chosen == [str(tmp_path / "tekken_a.json")]


_HUB_MODEL_IDS_4BIT_CUDA = (
    "Qwen/Qwen3-14B",
    "microsoft/Phi-4-reasoning",
    "microsoft/Phi-4-reasoning-plus",
)

_HUB_MODEL_IDS_CPU_BF16_FALLBACK = (
    "org/model",
    "microsoft/Phi-4-reasoning",
    "microsoft/Phi-4-reasoning-plus",
)


def _tokenizer_mock() -> MagicMock:
    """Build a tokenizer mock that satisfies ``HuggingFaceLLMClient.__post_init__``."""
    tok = MagicMock()
    tok.eos_token_id = 2
    tok.pad_token_id = 0
    tok.pad_token = object()  # non-None so ``__post_init__`` does not need ``eos_token``
    tok.padding_side = "right"
    tok.parse_response = None
    return tok


def test_huggingface_client_local_ministral_checkpoint_uses_mistral_common_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Filesystem paths to Ministral checkpoints use ``MistralCommonBackend``, not ``AutoTokenizer``."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "ministral"}), encoding="utf-8")
    tok = _tokenizer_mock()
    captured_mc: list[str] = []
    auto_calls: list[object] = []

    def fake_mistral_common(*args: object, **_kw: object) -> MagicMock:
        """Accept both class-bound (cls, path) and patched single-arg (path) calls."""
        min_classmethod_args = 2
        path_arg = args[1] if len(args) >= min_classmethod_args else args[0]
        captured_mc.append(str(path_arg))
        return tok

    def spy_auto_tokenizer(*_a: object, **_k: object) -> MagicMock:
        auto_calls.append(True)
        return tok

    def fake_pipeline(**kwargs: object) -> MagicMock:
        mock_gen = MagicMock()
        mock_gen.tokenizer = tok
        mock_gen.model = MagicMock()
        mock_gen.model.config = MagicMock()
        model_gc = MagicMock()
        model_gc.max_length = 20
        mock_gen.model.generation_config = model_gc
        pipe_gc = MagicMock()
        pipe_gc.max_length = 20
        mock_gen.generation_config = pipe_gc
        assert kwargs.get("tokenizer") is tok
        return mock_gen

    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm.torch.cuda.is_available",
        lambda: False,
    )
    monkeypatch.setattr("recsys_nle.nl_explanations.llm.pipeline", fake_pipeline)
    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm.MistralCommonBackend.from_pretrained",
        fake_mistral_common,
    )
    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm.AutoTokenizer.from_pretrained",
        spy_auto_tokenizer,
    )
    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm._validate_openai_gpt_oss_mxfp4_runtime",
        lambda *_a, **_k: None,
    )

    HuggingFaceLLMClient(model_id=str(tmp_path))
    assert captured_mc == [str(tmp_path)]
    assert not auto_calls


def _patch_hf_init(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda_available: bool,
) -> tuple[MagicMock, dict[str, object]]:
    """Patch tokenizer + pipeline; return tokenizer and kwargs passed to ``pipeline``."""
    tok = _tokenizer_mock()
    captured: dict[str, object] = {}

    def fake_pipeline(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        mock_gen = MagicMock()
        mock_gen.tokenizer = tok
        mock_gen.model = MagicMock()
        mock_gen.model.config = MagicMock()
        model_gc = MagicMock()
        hub_default_len = 20
        model_gc.max_length = hub_default_len
        mock_gen.model.generation_config = model_gc
        pipe_gc = MagicMock()
        pipe_gc.max_length = hub_default_len
        mock_gen.generation_config = pipe_gc
        captured["last_pipeline_mock"] = mock_gen
        return mock_gen

    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm.torch.cuda.is_available",
        lambda: cuda_available,
    )
    monkeypatch.setattr("recsys_nle.nl_explanations.llm.pipeline", fake_pipeline)
    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm.AutoTokenizer.from_pretrained",
        lambda *_a, **_k: tok,
    )
    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm._validate_openai_gpt_oss_mxfp4_runtime",
        lambda *_a, **_k: None,
    )
    return tok, captured


def _assert_pipeline_captured_4bit_default_cuda(captured: dict[str, object]) -> None:
    """Assert kwargs for 4-bit load on default CUDA (Accelerate ``device_map``)."""
    assert captured["torch_dtype"] is torch.bfloat16
    assert "device" not in captured
    model_kwargs = captured["model_kwargs"]
    assert isinstance(model_kwargs, dict)
    assert model_kwargs["device_map"] == "auto"
    qc = model_kwargs["quantization_config"]
    assert qc.load_in_4bit is True


def _assert_pipeline_captured_bf16_cpu_no_quant(captured: dict[str, object]) -> None:
    """Assert kwargs for bf16 CPU path without quantization."""
    assert captured["device"] == -1
    assert captured["torch_dtype"] is torch.bfloat16
    assert "model_kwargs" not in captured


@pytest.mark.parametrize("model_id", _HUB_MODEL_IDS_4BIT_CUDA)
def test_huggingface_client_4bit_models_use_bnb_on_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
) -> None:
    """On CUDA, 4-bit Hub ids use ``BitsAndBytesConfig`` and ``device_map`` ``auto``."""
    _, captured = _patch_hf_init(monkeypatch, cuda_available=True)
    HuggingFaceLLMClient(model_id=model_id)
    _assert_pipeline_captured_4bit_default_cuda(captured)


def test_huggingface_client_generic_cuda_no_quantization_bfloat16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hub models outside the 4-bit set load in ``bfloat16`` on CUDA without quantization."""
    _, captured = _patch_hf_init(monkeypatch, cuda_available=True)
    HuggingFaceLLMClient(model_id="org/other-model")

    assert captured["torch_dtype"] is torch.bfloat16
    assert captured["device"] == 0
    assert "model_kwargs" not in captured


def test_huggingface_client_openai_gpt_oss_uses_native_auto_loading_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On CUDA, openai gpt-oss uses native MXFP4 path with ``torch_dtype=auto`` + auto map."""
    _, captured = _patch_hf_init(monkeypatch, cuda_available=True)
    HuggingFaceLLMClient(model_id=_HF_MODEL_GPT_OSS_OPENAI)

    assert captured["torch_dtype"] == "auto"
    assert "device" not in captured
    assert captured["device_map"] == "auto"
    assert "model_kwargs" not in captured


def test_huggingface_client_openai_gpt_oss_explicit_cuda_device_maps_to_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit CUDA index maps openai gpt-oss to that single CUDA device."""
    _, captured = _patch_hf_init(monkeypatch, cuda_available=True)
    HuggingFaceLLMClient(model_id=_HF_MODEL_GPT_OSS_OPENAI, device=1)

    assert captured["torch_dtype"] == "auto"
    assert "device" not in captured
    assert captured["device_map"] == {"": 1}


def test_huggingface_client_openai_gpt_oss_device_minus_one_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``device=-1`` is rejected for openai gpt-oss (strict MXFP4 on CUDA only)."""
    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm.torch.cuda.is_available",
        lambda: True,
    )
    with pytest.raises(ValueError, match="strict MXFP4"):
        HuggingFaceLLMClient(model_id=_HF_MODEL_GPT_OSS_OPENAI, device=-1)


def test_huggingface_client_openai_gpt_oss_requires_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without CUDA, openai gpt-oss cannot load (no BF16 CPU fallback)."""
    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm.torch.cuda.is_available",
        lambda: False,
    )
    with pytest.raises(ValueError, match="CUDA GPU"):
        HuggingFaceLLMClient(model_id=_HF_MODEL_GPT_OSS_OPENAI)


@pytest.mark.parametrize("model_id", _HUB_MODEL_IDS_CPU_BF16_FALLBACK)
def test_huggingface_client_cpu_bf16_without_quantization(
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
) -> None:
    """Without CUDA, bf16 CPU path (device ``-1``) without quantization kwargs."""
    _, captured = _patch_hf_init(monkeypatch, cuda_available=False)
    HuggingFaceLLMClient(model_id=model_id)
    _assert_pipeline_captured_bf16_cpu_no_quant(captured)


@pytest.mark.parametrize("model_id", _HUB_MODEL_IDS_4BIT_CUDA)
def test_huggingface_client_explicit_cuda_device_maps_quantized_model(
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
) -> None:
    """Explicit non-negative device index maps quantized Hub models to that CUDA device."""
    _, captured = _patch_hf_init(monkeypatch, cuda_available=True)
    HuggingFaceLLMClient(model_id=model_id, device=1)

    model_kwargs = captured["model_kwargs"]
    assert isinstance(model_kwargs, dict)
    assert model_kwargs["device_map"] == {"": 1}
    qc = model_kwargs["quantization_config"]
    assert qc.load_in_4bit is True


def test_huggingface_client_device_minus_one_skips_4bit_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``device=-1`` forces the non-quantized path even when CUDA is available."""
    _, captured = _patch_hf_init(monkeypatch, cuda_available=True)
    HuggingFaceLLMClient(model_id="org/model", device=-1)

    assert captured["device"] == -1
    assert "model_kwargs" not in captured


def test_huggingface_client_clears_generation_config_max_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clear hub default ``max_length`` so only ``max_new_tokens`` controls decode length."""
    _, captured = _patch_hf_init(monkeypatch, cuda_available=False)
    HuggingFaceLLMClient(model_id="org/model")
    mock_gen = captured["last_pipeline_mock"]
    assert isinstance(mock_gen, MagicMock)
    assert mock_gen.model.generation_config.max_length is None
    assert mock_gen.generation_config.max_length is None


def test_build_generation_kwargs_greedy_clears_temperature_and_top_p() -> None:
    """Greedy decoding omits active sampling params so transformers does not warn."""
    kwargs = _build_generation_kwargs(
        tokens=64,
        sampling_temperature=0.0,
        sampling_top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=0,
        eos_token_id=2,
    )
    assert kwargs["do_sample"] is False
    assert kwargs["temperature"] is None
    assert kwargs["top_p"] is None


def test_build_generation_kwargs_sampling_includes_temperature_and_top_p() -> None:
    """Sampling mode passes ``temperature`` and ``top_p``."""
    sampling_temp = 0.7
    sampling_top_p_in = 0.95
    kwargs = _build_generation_kwargs(
        tokens=64,
        sampling_temperature=sampling_temp,
        sampling_top_p=sampling_top_p_in,
        repetition_penalty=1.05,
        pad_token_id=0,
        eos_token_id=2,
    )
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == sampling_temp
    assert kwargs["top_p"] == sampling_top_p_in


def test_openai_gpt_oss_chat_template_kwargs_set_low_reasoning_effort() -> None:
    """gpt-oss chat prompts explicitly request low reasoning effort."""
    kwargs = _hf_chat_template_kwargs(_HF_MODEL_GPT_OSS_OPENAI)
    assert kwargs["enable_thinking"] is False
    assert kwargs["reasoning_effort"] == "low"


def test_generic_chat_template_kwargs_disable_thinking_without_effort_override() -> None:
    """Non-gpt-oss models avoid injecting unsupported reasoning-effort overrides."""
    kwargs = _hf_chat_template_kwargs("org/other-model")
    assert kwargs["enable_thinking"] is False
    assert "reasoning_effort" not in kwargs


def test_build_chat_prompt_falls_back_when_chat_template_is_missing() -> None:
    """If tokenizer lacks chat_template, build a simple role-prefixed fallback prompt."""
    tok = MagicMock()
    tok.apply_chat_template.side_effect = ValueError("tokenizer.chat_template is not set")
    prompt = _build_chat_prompt(
        tok,
        [
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "hello"},
        ],
        "org/other-model",
    )
    assert "System: system rules" in prompt
    assert "User: hello" in prompt
    assert prompt.endswith("Assistant:")


def test_huggingface_generate_returns_pipeline_generated_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pipeline path returns ``generated_text`` after stop-sequence handling."""
    tok = _tokenizer_mock()
    tok.apply_chat_template.return_value = "<prompt>"

    def fake_pipeline(**_kwargs: object) -> MagicMock:
        mock_gen = MagicMock()
        mock_gen.tokenizer = tok
        mock_gen.model = MagicMock()
        mock_gen.model.config = MagicMock()
        model_gc = MagicMock()
        model_gc.max_length = 20
        mock_gen.model.generation_config = model_gc
        pipe_gc = MagicMock()
        pipe_gc.max_length = 20
        mock_gen.generation_config = pipe_gc
        mock_gen.return_value = [{"generated_text": "Hello from the model."}]
        return mock_gen

    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm.torch.cuda.is_available",
        lambda: False,
    )
    monkeypatch.setattr("recsys_nle.nl_explanations.llm.pipeline", fake_pipeline)
    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm.AutoTokenizer.from_pretrained",
        lambda *_a, **_k: tok,
    )
    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm._validate_openai_gpt_oss_mxfp4_runtime",
        lambda *_a, **_k: None,
    )

    client = HuggingFaceLLMClient(model_id="org/other-model")
    out = client.generate([{"role": "user", "content": "hi"}], max_new_tokens=8, temperature=0.0)
    assert out == "Hello from the model."


def test_huggingface_generate_batch_returns_pipeline_generated_text_per_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch pipeline path returns each row's ``generated_text``."""
    tok = _tokenizer_mock()
    tok.apply_chat_template.return_value = "<prompt>"

    def fake_pipeline(**_kwargs: object) -> MagicMock:
        mock_gen = MagicMock()
        mock_gen.tokenizer = tok
        mock_gen.model = MagicMock()
        mock_gen.model.config = MagicMock()
        model_gc = MagicMock()
        model_gc.max_length = 20
        mock_gen.model.generation_config = model_gc
        pipe_gc = MagicMock()
        pipe_gc.max_length = 20
        mock_gen.generation_config = pipe_gc
        mock_gen.return_value = [
            {"generated_text": "raw A"},
            {"generated_text": "raw B"},
        ]
        return mock_gen

    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm.torch.cuda.is_available",
        lambda: False,
    )
    monkeypatch.setattr("recsys_nle.nl_explanations.llm.pipeline", fake_pipeline)
    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm.AutoTokenizer.from_pretrained",
        lambda *_a, **_k: tok,
    )
    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm._validate_openai_gpt_oss_mxfp4_runtime",
        lambda *_a, **_k: None,
    )

    client = HuggingFaceLLMClient(model_id="org/other-model")
    outs = client.generate_batch(
        [
            [{"role": "user", "content": "a"}],
            [{"role": "user", "content": "b"}],
        ],
        max_new_tokens=8,
        temperature=0.0,
    )
    assert list(outs) == ["raw A", "raw B"]


def test_mistral_small_instruct_uses_mistral_encode_and_model_generate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mistral-Small-3.2 uses mistral-common token ids and ``model.generate``."""
    tok = _tokenizer_mock()
    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])

    assistant_token_id = 99

    def _fake_generate(**kwargs: object) -> torch.Tensor:
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        seq_len = int(input_ids.shape[1])
        out = torch.zeros(1, seq_len + 1, dtype=torch.long)
        out[0, seq_len] = assistant_token_id
        return out

    mock_model.generate.side_effect = _fake_generate

    def fake_pipeline(**_kwargs: object) -> MagicMock:
        mock_gen = MagicMock()
        mock_gen.tokenizer = tok
        mock_gen.model = mock_model
        mock_gen.model.config = MagicMock()
        model_gc = MagicMock()
        model_gc.max_length = 20
        mock_gen.model.generation_config = model_gc
        pipe_gc = MagicMock()
        pipe_gc.max_length = 20
        mock_gen.generation_config = pipe_gc
        return mock_gen

    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm.torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr("recsys_nle.nl_explanations.llm.pipeline", fake_pipeline)
    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm.MistralCommonBackend.from_pretrained",
        lambda *_a, **_k: tok,
    )
    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm._validate_openai_gpt_oss_mxfp4_runtime",
        lambda *_a, **_k: None,
    )

    encode_calls: list[tuple[str, list[dict[str, str]]]] = []

    def fake_encode(repo_id: str, messages: Sequence[Mapping[str, str]]) -> list[int]:
        encode_calls.append((repo_id, [dict(m) for m in messages]))
        return [7, 8, 9]

    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm._mistral_encode_chat_prompt_tokens",
        fake_encode,
    )

    fake_mt = MagicMock()
    fake_mt.decode.return_value = " model reply "
    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm._mistral_tokenizer_cached",
        lambda _repo_id: fake_mt,
    )

    client = HuggingFaceLLMClient(model_id=_MISTRAL_SMALL_HUB_ID)
    out = client.generate([{"role": "user", "content": "ping"}], max_new_tokens=4, temperature=0.0)
    assert out == "model reply"
    assert encode_calls == [
        (_MISTRAL_SMALL_HUB_ID, [{"role": "user", "content": "ping"}]),
    ]
    mock_model.generate.assert_called_once()
    fake_mt.decode.assert_called_once_with([assistant_token_id])


def test_mistral_small_generate_batch_calls_model_generate_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mistral batch path left-pads, runs one ``model.generate`` per micro-batch, decodes each row."""
    tok = _tokenizer_mock()
    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])

    assistant_token_id = 99

    def _fake_generate(**kwargs: object) -> torch.Tensor:
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        expected_batch_rows = 2
        assert int(input_ids.shape[0]) == expected_batch_rows
        batch, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        out = torch.zeros(batch, seq_len + 1, dtype=torch.long)
        out[:, seq_len] = assistant_token_id
        return out

    mock_model.generate.side_effect = _fake_generate

    def fake_pipeline(**_kwargs: object) -> MagicMock:
        mock_gen = MagicMock()
        mock_gen.tokenizer = tok
        mock_gen.model = mock_model
        mock_gen.model.config = MagicMock()
        model_gc = MagicMock()
        model_gc.max_length = 20
        mock_gen.model.generation_config = model_gc
        pipe_gc = MagicMock()
        pipe_gc.max_length = 20
        mock_gen.generation_config = pipe_gc
        return mock_gen

    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm.torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr("recsys_nle.nl_explanations.llm.pipeline", fake_pipeline)
    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm.MistralCommonBackend.from_pretrained",
        lambda *_a, **_k: tok,
    )
    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm._validate_openai_gpt_oss_mxfp4_runtime",
        lambda *_a, **_k: None,
    )

    encode_calls: list[tuple[str, list[dict[str, str]]]] = []

    def fake_encode(repo_id: str, messages: Sequence[Mapping[str, str]]) -> list[int]:
        encode_calls.append((repo_id, [dict(m) for m in messages]))
        if len(encode_calls) == 1:
            return [7, 8, 9]
        return [10, 11]

    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm._mistral_encode_chat_prompt_tokens",
        fake_encode,
    )

    fake_mt = MagicMock()
    fake_mt.decode.return_value = " model reply "
    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm._mistral_tokenizer_cached",
        lambda _repo_id: fake_mt,
    )

    client = HuggingFaceLLMClient(model_id=_MISTRAL_SMALL_HUB_ID)
    outs = list(
        client.generate_batch(
            [
                [{"role": "user", "content": "a"}],
                [{"role": "user", "content": "b"}],
            ],
            max_new_tokens=4,
            temperature=0.0,
            batch_size=2,
        )
    )
    assert outs == ["model reply", "model reply"]
    assert encode_calls == [
        (_MISTRAL_SMALL_HUB_ID, [{"role": "user", "content": "a"}]),
        (_MISTRAL_SMALL_HUB_ID, [{"role": "user", "content": "b"}]),
    ]
    mock_model.generate.assert_called_once()
    expected_decode_calls = 2
    assert fake_mt.decode.call_count == expected_decode_calls
    fake_mt.decode.assert_has_calls(
        [call([assistant_token_id]), call([assistant_token_id])],
        any_order=False,
    )


def test_mistral_small_generate_uses_inference_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mistral path wraps ``model.generate`` in ``torch.inference_mode`` to avoid autograd overhead."""
    tok = _tokenizer_mock()
    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])

    inference_mode_was_active = False

    def _fake_generate(**kwargs: object) -> torch.Tensor:
        nonlocal inference_mode_was_active
        inference_mode_was_active = torch.is_inference_mode_enabled()
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        batch, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        return torch.zeros(batch, seq_len + 1, dtype=torch.long)

    mock_model.generate.side_effect = _fake_generate

    def fake_pipeline(**_kwargs: object) -> MagicMock:
        mock_gen = MagicMock()
        mock_gen.tokenizer = tok
        mock_gen.model = mock_model
        mock_gen.model.config = MagicMock()
        model_gc = MagicMock()
        model_gc.max_length = 20
        mock_gen.model.generation_config = model_gc
        pipe_gc = MagicMock()
        pipe_gc.max_length = 20
        mock_gen.generation_config = pipe_gc
        return mock_gen

    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm.torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr("recsys_nle.nl_explanations.llm.pipeline", fake_pipeline)
    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm.MistralCommonBackend.from_pretrained",
        lambda *_a, **_k: tok,
    )
    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm._validate_openai_gpt_oss_mxfp4_runtime",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm._mistral_encode_chat_prompt_tokens",
        lambda _repo_id, _messages: [1, 2, 3],
    )
    fake_mt = MagicMock()
    fake_mt.decode.return_value = "ok"
    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm._mistral_tokenizer_cached",
        lambda _repo_id: fake_mt,
    )

    client = HuggingFaceLLMClient(model_id=_MISTRAL_SMALL_HUB_ID)
    client.generate([{"role": "user", "content": "hi"}], max_new_tokens=4, temperature=0.0)

    assert inference_mode_was_active, "model.generate() must run inside torch.inference_mode()"


def test_mistral_small_generate_batch_respects_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mistral ``generate_batch`` chunks prompts so ``model.generate`` respects ``batch_size``."""
    tok = _tokenizer_mock()
    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])

    def _fake_generate(**kwargs: object) -> torch.Tensor:
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        batch_cap = 2
        assert int(input_ids.shape[0]) <= batch_cap
        batch, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
        out = torch.zeros(batch, seq_len + 2, dtype=torch.long)
        out[:, seq_len:] = 7
        return out

    mock_model.generate.side_effect = _fake_generate

    def fake_pipeline(**_kwargs: object) -> MagicMock:
        mock_gen = MagicMock()
        mock_gen.tokenizer = tok
        mock_gen.model = mock_model
        mock_gen.model.config = MagicMock()
        model_gc = MagicMock()
        model_gc.max_length = 20
        mock_gen.model.generation_config = model_gc
        pipe_gc = MagicMock()
        pipe_gc.max_length = 20
        mock_gen.generation_config = pipe_gc
        return mock_gen

    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm.torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr("recsys_nle.nl_explanations.llm.pipeline", fake_pipeline)
    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm.MistralCommonBackend.from_pretrained",
        lambda *_a, **_k: tok,
    )
    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm._validate_openai_gpt_oss_mxfp4_runtime",
        lambda *_a, **_k: None,
    )

    enc_index = {"i": 0}
    nested_tokens = ([10], [20, 21], [30, 31, 32])

    def fake_encode(_repo_id: str, _messages: Sequence[Mapping[str, str]]) -> list[int]:
        idx = enc_index["i"]
        enc_index["i"] += 1
        return list(nested_tokens[idx])

    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm._mistral_encode_chat_prompt_tokens",
        fake_encode,
    )

    fake_mt = MagicMock()
    fake_mt.decode.return_value = "x"
    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm._mistral_tokenizer_cached",
        lambda _repo_id: fake_mt,
    )

    client = HuggingFaceLLMClient(model_id=_MISTRAL_SMALL_HUB_ID)
    outs = list(
        client.generate_batch(
            [
                [{"role": "user", "content": "0"}],
                [{"role": "user", "content": "1"}],
                [{"role": "user", "content": "2"}],
            ],
            max_new_tokens=4,
            temperature=0.0,
            batch_size=2,
        )
    )
    n_prompts = 3
    expected_generate_calls = 2
    assert len(outs) == n_prompts
    assert mock_model.generate.call_count == expected_generate_calls


def test_openai_gpt_oss_rejects_missing_triton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MXFP4 preflight fails when Triton is below the required version."""
    monkeypatch.setattr("recsys_nle.nl_explanations.llm.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("recsys_nle.nl_explanations.llm.is_triton_available", lambda *_a, **_k: False)
    monkeypatch.setattr("recsys_nle.nl_explanations.llm.is_kernels_available", lambda: True)
    monkeypatch.setattr("recsys_nle.nl_explanations.llm.is_accelerate_available", lambda: True)
    with pytest.raises(ValueError, match="Triton"):
        HuggingFaceLLMClient(model_id=_HF_MODEL_GPT_OSS_OPENAI)


def test_openai_gpt_oss_rejects_missing_optin_device_property(monkeypatch: pytest.MonkeyPatch) -> None:
    """MXFP4 preflight fails when PyTorch omits ``shared_memory_per_block_optin``."""
    monkeypatch.setattr("recsys_nle.nl_explanations.llm.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("recsys_nle.nl_explanations.llm.is_triton_available", lambda *_a, **_k: True)
    monkeypatch.setattr("recsys_nle.nl_explanations.llm.is_kernels_available", lambda: True)
    monkeypatch.setattr("recsys_nle.nl_explanations.llm.is_accelerate_available", lambda: True)

    class _Props:
        """Minimal device properties without the MXFP4 opt-in shared memory field."""

    monkeypatch.setattr(
        "recsys_nle.nl_explanations.llm.torch.cuda.get_device_properties",
        lambda _i: _Props(),
    )
    with pytest.raises(ValueError, match="shared_memory_per_block_optin"):
        HuggingFaceLLMClient(model_id=_HF_MODEL_GPT_OSS_OPENAI)
