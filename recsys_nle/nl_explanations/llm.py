"""Language model client abstractions for NL explanations."""

from __future__ import annotations

import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence

import torch
from mistral_common.protocol.instruct.converters import convert_openai_messages
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from openai import OpenAI
from transformers import AutoTokenizer, BitsAndBytesConfig, pipeline
from transformers.tokenization_mistral_common import MistralCommonBackend
from transformers.utils.import_utils import (
    is_accelerate_available,
    is_kernels_available,
    is_triton_available,
)

from recsys_nle.nl_explanations.metrics import record_hf_call

if TYPE_CHECKING:
    from transformers.pipelines.text_generation import TextGenerationPipeline

ChatMessage = Mapping[str, str]

_HF_MODEL_QWEN3_14B = "Qwen/Qwen3-14B"
_HF_MODEL_GPT_OSS_OPENAI = "openai/gpt-oss-20b"
_HF_MODEL_PHI4_REASONING = "microsoft/Phi-4-reasoning"
_HF_MODEL_PHI4_REASONING_PLUS = "microsoft/Phi-4-reasoning-plus"
_MISTRAL_MODEL_PREFIX = "mistralai/"
_HF_MODELS_4BIT_BNB: frozenset[str] = frozenset(
    {
        _HF_MODEL_QWEN3_14B,
        _HF_MODEL_PHI4_REASONING,
        _HF_MODEL_PHI4_REASONING_PLUS,
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    }
)

_TRITON_MXFP4_MIN_VERSION = "3.4.0"
_MISTRAL_FAMILY_MODEL_TYPES: frozenset[str] = frozenset({"mistral", "ministral"})

EINFRA_MODEL_PREFIX = "EINFRA/"
EINFRA_DEFAULT_BASE_URL = "https://llm.ai.e-infra.cz/v1"
EINFRA_QWEN_DISABLE_THINKING_MODEL = "qwen3.5-122b"


def einfra_api_model_id(prefixed: str) -> str:
    """Return the OpenAI-compatible API model id from an ``EINFRA/<id>`` CLI value."""
    if not prefixed.startswith(EINFRA_MODEL_PREFIX):
        msg = "Expected model id prefixed with EINFRA/."
        raise ValueError(msg)
    return prefixed[len(EINFRA_MODEL_PREFIX) :].strip()


def _require_openai_api_key() -> str:
    """Return OPENAI_API_KEY or raise with a clear configuration error."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        msg = "OPENAI_API_KEY must be set in the environment for EINFRA evaluation (e-INFRA OpenAI-compatible API)."
        raise ValueError(msg)
    return key


def _apply_stop_sequences(text: str, stop_sequences: Sequence[str] | None) -> str:
    """Truncate ``text`` at the first occurrence of any stop sequence."""
    final_text = text
    if stop_sequences:
        for stop in stop_sequences:
            stop_index = final_text.find(stop)
            if stop_index != -1:
                final_text = final_text[:stop_index]
    return final_text.strip()


def _openai_completion_text(response: object) -> str:
    """Extract assistant text from a chat completion response object."""
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    choice0 = choices[0]
    message = getattr(choice0, "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return str(content)


class LLMClient(Protocol):
    """Protocol describing a minimal chat generation client interface."""

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop_sequences: Sequence[str] | None = None,
    ) -> str:
        """Return generated text for the provided chat messages."""

    def generate_batch(
        self,
        messages_batch: Sequence[Sequence[ChatMessage]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop_sequences: Sequence[str] | None = None,
        batch_size: int | None = None,
    ) -> Sequence[str]:
        """Return generated text for each set of chat messages in the batch."""

    def close(self) -> None:
        """Release resources held by the client."""


@dataclass(slots=True)
class OpenAIChatLLMClient:
    """OpenAI-compatible chat completions client (e.g. e-INFRA LLM gateway)."""

    model: str
    base_url: str = EINFRA_DEFAULT_BASE_URL
    default_max_tokens: int = 512
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    _client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build a sync OpenAI client using OPENAI_API_KEY."""
        api_key = _require_openai_api_key()
        self._client = OpenAI(api_key=api_key, base_url=self.base_url)

    def close(self) -> None:
        """Close the underlying HTTP client when supported."""
        closer = getattr(self._client, "close", None)
        if callable(closer):
            closer()

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop_sequences: Sequence[str] | None = None,
    ) -> str:
        """Generate one chat completion via the OpenAI-compatible API."""
        tokens = self.default_max_tokens if max_new_tokens is None else max_new_tokens
        temp = self.default_temperature if temperature is None else temperature
        top_p_val = self.default_top_p if top_p is None else top_p
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "max_tokens": tokens,
            "temperature": temp,
            "top_p": top_p_val,
        }
        if stop_sequences:
            kwargs["stop"] = list(stop_sequences)
        if self.model == EINFRA_QWEN_DISABLE_THINKING_MODEL:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        response = self._client.chat.completions.create(**kwargs)
        return _apply_stop_sequences(_openai_completion_text(response), stop_sequences)

    def generate_batch(
        self,
        messages_batch: Sequence[Sequence[ChatMessage]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop_sequences: Sequence[str] | None = None,
        batch_size: int | None = None,
    ) -> Sequence[str]:
        """Run one completion per prompt in parallel with bounded concurrency."""
        n = len(messages_batch)
        if n == 0:
            return []
        workers = batch_size if batch_size is not None and batch_size > 0 else n
        workers = min(workers, n)
        results: list[str] = [""] * n
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {
                pool.submit(
                    self.generate,
                    messages_batch[i],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop_sequences=stop_sequences,
                ): i
                for i in range(n)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results


def _clear_max_length_from_hf_pipeline_generation_config(pipeline: object) -> None:
    """Unset hub default ``max_length`` so ``max_new_tokens`` alone controls decode length."""
    model = getattr(pipeline, "model", None)
    if model is not None:
        model_gc = getattr(model, "generation_config", None)
        if model_gc is not None:
            model_gc.max_length = None
    pipe_gc = getattr(pipeline, "generation_config", None)
    if pipe_gc is not None:
        pipe_gc.max_length = None


def _hf_cuda_requested(device: int | str | None) -> bool:
    """Return True when HF loading should use CUDA rather than forced CPU."""
    return device != -1 and not (isinstance(device, str) and device.lower() in {"cpu", "-1"})


def _openai_gpt_oss_resolve_cuda_device_index(device: int | str | None) -> int:
    """Return the CUDA device index used for MXFP4 kernel capability checks."""
    if device is None:
        return 0
    if isinstance(device, int):
        if device < 0:
            msg = "openai/gpt-oss-20b requires CUDA for strict MXFP4; do not use device=-1."
            raise ValueError(msg)
        return device
    if isinstance(device, str):
        normalized = device.strip().lower()
        if normalized in {"cpu", "-1"}:
            msg = "openai/gpt-oss-20b requires CUDA for strict MXFP4; do not force CPU."
            raise ValueError(msg)
        if normalized == "cuda":
            return 0
        if normalized.startswith("cuda:"):
            tail = normalized.removeprefix("cuda:").strip()
            return int(tail)
    msg = f"Unsupported device for openai/gpt-oss-20b: {device!r}."
    raise ValueError(msg)


def _validate_openai_gpt_oss_mxfp4_runtime(device: int | str | None) -> None:
    """Fail fast unless this host can run Transformers MXFP4 for ``openai/gpt-oss-20b``."""
    if not _hf_cuda_requested(device):
        msg = (
            "openai/gpt-oss-20b is configured for strict MXFP4 on CUDA only. "
            "CPU or device=-1 (BF16 dequantized weights) is disabled to avoid silent high-memory fallback. "
            "Use a CUDA device or pick another model for CPU."
        )
        raise ValueError(msg)
    if not torch.cuda.is_available():
        msg = "openai/gpt-oss-20b requires a CUDA GPU for strict MXFP4, but torch.cuda.is_available() is False."
        raise ValueError(msg)
    if not is_accelerate_available():
        msg = (
            "openai/gpt-oss-20b needs ``accelerate`` for ``device_map`` with MXFP4. "
            "Install a recent ``accelerate`` (required by Transformers for auto device mapping)."
        )
        raise ValueError(msg)
    if not is_kernels_available():
        msg = (
            "openai/gpt-oss-20b requires the ``kernels`` package for MXFP4 ops. "
            "Install ``kernels`` (see Hugging Face MXFP4 docs)."
        )
        raise ValueError(msg)
    if not is_triton_available(_TRITON_MXFP4_MIN_VERSION):
        msg = (
            f"openai/gpt-oss-20b requires Triton >= {_TRITON_MXFP4_MIN_VERSION} for MXFP4. "
            "Upgrade Triton (bundled with recent PyTorch CUDA wheels)."
        )
        raise ValueError(msg)
    cuda_index = _openai_gpt_oss_resolve_cuda_device_index(device)
    dev_props = torch.cuda.get_device_properties(cuda_index)
    if not hasattr(dev_props, "shared_memory_per_block_optin"):
        msg = (
            "This PyTorch build does not expose ``shared_memory_per_block_optin`` on "
            "``torch.cuda.get_device_properties``, which MXFP4 Triton kernels require. "
            "Use a newer PyTorch CUDA wheel (e.g. PyTorch 2.8+ with CUDA 12.6 wheels from pytorch.org) "
            "and ensure your NVIDIA driver supports that CUDA userland bundle."
        )
        raise ValueError(msg)


def _hf_chat_template_kwargs(model_id: str) -> dict[str, object]:
    """Return chat-template kwargs for the selected model."""
    kwargs: dict[str, object] = {"enable_thinking": False}
    if model_id == _HF_MODEL_GPT_OSS_OPENAI:
        kwargs["reasoning_effort"] = "low"
    return kwargs


def _is_mistral_model(model_id: str) -> bool:
    """Return True when ``model_id`` is a Mistral-family Hub id or a local checkpoint of that architecture.

    Detects full model dirs via ``config.json`` ``model_type`` and PEFT/LoRA checkpoints via
    ``adapter_config.json`` ``base_model_name_or_path`` (Trainer saves adapters without base ``config.json``).
    """
    if model_id.startswith(_MISTRAL_MODEL_PREFIX):
        return True
    model_path = Path(model_id)
    config_path = model_path / "config.json"
    if config_path.is_file():
        try:
            with config_path.open(encoding="utf-8") as config_file:
                config_obj = json.load(config_file)
        except (OSError, json.JSONDecodeError):
            pass
        else:
            model_type_any = config_obj.get("model_type", "")
            if str(model_type_any) in _MISTRAL_FAMILY_MODEL_TYPES:
                return True
    adapter_path = model_path / "adapter_config.json"
    if not adapter_path.is_file():
        return False
    try:
        with adapter_path.open(encoding="utf-8") as adapter_file:
            adapter_obj = json.load(adapter_file)
    except (OSError, json.JSONDecodeError):
        return False
    base_any = adapter_obj.get("base_model_name_or_path", "")
    return str(base_any).startswith(_MISTRAL_MODEL_PREFIX)


def _fallback_chat_prompt(messages: Sequence[ChatMessage]) -> str:
    """Render a simple role-tagged prompt when no chat template is available."""
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).strip().lower()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"User: {content}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


@lru_cache(maxsize=8)
def _mistral_tokenizer_cached(model_id: str) -> Any:
    """Return a cached ``MistralTokenizer`` from a Hub repo id or a local directory containing ``tekken*.json``."""
    model_path = Path(model_id)
    if model_path.is_dir():
        tekken_files = sorted(model_path.glob("tekken*.json"))
        if not tekken_files:
            msg = f"No tekken*.json tokenizer file found in {model_id}"
            raise FileNotFoundError(msg)
        return MistralTokenizer.from_file(str(tekken_files[0]))
    return MistralTokenizer.from_hf_hub(model_id)


def _mistral_encode_chat_prompt_tokens(repo_id: str, messages: Sequence[ChatMessage]) -> list[int]:
    """Encode chat turns with mistral-common (recommended for Mistral3 instruct checkpoints)."""
    mtok = _mistral_tokenizer_cached(repo_id)
    mistral_messages = convert_openai_messages([dict(m) for m in messages])
    encoded = mtok.encode_chat_completion(ChatCompletionRequest(messages=mistral_messages))
    return list(encoded.tokens)


def _build_chat_prompt(tokenizer: Any, messages: Sequence[ChatMessage], model_id: str) -> str:
    """Build a prompt from messages, with fallback when tokenizer chat templates are absent."""
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        return _fallback_chat_prompt(messages)
    try:
        return str(
            apply_chat_template(
                [dict(message) for message in messages],
                tokenize=False,
                add_generation_prompt=True,
                **_hf_chat_template_kwargs(model_id),
            )
        )
    except ValueError as exc:
        if "chat_template is not set" not in str(exc):
            raise
        return _fallback_chat_prompt(messages)


def _hf_text_generation_pipeline_kwargs(model_id: str, device: int | str | None, tokenizer: Any) -> dict[str, Any]:
    """Build ``transformers.pipeline`` kwargs for text generation (dtype, device map, optional 4-bit)."""
    if model_id == _HF_MODEL_GPT_OSS_OPENAI:
        kwargs: dict[str, Any] = {
            "task": "text-generation",
            "model": _HF_MODEL_GPT_OSS_OPENAI,
            "tokenizer": tokenizer,
            "torch_dtype": "auto",
        }
        if isinstance(device, int) or (isinstance(device, str) and device.lower().startswith("cuda")):
            kwargs["device_map"] = {"": device}
        else:
            kwargs["device_map"] = "auto"
        return kwargs

    cuda_requested = _hf_cuda_requested(device)
    quantize = cuda_requested and torch.cuda.is_available() and model_id in _HF_MODELS_4BIT_BNB
    pipeline_kwargs: dict[str, Any] = {
        "task": "text-generation",
        "model": model_id,
        "tokenizer": tokenizer,
        "torch_dtype": torch.bfloat16,
    }
    if quantize:
        if isinstance(device, int) and device < 0:
            msg = "Quantized loading requires a non-negative CUDA device index."
            raise ValueError(msg)
        device_map: str | dict[str, int | str] = "auto" if device is None else {"": device}
        model_kwargs: dict[str, Any] = {
            "quantization_config": BitsAndBytesConfig(  # type: ignore[no-untyped-call]
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            "device_map": device_map,
        }
        pipeline_kwargs["model_kwargs"] = model_kwargs
    else:
        pipeline_kwargs["device"] = (0 if torch.cuda.is_available() else -1) if device is None else device
    return pipeline_kwargs


@dataclass(slots=True)
class HuggingFaceLLMClient:
    """HF text-generation pipeline; optional 4-bit loads; MXFP4 gpt-oss; Mistral3 instruct via mistral-common."""

    model_id: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    device: int | str | None = None

    _generator: TextGenerationPipeline | None = field(init=False, repr=False)
    _pad_token_id: int | None = field(init=False, repr=False)
    _eos_token_id: int | None = field(init=False, repr=False)
    _closed: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        """Initialise the underlying text-generation pipeline."""
        if self.model_id == _HF_MODEL_GPT_OSS_OPENAI:
            _validate_openai_gpt_oss_mxfp4_runtime(self.device)
        tokenizer = (
            MistralCommonBackend.from_pretrained(self.model_id)
            if _is_mistral_model(self.model_id)
            else AutoTokenizer.from_pretrained(self.model_id, legacy=False)
        )
        generator = pipeline(**_hf_text_generation_pipeline_kwargs(self.model_id, self.device, tokenizer))

        tokenizer_any: Any = generator.tokenizer
        if tokenizer_any is None:
            msg = "Text-generation pipeline must define a tokenizer."
            raise ValueError(msg)

        pad_token_id_any = getattr(tokenizer_any, "pad_token_id", None)
        eos_token_id_any = getattr(tokenizer_any, "eos_token_id", None)
        if eos_token_id_any is None:
            msg = "Tokenizer must define an EOS token."
            raise ValueError(msg)

        eos_token_id_int = int(eos_token_id_any)
        pad_token_id_int = eos_token_id_int if pad_token_id_any is None else int(pad_token_id_any)

        if getattr(tokenizer_any, "pad_token", None) is None:
            eos_token = getattr(tokenizer_any, "eos_token", None)
            if eos_token is not None:
                tokenizer_any.pad_token = eos_token
        if getattr(tokenizer_any, "pad_token_id", None) is None:
            tokenizer_any.pad_token_id = pad_token_id_int
        if getattr(tokenizer_any, "padding_side", None) != "left":
            tokenizer_any.padding_side = "left"
        model_config = getattr(generator.model, "config", None)
        if model_config is not None:
            model_config_any: Any = model_config
            model_config_any.pad_token_id = pad_token_id_int

        _clear_max_length_from_hf_pipeline_generation_config(generator)

        self._generator = generator
        self._pad_token_id = pad_token_id_int
        self._eos_token_id = eos_token_id_int
        self._closed = False

    def close(self) -> None:
        """Release the underlying pipeline and free GPU memory."""
        if self._closed:
            return
        generator = self._generator
        self._generator = None
        self._pad_token_id = None
        self._eos_token_id = None
        self._closed = True
        if generator is not None:
            del generator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_open(self) -> TextGenerationPipeline:
        """Return the generator if available, raising if closed."""
        if self._closed or self._generator is None:
            msg = "LLM client has been closed and cannot be reused."
            raise RuntimeError(msg)
        return self._generator

    def _generate_batch_with_mistral_common(
        self,
        generator: TextGenerationPipeline,
        messages_batch: Sequence[Sequence[ChatMessage]],
        *,
        max_new_tokens: int,
        sampling_temperature: float,
        sampling_top_p: float,
        stop_sequences: Sequence[str] | None,
        batch_size: int,
    ) -> list[str]:
        """Batch-generate via mistral-common token ids and one ``model.generate`` per batch."""
        if not messages_batch:
            return []
        if batch_size <= 0:
            msg = "batch_size must be a positive integer."
            raise ValueError(msg)

        pad_token_id = self._pad_token_id
        eos_token_id = self._eos_token_id
        if pad_token_id is None or eos_token_id is None:
            msg = "LLM client is not initialised with token IDs."
            raise RuntimeError(msg)

        mistral_tok = _mistral_tokenizer_cached(self.model_id)
        model_any: Any = generator.model
        device = next(model_any.parameters()).device
        generation_kwargs = dict(
            _build_generation_kwargs(
                tokens=max_new_tokens,
                sampling_temperature=sampling_temperature,
                sampling_top_p=sampling_top_p,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
        )
        generation_kwargs.pop("return_full_text", None)
        generation_kwargs.pop("batch_size", None)

        repo_id = self.model_id
        results: list[str] = []
        for start in range(0, len(messages_batch), batch_size):
            chunk = messages_batch[start : start + batch_size]
            prompt_rows = [_mistral_encode_chat_prompt_tokens(repo_id, messages) for messages in chunk]
            prompt_lens = [len(prompt_ids) for prompt_ids in prompt_rows]
            max_len = max(prompt_lens)

            input_list: list[list[int]] = []
            mask_list: list[list[int]] = []
            for prompt_ids, prompt_len in zip(prompt_rows, prompt_lens, strict=True):
                pad_count = max_len - prompt_len
                input_list.append([pad_token_id] * pad_count + prompt_ids)
                mask_list.append([0] * pad_count + [1] * prompt_len)

            input_ids = torch.tensor(input_list, device=device, dtype=torch.long)
            attention_mask = torch.tensor(mask_list, device=device, dtype=torch.long)

            start_t = perf_counter()
            try:
                with torch.inference_mode():
                    output_ids = model_any.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **generation_kwargs,
                    )
            finally:
                elapsed = perf_counter() - start_t
                record_hf_call(elapsed)

            for row in range(len(prompt_rows)):
                continuation = output_ids[row, max_len:].tolist()
                decoded = mistral_tok.decode(continuation).strip()
                results.append(_apply_stop_sequences(decoded, stop_sequences))

        return results

    @staticmethod
    def _parse_batch_outputs(
        outputs: object,
        *,
        stop_sequences: Sequence[str] | None,
        expected_count: int,
    ) -> list[str]:
        """Parse raw pipeline outputs into cleaned response strings."""
        if not isinstance(outputs, list):
            outputs = [outputs]

        texts: list[str] = []
        for entry in outputs:
            candidate = entry[0] if isinstance(entry, list) and entry else entry
            if isinstance(candidate, Mapping):
                raw_text = str(candidate.get("generated_text", ""))
            else:
                raw_text = str(candidate or "")

            texts.append(_apply_stop_sequences(raw_text, stop_sequences))

        if len(texts) < expected_count:
            texts.extend([""] * (expected_count - len(texts)))
        return texts

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop_sequences: Sequence[str] | None = None,
    ) -> str:
        """Generate chat completions using the configured pipeline."""
        return self.generate_batch(
            [messages],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
            batch_size=None,
        )[0]

    def generate_batch(
        self,
        messages_batch: Sequence[Sequence[ChatMessage]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop_sequences: Sequence[str] | None = None,
        batch_size: int | None = None,
    ) -> Sequence[str]:
        """Generate chat completions for a batch of message sequences."""
        if not messages_batch:
            return []

        generator = self._ensure_open()
        tokenizer: Any = generator.tokenizer
        sampling_temperature = self.temperature if temperature is None else temperature
        sampling_top_p = self.top_p if top_p is None else top_p
        tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if self._pad_token_id is None or self._eos_token_id is None:
            msg = "LLM client is not initialised with token IDs."
            raise RuntimeError(msg)

        if _is_mistral_model(self.model_id):
            effective_batch_size = batch_size if batch_size is not None else len(messages_batch)
            if effective_batch_size <= 0:
                effective_batch_size = 1
            return self._generate_batch_with_mistral_common(
                generator,
                messages_batch,
                max_new_tokens=tokens,
                sampling_temperature=sampling_temperature,
                sampling_top_p=sampling_top_p,
                stop_sequences=stop_sequences,
                batch_size=effective_batch_size,
            )

        prompts = [_build_chat_prompt(tokenizer, msgs, self.model_id) for msgs in messages_batch]

        effective_batch_size = batch_size if batch_size is not None else len(prompts)
        if effective_batch_size <= 0:
            effective_batch_size = 1

        start = perf_counter()
        try:
            generation_kwargs = _build_generation_kwargs(
                tokens=tokens,
                sampling_temperature=sampling_temperature,
                sampling_top_p=sampling_top_p,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self._pad_token_id,
                eos_token_id=self._eos_token_id,
                batch_size=effective_batch_size,
            )
            outputs = generator(prompts, **generation_kwargs)
        finally:
            elapsed = perf_counter() - start
            record_hf_call(elapsed)

        return self._parse_batch_outputs(
            outputs,
            stop_sequences=stop_sequences,
            expected_count=len(messages_batch),
        )


def _build_generation_kwargs(
    *,
    tokens: int,
    sampling_temperature: float,
    sampling_top_p: float,
    repetition_penalty: float,
    pad_token_id: int,
    eos_token_id: int,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Assemble generation kwargs while omitting invalid sampling params."""
    do_sample = sampling_temperature > 0
    kwargs: dict[str, Any] = {
        "max_new_tokens": tokens,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
        "return_full_text": False,
    }
    if do_sample:
        kwargs["temperature"] = sampling_temperature
        kwargs["top_p"] = sampling_top_p
    else:
        kwargs["temperature"] = None
        kwargs["top_p"] = None
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    return kwargs
