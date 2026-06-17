# ruff: noqa: S101, S105, S108, PT019, SLF001, PLR2004
"""Tests for the run_dpo command-line script."""

from __future__ import annotations

import json
import logging
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from transformers.tokenization_mistral_common import MistralCommonBackend

from recsys_nle.pipeline.reward import RewardType
from scripts import run_dpo


def _build_args(tmp_path: Path, **overrides: Any) -> Namespace:
    """Build CLI argument namespaces with sensible defaults."""
    defaults: dict[str, Any] = {
        "model_id": "test/model",
        "datasets_dir_a": tmp_path / "dir_a",
        "datasets_dir_b": tmp_path / "dir_b",
        "output_dir": tmp_path / "output",
        "reward": "informativeness",
        "learning_rate": 5e-5,
        "beta": 0.05,
        "n_epochs": 5,
        "log_level": "INFO",
        "eval_dataset_split": None,
        "lora_r": 8,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


_SAMPLE_PAIR = {
    "chosen": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "a"}],
    "rejected": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "b"}],
    "score_chosen": 1.0,
    "score_rejected": 0.5,
}


def test_parse_args_requires_lora_r(monkeypatch: Any) -> None:
    """parse_args exits when --lora-r is omitted."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_dpo.py",
            "--model-id",
            "test/model",
            "--datasets-dir-a",
            "/tmp/a",
            "--datasets-dir-b",
            "/tmp/b",
            "--output-dir",
            "/tmp/output",
            "--reward",
            "informativeness",
            "--learning-rate",
            "5e-5",
            "--beta",
            "0.05",
            "--n-epochs",
            "5",
        ],
    )
    with pytest.raises(SystemExit):
        run_dpo.parse_args()


def test_parse_args_accepts_custom_lora_r(monkeypatch: Any) -> None:
    """parse_args maps --lora-r to attributes used for LoRA rank."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_dpo.py",
            "--model-id",
            "test/model",
            "--datasets-dir-a",
            "/tmp/a",
            "--datasets-dir-b",
            "/tmp/b",
            "--output-dir",
            "/tmp/output",
            "--reward",
            "informativeness",
            "--learning-rate",
            "5e-5",
            "--beta",
            "0.05",
            "--n-epochs",
            "5",
            "--lora-r",
            "16",
        ],
    )
    args = run_dpo.parse_args()
    assert args.lora_r == 16


def test_parse_args_output_dir(monkeypatch: Any) -> None:
    """parse_args returns the --output-dir value as a Path."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_dpo.py",
            "--model-id",
            "test/model",
            "--datasets-dir-a",
            "/tmp/a",
            "--datasets-dir-b",
            "/tmp/b",
            "--output-dir",
            "/tmp/output",
            "--reward",
            "informativeness",
            "--learning-rate",
            "5e-5",
            "--beta",
            "0.05",
            "--n-epochs",
            "5",
            "--lora-r",
            "8",
        ],
    )
    args = run_dpo.parse_args()
    assert args.output_dir == Path("/tmp/output")
    assert args.n_epochs == 5


def test_parse_args_output_dir_is_required(monkeypatch: Any) -> None:
    """parse_args exits when --output-dir is omitted."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_dpo.py",
            "--model-id",
            "test/model",
            "--datasets-dir-a",
            "/tmp/a",
            "--datasets-dir-b",
            "/tmp/b",
            "--reward",
            "informativeness",
            "--learning-rate",
            "5e-5",
            "--beta",
            "0.05",
            "--n-epochs",
            "5",
            "--lora-r",
            "8",
        ],
    )
    with pytest.raises(SystemExit):
        run_dpo.parse_args()


def test_parse_args_n_epochs_is_required(monkeypatch: Any) -> None:
    """parse_args exits when --n-epochs is omitted."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_dpo.py",
            "--model-id",
            "test/model",
            "--datasets-dir-a",
            "/tmp/a",
            "--datasets-dir-b",
            "/tmp/b",
            "--output-dir",
            "/tmp/output",
            "--reward",
            "informativeness",
            "--learning-rate",
            "5e-5",
            "--beta",
            "0.05",
            "--lora-r",
            "8",
        ],
    )
    with pytest.raises(SystemExit):
        run_dpo.parse_args()


@patch("scripts.run_dpo._DPOTrainerWithEvalMetrics")
@patch("scripts.run_dpo.DPOConfig")
@patch("scripts.run_dpo.load_tokenizer")
@patch("scripts.run_dpo.AutoModelForCausalLM")
@patch("scripts.run_dpo._validate_and_select_pairs")
@patch("scripts.run_dpo._load_dataset_dir")
@patch("scripts.run_dpo.parse_args")
def test_main_copies_best_checkpoint(
    mock_parse_args: MagicMock,
    mock_load_dataset_dir: MagicMock,
    mock_validate: MagicMock,
    _mock_auto_model: MagicMock,
    mock_load_tokenizer: MagicMock,
    _mock_dpo_config: MagicMock,
    mock_dpo_trainer_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """main() copies the best checkpoint directory to best_model/."""
    args = _build_args(tmp_path)
    mock_parse_args.return_value = args
    mock_load_dataset_dir.return_value = pd.DataFrame({"user_id": [1]})
    mock_validate.return_value = [_SAMPLE_PAIR]

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "PAD"
    mock_load_tokenizer.return_value = mock_tokenizer

    # Create a fake checkpoint directory with adapter files
    ckpt_dir = tmp_path / "output" / "checkpoint-3"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "adapter_config.json").write_text("{}")
    (ckpt_dir / "adapter_model.safetensors").write_text("fake-weights")

    mock_trainer = MagicMock()
    mock_trainer.state.best_model_checkpoint = str(ckpt_dir)
    mock_dpo_trainer_cls.return_value = mock_trainer

    result = run_dpo.main()

    assert result == 0
    best_model_dir = tmp_path / "output" / "best_model"
    assert best_model_dir.is_dir()
    assert (best_model_dir / "adapter_config.json").exists()
    assert (best_model_dir / "adapter_model.safetensors").read_text() == "fake-weights"


@patch("scripts.run_dpo._DPOTrainerWithEvalMetrics")
@patch("scripts.run_dpo.DPOConfig")
@patch("scripts.run_dpo.load_tokenizer")
@patch("scripts.run_dpo.AutoModelForCausalLM")
@patch("scripts.run_dpo._validate_and_select_pairs")
@patch("scripts.run_dpo._load_dataset_dir")
@patch("scripts.run_dpo.parse_args")
def test_main_raises_when_no_best_checkpoint(
    mock_parse_args: MagicMock,
    mock_load_dataset_dir: MagicMock,
    mock_validate: MagicMock,
    _mock_auto_model: MagicMock,
    mock_load_tokenizer: MagicMock,
    _mock_dpo_config: MagicMock,
    mock_dpo_trainer_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """main() raises RuntimeError when no best checkpoint was recorded."""
    args = _build_args(tmp_path)
    mock_parse_args.return_value = args
    mock_load_dataset_dir.return_value = pd.DataFrame({"user_id": [1]})
    mock_validate.return_value = [_SAMPLE_PAIR]

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "PAD"
    mock_load_tokenizer.return_value = mock_tokenizer

    mock_trainer = MagicMock()
    mock_trainer.state.best_model_checkpoint = None
    mock_dpo_trainer_cls.return_value = mock_trainer

    with pytest.raises(RuntimeError, match="No best checkpoint"):
        run_dpo.main()


@patch("scripts.run_dpo.LoraConfig")
@patch("scripts.run_dpo._DPOTrainerWithEvalMetrics")
@patch("scripts.run_dpo.DPOConfig")
@patch("scripts.run_dpo.load_tokenizer")
@patch("scripts.run_dpo.AutoModelForCausalLM")
@patch("scripts.run_dpo._validate_and_select_pairs")
@patch("scripts.run_dpo._load_dataset_dir")
@patch("scripts.run_dpo.parse_args")
def test_main_lora_config_sets_alpha_to_twice_rank(
    mock_parse_args: MagicMock,
    mock_load_dataset_dir: MagicMock,
    mock_validate: MagicMock,
    _mock_auto_model: MagicMock,
    mock_load_tokenizer: MagicMock,
    _mock_dpo_config: MagicMock,
    mock_dpo_trainer_cls: MagicMock,
    mock_lora_config: MagicMock,
    tmp_path: Path,
) -> None:
    """main() builds LoraConfig with r=args.lora_r and lora_alpha=r*2."""
    args = _build_args(tmp_path, lora_r=16)
    mock_parse_args.return_value = args
    mock_load_dataset_dir.return_value = pd.DataFrame({"user_id": [1]})
    mock_validate.return_value = [_SAMPLE_PAIR]

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "PAD"
    mock_load_tokenizer.return_value = mock_tokenizer

    ckpt_dir = tmp_path / "output" / "checkpoint-1"
    ckpt_dir.mkdir(parents=True)

    mock_trainer = MagicMock()
    mock_trainer.state.best_model_checkpoint = str(ckpt_dir)
    mock_dpo_trainer_cls.return_value = mock_trainer

    run_dpo.main()

    lora_kwargs = mock_lora_config.call_args.kwargs
    assert lora_kwargs["r"] == 16
    assert lora_kwargs["lora_alpha"] == 32


@patch("scripts.run_dpo._DPOTrainerWithEvalMetrics")
@patch("scripts.run_dpo.DPOConfig")
@patch("scripts.run_dpo.load_tokenizer")
@patch("scripts.run_dpo.AutoModelForCausalLM")
@patch("scripts.run_dpo._validate_and_select_pairs")
@patch("scripts.run_dpo._load_dataset_dir")
@patch("scripts.run_dpo.parse_args")
def test_main_passes_output_dir_and_n_epochs_to_dpo_config(
    mock_parse_args: MagicMock,
    mock_load_dataset_dir: MagicMock,
    mock_validate: MagicMock,
    _mock_auto_model: MagicMock,
    mock_load_tokenizer: MagicMock,
    mock_dpo_config: MagicMock,
    mock_dpo_trainer_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """main() passes the CLI output_dir and n_epochs to DPOConfig."""
    args = _build_args(tmp_path, n_epochs=7)
    mock_parse_args.return_value = args
    mock_load_dataset_dir.return_value = pd.DataFrame({"user_id": [1]})
    mock_validate.return_value = [_SAMPLE_PAIR]

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "PAD"
    mock_load_tokenizer.return_value = mock_tokenizer

    ckpt_dir = tmp_path / "output" / "checkpoint-1"
    ckpt_dir.mkdir(parents=True)

    mock_trainer = MagicMock()
    mock_trainer.state.best_model_checkpoint = str(ckpt_dir)
    mock_dpo_trainer_cls.return_value = mock_trainer

    run_dpo.main()

    config_kwargs = mock_dpo_config.call_args
    assert config_kwargs.kwargs["output_dir"] == str(tmp_path / "output")
    assert config_kwargs.kwargs["num_train_epochs"] == 7


# -- Tests for load_tokenizer --


@patch("scripts.run_dpo._DPOMistralTokenizer")
def test_load_tokenizer_uses_dpo_mistral_tokenizer_for_mistral_models(
    mock_mistral_cls: MagicMock,
) -> None:
    """load_tokenizer uses _DPOMistralTokenizer for mistralai/ models."""
    mock_tok = MagicMock()
    mock_mistral_cls.from_pretrained.return_value = mock_tok

    tokenizer = run_dpo.load_tokenizer("mistralai/Ministral-8B-Instruct-2410")

    assert tokenizer is mock_tok
    mock_mistral_cls.from_pretrained.assert_called_once_with("mistralai/Ministral-8B-Instruct-2410")


def test_dpo_mistral_tokenizer_sets_continue_final_message_for_assistant() -> None:
    """_DPOMistralTokenizer auto-sets continue_final_message when last turn is assistant."""
    tok = run_dpo._DPOMistralTokenizer.__new__(run_dpo._DPOMistralTokenizer)
    calls: list[dict[str, Any]] = []

    def fake_apply(_self: Any, _conversation: Any, *, continue_final_message: bool = False, **kwargs: Any) -> str:
        calls.append({"continue_final_message": continue_final_message, **kwargs})
        return "ok"

    with patch.object(MistralCommonBackend, "apply_chat_template", fake_apply):
        tok.apply_chat_template(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        )

    assert calls[0]["continue_final_message"] is True


def test_dpo_mistral_tokenizer_does_not_set_continue_for_user() -> None:
    """_DPOMistralTokenizer leaves continue_final_message=False when last turn is user."""
    tok = run_dpo._DPOMistralTokenizer.__new__(run_dpo._DPOMistralTokenizer)
    calls: list[dict[str, Any]] = []

    def fake_apply(_self: Any, _conversation: Any, *, continue_final_message: bool = False, **kwargs: Any) -> str:
        calls.append({"continue_final_message": continue_final_message, **kwargs})
        return "ok"

    with patch.object(MistralCommonBackend, "apply_chat_template", fake_apply):
        tok.apply_chat_template(
            [{"role": "user", "content": "hi"}],
        )

    assert calls[0]["continue_final_message"] is False


def test_dpo_mistral_tokenizer_respects_explicit_continue_true() -> None:
    """_DPOMistralTokenizer preserves an explicit continue_final_message=True."""
    tok = run_dpo._DPOMistralTokenizer.__new__(run_dpo._DPOMistralTokenizer)
    calls: list[dict[str, Any]] = []

    def fake_apply(_self: Any, _conversation: Any, *, continue_final_message: bool = False, **kwargs: Any) -> str:
        calls.append({"continue_final_message": continue_final_message, **kwargs})
        return "ok"

    with patch.object(MistralCommonBackend, "apply_chat_template", fake_apply):
        tok.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            continue_final_message=True,
        )

    assert calls[0]["continue_final_message"] is True


@patch("scripts.run_dpo.AutoTokenizer")
@patch("scripts.run_dpo.AutoProcessor")
def test_load_tokenizer_extracts_inner_tokenizer_from_processor(
    mock_auto_processor: MagicMock,
    mock_auto_tokenizer: MagicMock,
) -> None:
    """load_tokenizer extracts the inner tokenizer from a multimodal processor."""
    mock_inner_tokenizer = MagicMock()
    mock_processor = MagicMock()
    mock_processor.tokenizer = mock_inner_tokenizer
    mock_auto_processor.from_pretrained.return_value = mock_processor

    tokenizer = run_dpo.load_tokenizer("some/multimodal-model")

    assert tokenizer is mock_inner_tokenizer
    mock_auto_tokenizer.from_pretrained.assert_not_called()


@patch("scripts.run_dpo.AutoTokenizer")
@patch("scripts.run_dpo.AutoProcessor")
def test_load_tokenizer_returns_tokenizer_from_auto_processor(
    mock_auto_processor: MagicMock,
    mock_auto_tokenizer: MagicMock,
) -> None:
    """load_tokenizer returns the object directly when AutoProcessor yields a tokenizer."""
    mock_tokenizer = MagicMock(spec=[])  # no .tokenizer attribute
    mock_auto_processor.from_pretrained.return_value = mock_tokenizer

    tokenizer = run_dpo.load_tokenizer("some/text-model")

    assert tokenizer is mock_tokenizer
    mock_auto_tokenizer.from_pretrained.assert_not_called()


@patch("scripts.run_dpo.AutoTokenizer")
@patch("scripts.run_dpo.AutoProcessor")
def test_load_tokenizer_falls_back_to_auto_tokenizer(
    mock_auto_processor: MagicMock,
    mock_auto_tokenizer: MagicMock,
) -> None:
    """load_tokenizer falls back to AutoTokenizer when AutoProcessor fails."""
    mock_auto_processor.from_pretrained.side_effect = KeyError("no processor config")
    mock_tokenizer = MagicMock()
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    tokenizer = run_dpo.load_tokenizer("some/text-model")

    assert tokenizer is mock_tokenizer


@patch("scripts.run_dpo.AutoTokenizer")
@patch("scripts.run_dpo.AutoProcessor")
def test_load_tokenizer_falls_back_on_os_error(
    mock_auto_processor: MagicMock,
    mock_auto_tokenizer: MagicMock,
) -> None:
    """load_tokenizer also catches OSError from AutoProcessor."""
    mock_auto_processor.from_pretrained.side_effect = OSError("file not found")
    mock_tokenizer = MagicMock()
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

    tokenizer = run_dpo.load_tokenizer("some/local-model")

    assert tokenizer is mock_tokenizer


# -- Tests for multimodal model_type override in main() --


@patch("scripts.run_dpo._DPOTrainerWithEvalMetrics")
@patch("scripts.run_dpo.DPOConfig")
@patch("scripts.run_dpo.load_tokenizer")
@patch("scripts.run_dpo.AutoModelForCausalLM")
@patch("scripts.run_dpo._validate_and_select_pairs")
@patch("scripts.run_dpo._load_dataset_dir")
@patch("scripts.run_dpo.parse_args")
def test_main_overrides_model_type_for_multimodal_models(
    mock_parse_args: MagicMock,
    mock_load_dataset_dir: MagicMock,
    mock_validate: MagicMock,
    mock_auto_model: MagicMock,
    mock_load_tokenizer: MagicMock,
    _mock_dpo_config: MagicMock,
    mock_dpo_trainer_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """main() switches model_type to the text sub-config type for multimodal models."""
    args = _build_args(tmp_path)
    mock_parse_args.return_value = args
    mock_load_dataset_dir.return_value = pd.DataFrame({"user_id": [1]})
    mock_validate.return_value = [_SAMPLE_PAIR]

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "PAD"
    mock_load_tokenizer.return_value = mock_tokenizer

    mock_model = MagicMock()
    mock_model.config.model_type = "gemma3"
    mock_model.config.text_config.model_type = "gemma3_text"
    mock_auto_model.from_pretrained.return_value = mock_model

    ckpt_dir = tmp_path / "output" / "checkpoint-1"
    ckpt_dir.mkdir(parents=True)

    mock_trainer = MagicMock()
    mock_trainer.state.best_model_checkpoint = str(ckpt_dir)
    mock_dpo_trainer_cls.return_value = mock_trainer

    run_dpo.main()

    assert mock_model.config.model_type == "gemma3_text"


@patch("scripts.run_dpo._DPOTrainerWithEvalMetrics")
@patch("scripts.run_dpo.DPOConfig")
@patch("scripts.run_dpo.load_tokenizer")
@patch("scripts.run_dpo.AutoModelForCausalLM")
@patch("scripts.run_dpo._validate_and_select_pairs")
@patch("scripts.run_dpo._load_dataset_dir")
@patch("scripts.run_dpo.parse_args")
def test_main_leaves_model_type_for_text_only_models(
    mock_parse_args: MagicMock,
    mock_load_dataset_dir: MagicMock,
    mock_validate: MagicMock,
    mock_auto_model: MagicMock,
    mock_load_tokenizer: MagicMock,
    _mock_dpo_config: MagicMock,
    mock_dpo_trainer_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """main() does not modify model_type for text-only models without text_config."""
    args = _build_args(tmp_path)
    mock_parse_args.return_value = args
    mock_load_dataset_dir.return_value = pd.DataFrame({"user_id": [1]})
    mock_validate.return_value = [_SAMPLE_PAIR]

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "PAD"
    mock_load_tokenizer.return_value = mock_tokenizer

    mock_model = MagicMock()
    mock_model.config = MagicMock(spec=["model_type", "is_encoder_decoder", "use_cache"])
    mock_model.config.model_type = "llama"
    mock_auto_model.from_pretrained.return_value = mock_model

    ckpt_dir = tmp_path / "output" / "checkpoint-1"
    ckpt_dir.mkdir(parents=True)

    mock_trainer = MagicMock()
    mock_trainer.state.best_model_checkpoint = str(ckpt_dir)
    mock_dpo_trainer_cls.return_value = mock_trainer

    run_dpo.main()

    assert mock_model.config.model_type == "llama"


def test_dpo_trainer_with_eval_metrics_log_mutates_dict_in_place() -> None:
    """_DPOTrainerWithEvalMetrics.log() merges DPO metrics into the passed dict in-place."""
    trainer = run_dpo._DPOTrainerWithEvalMetrics.__new__(run_dpo._DPOTrainerWithEvalMetrics)
    trainer._metrics = defaultdict(lambda: defaultdict(list))
    trainer._metrics["eval"]["rewards/margins"] = [2.0, 6.0]
    trainer._metrics["eval"]["rewards/accuracies"] = [0.8, 0.9]
    trainer.model = MagicMock(training=False)

    logs: dict[str, float] = {"eval_loss": 1.5}
    with patch("transformers.Trainer.log"):
        trainer.log(logs)

    assert logs["eval_rewards/margins"] == pytest.approx(4.0)
    assert logs["eval_rewards/accuracies"] == pytest.approx(0.85)
    assert logs["eval_loss"] == pytest.approx(1.5)
    assert trainer._metrics["eval"] == {}


@patch("scripts.run_dpo._DPOTrainerWithEvalMetrics")
@patch("scripts.run_dpo.DPOConfig")
@patch("scripts.run_dpo.load_tokenizer")
@patch("scripts.run_dpo.AutoModelForCausalLM")
@patch("scripts.run_dpo._validate_and_select_pairs")
@patch("scripts.run_dpo._load_dataset_dir")
@patch("scripts.run_dpo.parse_args")
def test_main_passes_tokenizer_to_dpo_trainer(
    mock_parse_args: MagicMock,
    mock_load_dataset_dir: MagicMock,
    mock_validate: MagicMock,
    _mock_auto_model: MagicMock,
    mock_load_tokenizer: MagicMock,
    _mock_dpo_config: MagicMock,
    mock_dpo_trainer_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """main() passes the tokenizer to _DPOTrainerWithEvalMetrics."""
    args = _build_args(tmp_path)
    mock_parse_args.return_value = args
    mock_load_dataset_dir.return_value = pd.DataFrame({"user_id": [1]})
    mock_validate.return_value = [_SAMPLE_PAIR]

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "PAD"
    mock_load_tokenizer.return_value = mock_tokenizer

    ckpt_dir = tmp_path / "output" / "checkpoint-1"
    ckpt_dir.mkdir(parents=True)

    mock_trainer = MagicMock()
    mock_trainer.state.best_model_checkpoint = str(ckpt_dir)
    mock_dpo_trainer_cls.return_value = mock_trainer

    run_dpo.main()

    call_kwargs = mock_dpo_trainer_cls.call_args
    assert call_kwargs.kwargs["processing_class"] is mock_tokenizer


@patch("scripts.run_dpo._DPOTrainerWithEvalMetrics")
@patch("scripts.run_dpo.DPOConfig")
@patch("scripts.run_dpo.load_tokenizer")
@patch("scripts.run_dpo.AutoModelForCausalLM")
@patch("scripts.run_dpo._validate_and_select_pairs")
@patch("scripts.run_dpo._load_dataset_dir")
@patch("scripts.run_dpo.parse_args")
def test_main_enables_input_require_grads(
    mock_parse_args: MagicMock,
    mock_load_dataset_dir: MagicMock,
    mock_validate: MagicMock,
    mock_auto_model: MagicMock,
    mock_load_tokenizer: MagicMock,
    _mock_dpo_config: MagicMock,
    mock_dpo_trainer_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """main() calls enable_input_require_grads so gradients reach LoRA under gradient checkpointing."""
    args = _build_args(tmp_path)
    mock_parse_args.return_value = args
    mock_load_dataset_dir.return_value = pd.DataFrame({"user_id": [1]})
    mock_validate.return_value = [_SAMPLE_PAIR]

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "PAD"
    mock_load_tokenizer.return_value = mock_tokenizer

    mock_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model

    ckpt_dir = tmp_path / "output" / "checkpoint-1"
    ckpt_dir.mkdir(parents=True)

    mock_trainer = MagicMock()
    mock_trainer.state.best_model_checkpoint = str(ckpt_dir)
    mock_dpo_trainer_cls.return_value = mock_trainer

    run_dpo.main()

    mock_model.gradient_checkpointing_enable.assert_called_once()
    mock_model.enable_input_require_grads.assert_called_once()


# -- Tests for _validate_and_select_pairs win-source logging --


def _make_pair_dataframes(
    scores_a: list[float],
    scores_b: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build minimal DataFrames for _validate_and_select_pairs with informativeness scoring."""
    rows_a = []
    rows_b = []
    for i, (sa, sb) in enumerate(zip(scores_a, scores_b, strict=True)):
        conv_a = json.dumps([{"role": "user", "content": "hi"}, {"role": "assistant", "content": f"a{i}"}])
        conv_b = json.dumps([{"role": "user", "content": "hi"}, {"role": "assistant", "content": f"b{i}"}])
        rows_a.append(
            {
                "user_id": i,
                "conversation": conv_a,
                "explanation_cfx_pattern_match_mean": sa,
                "explanation_non_cfx_pattern_match_mean": 0.0,
            }
        )
        rows_b.append(
            {
                "user_id": i,
                "conversation": conv_b,
                "explanation_cfx_pattern_match_mean": sb,
                "explanation_non_cfx_pattern_match_mean": 0.0,
            }
        )
    df_a = pd.DataFrame(rows_a)
    df_b = pd.DataFrame(rows_b)
    return df_a, df_b


def test_validate_and_select_pairs_logs_win_source_counts(caplog: pytest.LogCaptureFixture) -> None:
    """_validate_and_select_pairs logs how many winners come from each dataset."""
    # 2 wins for A (scores 0.8 > 0.2, 0.6 > 0.1), 1 win for B (0.1 < 0.9), 1 tie skipped
    df_a, df_b = _make_pair_dataframes(
        scores_a=[0.8, 0.6, 0.1, 0.5],
        scores_b=[0.2, 0.1, 0.9, 0.5],
    )

    with caplog.at_level(logging.INFO, logger="scripts.run_dpo"):
        pairs = run_dpo._validate_and_select_pairs(df_a, df_b, RewardType.INFORMATIVENESS)

    assert len(pairs) == 3  # 2 A-wins + 1 B-win; tie excluded

    matching = [r for r in caplog.records if "Pairwise winners" in r.message]
    assert len(matching) == 1
    record = matching[0]
    assert "2 from dataset A" in record.message
    assert "1 from dataset B" in record.message
    assert "3 pairs" in record.message
    assert "skipped 0 pairs" in record.message


def test_validate_and_select_pairs_correctness_informativeness_combines_both_terms() -> None:
    """correctness_informativeness scores 1.5*cfx - non_cfx; both terms must contribute to the winner."""
    df_a = pd.DataFrame(
        [
            {
                "user_id": 0,
                "conversation": json.dumps([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "a0"}]),
                "explanation_cfx_pattern_match_mean": 0.5,
                "explanation_non_cfx_pattern_match_mean": 0.0,  # score = 0.75
            },
        ]
    )
    df_b = pd.DataFrame(
        [
            {
                "user_id": 0,
                "conversation": json.dumps([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "b0"}]),
                "explanation_cfx_pattern_match_mean": 0.6,
                "explanation_non_cfx_pattern_match_mean": 0.5,  # score = 0.4
            },
        ]
    )

    pairs = run_dpo._validate_and_select_pairs(df_a, df_b, RewardType.CORRECTNESS_INFORMATIVENESS)

    assert len(pairs) == 1
    assert pairs[0]["score_chosen"] == pytest.approx(0.75)
    assert pairs[0]["score_rejected"] == pytest.approx(0.4)
    assert pairs[0]["chosen"][-1]["content"] == "a0"
    assert pairs[0]["rejected"][-1]["content"] == "b0"


def test_validate_and_select_pairs_skips_rows_with_missing_scores(caplog: pytest.LogCaptureFixture) -> None:
    """Rows where any required score column is NaN on either side are dropped, not zero-imputed."""
    df_a, df_b = _make_pair_dataframes(scores_a=[0.8, 0.6], scores_b=[0.2, 0.1])
    df_a.loc[0, "explanation_non_cfx_pattern_match_mean"] = float("nan")
    df_b.loc[1, "explanation_cfx_pattern_match_mean"] = float("nan")

    with caplog.at_level(logging.INFO, logger="scripts.run_dpo"):
        pairs = run_dpo._validate_and_select_pairs(df_a, df_b, RewardType.INFORMATIVENESS)

    assert pairs == []
    assert all(not pd.isna(p["score_chosen"]) and not pd.isna(p["score_rejected"]) for p in pairs)

    summary = next(r for r in caplog.records if "Pairwise winners" in r.message)
    assert "skipped 2 pairs" in summary.message


def test_validate_and_select_pairs_correctness_informativeness_readability_combines_all_terms() -> None:
    """correctness_informativeness_readability scores 1.5*cfx - non_cfx + 1.5*readability."""
    df_a = pd.DataFrame(
        [
            {
                "user_id": 0,
                "conversation": json.dumps([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "a0"}]),
                "explanation_cfx_pattern_match_mean": 0.5,
                "explanation_non_cfx_pattern_match_mean": 0.0,
                "readability_overall_mean": 0.8,  # score = 0.75 + 1.2 = 1.95
            },
        ]
    )
    df_b = pd.DataFrame(
        [
            {
                "user_id": 0,
                "conversation": json.dumps([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "b0"}]),
                "explanation_cfx_pattern_match_mean": 0.6,
                "explanation_non_cfx_pattern_match_mean": 0.5,
                "readability_overall_mean": 0.2,  # score = 0.4 + 0.3 = 0.7
            },
        ]
    )

    pairs = run_dpo._validate_and_select_pairs(df_a, df_b, RewardType.CORRECTNESS_INFORMATIVENESS_READABILITY)

    assert len(pairs) == 1
    assert pairs[0]["score_chosen"] == pytest.approx(1.95)
    assert pairs[0]["score_rejected"] == pytest.approx(0.7)
    assert pairs[0]["chosen"][-1]["content"] == "a0"
    assert pairs[0]["rejected"][-1]["content"] == "b0"


@patch("scripts.run_dpo._DPOTrainerWithEvalMetrics")
@patch("scripts.run_dpo.DPOConfig")
@patch("scripts.run_dpo.load_tokenizer")
@patch("scripts.run_dpo.AutoModelForCausalLM")
@patch("scripts.run_dpo._validate_and_select_pairs")
@patch("scripts.run_dpo._load_dataset_dir")
@patch("scripts.run_dpo.parse_args")
def test_main_passes_precompute_ref_log_probs_to_dpo_config(
    mock_parse_args: MagicMock,
    mock_load_dataset_dir: MagicMock,
    mock_validate: MagicMock,
    _mock_auto_model: MagicMock,
    mock_load_tokenizer: MagicMock,
    mock_dpo_config: MagicMock,
    mock_dpo_trainer_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """main() passes precompute_ref_log_probs=True to DPOConfig to reduce GPU memory."""
    args = _build_args(tmp_path)
    mock_parse_args.return_value = args
    mock_load_dataset_dir.return_value = pd.DataFrame({"user_id": [1]})
    mock_validate.return_value = [_SAMPLE_PAIR]

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "PAD"
    mock_load_tokenizer.return_value = mock_tokenizer

    ckpt_dir = tmp_path / "output" / "checkpoint-1"
    ckpt_dir.mkdir(parents=True)

    mock_trainer = MagicMock()
    mock_trainer.state.best_model_checkpoint = str(ckpt_dir)
    mock_dpo_trainer_cls.return_value = mock_trainer

    run_dpo.main()

    config_kwargs = mock_dpo_config.call_args.kwargs
    assert config_kwargs["precompute_ref_log_probs"] is True


def test_validate_and_select_pairs_correctness_uses_cfx_match_only() -> None:
    """The correctness reward selects winners purely on CFX pattern match, ignoring non-CFX."""
    # A's higher non-CFX score must NOT subtract under correctness scoring; cfx scores decide.
    df_a = pd.DataFrame(
        [
            {
                "user_id": 0,
                "conversation": json.dumps([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "a0"}]),
                "explanation_cfx_pattern_match_mean": 0.4,
                "explanation_non_cfx_pattern_match_mean": 1.0,
            },
        ]
    )
    df_b = pd.DataFrame(
        [
            {
                "user_id": 0,
                "conversation": json.dumps([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "b0"}]),
                "explanation_cfx_pattern_match_mean": 0.6,
                "explanation_non_cfx_pattern_match_mean": 0.0,
            },
        ]
    )

    pairs = run_dpo._validate_and_select_pairs(df_a, df_b, RewardType.CORRECTNESS)

    assert len(pairs) == 1
    assert pairs[0]["score_chosen"] == 0.6
    assert pairs[0]["score_rejected"] == 0.4
    assert pairs[0]["chosen"][-1]["content"] == "b0"
    assert pairs[0]["rejected"][-1]["content"] == "a0"
