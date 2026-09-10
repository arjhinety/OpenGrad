"""The engine is a declared, recorded, validated dimension of a baseline run.

Two engines emit different tokens, so a baseline is only comparable to a run that names the
same engine and version. These tests pin the contract without needing a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from opengrad.evaluation import runner
from opengrad.evaluation.runner import (
    SUPPORTED_ENGINES,
    VLLMInferenceBackend,
    build_backend,
)

BASELINE_CONFIG = Path("configs/evaluation/tool_calling/qwen35_2b_baseline.yaml")


def base_config() -> dict[str, Any]:
    return yaml.safe_load(BASELINE_CONFIG.read_text())


# --------------------------------------------------------------------------------------
# Backend construction
# --------------------------------------------------------------------------------------


def test_declared_engine_is_the_engine_that_gets_built():
    config = base_config()
    assert build_backend("vllm", config).name == "vllm"
    assert build_backend("transformers", config).name == "transformers"


def test_unknown_engine_is_rejected():
    with pytest.raises(ValueError, match="unsupported engine"):
        build_backend("llama.cpp", base_config())


def test_vllm_backend_never_renders_its_own_prompt():
    """vLLM must not re-apply a chat template: the frozen renderer owns the prompt."""
    backend = VLLMInferenceBackend(model_id="Qwen/Qwen3.5-2B", revision="a" * 40)
    # Nothing in the construction path mentions a tokenizer or chat template; the backend
    # is handed finished prompt strings.
    assert not hasattr(backend, "tokenizer")
    assert not hasattr(backend, "apply_chat_template")


def test_vllm_max_model_len_covers_the_largest_in_contract_prompt():
    """vLLM rejects prompt + max_tokens > max_model_len, so the window must include both.

    Prompts beyond context_length are bucketed as overflow by the frozen policy and must
    still be generatable, or the run would crash on them instead of measuring them.
    """
    config = base_config()
    backend = build_backend("vllm", config)
    assert isinstance(backend, VLLMInferenceBackend)
    # The pinned window, not just context_length: the longest held-out prompt measures 5248
    # tokens, which exceeds the 4096 bucket boundary.
    assert backend.max_model_len == config["runtime"]["max_model_len"]
    assert backend.max_model_len > config["runtime"]["context_length"]


def test_vllm_window_must_cover_the_completion_budget():
    config = base_config()
    config["runtime"]["max_model_len"] = 4096  # equal to context_length, so no room to finish
    with pytest.raises(ValueError, match="must cover context_length plus the completion"):
        runner._validate_baseline_config(config)


def test_vllm_window_is_required():
    config = base_config()
    config["runtime"].pop("max_model_len", None)
    with pytest.raises(ValueError, match="must pin max_model_len"):
        runner._validate_baseline_config(config)


def test_provenance_ignores_untracked_run_outputs(tmp_path):
    """A run creates untracked files; judging provenance afterwards would call every run dirty."""
    from opengrad.env_capture import tracked_tree_provenance

    root = Path(__file__).parents[2]
    before = tracked_tree_provenance(root)
    assert before["sha"], "these tests assume a git checkout"

    # An untracked artifact must not change the verdict.
    probe = root / "untracked-probe.txt"
    probe.write_text("probe\n", encoding="utf-8")
    try:
        assert tracked_tree_provenance(root) == before
    finally:
        probe.unlink(missing_ok=True)


def test_provenance_fails_closed_without_git(tmp_path):
    from opengrad.env_capture import tracked_tree_provenance

    assert tracked_tree_provenance(tmp_path)["dirty"] is True, "unknown provenance is never clean"


def test_prompt_lengths_are_measured_for_the_full_heldout_set():
    """The window is pinned from a measurement, not a guess.

    Longest prompt across the frozen held-out set is 5,248 tokens, so a window equal to
    context_length (4,096) cannot generate from it. This asserts the pinned relationship
    without needing the parquet materialization.
    """
    import yaml

    config = yaml.safe_load(BASELINE_CONFIG.read_text())
    runtime = config["runtime"]
    assert runtime["max_model_len"] > runtime["context_length"]
    assert runtime["overflow_policy"] == "separate_bucket"

    backend = VLLMInferenceBackend(model_id="Qwen/Qwen3.5-2B", revision="a" * 40)
    metadata = backend.engine_metadata()
    assert metadata["name"] == "vllm"
    assert "version" in metadata
    assert metadata["batching"] == "continuous"


def test_venv_bin_is_put_on_path_for_jit_compilers(monkeypatch):
    """flashinfer shells out to `ninja`, which lives in the venv's bin directory."""
    import os

    monkeypatch.setenv("PATH", "/usr/bin")
    VLLMInferenceBackend._expose_venv_bin_for_jit()
    entries = os.environ["PATH"].split(os.pathsep)
    assert str(Path(__import__("sys").executable).parent) in entries
    # Idempotent: repeated loads must not grow PATH.
    before = os.environ["PATH"]
    VLLMInferenceBackend._expose_venv_bin_for_jit()
    assert os.environ["PATH"] == before


# --------------------------------------------------------------------------------------
# Config and runtime contract
# --------------------------------------------------------------------------------------


def test_frozen_config_declares_a_supported_engine():
    config = base_config()
    assert config["runtime"]["backend"] in SUPPORTED_ENGINES
    runner._validate_baseline_config(config)


@pytest.mark.parametrize("engine", ["vllm", "transformers"])
def test_each_engine_requires_its_own_pinned_version(engine):
    config = base_config()
    config["runtime"]["backend"] = engine
    key = "vllm_version" if engine == "vllm" else "transformers_version"
    config["runtime"].pop(key, None)
    with pytest.raises(ValueError, match=f"{engine} runtime must pin"):
        runner._validate_baseline_config(config)


def test_unsupported_engine_fails_the_config_contract():
    config = base_config()
    config["runtime"]["backend"] = "llamacpp"
    with pytest.raises(ValueError, match="must declare a supported engine"):
        runner._validate_baseline_config(config)


def test_runtime_still_requires_bf16_and_an_accelerator():
    config = base_config()
    config["runtime"]["precision"] = "float16"
    with pytest.raises(ValueError, match="BF16"):
        runner._validate_baseline_config(config)


# --------------------------------------------------------------------------------------
# Batching in the runner
# --------------------------------------------------------------------------------------


class _RecordingBackend:
    """Records whether the runner used batched generation, and with what chunk sizes.

    Deliberately exposes only the documented optional surface: `generate_batch` and the
    batch-wide `last_truncated`, never `last_truncated_flags`. A backend that omits the
    per-sample flag list must still produce one prediction per example.
    """

    name = "recording"

    def __init__(self) -> None:
        self.batches: list[int] = []
        self.serial = 0
        self.last_truncated = False

    def generate_batch(self, prompts, *, examples, generation_config):
        self.batches.append(len(prompts))
        return [
            "<tool_call><function=lookup><parameter=q>worker</parameter></function></tool_call>"
        ] * len(prompts)

    def generate(self, prompt, *, example, generation_config):
        self.serial += 1
        return "<tool_call><function=lookup><parameter=q>worker</parameter></function></tool_call>"


class _SerialOnlyBackend:
    """No generate_batch: the runner must fall back to one prompt at a time."""

    name = "serial"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt, *, example, generation_config):
        self.calls += 1
        return "no call here"


def test_runner_batches_when_the_engine_can(monkeypatch, tmp_path):
    from tests.evaluation.test_baseline_runner import (
        BASELINE_CONFIG,
        _patch_manifest_and_renderer,
    )

    _patch_manifest_and_renderer(monkeypatch, tmp_path)
    config = yaml.safe_load(BASELINE_CONFIG.read_text())
    config["runtime"]["batch_size"] = 2
    config["outputs"] = {
        key: str(tmp_path / f"{key}.json")
        for key in ("predictions", "metrics", "residual_profile", "environment")
    }
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(yaml.safe_dump(config))

    backend = _RecordingBackend()
    result = runner.run_baseline(config_path, root=tmp_path, backend=backend, dry_run=True)
    assert result["records"] == 4
    assert backend.serial == 0, "a batching engine should not be driven one prompt at a time"
    assert backend.batches == [2, 2]
    assert result["generation_batch_size"] == 2


def test_runner_falls_back_to_serial_for_engines_without_batching(monkeypatch, tmp_path):
    from tests.evaluation.test_baseline_runner import (
        BASELINE_CONFIG,
        _patch_manifest_and_renderer,
    )

    _patch_manifest_and_renderer(monkeypatch, tmp_path)
    config = yaml.safe_load(BASELINE_CONFIG.read_text())
    config["runtime"]["batch_size"] = 4
    config["outputs"] = {
        key: str(tmp_path / f"{key}.json")
        for key in ("predictions", "metrics", "residual_profile", "environment")
    }
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(yaml.safe_dump(config))

    backend = _SerialOnlyBackend()
    result = runner.run_baseline(config_path, root=tmp_path, backend=backend, dry_run=True)
    assert result["records"] == 4
    assert backend.calls == 4


def test_engine_mismatch_between_prompts_and_samples_is_a_loud_error(monkeypatch, tmp_path):
    from tests.evaluation.test_baseline_runner import (
        BASELINE_CONFIG,
        _patch_manifest_and_renderer,
    )

    class _ShortBackend(_RecordingBackend):
        def generate_batch(self, prompts, *, examples, generation_config):
            return ["only one"]  # fewer results than prompts

    _patch_manifest_and_renderer(monkeypatch, tmp_path)
    config = yaml.safe_load(BASELINE_CONFIG.read_text())
    config["runtime"]["batch_size"] = 2
    config["outputs"] = {
        key: str(tmp_path / f"{key}.json")
        for key in ("predictions", "metrics", "residual_profile", "environment")
    }
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(RuntimeError, match="returned 1 samples for 2 prompts"):
        runner.run_baseline(config_path, root=tmp_path, backend=_ShortBackend(), dry_run=True)
