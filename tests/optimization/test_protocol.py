"""Protocol-level tests: recipe hashing, provenance completeness, and the overwrite guard.

Everything here is CPU-only and needs neither ModelOpt nor a GPU. The mock backend is
used to exercise the full producer boundary deterministically.
"""

from __future__ import annotations

import json

import pytest

from opengrad.optimization import MockOptimizationBackend, select_optimization_backend
from opengrad.optimization.protocol import (
    ExecutionStatus,
    OptimizationRecipe,
    OptimizationResult,
    SourceCheckpoint,
    ensure_distinct_output,
    validate_recipe_shape,
)


def _recipe(**overrides) -> OptimizationRecipe:
    base = {
        "technique": "ptq",
        "format": "fp8",
        "calibration_dataset_id": "behavioral-heldout-v2",
        "calibration_sample_count": 512,
        "calibration_seed": 42,
        "export_format": "hf",
        "parameters": {"calibration_dataset_fingerprint": "abc123"},
    }
    base.update(overrides)
    return OptimizationRecipe(**base)


def _source(path, **overrides) -> SourceCheckpoint:
    base = {
        "checkpoint_id": "exp::checkpoint-1200",
        "path": str(path),
        "model_id": "Qwen/Qwen3.5-2B",
        "model_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "hash": "deadbeef" * 8,
        "experiment_id": "qwen35_2b_m0_sft_v2corpus",
        "lineage": ("tool_calling/qwen35_2b/baseline", "qwen35_2b_m0_sft_v2corpus"),
    }
    base.update(overrides)
    return SourceCheckpoint(**base)


# --------------------------------------------------------------------------------------
# Recipe hashing
# --------------------------------------------------------------------------------------


def test_recipe_hash_is_deterministic_and_key_order_independent():
    a = _recipe(parameters={"b": 2, "a": 1, "nested": {"y": 1, "x": 0}})
    b = _recipe(parameters={"nested": {"x": 0, "y": 1}, "a": 1, "b": 2})
    assert a.recipe_hash == b.recipe_hash, "dict insertion order must not change the recipe pin"
    assert len(a.recipe_hash) == 64


def test_recipe_hash_changes_when_intent_changes():
    base = _recipe()
    for changed in (
        _recipe(format="nvfp4"),
        _recipe(calibration_seed=43),
        _recipe(calibration_sample_count=513),
        _recipe(technique="qat"),
        _recipe(parameters={"calibration_dataset_fingerprint": "zzz"}),
    ):
        assert changed.recipe_hash != base.recipe_hash


def test_recipe_round_trips_through_dict_with_stable_hash():
    recipe = _recipe()
    restored = OptimizationRecipe.from_dict(recipe.to_dict())
    assert restored == recipe
    assert restored.recipe_hash == recipe.recipe_hash
    json.dumps(recipe.to_dict())  # serializable


# --------------------------------------------------------------------------------------
# Recipe validation
# --------------------------------------------------------------------------------------


def test_unknown_technique_is_refused():
    with pytest.raises(ValueError, match="unknown optimization technique"):
        validate_recipe_shape(_recipe(technique="teleportation"))


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"format": None}, "requires a target format"),
        ({"calibration_dataset_id": None}, "calibration dataset identity"),
        ({"calibration_sample_count": None}, "positive calibration sample count"),
        ({"calibration_sample_count": 0}, "positive calibration sample count"),
    ],
)
def test_ptq_requires_format_and_calibration(overrides, match):
    with pytest.raises(ValueError, match=match):
        validate_recipe_shape(_recipe(**overrides))


def test_distillation_needs_no_calibration_block():
    validate_recipe_shape(OptimizationRecipe(technique="distillation"))


# --------------------------------------------------------------------------------------
# Provenance completeness
# --------------------------------------------------------------------------------------

REQUIRED_PROVENANCE_KEYS = {
    "optimization_id",
    "source_checkpoint_id",
    "source_checkpoint_hash",
    "source_experiment_id",
    "source_experiment_lineage",
    "backend",
    "backend_version",
    "technique",
    "recipe_hash",
    "recipe",
    "calibration_dataset_id",
    "calibration_dataset_fingerprint",
    "calibration_sample_count",
    "calibration_seed",
    "quantization_format",
    "hardware",
    "cuda_version",
    "torch_version",
    "transformers_version",
    "export_format",
    "output_artifact_path",
    "output_artifact_hash",
    "runtime_used",
    "benchmark_ids",
    "capability_deltas",
    "efficiency_deltas",
    "failures",
    "unsupported_features",
    "execution_status",
    "evidence",
    "created_timestamp",
}


def test_mock_result_carries_full_provenance_and_round_trips(tmp_path):
    source = _source(tmp_path / "checkpoints" / "checkpoint-1200")
    (tmp_path / "checkpoints" / "checkpoint-1200").mkdir(parents=True)
    recipe = _recipe()
    result = MockOptimizationBackend().optimize(
        source, recipe, tmp_path / "out", runtime="vllm", benchmark_ids=["tool_use_core"]
    )
    payload = result.to_dict()
    assert REQUIRED_PROVENANCE_KEYS <= set(payload), REQUIRED_PROVENANCE_KEYS - set(payload)

    # Provenance values are populated, not null placeholders.
    assert payload["source_checkpoint_id"] == source.checkpoint_id
    assert payload["source_checkpoint_hash"] == source.hash
    assert payload["source_experiment_lineage"] == list(source.lineage)
    assert payload["backend"] == "mock"
    assert payload["technique"] == "ptq"
    assert payload["recipe_hash"] == recipe.recipe_hash
    assert payload["calibration_dataset_fingerprint"] == "abc123"
    assert payload["quantization_format"] == "fp8"
    assert payload["export_format"] == "hf"
    assert payload["runtime_used"] == "vllm"
    assert payload["benchmark_ids"] == ["tool_use_core"]
    assert payload["output_artifact_hash"]
    assert payload["execution_status"] == ExecutionStatus.MOCK.value
    assert payload["evidence"] is False

    restored = OptimizationResult.from_dict(payload)
    assert restored == result, "result must round-trip without loss"


def test_mock_artifact_hash_is_deterministic(tmp_path):
    source = _source(tmp_path / "src")
    (tmp_path / "src").mkdir()
    recipe = _recipe()
    first = MockOptimizationBackend().optimize(source, recipe, tmp_path / "out-a")
    second = MockOptimizationBackend().optimize(source, recipe, tmp_path / "out-b")
    assert first.output_artifact_hash == second.output_artifact_hash
    assert first.optimization_id == second.optimization_id


# --------------------------------------------------------------------------------------
# Source-overwrite guard
# --------------------------------------------------------------------------------------


def test_guard_rejects_identical_and_nested_paths(tmp_path):
    src = tmp_path / "ckpt"
    src.mkdir()
    with pytest.raises(ValueError, match="must not overwrite its source"):
        ensure_distinct_output(src, src)
    with pytest.raises(ValueError, match="inside the source checkpoint"):
        ensure_distinct_output(src, src / "child")
    with pytest.raises(ValueError, match="inside the optimization output"):
        ensure_distinct_output(src / "child", src)


def test_mock_optimize_refuses_to_write_into_the_source(tmp_path):
    src = tmp_path / "ckpt"
    src.mkdir()
    (src / "model.safetensors").write_bytes(b"weights")
    source = _source(src)
    with pytest.raises(ValueError):
        MockOptimizationBackend().optimize(source, _recipe(), src)
    # The source checkpoint is untouched: no new files, original weights intact.
    assert (src / "model.safetensors").read_bytes() == b"weights"
    assert sorted(p.name for p in src.iterdir()) == ["model.safetensors"]


def test_mock_optimize_never_modifies_the_source(tmp_path):
    src = tmp_path / "ckpt"
    src.mkdir()
    marker = src / "checkpoint_metadata.json"
    marker.write_text('{"status": "CANDIDATE"}\n', encoding="utf-8")
    before = marker.read_bytes()
    MockOptimizationBackend().optimize(_source(src), _recipe(), tmp_path / "out")
    assert marker.read_bytes() == before


# --------------------------------------------------------------------------------------
# Selector
# --------------------------------------------------------------------------------------


def test_selector_maps_names_and_rejects_unknown():
    assert select_optimization_backend("mock").name == "mock"
    assert select_optimization_backend("deterministic").name == "mock"
    assert select_optimization_backend("modelopt").name == "modelopt"
    assert select_optimization_backend("not-a-backend") is None
