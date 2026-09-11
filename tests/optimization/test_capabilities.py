"""Capability-discovery tests: per-model answers, no cross-size inference, UNKNOWN default."""

from __future__ import annotations

from opengrad.optimization.capabilities import (
    TECHNIQUE_KEYS,
    CapabilityMatrix,
    CapabilityStatus,
    probe_capabilities,
)

REQUIRED_TECHNIQUES = {
    "fp8_ptq",
    "nvfp4",
    "ptq_other",
    "qat",
    "qad",
    "distillation",
    "pruning_sparsity",
    "speculative_decoding",
    "hf_export",
    "vllm_deployment",
}


def test_every_covered_technique_is_probed():
    assert REQUIRED_TECHNIQUES <= set(TECHNIQUE_KEYS)


def test_unrecorded_model_reports_unknown_everywhere():
    matrix = probe_capabilities(
        "Qwen/Qwen3.5-2B",
        model_revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
        backend="modelopt",
        backend_revision=None,
    )
    assert matrix.status_of("fp8_ptq") == CapabilityStatus.UNKNOWN.value
    assert all(probe.status == CapabilityStatus.UNKNOWN.value for probe in matrix.probes)


def test_evidence_names_the_backend_revision_and_refuses_in_memory_claims():
    matrix = probe_capabilities("Qwen/Qwen3.5-2B", backend="modelopt", backend_revision="0.42.0")
    probe = next(p for p in matrix.probes if p.technique == "fp8_ptq")
    assert "0.42.0" in probe.evidence
    assert "executing the backend" in probe.evidence
    assert probe.status == CapabilityStatus.UNKNOWN.value


def test_support_for_one_model_is_not_inferred_for_another_size():
    """The critical rule: a fact about Qwen3.5-7B must never answer for Qwen3.5-2B."""
    knowledge = {
        ("Qwen/Qwen3.5-7B", "fp8_ptq"): ("SUPPORTED", "measured on this exact model"),
    }
    seven = probe_capabilities("Qwen/Qwen3.5-7B", knowledge=knowledge)
    two = probe_capabilities("Qwen/Qwen3.5-2B", knowledge=knowledge)
    assert seven.status_of("fp8_ptq") == CapabilityStatus.SUPPORTED.value
    assert two.status_of("fp8_ptq") == CapabilityStatus.UNKNOWN.value
    # And the reverse direction is equally strict.
    reverse = {("Qwen/Qwen3.5-2B", "nvfp4"): ("UNSUPPORTED", "measured incompatible")}
    assert (
        probe_capabilities("Qwen/Qwen3.5-2B", knowledge=reverse).status_of("nvfp4")
        == CapabilityStatus.UNSUPPORTED.value
    )
    assert (
        probe_capabilities("Qwen/Qwen3.5-7B", knowledge=reverse).status_of("nvfp4")
        == CapabilityStatus.UNKNOWN.value
    )


def test_matrix_round_trips_through_dict():
    matrix = probe_capabilities("Qwen/Qwen3.5-2B", backend="mock", backend_revision="mock-1")
    restored = CapabilityMatrix.from_dict(matrix.to_dict())
    assert restored == matrix
    assert restored.status_of("fp8_ptq") == CapabilityStatus.UNKNOWN.value
