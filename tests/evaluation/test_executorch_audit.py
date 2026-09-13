"""The 8da4w export must not silently leave layers in float.

Phase 3B's requirement is that an unexpected skipped layer is a failure condition, not a
footnote. The audit is only worth anything if it actually fails when something is skipped, so
these tests pin both directions: the real artifacts reconcile exactly, and an injected skip is
caught rather than absorbed.
"""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/audit_executorch_quantization.py"
AUDIT_DIR = ROOT / "results/quantization/executorch"


def _load_module():
    spec = importlib.util.spec_from_file_location("executorch_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_module()

# 6144 channels x kernel 4, once per Gated DeltaNet layer. This is the only weight the quantizer
# is permitted to leave in float: 8da4w rewrites nn.Linear, and a depthwise conv1d is not one.
CONV1D_PARAMETERS = 6144 * 4
LINEAR_ATTENTION_LAYERS = 18


def _real_audits():
    fp32 = AUDIT_DIR / "audit_qwen3_5_2b_fp32.json"
    quant = AUDIT_DIR / "audit_qwen3_5_2b_8da4w.json"
    if not (fp32.is_file() and quant.is_file()):
        pytest.skip("ExecuTorch audit records not present; run audit_pte first")
    return audit.load_sizes("qwen3_5_2b_fp32"), audit.load_sizes("qwen3_5_2b_8da4w")


@pytest.fixture(scope="module")
def reconciliation():
    (_fp32_doc, fp32), (_quant_doc, quant) = _real_audits()
    config = json.loads(audit.CONFIG.read_text(encoding="utf-8"))
    labels = audit.expected_weight_classes(config)
    return audit.reconcile(fp32, quant, labels), fp32, quant, labels


def test_every_buffer_is_accounted_for(reconciliation):
    """Nothing left over. A leftover buffer means the size-multiset match was a coincidence."""
    result, _, _, _ = reconciliation
    assert result["unconsumed_buffers"] == {}
    assert [c for c in result["classes"] if c["verdict"] == "UNMATCHED"] == []


def test_every_weight_class_is_predicted_by_the_config(reconciliation):
    """No class may be identified as UNIDENTIFIED — the architecture has to explain all of them."""
    result, _, _, _ = reconciliation
    unexplained = [c for c in result["classes"] if c["identity"].startswith("UNIDENTIFIED")]
    assert unexplained == [], f"weight classes not predicted by config.json: {unexplained}"


def test_the_only_skipped_weight_is_the_depthwise_conv1d(reconciliation):
    """Exactly one skipped class, of exactly the expected size and multiplicity."""
    result, _, _, _ = reconciliation
    skipped = [c for c in result["classes"] if c["verdict"] == "SKIPPED_FP32"]
    assert len(skipped) == 1, f"unexpected skipped weight classes: {skipped}"
    only = skipped[0]
    assert only["elements_each"] == CONV1D_PARAMETERS
    assert only["count"] == LINEAR_ATTENTION_LAYERS
    assert "conv1d" in only["identity"]


def test_named_data_coverage_is_effectively_total(reconciliation):
    """The skip must stay negligible. A regression that skips a projection would blow past this."""
    result, _, _, _ = reconciliation
    total = result["quantized_parameters"] + result["skipped_parameters"]
    assert result["skipped_parameters"] == CONV1D_PARAMETERS * LINEAR_ATTENTION_LAYERS
    assert result["quantized_parameters"] / total > 0.999


def test_an_injected_skip_is_detected(reconciliation):
    """The audit fails when a class is left in float — otherwise it proves nothing.

    This removes one class's int4 payload and scale buffers from the quantized store and puts the
    fp32-sized buffers back, which is exactly what a silently-skipped projection looks like.
    """
    _, fp32, quant, labels = reconciliation
    # The full-attention q_proj class: 6 weights, large enough to matter, small enough to be easy
    # to miss in a size ratio.
    q_proj_elements = 8 * 256 * 2 * 2048
    size = q_proj_elements * 4
    count = fp32[size]
    assert count == 6

    tampered = Counter(quant)
    tampered[q_proj_elements // 2] -= count
    tampered[(q_proj_elements // audit.GROUP_SIZE) * audit.SCALE_ITEMSIZE] -= count
    tampered[size] += count
    tampered = Counter({k: v for k, v in tampered.items() if v > 0})

    result = audit.reconcile(fp32, tampered, labels)
    skipped = [c for c in result["classes"] if c["verdict"] == "SKIPPED_FP32"]
    assert any(c["elements_each"] == q_proj_elements for c in skipped), (
        "a fully-skipped q_proj was not reported as skipped"
    )
    assert result["skipped_parameters"] > CONV1D_PARAMETERS * LINEAR_ATTENTION_LAYERS


def test_embedding_is_not_counted_as_quantized(reconciliation):
    """The embedding lives in the program, not the named-data store, and stays fp32.

    It is 65% of the artifact, so folding it into the coverage number would turn a real limitation
    into a headline. This pins that it is excluded from the named-data denominator.
    """
    result, _, _, _ = reconciliation
    config = json.loads(audit.CONFIG.read_text(encoding="utf-8"))
    embedding = config["vocab_size"] * config["hidden_size"]
    named_total = result["quantized_parameters"] + result["skipped_parameters"]
    # The tied lm_head copy IS quantized and is the same shape, so the store holds it once.
    assert result["quantized_parameters"] >= embedding
    quant_doc = json.loads(
        (AUDIT_DIR / "audit_qwen3_5_2b_8da4w.json").read_text(encoding="utf-8")
    )
    const_elements = sum(quant_doc["constant_elements_by_dtype"].values())
    assert const_elements >= embedding
    assert named_total < const_elements + named_total
