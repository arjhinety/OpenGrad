from __future__ import annotations

import json
from pathlib import Path

import pytest

from opengrad.promotion.artifacts import (
    ArtifactLineageError,
    RequantizationError,
    assert_m1_v2_lineage,
    assert_quantizable_source,
    assert_releasable,
    can_release,
    is_low_bit_name,
)
from opengrad.promotion.quantization import compute_preservation_thresholds

ROOT = Path(__file__).resolve().parents[2]

REFERENCE = ROOT / "results/quantization/m1_v2_reference.json"
PARTITION = ROOT / "reports/evaluation/behavioral-heldout-v2-partition.json"
QAD_CORPUS = ROOT / "manifests/quantization/m1_v2_qad_recovery_v1.jsonl"
IMATRIX_CORPUS = ROOT / "manifests/quantization/m1_v2_imatrix_calibration_v2.jsonl"
LEDGER = ROOT / "results/quantization/findings.jsonl"

VALID_STATUSES = {
    "BF16_REFERENCE",
    "PTQ_ACCEPTED",
    "QAD_REQUIRED",
    "QAD_ACCEPTED",
    "REJECTED_ACCURACY",
    "REJECTED_RUNTIME",
    "REJECTED_EXPORT",
    "REJECTED_PARITY",
    "EXPORTED_PENDING_EVALUATION",
    "BLOCKED_SDK_ACCESS",
    # The SDK gate and the port gate are different blockers and collapsing them would lose the
    # distinction that matters: MediaTek's NeuroPilot SDK was obtained, so BLOCKED_SDK_ACCESS is no
    # longer true, but the Gated DeltaNet layer is still unimplemented so no export exists either.
    "BLOCKED_PORT_INCOMPLETE",
    # A rung that was quantized but whose behavioural score is absent. Distinct from
    # REJECTED_ACCURACY, which means it was scored and failed — recording an unscored rung as
    # rejected would fabricate evidence, and omitting it would leave a silent hole in the ladder.
    "NOT_SCORED",
}


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def held_out_ids() -> set[str]:
    partition = json.loads(PARTITION.read_text(encoding="utf-8"))
    return set(partition["example_ids"]["dev"]) | set(partition["example_ids"]["confirmatory"])


# --- requantization -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["m1-v2-Q4_K_M.gguf", "model-q8_0.gguf", "x-IQ3_XXS.gguf", "qwen3_5_2b_8da4w.pte",
     "qwen_qnn_16a4w.pte", "student-int4.pt"],
)
def test_low_bit_names_are_detected(name):
    assert is_low_bit_name(name)


@pytest.mark.parametrize("name", ["m1-v2-bf16.gguf", "m1-v2-f16.gguf", "qwen3_5_2b_fp32.pte"])
def test_high_precision_names_are_not_flagged(name):
    assert not is_low_bit_name(name)


def test_quantizing_from_a_quantized_source_is_refused():
    with pytest.raises(RequantizationError) as excinfo:
        assert_quantizable_source("m1-v2-Q8_0.gguf")
    assert "already-quantized" in str(excinfo.value)


def test_quantizing_from_bf16_is_allowed():
    assert assert_quantizable_source("m1-v2-bf16.gguf", source_type="bf16") is None


def test_unknown_source_type_is_refused():
    with pytest.raises(RequantizationError):
        assert_quantizable_source("m1-v2-mystery.gguf", source_type="q4_k_m")


# --- lineage ------------------------------------------------------------------------------


def test_m1_v2_lineage_is_accepted():
    assert_m1_v2_lineage(
        {
            "parent_experiment_id": "m1_dpo_canonical_v2_final_v2",
            "parent_checkpoint": "m1_dpo_canonical_v2_final_v2::dpo-checkpoint-30",
        }
    )


def test_foreign_parent_is_rejected():
    with pytest.raises(ArtifactLineageError):
        assert_m1_v2_lineage(
            {
                "parent_experiment_id": "m0_sft_canonical_v2_final",
                "parent_checkpoint": "m0_sft_canonical_v2_final::checkpoint-1800",
            }
        )


def test_wrong_checkpoint_is_rejected():
    with pytest.raises(ArtifactLineageError):
        assert_m1_v2_lineage(
            {
                "parent_experiment_id": "m1_dpo_canonical_v2_final_v2",
                "parent_checkpoint": "m1_dpo_canonical_v2_final_v2::dpo-checkpoint-10",
            }
        )


def test_the_frozen_reference_artifact_declares_m1_v2_lineage():
    """The reference IS M1-v2, so it satisfies the check through experiment_id, not parentage.

    Its parent_experiment_id correctly names M0; accepting that as M1-v2 lineage would let an M0
    descendant masquerade as one, which is exactly what the two-shape check prevents.
    """
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert reference["experiment_id"] == "m1_dpo_canonical_v2_final_v2"
    assert reference["parent_experiment_id"] == "m0_sft_canonical_v2_final"
    assert_m1_v2_lineage(reference)


def test_an_m0_descendant_is_not_accepted_as_m1_v2_lineage():
    with pytest.raises(ArtifactLineageError):
        assert_m1_v2_lineage(
            {
                "experiment_id": "m0_sft_canonical_v2_final",
                "selected_checkpoint": "checkpoint-1800",
                "parent_experiment_id": "qwen35_2b_base",
            }
        )


# --- release gating -----------------------------------------------------------------------


def test_an_artifact_cannot_be_released_without_a_passing_verdict():
    assert not can_release("PTQ_ACCEPTED", {"failed_dimensions": ["call_recall"]})
    assert not can_release("PTQ_ACCEPTED", None)
    assert not can_release("REJECTED_ACCURACY", {"failed_dimensions": []})
    assert not can_release("EXPORTED_PENDING_EVALUATION", {"failed_dimensions": []})
    assert can_release("PTQ_ACCEPTED", {"failed_dimensions": [], "decision": "PTQ_ACCEPTED"})
    assert can_release("QAD_ACCEPTED", {"failed_dimensions": [], "decision": "PTQ_ACCEPTED"})


def test_assert_releasable_names_the_failing_dimensions():
    with pytest.raises(ValueError) as excinfo:
        assert_releasable("PTQ_ACCEPTED", {"failed_dimensions": ["unsupported_accuracy"]})
    assert "unsupported_accuracy" in str(excinfo.value)


# --- gate determinism ---------------------------------------------------------------------


def test_threshold_computation_is_deterministic_and_matches_the_frozen_file():
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))["metrics"]
    first = compute_preservation_thresholds(reference)
    second = compute_preservation_thresholds(dict(reference))
    assert first == second

    frozen = json.loads(
        (ROOT / "results/quantization/quantization_preservation_v1.json").read_text(
            encoding="utf-8"
        )
    )["thresholds"]
    assert first == frozen


def test_the_gate_floor_is_exactly_ninety_nine_percent_of_the_reference():
    """One literal, hand-checkable value so a silent change to the retention rate is caught.

    0.7548387096774194 * 0.99 = 0.7472903225806452, which is the committed call_f1 threshold.
    """
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))["metrics"]
    assert reference["call_f1"] == 0.7548387096774194
    assert compute_preservation_thresholds(reference)["call_f1"] == 0.7472903225806452


# --- contamination ------------------------------------------------------------------------


def test_qad_recovery_corpus_has_zero_overlap_with_the_frozen_heldout():
    excluded = held_out_ids()
    rows = read_jsonl(QAD_CORPUS)
    assert rows, "QAD recovery corpus is empty"
    overlap = {row["canonical_id"] for row in rows} & excluded
    assert overlap == set(), f"QAD corpus leaks held-out ids: {sorted(overlap)[:10]}"


def test_imatrix_calibration_corpus_has_zero_overlap_with_the_frozen_heldout():
    excluded = held_out_ids()
    rows = read_jsonl(IMATRIX_CORPUS)
    assert rows, "imatrix calibration corpus is empty"
    overlap = {row["calibration_id"] for row in rows} & excluded
    assert overlap == set(), f"calibration corpus leaks held-out ids: {sorted(overlap)[:10]}"


def test_imatrix_calibration_covers_every_behaviour():
    """Calibrating only on CALL and ANSWER would starve the two weakest reference metrics."""
    rows = read_jsonl(IMATRIX_CORPUS)
    behaviours = {row["behavior"] for row in rows}
    assert behaviours == {"ANSWER", "CALL", "CLARIFY", "UNSUPPORTED"}
    for behaviour in behaviours:
        assert sum(1 for row in rows if row["behavior"] == behaviour) >= 50


# --- ledger -------------------------------------------------------------------------------


def test_every_ledger_row_carries_a_recognised_status():
    if not LEDGER.is_file():
        pytest.skip("no candidates recorded yet")
    rows = read_jsonl(LEDGER)
    for row in rows:
        assert row["status"] in VALID_STATUSES, f"unknown status {row['status']!r}"
        assert row["branch"] and row["artifact"]


def test_no_scored_ledger_row_drops_examples():
    """A shrinking denominator is the failure mode the runtime accounting exists to prevent."""
    if not LEDGER.is_file():
        pytest.skip("no candidates recorded yet")
    for row in read_jsonl(LEDGER):
        if row.get("records") is None:
            continue
        assert row["records"] == row["submitted"], (
            f"{row['branch']}/{row['artifact']} scored {row['records']} of "
            f"{row['submitted']} submitted examples"
        )
