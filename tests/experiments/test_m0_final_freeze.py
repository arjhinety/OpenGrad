"""Tests for the M0-final result freeze and the rendered-sample sanity check.

The freeze exists so a numeric claim in the reports can be checked against the artifact that
produced it. A freeze that is not automatically re-checked decays into a file of hashes that were
true once, so the verification runs here rather than only when someone remembers to run it.

The sanity check exists because the pre-launch check was specified but left no artifact, so the
claim that the corpus reached the trainer as intended rested on inspection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

FREEZE = ROOT / "reports/data/m0-final-freeze.json"
SANITY = ROOT / "reports/data/m0-final-rendered-sanity.json"


def _freeze_module():
    import freeze_m0_final

    return freeze_m0_final


def _sanity_module():
    import sanity_check_rendered_samples

    return sanity_check_rendered_samples


def test_freeze_manifest_exists_and_declares_its_role() -> None:
    manifest = json.loads(FREEZE.read_text())
    assert manifest["artifact_kind"] == "M0_FINAL_RESULT_FREEZE"
    assert manifest["experiment_id"] == "m0_sft_canonical_v2_final"
    assert manifest["experiment_status"] == "REJECTED"
    # A reader must be able to tell why the file exists from the file alone.
    assert "Pins the artifacts" in manifest["role"]


def test_freeze_covers_every_claim_in_the_reports() -> None:
    """A group missing here means some reported number has nothing behind it."""
    manifest = json.loads(FREEZE.read_text())
    assert set(manifest["evidence"]) >= {
        "corpus",
        "contamination",
        "evaluation",
        "config",
        "training",
        "selection",
        "sanity",
        "confirmatory",
        "decision",
        "reports",
    }
    # Both partitions must be pinned: selection is not interpretable without them.
    assert (
        "reports/evaluation/behavioral-heldout-v2-partition.json"
        in manifest["evidence"]["evaluation"]
    )
    # The promotion verdict is only authoritative if its records are pinned too.
    assert "runs/checkpoint_registry.json" in manifest["evidence"]["decision"]
    assert "runs/central_ledger.jsonl" in manifest["evidence"]["decision"]


def test_freeze_records_weights_by_hash_without_committing_them() -> None:
    """Weights are gigabytes and stay off git, but their identity must still be checkable."""
    manifest = json.loads(FREEZE.read_text())
    weights = manifest["unversioned_by_design"]["weights"]
    assert len(weights) == 4
    for meta in weights.values():
        assert len(meta["sha256"]) == 64
        assert meta["bytes"] > 1_000_000_000


@pytest.mark.skipif(not FREEZE.is_file(), reason="freeze manifest not built")
def test_freeze_verifies_against_disk(capsys) -> None:
    """Every recorded hash must still match the file it names.

    Deliberately invokes the real script: a reimplementation here could pass while the tool
    operators actually run fails.
    """
    module = _freeze_module()
    status = module.verify()
    out = capsys.readouterr().out
    assert status == 0, out
    assert "VERIFY PASSED" in out


def test_sanity_check_covered_every_source_and_found_no_failures() -> None:
    report = json.loads(SANITY.read_text())
    assert report["failures"] == []
    assert report["rendered_records"] == 161_966
    sources = {example["source"] for example in report["examples"]}
    assert sources == {
        "glaive-function-calling-v2",
        "toolace",
        "when2call",
        "xlam-function-calling-60k",
    }


def test_sanity_check_asserts_no_record_reaches_the_trainer_unsupervised() -> None:
    """A record with no supervised token consumes a step and contributes no gradient, so the step
    count would not mean what it was intended to mean."""
    report = json.loads(SANITY.read_text())
    finding = next(
        f for f in report["findings"] if f["check"] == "no_record_reaches_the_trainer_unsupervised"
    )
    assert finding["passed"] is True


def test_sanity_check_confirms_call_prediction_targets_have_no_fabricated_result() -> None:
    report = json.loads(SANITY.read_text())
    finding = next(
        f
        for f in report["findings"]
        if f["group"] == "xlam-function-calling-60k"
        and f["check"] == "call_prediction_records_supervise_a_call"
    )
    assert finding["passed"] is True
    # And the decoded targets must actually be calls rather than the check being vacuous.
    xlam = next(e for e in report["examples"] if e["source"] == "xlam-function-calling-60k")
    assert all("<tool_call>" in s["target_head"] for s in xlam["sample"])


def test_sanity_check_sampling_is_deterministic() -> None:
    """The inspected set must not depend on parquet row order, so a re-render cannot change it."""
    module = _sanity_module()
    records = [{"record_id": f"r{i}"} for i in range(200)]
    first = [r["record_id"] for r in module.deterministic_sample(records, 10)]
    shuffled = list(reversed(records))
    second = [r["record_id"] for r in module.deterministic_sample(shuffled, 10)]
    assert first == second
    assert len(first) == 10
