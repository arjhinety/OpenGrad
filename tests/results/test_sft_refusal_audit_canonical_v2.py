"""Internal arithmetic of the Canonical-v2 SFT refusal-label audit.

The audit itself needs the 176 release shards, which are not in the repository. These tests need
only the committed JSON: they pin that the per-source counts add up to the totals, that every rate
is its own count over its own denominator, and that the file names the corpus M0 trained on.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CAP = ROOT / "results/benchmarks/h200/capability_v1"
AUDIT = CAP / "sft_refusal_supervision_audit_canonical_v2.json"
SUPERSEDED = CAP / "sft_refusal_supervision_audit.json"
MANIFEST = ROOT / ".release/hf/toolpolicy-canonical-v2-final/release-manifest.json"
MANIFEST_SHA256 = "8ced403b996e563d6e279aee7fdb346fc829fe5ff6af9daf8ef47c0a4007e161"


@pytest.fixture(scope="module")
def audit() -> dict:
    return json.loads(AUDIT.read_text(encoding="utf-8"))


def test_audit_names_the_canonical_v2_release(audit):
    ident = audit["corpus_identity"]
    assert ident["release_manifest_sha256"] == MANIFEST_SHA256
    assert ident["release_name"] == "OpenGrad ToolPolicy Canonical (v2 final)"
    assert ident["output_shards_verified"] == 176
    assert sum(ident["per_source_shards"].values()) == ident["output_shards_verified"]
    assert audit["totals"]["records_audited"] == ident["manifest_record_count"] == 173237


def test_committed_manifest_is_the_audited_one(audit):
    if not MANIFEST.exists():
        pytest.skip("release manifest not present")
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == MANIFEST_SHA256
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert set(audit["per_source"]) == {s["source"] for s in manifest["sources"]}
    assert manifest["record_count"] == audit["totals"]["records_audited"]


def test_per_source_counts_sum_to_totals(audit):
    per_source = audit["per_source"].values()
    totals = audit["totals"]
    assert all(v["status"] == "AUDITED" for v in per_source)
    assert sum(v["records"] for v in per_source) == totals["records_audited"]
    assert sum(v["refusal_targets"] for v in per_source) == totals["refusal_targets"]

    by_label: dict[str, int] = {}
    for v in per_source:
        assert sum(v["refusal_targets_by_decision_label"].values()) == v["refusal_targets"]
        assert v["refusals_with_tools_offered"] + v["refusals_with_no_tools_offered"] == v["refusal_targets"]
        assert sum(v["decision_distribution"].values()) <= v["records"]
        for label, n in v["refusal_targets_by_decision_label"].items():
            by_label[label] = by_label.get(label, 0) + n
    assert by_label == totals["refusal_targets_by_decision_label"]


def test_rates_are_count_over_their_own_denominator(audit):
    totals = audit["totals"]
    assert totals["refusal_target_rate"] == pytest.approx(
        totals["refusal_targets"] / totals["records_audited"], abs=1e-15)
    for source, v in audit["per_source"].items():
        assert v["refusal_rate"] == pytest.approx(v["refusal_targets"] / v["records"], abs=1e-15), source


def test_headline_numbers(audit):
    """Literal values, so a units or scope change cannot pass by staying self-consistent."""
    totals = audit["totals"]
    assert (totals["refusal_targets"], totals["records_audited"]) == (18114, 173237)
    assert round(totals["refusal_target_rate"] * 100, 1) == 10.5
    assert totals["refusal_targets_by_decision_label"] == {"ANSWER": 18114}
    counts = {s: (v["refusal_targets"], v["records"]) for s, v in audit["per_source"].items()}
    assert counts == {
        "when2call": (4038, 6505),
        "glaive-function-calling-v2": (14066, 98339),
        "toolace": (10, 11051),
        "xlam-function-calling-60k": (0, 57342),
    }
    assert round(4038 / 6505 * 100, 1) == 62.1


def test_superseded_audit_points_here():
    old = json.loads(SUPERSEDED.read_text(encoding="utf-8"))
    pointer = old["superseded_by"]
    assert (ROOT / pointer["path"]).resolve() == AUDIT.resolve()
    assert "normalization-v1" in pointer["reason"]
