"""The closed PTQ phase must stay closed, consistent, and honestly labelled.

Three things are locked here:

* **Release roles are not pass claims.** Q6_K and Q8_0 both passed the primary gate and both
  FAILED the secondary release bar. A role like `RECOMMENDED_RELEASE` must never be readable as
  "met the secondary bar", so every accepted artifact carries its bar failure alongside its role.
* **Every artifact is hashed.** A closure record that references an unhashed artifact is not a
  reproduction record.
* **The ledger and the closure manifest agree.** They are written by different scripts; if they
  drift, one of them is lying about what shipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CLOSURE = ROOT / "manifests/quantization/ptq_phase_closure_v1.json"
LEDGER = ROOT / "results/quantization/findings.jsonl"

ACCEPTED_ROLES = {"RECOMMENDED_RELEASE", "MEMORY_OPTIMIZED_RELEASE"}
PASSING_GATE = {"PTQ_ACCEPTED", "PASS", "PASSED", "ACCEPTED"}


@pytest.fixture(scope="module")
def closure():
    if not CLOSURE.is_file():
        pytest.skip("PTQ phase not closed")
    return json.loads(CLOSURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ledger():
    if not LEDGER.is_file():
        pytest.skip("no ledger")
    return [json.loads(l) for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_phase_is_closed(closure):
    assert closure["phase"] == "ptq"
    assert closure["phase_status"] == "CLOSED"


def test_every_artifact_carries_a_hash(closure):
    unhashed = [a["artifact"] for a in closure["artifacts"] if not a.get("sha256")]
    assert unhashed == [], f"artifacts without sha256: {unhashed}"


def test_no_artifact_claims_the_secondary_release_bar(closure):
    """The bar was met by nothing. Any artifact claiming otherwise is a false claim."""
    assert closure["gates"]["secondary_release_bar"]["result"] == "NO RUNG MET IT"
    for a in closure["artifacts"]:
        bar = str(a.get("secondary_release_bar", ""))
        assert not bar.startswith("PASSED"), (
            f"{a['artifact']} claims the secondary release bar, which no PTQ artifact met"
        )


def test_accepted_roles_carry_their_bar_failures(closure):
    """A release role must never be readable as a pass on the stricter bar."""
    accepted = [a for a in closure["artifacts"] if a["role"] in ACCEPTED_ROLES]
    assert {a["quantization"] for a in accepted} == {"Q6_K", "Q8_0"}
    for a in accepted:
        assert a["gate_decision"] in PASSING_GATE
        assert str(a["secondary_release_bar"]).startswith("FAILED:"), (
            f"{a['quantization']} holds a release role without recording its secondary-bar failure"
        )


def test_roles_match_gate_outcomes(closure):
    """Nothing that failed the primary gate may hold a release role, and vice versa."""
    _items = list(closure["artifacts"])
    assert _items, "nothing to check: an empty collection would pass this test vacuously"
    for a in _items:
        passed = a["gate_decision"] in PASSING_GATE
        if a["role"] in ACCEPTED_ROLES:
            assert passed, f"{a['quantization']} has a release role but failed the primary gate"
        if a["role"] == "REJECTED_ACCURACY":
            assert not passed, f"{a['quantization']} labelled rejected but passed the gate"
            assert a.get("failed_gate_dimensions"), "a rejected rung must name what it failed"


def test_exactly_two_rungs_passed_and_they_are_q6k_and_q8_0(closure):
    passed = {
        a["quantization"] for a in closure["artifacts"]
        if a["quantization"] != "BF16" and a["gate_decision"] in PASSING_GATE
    }
    assert passed == {"Q6_K", "Q8_0"}


def test_tokenizer_parity_failure_is_preserved_not_restated_as_passing(closure):
    parity = closure["gates"]["engine_tokenizer_parity"]
    assert parity["status"] == "ENGINE_TOKENIZER_PARITY_FAILED"
    assert parity["mismatches"] == 6
    assert parity["exact_matches"] == 1271
    assert parity["materiality"] == "TOKENIZER_DIVERGENCE_BEHAVIORALLY_MATERIAL"


def test_q4_k_m_beats_bf16_on_f1_while_failing_the_gate(closure):
    """The concrete reason call_f1 alone is not an acceptance criterion. Locked so it stays visible."""
    by = {a["quantization"]: a for a in closure["artifacts"]}
    bf16, q4 = by["BF16"]["metrics"], by["Q4_K_M"]["metrics"]
    assert q4["call_f1"] > bf16["call_f1"], "the cautionary example has changed; re-check the report"
    assert q4["over_call_rate"] > bf16["over_call_rate"]
    assert q4["clarification_accuracy"] < bf16["clarification_accuracy"]
    assert by["Q4_K_M"]["gate_decision"] not in PASSING_GATE


def test_ledger_agrees_with_closure_on_roles_and_hashes(closure, ledger):
    rows = {r["quantization"]: r for r in ledger if r.get("branch") == "gguf"}
    _items = list(closure["artifacts"])
    assert _items, "nothing to check: an empty collection would pass this test vacuously"
    for a in _items:
        row = rows.get(a["quantization"])
        assert row is not None, f"{a['quantization']} missing from the ledger"
        assert row.get("release_role") == a["role"], (
            f"{a['quantization']}: ledger role {row.get('release_role')!r} != "
            f"closure role {a['role']!r}"
        )
        if a.get("sha256") and row.get("artifact_sha256"):
            assert row["artifact_sha256"] == a["sha256"], f"{a['quantization']} hash drift"


def test_frozen_reference_caveat_is_recorded(closure):
    """The aggregate-only limitation must travel with the phase, not live only in prose."""
    caveat = closure["evaluation"]["frozen_reference_caveat"]
    assert "aggregate" in caveat.lower()
    assert "per-example" in caveat.lower()
