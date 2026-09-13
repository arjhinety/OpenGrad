"""Release selection must apply the frozen criteria, not invent convenient ones.

The hazard this guards is subtle: after seeing a ladder it is tempting to pick the rung that looks
good and justify it afterwards. The criteria file was frozen before any rung was quantized, and
these tests pin that the selector reads it, honours the fallback rather than relaxing a threshold,
and keeps "smallest that passes the gate" distinct from "what should ship".
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/build_ptq_evaluation_report.py"
CRITERIA_PATH = ROOT / "results/quantization/gguf/release_selection_criteria_v1.json"


def _load():
    spec = importlib.util.spec_from_file_location("build_ptq_evaluation_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = _load()
CRITERIA = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
BASELINE = {
    "call_f1": 0.75,
    "call_precision": 0.80,
    "call_recall": 0.70,
    "clarification_accuracy": 0.60,
    "unsupported_accuracy": 0.90,
    "over_call_rate": 0.15,
    "parse_valid_rate": 1.0,
}


def rung(name, *, size, decision="PTQ_ACCEPTED", agreement=1.0, retention=1.0,
         over_call=0.15, parse_valid=1.0, broke=0, records=1277):
    return {
        "rung": name,
        "present": True,
        "records": records,
        "artifact_bytes": size,
        "metrics": {**BASELINE, "over_call_rate": over_call, "parse_valid_rate": parse_valid},
        "gate": {"decision": decision, "checks": []},
        "retention_vs_bf16": {m: retention for m in report.RETENTION},
        "decision_agreement": agreement,
        "flip_effects": {"fixed": 0, "broke": broke, "lateral": 0},
    }


def test_criteria_file_is_the_source_of_thresholds():
    """Nothing may be hardcoded in the selector that the frozen file also defines."""
    sec = CRITERIA["recommended_release_rung"]["secondary_criteria"]
    assert sec["decision_agreement_vs_llamacpp_bf16_min"] == 0.99
    assert sec["retention_vs_llamacpp_bf16_min"] == 0.995
    # A rung that fails only because of the file's threshold must pass when the file is relaxed,
    # proving the selector actually reads it rather than using a constant of its own.
    candidate = rung("X", size=1, agreement=0.98)
    ok, _ = report.meets_release_bar(candidate, BASELINE, CRITERIA)
    assert ok is False
    relaxed = json.loads(json.dumps(CRITERIA))
    relaxed["recommended_release_rung"]["secondary_criteria"][
        "decision_agreement_vs_llamacpp_bf16_min"
    ] = 0.5
    ok_relaxed, _ = report.meets_release_bar(candidate, BASELINE, relaxed)
    assert ok_relaxed is True


def test_freeze_evidence_shows_no_rung_was_scored_when_criteria_were_written():
    assert CRITERIA["frozen_at_state"]["score_files_present_at_freeze"] == []
    assert CRITERIA["frozen_at_state"]["bf16_gguf_scored"] is False


def test_smallest_passing_and_recommended_can_differ():
    """The distinction the study is required to report."""
    rungs = [
        rung("TINY", size=1_000, agreement=0.95),          # passes gate, thin margin
        rung("MID", size=2_000, agreement=0.999),          # passes gate and release bar
        rung("BIG", size=3_000, agreement=1.0),
    ]
    selection = report.select(rungs, BASELINE, CRITERIA)
    assert selection["highest_compression_passing_rung"] == "TINY"
    assert selection["recommended_release_rung"] == "MID"
    assert selection["same_rung"] is False
    assert selection["fallback_used"] is False


def test_identical_when_smallest_also_has_margin():
    rungs = [rung("A", size=1_000, agreement=0.999), rung("B", size=2_000)]
    selection = report.select(rungs, BASELINE, CRITERIA)
    assert selection["highest_compression_passing_rung"] == "A"
    assert selection["recommended_release_rung"] == "A"
    assert selection["same_rung"] is True


def test_fallback_is_flagged_not_hidden():
    """If nothing clears the release bar, that must be visible, not silently downgraded."""
    rungs = [rung("A", size=1_000, agreement=0.90), rung("B", size=2_000, agreement=0.91)]
    selection = report.select(rungs, BASELINE, CRITERIA)
    assert selection["release_bar_rungs"] == []
    assert selection["fallback_used"] is True
    assert selection["recommended_release_rung"] == "A"


def test_failing_gate_excludes_a_rung_entirely():
    rungs = [
        rung("SMALL", size=1_000, decision="REJECTED_ACCURACY", agreement=1.0),
        rung("LARGE", size=5_000, agreement=1.0),
    ]
    selection = report.select(rungs, BASELINE, CRITERIA)
    assert "SMALL" not in selection["gate_passing_rungs"]
    assert selection["highest_compression_passing_rung"] == "LARGE"


def test_broke_flips_block_release_even_when_aggregates_hold():
    """Correct answers swapped for incorrect ones at the same rate keeps metrics flat."""
    bad = rung("A", size=1_000, agreement=1.0, broke=200, records=1277)
    # Force the flip fraction to bite while every aggregate stays at baseline.
    bad["decision_agreement"] = 0.999
    ok, failures = report.meets_release_bar(bad, BASELINE, CRITERIA)
    assert ok is False
    assert any("broke-flip" in f for f in failures)


def test_over_call_headroom_is_half_the_gate_tolerance():
    sec = CRITERIA["recommended_release_rung"]["secondary_criteria"]
    assert sec["over_call_rate_absolute_headroom_vs_llamacpp_bf16_max"] == 0.005
    just_over = rung("A", size=1, over_call=BASELINE["over_call_rate"] + 0.006)
    ok, failures = report.meets_release_bar(just_over, BASELINE, CRITERIA)
    assert ok is False
    assert any("over_call_rate" in f for f in failures)
    just_under = rung("B", size=1, over_call=BASELINE["over_call_rate"] + 0.004)
    ok2, _ = report.meets_release_bar(just_under, BASELINE, CRITERIA)
    assert ok2 is True


def test_unscored_rungs_are_never_treated_as_passing():
    rungs = [{"rung": "GHOST", "present": False, "artifact_bytes": 1, "status": "NOT_SCORED"}]
    selection = report.select(rungs, BASELINE, CRITERIA)
    assert selection["gate_passing_rungs"] == []
    assert selection["recommended_release_rung"] is None
