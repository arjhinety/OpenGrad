"""Tests for the renderability/trainability gate.

The gate exists because canonical validity does not imply trainability.  These tests pin the
collapse detection that would have caught the Canonical-v1 defect (213,951 canonical records,
9 tool-call targets) and the xLAM case (canonically valid, 0 trainable).
"""

from __future__ import annotations

import pytest

from opengrad.data.yield_gate import (
    DEFAULT_EXPECTATIONS,
    YIELD_ANOMALY,
    YIELD_COLLAPSE,
    YIELD_NOT_EXPECTED,
    YIELD_OK,
    SourceYield,
    aggregate_yields,
    evaluate_yield_gate,
    gate_is_blocking,
    load_expectations,
    render_yield_table,
)


def test_aggregate_counts_canonical_and_trainable_separately() -> None:
    rows = [
        ("a", "OK", True, "CALL_PREDICTION"),
        ("a", "OK", False, "COMPLETE_TRAJECTORY"),
        ("a", "UNRENDERABLE", False, "COMPLETE_TRAJECTORY"),
        ("a", "TARGET_TRUNCATED", False, "COMPLETE_TRAJECTORY"),
    ]
    yields = aggregate_yields(rows)
    entry = yields["a"]
    assert entry.canonical_records == 4
    assert entry.trainable_records == 2
    assert entry.targets_with_tool_calls == 1
    assert entry.targets_without_tool_calls == 1
    assert entry.failure_reasons == {"UNRENDERABLE": 1, "TARGET_TRUNCATED": 1}


def test_context_tail_truncated_is_trainable() -> None:
    yields = aggregate_yields([("a", "CONTEXT_TAIL_TRUNCATED", True, "COMPLETE_TRAJECTORY")])
    assert yields["a"].trainable_records == 1


def test_ratios_are_computed_against_canonical_records() -> None:
    yields = aggregate_yields(
        [("a", "OK", True, "CALL_PREDICTION")] * 3
        + [("a", "UNRENDERABLE", False, "CALL_PREDICTION")] * 1
    )
    entry = yields["a"]
    assert entry.canonical_records == 4
    assert entry.yield_ratio == pytest.approx(0.75)
    assert entry.quarantine_ratio == pytest.approx(0.25)
    assert entry.tool_call_target_ratio == pytest.approx(1.0)


def test_zero_canonical_records_does_not_divide_by_zero() -> None:
    entry = SourceYield(source="empty")
    assert entry.yield_ratio == 0.0
    assert entry.tool_call_target_ratio == 0.0
    assert entry.quarantine_ratio == 0.0


# --- collapse detection -------------------------------------------------------------


def test_xlam_shape_is_a_collapse() -> None:
    """Canonically valid, zero trainable: the case that must never pass silently."""
    yields = {"xlam": SourceYield(source="xlam", canonical_records=59370, trainable_records=0)}
    findings = evaluate_yield_gate(
        yields,
        {"defaults": DEFAULT_EXPECTATIONS, "sources": {"xlam": {"expects_tool_calls": True}}},
    )
    assert findings[0]["status"] == YIELD_COLLAPSE
    assert gate_is_blocking(findings) is True


def test_canonical_v1_shape_is_a_collapse() -> None:
    """213,951 canonical records with almost no tool-call targets is a collapse on the
    dimension the source exists to teach."""
    yields = {
        "canonical_v1": SourceYield(
            source="canonical_v1",
            canonical_records=55719,
            trainable_records=55719,
            targets_with_tool_calls=9,
            targets_without_tool_calls=55710,
        )
    }
    findings = evaluate_yield_gate(
        yields,
        {
            "defaults": DEFAULT_EXPECTATIONS,
            "sources": {
                "canonical_v1": {"expects_tool_calls": True, "min_tool_call_target_ratio": 0.2}
            },
        },
    )
    assert findings[0]["status"] == YIELD_ANOMALY
    assert "tool-call target ratio" in findings[0]["reasons"][0]


def test_a_source_teaching_tool_calls_with_zero_calls_collapses() -> None:
    yields = {
        "s": SourceYield(
            source="s", canonical_records=100, trainable_records=100, targets_with_tool_calls=0
        )
    }
    findings = evaluate_yield_gate(
        yields, {"defaults": DEFAULT_EXPECTATIONS, "sources": {"s": {"expects_tool_calls": True}}}
    )
    assert findings[0]["status"] == YIELD_COLLAPSE


def test_low_yield_below_floor_is_an_anomaly() -> None:
    yields = {"s": SourceYield(source="s", canonical_records=100, trainable_records=10)}
    findings = evaluate_yield_gate(yields, {"defaults": DEFAULT_EXPECTATIONS, "sources": {}})
    assert findings[0]["status"] == YIELD_ANOMALY
    assert findings[0]["yield_ratio"] == pytest.approx(0.1)


def test_healthy_source_is_ok() -> None:
    yields = {
        "s": SourceYield(
            source="s",
            canonical_records=100,
            trainable_records=90,
            targets_with_tool_calls=80,
            targets_without_tool_calls=10,
        )
    }
    findings = evaluate_yield_gate(
        yields, {"defaults": DEFAULT_EXPECTATIONS, "sources": {"s": {"expects_tool_calls": True}}}
    )
    assert findings[0]["status"] == YIELD_OK
    assert findings[0]["reasons"] == []


def test_absent_source_is_not_expected() -> None:
    yields = {"s": SourceYield(source="s", canonical_records=0)}
    findings = evaluate_yield_gate(yields, {"defaults": DEFAULT_EXPECTATIONS, "sources": {}})
    assert findings[0]["status"] == YIELD_NOT_EXPECTED
    assert gate_is_blocking(findings) is False


def test_non_tool_source_is_not_required_to_emit_calls() -> None:
    """A source not expected to teach tool calling may legitimately have no call targets."""
    yields = {
        "s": SourceYield(
            source="s",
            canonical_records=100,
            trainable_records=90,
            targets_with_tool_calls=0,
            targets_without_tool_calls=90,
        )
    }
    findings = evaluate_yield_gate(yields, {"defaults": DEFAULT_EXPECTATIONS, "sources": {}})
    assert findings[0]["status"] == YIELD_OK


# --- expectations -------------------------------------------------------------------


def test_shipped_expectations_load_and_mark_tool_sources(tmp_path) -> None:
    from pathlib import Path

    shipped = Path("configs/data/yield_expectations.yaml")
    expectations = load_expectations(shipped)
    assert expectations["sources"]["xlam-function-calling-60k"]["expects_tool_calls"] is True
    assert expectations["sources"]["glaive-function-calling-v2"]["expects_tool_calls"] is True
    # When2Call supervises the *decision* in prose, not a structured invocation: measured, 0 of
    # 6,505 canonical records carry a tool call or a result. Requiring structured calls of it
    # would fail a healthy source instead of detecting a collapse, so `min_yield_ratio` is what
    # guards it.
    assert expectations["sources"]["when2call"]["expects_tool_calls"] is False
    assert expectations["sources"]["when2call"]["min_yield_ratio"] > 0


def test_partial_expectation_inherits_defaults(tmp_path) -> None:
    path = tmp_path / "e.yaml"
    path.write_text("schema_version: 1\nsources:\n  s:\n    expects_tool_calls: true\n")
    expectations = load_expectations(path)
    assert (
        expectations["sources"]["s"]["min_yield_ratio"] == DEFAULT_EXPECTATIONS["min_yield_ratio"]
    )


def test_render_table_includes_reasons() -> None:
    yields = {"x": SourceYield(source="x", canonical_records=10, trainable_records=0)}
    findings = evaluate_yield_gate(yields, {"defaults": DEFAULT_EXPECTATIONS, "sources": {}})
    table = render_yield_table(findings)
    assert "COLLAPSE" in table
    assert "0 of 10 canonical records are trainable" in table


def test_supervision_kinds_are_counted_per_source() -> None:
    """A source's trainable records are attributable to a supervision kind, not just a total."""
    rows = [
        ("x", "OK", True, "CALL_PREDICTION"),
        ("x", "OK", True, "CALL_PREDICTION"),
        ("x", "OK", True, "COMPLETE_TRAJECTORY"),
        ("x", "UNRENDERABLE", False, "COMPLETE_TRAJECTORY"),
    ]
    entry = aggregate_yields(rows)["x"]
    assert entry.supervision_kinds == {"CALL_PREDICTION": 2, "COMPLETE_TRAJECTORY": 1}
    assert entry.supervision_kinds_total == {"CALL_PREDICTION": 2, "COMPLETE_TRAJECTORY": 2}
    assert entry.as_dict()["supervision_kinds_trainable"] == {
        "CALL_PREDICTION": 2,
        "COMPLETE_TRAJECTORY": 1,
    }


def test_unclassified_records_are_not_folded_into_a_real_kind() -> None:
    entry = aggregate_yields([("x", "OK", True, "")])["x"]
    assert entry.supervision_kinds == {"UNCLASSIFIED": 1}


def test_yield_table_shows_the_kind_breakdown() -> None:
    yields = {"x": SourceYield(source="x", canonical_records=1, trainable_records=1)}
    yields["x"].supervision_kinds = {"CALL_PREDICTION": 1}
    findings = evaluate_yield_gate(yields, {"defaults": DEFAULT_EXPECTATIONS, "sources": {}})
    table = render_yield_table(findings)
    assert "trainable by supervision kind: CALL_PREDICTION=1" in table
