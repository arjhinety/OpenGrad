"""Renderability / trainability gate for dataset builds.

Why this exists
---------------
Canonical validity and trainability are **different properties**.  A record can satisfy the
canonical schema, pass contamination screening, and be released, while still producing no
usable supervised training target.  The Canonical-v1 incident is the worked example: the
released corpus contained 213,951 canonical records, the training boundary accepted 55,719,
and only **9** of those carried a tool call in their supervised target.  Every gate that
looked at canonical counts passed.

A second, sharper example is xLAM: after its parameter maps are normalized it is 100%
canonically valid, yet 0 records are trainable, because its trajectory is a single-turn call
prediction with no tool result and the complete-target policy requires one.  A gate that only
counted canonical records would call that source healthy.

So this module reports, per source, what the *training boundary* actually yields, and fails
closed when a source that is supposed to teach function calling yields almost no tool-call
targets -- rather than silently contributing nothing to the mixture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

# Gate outcomes.
YIELD_OK = "OK"
YIELD_ANOMALY = "ANOMALY"
YIELD_COLLAPSE = "COLLAPSE"
YIELD_NOT_EXPECTED = "NOT_EXPECTED"


@dataclass
class SourceYield:
    """Per-source canonical and training-boundary yield."""

    source: str
    canonical_records: int = 0
    rendered_records: int = 0
    trainable_records: int = 0
    targets_with_tool_calls: int = 0
    targets_without_tool_calls: int = 0
    failure_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def yield_ratio(self) -> float:
        return self.trainable_records / self.canonical_records if self.canonical_records else 0.0

    @property
    def render_ratio(self) -> float:
        return self.rendered_records / self.canonical_records if self.canonical_records else 0.0

    @property
    def tool_call_target_ratio(self) -> float:
        if not self.trainable_records:
            return 0.0
        return self.targets_with_tool_calls / self.trainable_records

    @property
    def quarantine_ratio(self) -> float:
        if not self.canonical_records:
            return 0.0
        return 1.0 - self.yield_ratio

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "canonical_records": self.canonical_records,
            "rendered_records": self.rendered_records,
            "trainable_records": self.trainable_records,
            "targets_with_tool_calls": self.targets_with_tool_calls,
            "targets_without_tool_calls": self.targets_without_tool_calls,
            "yield_ratio": round(self.yield_ratio, 6),
            "render_ratio": round(self.render_ratio, 6),
            "tool_call_target_ratio": round(self.tool_call_target_ratio, 6),
            "quarantine_ratio": round(self.quarantine_ratio, 6),
            "failure_reasons": dict(sorted(self.failure_reasons.items())),
        }


def aggregate_yields(
    rows: Iterable[tuple[str, str, bool]],
) -> dict[str, SourceYield]:
    """Aggregate ``(source, boundary_status, has_tool_call_target)`` tuples.

    ``boundary_status`` is the training-boundary status string; ``OK`` and
    ``CONTEXT_TAIL_TRUNCATED`` are the trainable states.
    """
    from opengrad.training.sft_data import TRAINABLE

    out: dict[str, SourceYield] = {}
    for source, status, has_calls in rows:
        entry = out.setdefault(source, SourceYield(source=source))
        entry.canonical_records += 1
        if status in TRAINABLE:
            entry.trainable_records += 1
            if has_calls:
                entry.targets_with_tool_calls += 1
            else:
                entry.targets_without_tool_calls += 1
            entry.rendered_records += 1
        else:
            entry.failure_reasons[status] = entry.failure_reasons.get(status, 0) + 1
    return dict(sorted(out.items()))


# --- expectations -------------------------------------------------------------------

DEFAULT_EXPECTATIONS: dict[str, Any] = {
    "min_yield_ratio": 0.5,
    "min_tool_call_target_ratio": 0.0,
    "expects_tool_calls": False,
}


def load_expectations(path: Any) -> dict[str, Any]:
    """Load the per-source expectation config, falling back to conservative defaults."""
    import yaml

    from pathlib import Path

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise TypeError("yield expectations must be a mapping")
    sources = config.get("sources") or {}
    merged: dict[str, Any] = {}
    for name, entry in sources.items():
        if not isinstance(entry, dict):
            raise TypeError(f"yield expectation for {name} must be a mapping")
        merged[str(name)] = {**DEFAULT_EXPECTATIONS, **entry}
    return {"defaults": {**DEFAULT_EXPECTATIONS, **(config.get("defaults") or {})}, "sources": merged}


def evaluate_yield_gate(
    yields: dict[str, SourceYield], expectations: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return one finding per source. Statuses: OK, ANOMALY, COLLAPSE, NOT_EXPECTED.

    A source that declares it should teach function calling and yields no tool-call targets
    is a COLLAPSE -- the exact xLAM ``59,370 canonical -> 0 trainable`` case.
    """
    defaults = expectations.get("defaults", DEFAULT_EXPECTATIONS)
    per_source = expectations.get("sources", {})
    findings: list[dict[str, Any]] = []
    for name, measured in sorted(yields.items()):
        rules = {**defaults, **per_source.get(name, {})}
        reasons: list[str] = []
        status = YIELD_OK
        if measured.canonical_records == 0:
            status = YIELD_NOT_EXPECTED
            reasons.append("no canonical records")
        else:
            if measured.trainable_records == 0:
                status = YIELD_COLLAPSE
                reasons.append(
                    f"0 of {measured.canonical_records} canonical records are trainable"
                )
            elif measured.yield_ratio < float(rules["min_yield_ratio"]):
                status = YIELD_ANOMALY
                reasons.append(
                    f"trainable yield {measured.yield_ratio:.3f} below floor "
                    f"{float(rules['min_yield_ratio']):.3f}"
                )
            if rules.get("expects_tool_calls") and measured.trainable_records:
                if measured.targets_with_tool_calls == 0:
                    # Not a floor comparison: a source whose purpose is to teach tool calling
                    # and which emits no call target at all has collapsed outright.
                    status = YIELD_COLLAPSE
                    reasons.append(
                        "a source expected to teach tool calling yields zero tool-call targets"
                    )
                elif measured.tool_call_target_ratio < float(rules["min_tool_call_target_ratio"]):
                    if status == YIELD_OK:
                        status = YIELD_ANOMALY
                    reasons.append(
                        f"tool-call target ratio {measured.tool_call_target_ratio:.3f} below "
                        f"floor {float(rules['min_tool_call_target_ratio']):.3f} for a source "
                        "expected to teach tool calling"
                    )
        findings.append({"source": name, "status": status, "reasons": reasons, **measured.as_dict()})
    return findings


def gate_is_blocking(findings: list[dict[str, Any]]) -> bool:
    """COLLAPSE blocks; ANOMALY is reported without blocking an unrelated build."""
    return any(finding["status"] == YIELD_COLLAPSE for finding in findings)


def render_yield_table(findings: list[dict[str, Any]]) -> str:
    """A compact human-readable table for reports and CLI output."""
    header = (
        f"{'source':<34}{'canon':>8}{'train':>8}{'calls':>8}"
        f"{'yield':>8}{'call%':>8}  status"
    )
    lines = [header, "-" * len(header)]
    for finding in findings:
        lines.append(
            f"{finding['source']:<34}{finding['canonical_records']:>8}"
            f"{finding['trainable_records']:>8}{finding['targets_with_tool_calls']:>8}"
            f"{finding['yield_ratio']:>8.3f}{finding['tool_call_target_ratio']:>8.3f}"
            f"  {finding['status']}"
        )
        for reason in finding["reasons"]:
            lines.append(f"{'':<34}  - {reason}")
    return "\n".join(lines)


def measure_materialized_corpus(
    input_dir: Any,
    *,
    model: str = "Qwen/Qwen3.5-2B",
    max_seq_length: int = 2048,
    limit: int | None = None,
) -> dict[str, SourceYield]:
    """Measure per-source trainability over a materialized canonical corpus directory."""
    from pathlib import Path

    from opengrad.data.canonical import ToolConversation
    from opengrad.data.materialize import iter_materialized_rows
    from opengrad.data.real_analysis import _restore_row
    from opengrad.data.renderers import renderer_for
    from opengrad.training.sft_data import build_sample

    renderer = renderer_for(model)
    rows: list[tuple[str, str, bool]] = []
    for index, raw in enumerate(iter_materialized_rows(Path(input_dir))):
        if limit is not None and index >= limit:
            break
        row = _restore_row(raw)
        source = str(row.get("source", "unknown"))
        try:
            conversation = ToolConversation(
                row["id"], row["source"], row["tools"], row["messages"], row["metadata"]
            )
        except Exception as exc:  # noqa: BLE001 - malformed rows are quarantined, not repaired
            rows.append((source, f"CONSTRUCTION_FAILED:{type(exc).__name__}", False))
            continue
        has_calls = any(message.get("tool_calls") for message in conversation.messages)
        sample = build_sample(
            renderer, conversation, max_seq_length=max_seq_length, source_dataset=source
        )
        rows.append((source, sample.status, has_calls))
    return aggregate_yields(rows)
