"""Speculative decoding and MTP evaluation package."""

from opengrad.benchmarks.speculative.metrics import (
    LatencyMetrics,
    MTPDepthMetrics,
    SpeculationTelemetry,
    SpeedupAccounting,
    ThroughputMetrics,
    compute_mtp_depth_metrics,
    compute_speedup,
)
from opengrad.benchmarks.speculative.modes import SpeculativeMode
from opengrad.benchmarks.speculative.pareto import (
    ParetoPoint,
    analyze_pareto,
    render_pareto_table,
)
from opengrad.benchmarks.speculative.parity import (
    QualityParitySummary,
    ToolCallEquivalence,
    check_tool_call_equivalence,
    evaluate_quality_parity,
)

__all__ = [
    "LatencyMetrics",
    "MTPDepthMetrics",
    "ParetoPoint",
    "QualityParitySummary",
    "SpeculationTelemetry",
    "SpeculativeMode",
    "SpeedupAccounting",
    "ThroughputMetrics",
    "ToolCallEquivalence",
    "analyze_pareto",
    "check_tool_call_equivalence",
    "compute_mtp_depth_metrics",
    "compute_speedup",
    "evaluate_quality_parity",
    "render_pareto_table",
]
