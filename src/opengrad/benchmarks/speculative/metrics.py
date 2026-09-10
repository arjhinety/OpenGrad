"""Speculative decoding and native MTP telemetry and speedup accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LatencyMetrics:
    ttft_ms: float
    mean_itl_ms: float
    p50_itl_ms: float
    p95_itl_ms: float
    total_generation_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "ttft_ms": round(self.ttft_ms, 2),
            "mean_itl_ms": round(self.mean_itl_ms, 2),
            "p50_itl_ms": round(self.p50_itl_ms, 2),
            "p95_itl_ms": round(self.p95_itl_ms, 2),
            "total_generation_latency_ms": round(self.total_generation_latency_ms, 2),
        }


@dataclass
class ThroughputMetrics:
    output_tokens_per_sec: float
    total_tokens_per_sec: float
    requests_per_sec: float
    completed_tasks_per_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_tokens_per_sec": round(self.output_tokens_per_sec, 2),
            "total_tokens_per_sec": round(self.total_tokens_per_sec, 2),
            "requests_per_sec": round(self.requests_per_sec, 2),
            "completed_tasks_per_sec": round(self.completed_tasks_per_sec, 2),
        }


@dataclass
class SpeculationTelemetry:
    speculation_depth_requested: int
    speculation_depth_achieved: int
    proposed_draft_tokens: int
    accepted_draft_tokens: int
    rejected_draft_tokens: int
    acceptance_rate: float
    accepted_tokens_per_step: float
    verifier_steps: int
    rollback_count: int
    recomputed_tokens: int
    effective_depth: float
    verification_overhead_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "speculation_depth_requested": self.speculation_depth_requested,
            "speculation_depth_achieved": self.speculation_depth_achieved,
            "proposed_draft_tokens": self.proposed_draft_tokens,
            "accepted_draft_tokens": self.accepted_draft_tokens,
            "rejected_draft_tokens": self.rejected_draft_tokens,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "accepted_tokens_per_step": round(self.accepted_tokens_per_step, 2),
            "verifier_steps": self.verifier_steps,
            "rollback_count": self.rollback_count,
            "recomputed_tokens": self.recomputed_tokens,
            "effective_depth": round(self.effective_depth, 2),
            "verification_overhead_ms": round(self.verification_overhead_ms, 2),
        }


@dataclass
class MTPDepthMetrics:
    per_depth_acceptance: dict[int, float] = field(default_factory=dict)
    conditional_acceptance: dict[int, float] = field(default_factory=dict)
    head_confidence: dict[int, float] = field(default_factory=dict)
    useful_speculation_depth: int = 1
    wasted_speculative_computation_pct: float = 0.0
    average_accepted_prefix_length: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_depth_acceptance": {
                f"depth_{k}": round(v, 4) for k, v in sorted(self.per_depth_acceptance.items())
            },
            "conditional_acceptance": {
                f"depth_{k}": round(v, 4) for k, v in sorted(self.conditional_acceptance.items())
            },
            "head_confidence": {
                f"head_{k}": round(v, 4) for k, v in sorted(self.head_confidence.items())
            },
            "useful_speculation_depth": self.useful_speculation_depth,
            "wasted_speculative_computation_pct": round(self.wasted_speculative_computation_pct, 2),
            "average_accepted_prefix_length": round(self.average_accepted_prefix_length, 2),
        }


@dataclass
class SpeedupAccounting:
    throughput_speedup: float  # speculative_tokens_per_sec / AR_tokens_per_sec
    latency_speedup: float  # AR_wall_time / speculative_wall_time
    ttft_delta_ms: float
    p95_itl_delta_ms: float
    vram_delta_mb: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "throughput_speedup": round(self.throughput_speedup, 3),
            "latency_speedup": round(self.latency_speedup, 3),
            "ttft_delta_ms": round(self.ttft_delta_ms, 2),
            "p95_itl_delta_ms": round(self.p95_itl_delta_ms, 2),
            "vram_delta_mb": round(self.vram_delta_mb, 2),
        }


def compute_speedup(
    ar_tokens_per_sec: float,
    spec_tokens_per_sec: float,
    ar_wall_time: float,
    spec_wall_time: float,
    ar_ttft_ms: float = 0.0,
    spec_ttft_ms: float = 0.0,
    ar_p95_itl_ms: float = 0.0,
    spec_p95_itl_ms: float = 0.0,
    ar_vram_mb: float = 0.0,
    spec_vram_mb: float = 0.0,
) -> SpeedupAccounting:
    """Calculate speedup strictly according to Section 14.

    Never infers speedup from acceptance rate alone.
    """
    throughput_speedup = round(spec_tokens_per_sec / max(1e-6, ar_tokens_per_sec), 3)
    latency_speedup = round(ar_wall_time / max(1e-6, spec_wall_time), 3)
    ttft_delta = round(spec_ttft_ms - ar_ttft_ms, 2)
    p95_delta = round(spec_p95_itl_ms - ar_p95_itl_ms, 2)
    vram_delta = round(spec_vram_mb - ar_vram_mb, 2)

    return SpeedupAccounting(
        throughput_speedup=throughput_speedup,
        latency_speedup=latency_speedup,
        ttft_delta_ms=ttft_delta,
        p95_itl_delta_ms=p95_delta,
        vram_delta_mb=vram_delta,
    )


def compute_mtp_depth_metrics(
    depth_acceptance_counts: dict[int, tuple[int, int]],  # depth -> (accepted, total_proposals)
) -> MTPDepthMetrics:
    """Compute per-depth acceptance, conditional acceptance, and useful depth."""
    per_depth: dict[int, float] = {}
    conditional: dict[int, float] = {}

    prev_acc = 1.0
    for depth in sorted(depth_acceptance_counts.keys()):
        acc, tot = depth_acceptance_counts[depth]
        rate = acc / max(1, tot)
        per_depth[depth] = rate
        cond = rate / max(1e-6, prev_acc) if prev_acc > 0 else 0.0
        conditional[depth] = min(1.0, cond)
        prev_acc = rate

    useful_depth = 1
    for depth, rate in sorted(per_depth.items()):
        if rate >= 0.30:  # If acceptance drops below 30%, depth is rarely cost-effective
            useful_depth = depth

    # Calculate wasted computation
    total_proposed = sum(tot for _, tot in depth_acceptance_counts.values())
    total_accepted = sum(acc for acc, _ in depth_acceptance_counts.values())
    wasted = (
        round((total_proposed - total_accepted) / max(1, total_proposed) * 100.0, 2)
        if total_proposed
        else 0.0
    )

    avg_prefix = round(total_accepted / max(1, len(depth_acceptance_counts)), 2)

    return MTPDepthMetrics(
        per_depth_acceptance=per_depth,
        conditional_acceptance=conditional,
        useful_speculation_depth=useful_depth,
        wasted_speculative_computation_pct=wasted,
        average_accepted_prefix_length=avg_prefix,
    )
