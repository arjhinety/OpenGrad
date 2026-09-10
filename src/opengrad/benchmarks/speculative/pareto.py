"""Pareto analysis for speculative decoding speedup vs capability preservation tradeoffs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParetoPoint:
    config_name: str
    speedup: float  # e.g. 1.8x (higher is better)
    quality_delta: float  # e.g. -0.1 (closer to 0 or positive is better)
    benchmark_score: float  # e.g. 72.3
    peak_vram_mb: float = 0.0
    acceptable: bool = True
    dominated: bool = False
    dominated_by: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_name": self.config_name,
            "speedup": round(self.speedup, 3),
            "quality_delta": round(self.quality_delta, 2),
            "benchmark_score": round(self.benchmark_score, 2),
            "peak_vram_mb": round(self.peak_vram_mb, 1),
            "acceptable": self.acceptable,
            "dominated": self.dominated,
            "dominated_by": self.dominated_by,
            "notes": self.notes,
        }


def analyze_pareto(
    points: list[ParetoPoint], max_acceptable_quality_drop: float = 1.0
) -> list[ParetoPoint]:
    """Identify Pareto-optimal and Pareto-dominated configurations.
    
    A configuration X dominates Y if:
    - X has >= speedup AND X has >= quality_delta (less regression)
    - AND at least one metric is strictly superior.
    Configurations dropping more than max_acceptable_quality_drop are flagged unacceptable.
    """
    results: list[ParetoPoint] = []

    for pt in points:
        # Check acceptability threshold
        acceptable = pt.quality_delta >= -max_acceptable_quality_drop
        notes = []
        if not acceptable:
            notes.append(
                f"UNACCEPTABLE: Quality drop {pt.quality_delta:.2f} exceeds threshold -{max_acceptable_quality_drop:.2f}"
            )

        dominated_by: list[str] = []
        for other in points:
            if other.config_name == pt.config_name:
                continue
            # other dominates pt if other has >= speedup and >= quality_delta, with at least one strictly greater
            if (other.speedup >= pt.speedup and other.quality_delta >= pt.quality_delta) and (
                other.speedup > pt.speedup or other.quality_delta > pt.quality_delta
            ):
                dominated_by.append(other.config_name)

        is_dominated = len(dominated_by) > 0
        if is_dominated:
            notes.append(f"Pareto-dominated by: {', '.join(dominated_by)}")

        results.append(
            ParetoPoint(
                config_name=pt.config_name,
                speedup=pt.speedup,
                quality_delta=pt.quality_delta,
                benchmark_score=pt.benchmark_score,
                peak_vram_mb=pt.peak_vram_mb,
                acceptable=acceptable,
                dominated=is_dominated,
                dominated_by=dominated_by,
                notes=notes,
            )
        )

    return results


def render_pareto_table(points: list[ParetoPoint]) -> str:
    """Render human-readable markdown table of Pareto analysis."""
    lines = [
        "| Configuration | Speedup | Quality Delta | Score | Pareto Status | Acceptable | Notes |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for p in sorted(points, key=lambda x: x.speedup, reverse=True):
        status = "Dominated" if p.dominated else "**Optimal (Frontier)**"
        acc_str = "YES" if p.acceptable else "**NO (REJECT)**"
        note_str = "; ".join(p.notes) if p.notes else "On Pareto frontier"
        lines.append(
            f"| `{p.config_name}` | {p.speedup:.2f}x | {p.quality_delta:+.2f} | {p.benchmark_score:.1f} | {status} | {acc_str} | {note_str} |"
        )
    return "\n".join(lines)
