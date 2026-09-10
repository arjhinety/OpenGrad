"""Resource and cost telemetry tracking for experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CostResourceTelemetry:
    gpu_hours: float = 0.0
    teacher_input_tokens: int = 0
    teacher_output_tokens: int = 0
    judge_tokens: int = 0
    estimated_cost_usd: float = 0.0
    wall_time_seconds: float = 0.0
    peak_vram_mb: float = 0.0
    peak_ram_mb: float = 0.0
    checkpoint_size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_hours": round(self.gpu_hours, 3),
            "teacher_input_tokens": self.teacher_input_tokens,
            "teacher_output_tokens": self.teacher_output_tokens,
            "judge_tokens": self.judge_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
            "wall_time_seconds": round(self.wall_time_seconds, 2),
            "peak_vram_mb": round(self.peak_vram_mb, 1),
            "peak_ram_mb": round(self.peak_ram_mb, 1),
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostResourceTelemetry:
        return cls(
            gpu_hours=float(data.get("gpu_hours", 0.0)),
            teacher_input_tokens=int(data.get("teacher_input_tokens", 0)),
            teacher_output_tokens=int(data.get("teacher_output_tokens", 0)),
            judge_tokens=int(data.get("judge_tokens", 0)),
            estimated_cost_usd=float(data.get("estimated_cost_usd", 0.0)),
            wall_time_seconds=float(data.get("wall_time_seconds", 0.0)),
            peak_vram_mb=float(data.get("peak_vram_mb", 0.0)),
            peak_ram_mb=float(data.get("peak_ram_mb", 0.0)),
            checkpoint_size_bytes=int(data.get("checkpoint_size_bytes", 0)),
        )
