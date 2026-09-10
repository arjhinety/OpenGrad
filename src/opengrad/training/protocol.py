"""Trainer backend protocol and training metadata contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol


@dataclass
class TrainingMetadata:
    micro_batch_size: int
    gradient_accumulation_steps: int
    world_size: int
    effective_global_batch_size: int
    learning_rate: float
    max_steps: int
    warmup_steps: int
    tokens_per_update: int
    estimated_total_tokens: int
    precision: str = "bfloat16"
    optimizer: str = "AdamW"
    scheduler: str = "cosine"

    def to_dict(self) -> dict[str, Any]:
        return {
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "world_size": self.world_size,
            "effective_global_batch_size": self.effective_global_batch_size,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "tokens_per_update": self.tokens_per_update,
            "estimated_total_tokens": self.estimated_total_tokens,
            "precision": self.precision,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
        }


@dataclass
class TrainingRunResult:
    experiment_id: str
    algorithm: str  # "sft", "dpo", "on_policy_distillation"
    final_checkpoint_path: str
    checkpoints_created: list[str]
    total_steps: int
    total_tokens_seen: int
    final_loss: float
    elapsed_seconds: float
    metrics_history: list[dict[str, Any]] = field(default_factory=list)
    algorithm_diagnostics: dict[str, Any] = field(default_factory=dict)
    completed_timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "algorithm": self.algorithm,
            "final_checkpoint_path": self.final_checkpoint_path,
            "checkpoints_created": self.checkpoints_created,
            "total_steps": self.total_steps,
            "total_tokens_seen": self.total_tokens_seen,
            "final_loss": round(self.final_loss, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "algorithm_diagnostics": self.algorithm_diagnostics,
            "completed_timestamp": self.completed_timestamp,
        }


class TrainerBackend(Protocol):
    """Stable interface shared by all training algorithms."""

    name: str

    def train(
        self,
        experiment_id: str,
        config: dict[str, Any],
        output_dir: Path,
        *,
        dry_run: bool = False,
    ) -> TrainingRunResult: ...
