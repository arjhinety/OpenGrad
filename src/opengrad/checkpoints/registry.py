"""Checkpoint registry managing lifecycle, metadata, evaluation status, and promotion states."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any


class CheckpointLifecycle(str, Enum):
    TRAINING = "TRAINING"
    CANDIDATE = "CANDIDATE"
    EVALUATING = "EVALUATING"
    REJECTED = "REJECTED"
    PROMOTED = "PROMOTED"
    ARCHIVED = "ARCHIVED"


@dataclass
class CheckpointRecord:
    checkpoint_id: str
    experiment_id: str
    path: str
    global_step: int = 0
    tokens_seen: int = 0
    training_loss: float = 0.0
    parent_checkpoint: str | None = None
    hash: str = ""
    model_id: str = "Qwen/Qwen3.5-2B"
    model_revision: str = "15852e8c16360a2fea060d615a32b45270f8a8fc"
    adapter_info: dict[str, Any] = field(default_factory=dict)
    creation_timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    evaluation_status: str = "UNEVALUATED"
    promotion_status: str = CheckpointLifecycle.CANDIDATE.value
    benchmarks_evaluated: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "experiment_id": self.experiment_id,
            "path": self.path,
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "training_loss": round(self.training_loss, 4),
            "parent_checkpoint": self.parent_checkpoint,
            "hash": self.hash,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "adapter_info": self.adapter_info,
            "creation_timestamp": self.creation_timestamp,
            "evaluation_status": self.evaluation_status,
            "promotion_status": self.promotion_status,
            "benchmarks_evaluated": self.benchmarks_evaluated,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointRecord:
        return cls(
            checkpoint_id=str(data["checkpoint_id"]),
            experiment_id=str(data["experiment_id"]),
            path=str(data["path"]),
            global_step=int(data.get("global_step", 0)),
            tokens_seen=int(data.get("tokens_seen", 0)),
            training_loss=float(data.get("training_loss", 0.0)),
            parent_checkpoint=data.get("parent_checkpoint"),
            hash=str(data.get("hash", "")),
            model_id=str(data.get("model_id", "Qwen/Qwen3.5-2B")),
            model_revision=str(data.get("model_revision", "")),
            adapter_info=dict(data.get("adapter_info") or {}),
            creation_timestamp=str(data.get("creation_timestamp", "")),
            evaluation_status=str(data.get("evaluation_status", "UNEVALUATED")),
            promotion_status=str(data.get("promotion_status", CheckpointLifecycle.CANDIDATE.value)),
            benchmarks_evaluated=dict(data.get("benchmarks_evaluated") or {}),
            metadata=dict(data.get("metadata") or {}),
        )


class CheckpointRegistry:
    """Central registry tracking all model checkpoints and promotion status."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path.cwd()
        self.registry_file = self.root / "runs" / "checkpoint_registry.json"
        self._checkpoints: dict[str, CheckpointRecord] = {}
        self._load()

    def _load(self) -> None:
        if self.registry_file.exists():
            try:
                data = json.loads(self.registry_file.read_text(encoding="utf-8"))
                for row in data.get("checkpoints", []):
                    rec = CheckpointRecord.from_dict(row)
                    self._checkpoints[rec.checkpoint_id] = rec
            except (json.JSONDecodeError, KeyError):
                pass

    def register_checkpoint(self, record: CheckpointRecord) -> CheckpointRecord:
        existing = self._checkpoints.get(record.checkpoint_id)
        if existing is not None:
            # Checkpoint identities are immutable. Re-registering the same ID is
            # only valid when it is the exact same artifact/experiment lineage;
            # silently replacing a record would destroy provenance.
            if existing.to_dict() != record.to_dict():
                raise FileExistsError(
                    f"Checkpoint already registered with different provenance: {record.checkpoint_id}"
                )
            return existing
        self._checkpoints[record.checkpoint_id] = record
        self._save()
        return record

    def get_checkpoint(self, checkpoint_id: str) -> CheckpointRecord:
        if checkpoint_id not in self._checkpoints:
            raise KeyError(f"Checkpoint not registered: '{checkpoint_id}'")
        return self._checkpoints[checkpoint_id]

    def list_checkpoints(
        self, experiment_id: str | None = None, status: str | None = None
    ) -> list[CheckpointRecord]:
        items = list(self._checkpoints.values())
        if experiment_id:
            items = [c for c in items if c.experiment_id == experiment_id]
        if status:
            items = [c for c in items if c.promotion_status == status]
        return items

    def update_status(
        self, checkpoint_id: str, new_status: CheckpointLifecycle | str, note: str | None = None
    ) -> CheckpointRecord:
        rec = self.get_checkpoint(checkpoint_id)
        status_str = (
            new_status.value if isinstance(new_status, CheckpointLifecycle) else str(new_status)
        )
        rec.promotion_status = status_str
        if note:
            rec.metadata["status_note"] = note
            rec.metadata["status_updated_at"] = datetime.now(UTC).isoformat()
        self._save()
        return rec

    def get_promoted_checkpoint(self) -> CheckpointRecord | None:
        promoted = [
            c
            for c in self._checkpoints.values()
            if c.promotion_status == CheckpointLifecycle.PROMOTED.value
        ]
        if not promoted:
            return None
        # Pick highest step or newest creation timestamp
        return max(promoted, key=lambda x: (x.global_step, x.creation_timestamp))

    def _save(self) -> None:
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "last_updated": datetime.now(UTC).isoformat(),
            "total_checkpoints": len(self._checkpoints),
            "checkpoints": [c.to_dict() for c in self._checkpoints.values()],
        }
        temp = self.registry_file.with_name(self.registry_file.name + ".tmp")
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp.replace(self.registry_file)
