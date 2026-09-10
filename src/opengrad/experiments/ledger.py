"""Append-only experiment ledger tracking lifecycle transitions and milestones."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any


class LedgerEventType(str, Enum):
    EXPERIMENT_CREATED = "EXPERIMENT_CREATED"
    PREFLIGHT_PASSED = "PREFLIGHT_PASSED"
    PREFLIGHT_FAILED = "PREFLIGHT_FAILED"
    TRAINING_STARTED = "TRAINING_STARTED"
    CHECKPOINT_CREATED = "CHECKPOINT_CREATED"
    TRAINING_FINISHED = "TRAINING_FINISHED"
    EVALUATION_STARTED = "EVALUATION_STARTED"
    BENCHMARK_FINISHED = "BENCHMARK_FINISHED"
    REGRESSION_FOUND = "REGRESSION_FOUND"
    PROMOTION_REVIEW = "PROMOTION_REVIEW"
    CHECKPOINT_PROMOTED = "CHECKPOINT_PROMOTED"
    CHECKPOINT_REJECTED = "CHECKPOINT_REJECTED"


@dataclass(frozen=True)
class LedgerEvent:
    event_type: str
    experiment_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "experiment_id": self.experiment_id,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LedgerEvent:
        return cls(
            timestamp=str(data["timestamp"]),
            event_type=str(data["event_type"]),
            experiment_id=str(data["experiment_id"]),
            details=dict(data.get("details") or {}),
        )


class ExperimentLedger:
    """Append-only event ledger for an individual experiment or central repository history."""

    def __init__(self, ledger_file: Path) -> None:
        self.ledger_file = ledger_file
        self.ledger_file.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        event_type: LedgerEventType | str,
        experiment_id: str,
        details: dict[str, Any] | None = None,
    ) -> LedgerEvent:
        type_str = event_type.value if isinstance(event_type, LedgerEventType) else str(event_type)
        event = LedgerEvent(
            event_type=type_str,
            experiment_id=experiment_id,
            details=details or {},
        )
        line = json.dumps(event.to_dict(), ensure_ascii=False) + "\n"
        with self.ledger_file.open("a", encoding="utf-8") as f:
            f.write(line)
        return event

    def read_events(self) -> list[LedgerEvent]:
        if not self.ledger_file.exists():
            return []
        events: list[LedgerEvent] = []
        with self.ledger_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        events.append(LedgerEvent.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        continue
        return events
