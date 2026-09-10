"""Preference pair data schemas, candidate generation records, and quality gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class PreferenceCandidate:
    candidate_id: str
    response_text: str
    parsed_action: dict[str, Any] | None = None
    parser_status: str = "RAW_VALID"
    deterministic_score: float = 0.0
    seed: int = 42
    temperature: float = 0.7
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "response_text": self.response_text,
            "parsed_action": self.parsed_action,
            "parser_status": self.parser_status,
            "deterministic_score": round(self.deterministic_score, 4),
            "seed": self.seed,
            "temperature": self.temperature,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreferenceCandidate:
        return cls(
            candidate_id=str(data["candidate_id"]),
            response_text=str(data["response_text"]),
            parsed_action=data.get("parsed_action"),
            parser_status=str(data.get("parser_status", "RAW_VALID")),
            deterministic_score=float(data.get("deterministic_score", 0.0)),
            seed=int(data.get("seed", 42)),
            temperature=float(data.get("temperature", 0.7)),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class PreferencePair:
    prompt_id: str
    canonical_id: str
    prompt: str
    chosen: str
    rejected: str
    preference_source: str  # "deterministic", "openai_adjudicated", "when2call_real"
    confidence: float = 1.0
    reason_codes: list[str] = field(default_factory=list)
    behavior_category: str = "tool_policy"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError(f"PreferencePair {self.prompt_id}: prompt is empty")
        if not self.chosen.strip():
            raise ValueError(f"PreferencePair {self.prompt_id}: chosen response is empty")
        if not self.rejected.strip():
            raise ValueError(f"PreferencePair {self.prompt_id}: rejected response is empty")
        if self.chosen.strip() == self.rejected.strip():
            raise ValueError(f"PreferencePair {self.prompt_id}: chosen equals rejected")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "prompt_id": self.prompt_id,
            "canonical_id": self.canonical_id,
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "preference_source": self.preference_source,
            "confidence": round(self.confidence, 4),
            "reason_codes": self.reason_codes,
            "behavior_category": self.behavior_category,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreferencePair:
        return cls(
            prompt_id=str(data["prompt_id"]),
            canonical_id=str(data.get("canonical_id", data["prompt_id"])),
            prompt=str(data["prompt"]),
            chosen=str(data["chosen"]),
            rejected=str(data["rejected"]),
            preference_source=str(data.get("preference_source", "deterministic")),
            confidence=float(data.get("confidence", 1.0)),
            reason_codes=list(data.get("reason_codes") or []),
            behavior_category=str(data.get("behavior_category", "tool_policy")),
            metadata=dict(data.get("metadata") or {}),
            created_at=str(data.get("created_at", "")),
        )
