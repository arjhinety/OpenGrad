"""Teacher provider abstraction and caching for on-policy distillation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol


@dataclass
class TeacherResponse:
    model_id: str
    text: str
    score: float = 1.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0
    error_code: str | None = None
    cached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "text": self.text,
            "score": round(self.score, 4),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency": round(self.latency, 4),
            "error_code": self.error_code,
            "cached": self.cached,
            "metadata": self.metadata,
        }


class TeacherProvider(Protocol):
    """Protocol for distillation teacher models."""

    model_id: str

    def generate_feedback(
        self,
        prompt: str,
        student_response: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> TeacherResponse: ...


class MockTeacherProvider:
    """Deterministic teacher provider for GPU-free testing."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-72B-Instruct") -> None:
        self.model_id = model_id

    def generate_feedback(
        self,
        prompt: str,
        student_response: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> TeacherResponse:
        # Check if student output is empty or malformed
        if not student_response.strip():
            return TeacherResponse(
                model_id=self.model_id,
                text="The response was empty.",
                score=0.0,
                prompt_tokens=32,
                completion_tokens=8,
                latency=0.05,
            )
        return TeacherResponse(
            model_id=self.model_id,
            text=f"Correct tool usage verified by {self.model_id}.",
            score=0.95,
            prompt_tokens=48,
            completion_tokens=16,
            latency=0.08,
            metadata={"verified": True},
        )


class CachedTeacherProvider:
    """Wraps any TeacherProvider with persistent disk caching to prevent recomputation."""

    def __init__(self, inner: TeacherProvider, cache_dir: Path) -> None:
        self.inner = inner
        self.cache_dir = cache_dir / "teacher_cache" / inner.model_id.replace("/", "_")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = inner.model_id

    def _cache_key(self, prompt: str, student_response: str) -> str:
        h = hashlib.sha256(f"{prompt}###{student_response}".encode()).hexdigest()
        return h

    def generate_feedback(
        self,
        prompt: str,
        student_response: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> TeacherResponse:
        key = self._cache_key(prompt, student_response)
        target = self.cache_dir / f"{key}.json"
        if target.exists():
            try:
                data = json.loads(target.read_text(encoding="utf-8"))
                return TeacherResponse(
                    model_id=data["model_id"],
                    text=data["text"],
                    score=float(data.get("score", 1.0)),
                    prompt_tokens=int(data.get("prompt_tokens", 0)),
                    completion_tokens=int(data.get("completion_tokens", 0)),
                    latency=0.0,
                    cached=True,
                    metadata=dict(data.get("metadata") or {}),
                )
            except (json.JSONDecodeError, KeyError):
                pass

        res = self.inner.generate_feedback(prompt, student_response, metadata=metadata)
        entry = res.to_dict()
        entry["cached_at"] = datetime.now(UTC).isoformat()
        target.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return res
