"""Inference backend protocol and generation result contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0
    ttft: float = 0.0
    token_timestamps: list[float] = field(default_factory=list)
    backend_metadata: dict[str, Any] = field(default_factory=dict)
    speculative_metadata: dict[str, Any] | None = None

    def __str__(self) -> str:
        return self.text


class InferenceBackend(Protocol):
    """Stable generation protocol supporting common timing and speculative metadata."""

    name: str

    def generate(
        self,
        prompt: str,
        *,
        generation_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GenerationResult: ...
