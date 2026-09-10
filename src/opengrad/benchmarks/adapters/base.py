"""Base contract for benchmark adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opengrad.benchmarks.backends.protocol import GenerationResult
from opengrad.benchmarks.schema import NormalizedTaskResult


@dataclass
class BenchmarkTask:
    task_id: str
    category: str
    prompt: str
    tools: list[dict[str, Any]] = field(default_factory=list)
    expected: Any = None
    expected_decision: str = "CALL"
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkAdapter:
    """Base class for benchmark adapters. Provides task loading, evaluation, and aggregation."""

    benchmark_id: str
    name: str

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path.cwd()

    def load_tasks(self, split: str = "test", limit: int | None = None) -> list[BenchmarkTask]:
        """Load benchmark tasks for the requested split."""
        raise NotImplementedError

    def render_prompt(self, task: BenchmarkTask, renderer_name: str = "default") -> str:
        """Render prompt using requested renderer conventions."""
        return task.prompt

    def evaluate_task(self, task: BenchmarkTask, generation: GenerationResult) -> NormalizedTaskResult:
        """Evaluate a single generation output against task expectation."""
        raise NotImplementedError

    def aggregate_metrics(self, tasks: list[NormalizedTaskResult]) -> dict[str, Any]:
        """Compute summary and category-level metrics across all evaluated tasks."""
        total = len(tasks)
        if total == 0:
            return {"total_tasks": 0, "accuracy": 0.0}

        successful = len([t for t in tasks if t.success])
        overall_accuracy = round(successful / total * 100.0, 2)

        # Breakdown by category
        cat_counts: dict[str, int] = {}
        cat_success: dict[str, int] = {}
        failure_counts: dict[str, int] = {}

        for t in tasks:
            cat = str(t.metadata.get("category", "general"))
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            if t.success:
                cat_success[cat] = cat_success.get(cat, 0) + 1
            elif t.failure_category:
                failure_counts[t.failure_category] = failure_counts.get(t.failure_category, 0) + 1

        category_accuracy: dict[str, float] = {}
        for cat, count in cat_counts.items():
            category_accuracy[cat] = round(cat_success.get(cat, 0) / count * 100.0, 2)

        # System performance summary
        latencies = [t.latency for t in tasks if t.latency > 0]
        prompt_tokens = [t.prompt_tokens for t in tasks if t.prompt_tokens > 0]
        comp_tokens = [t.completion_tokens for t in tasks if t.completion_tokens > 0]

        mean_latency = round(sum(latencies) / len(latencies), 4) if latencies else 0.0
        total_comp = sum(comp_tokens)
        total_time = sum(latencies)
        tokens_per_sec = round(total_comp / total_time, 2) if total_time > 0 else 0.0

        return {
            "total_tasks": total,
            "successful_tasks": successful,
            "overall_accuracy": overall_accuracy,
            "category_accuracy": category_accuracy,
            "failure_breakdown": dict(sorted(failure_counts.items())),
            "system_metrics": {
                "mean_latency_sec": mean_latency,
                "total_prompt_tokens": sum(prompt_tokens),
                "total_completion_tokens": total_comp,
                "output_tokens_per_second": tokens_per_sec,
            },
        }
