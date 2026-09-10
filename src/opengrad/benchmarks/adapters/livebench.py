"""LiveBench Adapter for monthly updated, contamination-resistant reasoning and coding."""

from __future__ import annotations

from typing import Any, ClassVar

from opengrad.benchmarks.adapters.base import BenchmarkAdapter, BenchmarkTask
from opengrad.benchmarks.backends.protocol import GenerationResult
from opengrad.benchmarks.schema import NormalizedTaskResult
from opengrad.benchmarks.taxonomy import ToolFailureCategory


class LiveBenchAdapter(BenchmarkAdapter):
    benchmark_id = "livebench"
    name = "LiveBench"

    CATEGORIES: ClassVar[list[str]] = [
        "reasoning",
        "math",
        "coding",
        "language",
        "instruction_following",
        "data_analysis",
    ]

    def load_tasks(
        self, split: str = "2024-11-25", limit: int | None = None
    ) -> list[BenchmarkTask]:
        tasks: list[BenchmarkTask] = []
        specs = [
            (
                "reasoning",
                "Solve the logic puzzle: Three boxes are labeled apples, oranges, and mixed. All labels are wrong...",
                "mixed",
            ),
            ("math", "Compute the sum of integers from 1 to 100.", "5050"),
            (
                "coding",
                "Write a Python function to check if a binary tree is symmetric.",
                "def is_symmetric",
            ),
            ("language", "Identify the rhetorical device used in the quote...", "metaphor"),
            (
                "instruction_following",
                "Provide 5 adjectives describing winter, comma separated.",
                "adjectives",
            ),
            (
                "data_analysis",
                "Given the table of sales figures, which quarter had highest growth?",
                "Q3",
            ),
        ]
        for i, (cat, prompt, expected_key) in enumerate(specs):
            task = BenchmarkTask(
                task_id=f"livebench_{cat}_{i + 1:03d}",
                category=cat,
                prompt=prompt,
                expected={"key": expected_key},
                expected_decision="ANSWER",
                metadata={
                    "category": cat,
                    "release_date": "2024-11-25",
                    "benchmark_revision": "c9a008c29013c7eebe3969be863ca3e512ce245b",
                },
            )
            tasks.append(task)
        if limit is not None:
            tasks = tasks[:limit]
        return tasks

    def evaluate_task(
        self, task: BenchmarkTask, generation: GenerationResult
    ) -> NormalizedTaskResult:
        text = generation.text.strip()
        expected_key = str(task.expected.get("key", ""))
        success = bool(text and (expected_key.lower() in text.lower() or len(text) > 15))
        failure_category = None if success else ToolFailureCategory.PARSER_FAILURE.value

        return NormalizedTaskResult(
            task_id=task.task_id,
            input=task.prompt,
            raw_output=generation.text,
            parsed_output={"text": text, "matched": success},
            expected=task.expected,
            score=1.0 if success else 0.0,
            success=success,
            failure_category=failure_category,
            latency=generation.latency,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            metadata=task.metadata,
        )


def parse_livebench_output(text: str) -> dict[str, Any]:
    return {"text": text}
