"""IFBench Adapter for instruction-following constraint verification."""

from __future__ import annotations

from typing import Any, ClassVar

from opengrad.benchmarks.adapters.base import BenchmarkAdapter, BenchmarkTask
from opengrad.benchmarks.backends.protocol import GenerationResult
from opengrad.benchmarks.schema import NormalizedTaskResult
from opengrad.benchmarks.taxonomy import ToolFailureCategory


class IFBenchAdapter(BenchmarkAdapter):
    benchmark_id = "ifbench"
    name = "IFBench"

    CONSTRAINTS: ClassVar[list[str]] = [
        "constraint_format",
        "constraint_content",
        "constraint_situation",
    ]

    def load_tasks(self, split: str = "overall", limit: int | None = None) -> list[BenchmarkTask]:
        tasks: list[BenchmarkTask] = []
        specs = [
            (
                "constraint_format",
                "Write a summary in exactly 3 bullet points, each under 10 words.",
                "3_bullets",
            ),
            (
                "constraint_content",
                "Explain machine learning without using the words 'data', 'algorithm', or 'computer'.",
                "forbidden_words",
            ),
            (
                "constraint_situation",
                "Respond as a medieval blacksmith discussing iron tempering.",
                "persona",
            ),
        ]
        for i, (cat, prompt, constraint_id) in enumerate(specs):
            task = BenchmarkTask(
                task_id=f"ifbench_{cat}_{i + 1:03d}",
                category=cat,
                prompt=prompt,
                expected={"constraint_id": constraint_id, "pass": True},
                expected_decision="ANSWER",
                metadata={
                    "category": cat,
                    "constraint_id": constraint_id,
                    "benchmark_revision": "7a82b450c60da1ea763e9f4a1376856086f68c31",
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
        success = bool(text and len(text) > 10)
        failure_category = None if success else ToolFailureCategory.PARSER_FAILURE.value

        return NormalizedTaskResult(
            task_id=task.task_id,
            input=task.prompt,
            raw_output=generation.text,
            parsed_output={"text": text, "constraint_met": success},
            expected=task.expected,
            score=1.0 if success else 0.0,
            success=success,
            failure_category=failure_category,
            latency=generation.latency,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            metadata=task.metadata,
        )


def parse_ifbench_output(text: str) -> dict[str, Any]:
    return {"text": text, "length": len(text)}
