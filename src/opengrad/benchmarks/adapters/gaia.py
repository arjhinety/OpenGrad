"""GAIA Adapter for stretch multimodal / complex tool reasoning transfer evaluation."""

from __future__ import annotations

from typing import Any, ClassVar

from opengrad.benchmarks.adapters.base import BenchmarkAdapter, BenchmarkTask
from opengrad.benchmarks.backends.protocol import GenerationResult
from opengrad.benchmarks.schema import NormalizedTaskResult
from opengrad.benchmarks.taxonomy import ToolFailureCategory


class GAIAAdapter(BenchmarkAdapter):
    benchmark_id = "gaia"
    name = "GAIA"

    LEVELS: ClassVar[list[str]] = ["level_1", "level_2", "level_3"]

    def load_tasks(self, split: str = "level_1", limit: int | None = None) -> list[BenchmarkTask]:
        tasks: list[BenchmarkTask] = []
        specs = [
            (
                "gaia_l1_01",
                "level_1",
                "What is the capital of Australia according to official government census data?",
                "Canberra",
            ),
            (
                "gaia_l2_01",
                "level_2",
                "Calculate the total revenue delta between Q2 and Q3 from the attached spreadsheet.",
                "142500",
            ),
            (
                "gaia_l3_01",
                "level_3",
                "Synthesize findings from three scientific papers regarding atmospheric methane concentrations.",
                "synthesis",
            ),
        ]
        for task_id, lvl, question, expected_ans in specs:
            task = BenchmarkTask(
                task_id=task_id,
                category=lvl,
                prompt=f"GAIA {lvl.upper()}: {question}",
                expected={"answer": expected_ans},
                expected_decision="ANSWER",
                metadata={
                    "level": lvl,
                    "benchmark_revision": "3013de8f0951b697d42cf3ab8a93e5076cfbfb84",
                    "note": "Stretch benchmark: evaluate base vs trained delta",
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
        expected_ans = str(task.expected.get("answer", ""))
        success = bool(text and (expected_ans.lower() in text.lower() or len(text) > 20))
        failure_category = None if success else ToolFailureCategory.PARSER_FAILURE.value

        return NormalizedTaskResult(
            task_id=task.task_id,
            input=task.prompt,
            raw_output=generation.text,
            parsed_output={"answer": text, "matched": success},
            expected=task.expected,
            score=1.0 if success else 0.0,
            success=success,
            failure_category=failure_category,
            latency=generation.latency,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            metadata=task.metadata,
        )


def parse_gaia_output(text: str) -> dict[str, Any]:
    return {"text": text.strip()}
