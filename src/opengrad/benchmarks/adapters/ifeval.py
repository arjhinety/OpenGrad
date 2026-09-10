"""IFEval Adapter for deterministic instruction-following regression testing."""

from __future__ import annotations

from typing import Any

from opengrad.benchmarks.adapters.base import BenchmarkAdapter, BenchmarkTask
from opengrad.benchmarks.backends.protocol import GenerationResult
from opengrad.benchmarks.schema import NormalizedTaskResult
from opengrad.benchmarks.taxonomy import ToolFailureCategory


class IFEvalAdapter(BenchmarkAdapter):
    benchmark_id = "ifeval"
    name = "IFEval"

    def load_tasks(self, split: str = "test", limit: int | None = None) -> list[BenchmarkTask]:
        tasks: list[BenchmarkTask] = []
        specs = [
            ("ifeval_word_count", "Write a short paragraph about Mars with at least 30 words.", lambda t: len(t.split()) >= 30),
            ("ifeval_capital_letters", "Write a sentence entirely in capital letters about space exploration.", lambda t: t.isupper()),
            ("ifeval_json_format", "Respond strictly with a JSON array of three colors.", lambda t: t.strip().startswith("[") and t.strip().endswith("]")),
            ("ifeval_section_header", "Write an article containing a section titled '## Background'.", lambda t: "## Background" in t),
        ]
        for i, (task_id, prompt, checker) in enumerate(specs):
            task = BenchmarkTask(
                task_id=f"{task_id}_{i+1:03d}",
                category="deterministic_rule",
                prompt=prompt,
                expected={"deterministic_rule": task_id},
                expected_decision="ANSWER",
                metadata={"checker": task_id, "benchmark_revision": "cb27b233a763806fcf8ff9a3411bcfcae67c87c0"},
            )
            tasks.append(task)
        if limit is not None:
            tasks = tasks[:limit]
        return tasks

    def evaluate_task(self, task: BenchmarkTask, generation: GenerationResult) -> NormalizedTaskResult:
        text = generation.text.strip()
        checker_id = task.metadata.get("checker", "")
        success = False

        if checker_id == "ifeval_word_count":
            success = len(text.split()) >= 30
        elif checker_id == "ifeval_capital_letters":
            success = text.isupper()
        elif checker_id == "ifeval_json_format":
            success = text.startswith("[") and text.endswith("]")
        elif checker_id == "ifeval_section_header":
            success = "## Background" in text
        else:
            success = len(text) > 5

        failure_category = None if success else ToolFailureCategory.PARSER_FAILURE.value

        return NormalizedTaskResult(
            task_id=task.task_id,
            input=task.prompt,
            raw_output=generation.text,
            parsed_output={"text": text, "strict_pass": success, "loose_pass": success},
            expected=task.expected,
            score=1.0 if success else 0.0,
            success=success,
            failure_category=failure_category,
            latency=generation.latency,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            metadata={**task.metadata, "strict_pass": success, "loose_pass": success},
        )


def parse_ifeval_output(text: str) -> dict[str, Any]:
    return {"text": text, "word_count": len(text.split())}
