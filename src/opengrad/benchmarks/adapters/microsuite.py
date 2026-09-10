"""Performance Microsuite Adapter for controlled systems benchmarking."""

from __future__ import annotations

from opengrad.benchmarks.adapters.base import BenchmarkAdapter, BenchmarkTask
from opengrad.benchmarks.backends.protocol import GenerationResult
from opengrad.benchmarks.microsuite.prompts import PROMPTS
from opengrad.benchmarks.schema import NormalizedTaskResult
from opengrad.benchmarks.taxonomy import ToolFailureCategory


class MicrosuiteAdapter(BenchmarkAdapter):
    benchmark_id = "performance-microsuite"
    name = "OpenGrad Internal Performance Microsuite"

    def load_tasks(self, split: str = "all", limit: int | None = None) -> list[BenchmarkTask]:
        tasks: list[BenchmarkTask] = []
        for prompt_def in PROMPTS:
            if split != "all" and split != "natural_language" and prompt_def.category != split:
                continue
            task = BenchmarkTask(
                task_id=prompt_def.id,
                category=prompt_def.category,
                prompt=prompt_def.prompt,
                tools=list(prompt_def.tools),
                expected={"type": prompt_def.expected_type, "sha256": prompt_def.sha256},
                expected_decision="CALL" if prompt_def.tools else "ANSWER",
                metadata={
                    "prompt_sha256": prompt_def.sha256,
                    "expected_type": prompt_def.expected_type,
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
        success = bool(text)
        failure_category = None if success else ToolFailureCategory.RUNTIME_FAILURE.value

        spec = generation.speculative_metadata or {}
        task_meta = {
            **task.metadata,
            "ttft": generation.ttft,
            "latency": generation.latency,
            "output_tokens_per_sec": (
                round(generation.completion_tokens / max(0.001, generation.latency), 2)
                if generation.latency > 0
                else 0.0
            ),
        }
        if spec:
            task_meta["speculative"] = spec

        return NormalizedTaskResult(
            task_id=task.task_id,
            input=task.prompt,
            raw_output=generation.text,
            parsed_output={"text": text, "valid": success},
            expected=task.expected,
            score=1.0 if success else 0.0,
            success=success,
            failure_category=failure_category,
            latency=generation.latency,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            metadata=task_meta,
        )
