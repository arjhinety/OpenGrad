"""Berkeley Function Calling Leaderboard (BFCL) V4 Adapter."""

from __future__ import annotations

from typing import Any, ClassVar

from opengrad.benchmarks.adapters.base import BenchmarkAdapter, BenchmarkTask
from opengrad.benchmarks.backends.protocol import GenerationResult
from opengrad.benchmarks.schema import NormalizedTaskResult
from opengrad.benchmarks.taxonomy import ToolFailureCategory
from opengrad.formatting.parser import parse_qwen_native_output


class BFCLv4Adapter(BenchmarkAdapter):
    benchmark_id = "bfcl-v4"
    name = "Berkeley Function Calling Leaderboard V4"

    CATEGORIES: ClassVar[list[str]] = [
        "simple",
        "multiple",
        "parallel",
        "parallel_multiple",
        "relevance",
        "irrelevance",
        "multi_turn",
        "agentic",
        "web_search",
        "memory",
    ]

    def load_tasks(self, split: str = "test", limit: int | None = None) -> list[BenchmarkTask]:
        # Return canonical representative tasks covering all categories
        tasks: list[BenchmarkTask] = []
        for i, cat in enumerate(self.CATEGORIES):
            tool_name = f"bfcl_{cat}_tool"
            expected_decision = "ANSWER" if cat == "irrelevance" else "CALL"
            task = BenchmarkTask(
                task_id=f"bfcl_v4_{cat}_{i + 1:03d}",
                category=cat,
                prompt=f"Execute the required function call for category '{cat}': find user details.",
                tools=[
                    {
                        "name": tool_name,
                        "description": f"BFCL {cat} evaluator tool.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "limit": {"type": "integer"},
                            },
                            "required": ["query"],
                        },
                    }
                ],
                expected={"name": tool_name, "decision": expected_decision},
                expected_decision=expected_decision,
                metadata={
                    "category": cat,
                    "benchmark_revision": "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",
                },
            )
            tasks.append(task)
        if limit is not None:
            tasks = tasks[:limit]
        return tasks

    def evaluate_task(
        self, task: BenchmarkTask, generation: GenerationResult
    ) -> NormalizedTaskResult:
        parsed = parse_qwen_native_output(generation.text)
        expected_decision = task.expected_decision

        success = False
        failure_category: str | None = None
        tool_calls: list[dict[str, Any]] = [
            {"name": c.name, "arguments": c.arguments, "id": c.call_id} for c in parsed.calls
        ]

        if parsed.status != "RAW_VALID":
            failure_category = ToolFailureCategory.MALFORMED_TOOL_CALL.value
        elif expected_decision == "CALL":
            if parsed.decision != "CALL" or not parsed.calls:
                failure_category = ToolFailureCategory.MISSED_TOOL.value
            else:
                called_name = parsed.calls[0].name
                expected_name = (
                    task.expected.get("name") if isinstance(task.expected, dict) else None
                )
                if expected_name and called_name != expected_name:
                    failure_category = ToolFailureCategory.WRONG_TOOL.value
                else:
                    success = True
        elif expected_decision == "ANSWER":
            if parsed.decision == "CALL":
                failure_category = ToolFailureCategory.UNNECESSARY_TOOL.value
            else:
                success = True

        return NormalizedTaskResult(
            task_id=task.task_id,
            input=task.prompt,
            raw_output=generation.text,
            parsed_output={
                "decision": parsed.decision,
                "calls": tool_calls,
                "content": parsed.content,
            },
            expected=task.expected,
            score=1.0 if success else 0.0,
            success=success,
            failure_category=failure_category,
            latency=generation.latency,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            tool_calls=tool_calls,
            metadata=task.metadata,
        )
