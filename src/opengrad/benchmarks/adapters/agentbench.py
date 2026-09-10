"""AgentBench FC Adapter for multi-environment agentic evaluation."""

from __future__ import annotations

from typing import Any, ClassVar

from opengrad.benchmarks.adapters.base import BenchmarkAdapter, BenchmarkTask
from opengrad.benchmarks.backends.protocol import GenerationResult
from opengrad.benchmarks.schema import NormalizedTaskResult
from opengrad.benchmarks.taxonomy import ToolFailureCategory
from opengrad.formatting.parser import parse_qwen_native_output


class AgentBenchFCAdapter(BenchmarkAdapter):
    benchmark_id = "agentbench-fc"
    name = "AgentBench FC"

    ENVIRONMENTS: ClassVar[list[str]] = ["os", "db", "web", "kg"]

    def load_tasks(self, split: str = "os", limit: int | None = None) -> list[BenchmarkTask]:
        tasks: list[BenchmarkTask] = []
        specs = [
            ("os", "Inspect disk space on /var partition.", "os_bash"),
            ("db", "Find students enrolled in CS101 in database.", "db_sql"),
            ("web", "Search documentation for error code 403.", "web_search"),
            ("kg", "Find relationship between entities 'Turing' and 'Bletchley'.", "kg_sparql"),
        ]
        for i, (e_name, prompt, expected_tool) in enumerate(specs):
            task = BenchmarkTask(
                task_id=f"agentbench_{e_name}_{i+1:03d}",
                category=e_name,
                prompt=f"Environment: {e_name}\nTask: {prompt}",
                tools=[
                    {
                        "name": expected_tool,
                        "description": f"Tool for {e_name}",
                        "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                    }
                ],
                expected={"tool": expected_tool, "environment": e_name},
                expected_decision="CALL",
                metadata={"environment": e_name, "benchmark_revision": "1b9e31a84f3c05988ad7890f5bca450cf9659b86"},
            )
            tasks.append(task)
        if limit is not None:
            tasks = tasks[:limit]
        return tasks

    def evaluate_task(self, task: BenchmarkTask, generation: GenerationResult) -> NormalizedTaskResult:
        parsed = parse_qwen_native_output(generation.text)
        success = False
        failure_category: str | None = None
        tool_calls = [
            {"name": c.name, "arguments": c.arguments, "id": c.call_id} for c in parsed.calls
        ]

        if parsed.status != "RAW_VALID":
            failure_category = ToolFailureCategory.MALFORMED_TOOL_CALL.value
        elif parsed.decision != "CALL":
            failure_category = ToolFailureCategory.MISSED_TOOL.value
        else:
            called = parsed.calls[0].name
            expected_tool = task.expected.get("tool") if isinstance(task.expected, dict) else None
            if expected_tool and called != expected_tool:
                failure_category = ToolFailureCategory.WRONG_TOOL.value
            else:
                success = True

        return NormalizedTaskResult(
            task_id=task.task_id,
            input=task.prompt,
            raw_output=generation.text,
            parsed_output={"decision": parsed.decision, "calls": tool_calls, "content": parsed.content},
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


def parse_agentbench_output(text: str) -> dict[str, Any]:
    parsed = parse_qwen_native_output(text)
    return {"decision": parsed.decision, "calls": [c.name for c in parsed.calls]}
