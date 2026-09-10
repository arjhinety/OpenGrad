"""MCPMark / MCP-Universe Adapter for Model Context Protocol ecosystem evaluation."""

from __future__ import annotations

from typing import Any, ClassVar

from opengrad.benchmarks.adapters.base import BenchmarkAdapter, BenchmarkTask
from opengrad.benchmarks.backends.protocol import GenerationResult
from opengrad.benchmarks.schema import NormalizedTaskResult
from opengrad.benchmarks.taxonomy import ToolFailureCategory
from opengrad.formatting.parser import parse_qwen_native_output


class MCPMarkAdapter(BenchmarkAdapter):
    benchmark_id = "mcpmark"
    name = "MCPMark / MCP-Universe"

    ENVIRONMENTS: ClassVar[list[str]] = ["filesystem", "github", "postgresql", "browser", "notion"]

    def load_tasks(
        self, split: str = "filesystem", limit: int | None = None
    ) -> list[BenchmarkTask]:
        tasks: list[BenchmarkTask] = []
        specs = [
            (
                "filesystem",
                "Find all files matching '*.py' in directory /project/src.",
                "mcp__filesystem__list_directory",
            ),
            (
                "github",
                "Create an issue with title 'Bug in tokenizer' in repo 'opengrad/core'.",
                "mcp__github__create_issue",
            ),
            (
                "postgresql",
                "Query active users from table 'accounts' created in last 7 days.",
                "mcp__postgres__query",
            ),
            (
                "browser",
                "Navigate to docs page and extract the API endpoint specification.",
                "mcp__browser__navigate",
            ),
            (
                "notion",
                "Append a bulleted summary to the weekly sprint page.",
                "mcp__notion__append_block",
            ),
        ]
        for i, (e_name, prompt, expected_tool) in enumerate(specs):
            task = BenchmarkTask(
                task_id=f"mcpmark_{e_name}_{i + 1:03d}",
                category=e_name,
                prompt=f"MCP Environment: {e_name}.\nTask: {prompt}",
                tools=[
                    {
                        "name": expected_tool,
                        "description": f"MCP tool for {e_name}",
                        "parameters": {
                            "type": "object",
                            "properties": {"target": {"type": "string"}},
                        },
                    }
                ],
                expected={"tool": expected_tool, "environment": e_name},
                expected_decision="CALL",
                metadata={
                    "environment": e_name,
                    "benchmark_revision": "a18a93e5076cfbfb84d4715f168fb9bc20163359",
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
            called_name = parsed.calls[0].name
            expected_tool = task.expected.get("tool") if isinstance(task.expected, dict) else None
            if expected_tool and called_name != expected_tool:
                failure_category = ToolFailureCategory.WRONG_TOOL.value
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
            metadata={
                **task.metadata,
                "step_count": 1,
                "invalid_actions": 0 if success else 1,
                "loops": 0,
                "recoveries": 0,
            },
        )


def parse_mcp_output(text: str) -> dict[str, Any]:
    parsed = parse_qwen_native_output(text)
    return {"decision": parsed.decision, "calls": [c.name for c in parsed.calls]}
