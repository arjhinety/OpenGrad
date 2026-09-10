"""ACEBench Adapter for fine-grained tool decision and clarification benchmarking."""

from __future__ import annotations

from typing import Any, ClassVar

from opengrad.benchmarks.adapters.base import BenchmarkAdapter, BenchmarkTask
from opengrad.benchmarks.backends.protocol import GenerationResult
from opengrad.benchmarks.schema import NormalizedTaskResult
from opengrad.benchmarks.taxonomy import ToolFailureCategory
from opengrad.formatting.parser import parse_qwen_native_output


class ACEBenchAdapter(BenchmarkAdapter):
    benchmark_id = "acebench"
    name = "ACEBench"

    CLASSES: ClassVar[list[str]] = [
        "normal",
        "special",
        "ambiguous",
        "incomplete",
        "impossible",
        "agent_multi_turn",
    ]

    def load_tasks(self, split: str = "normal", limit: int | None = None) -> list[BenchmarkTask]:
        tasks: list[BenchmarkTask] = []
        spec = [
            ("normal", "Fetch stock price for AAPL.", "CALL", "get_stock_price"),
            ("special", "Send an email to user without recipient address.", "CLARIFY", None),
            ("ambiguous", "Book a flight to Springfield (multiple cities named Springfield exist).", "CLARIFY", None),
            ("incomplete", "Transfer funds from checking account but amount is omitted.", "CLARIFY", None),
            ("impossible", "Predict the winning lottery numbers for tomorrow night.", "ANSWER", None),
            ("agent_multi_turn", "First check database record, then update if valid.", "CALL", "db_check"),
        ]
        for i, (cls_name, prompt, expected_dec, expected_tool) in enumerate(spec):
            tools = []
            if expected_tool:
                tools.append({
                    "name": expected_tool,
                    "description": f"Tool for {cls_name}",
                    "parameters": {"type": "object", "properties": {"target": {"type": "string"}}},
                })
            task = BenchmarkTask(
                task_id=f"acebench_{cls_name}_{i+1:03d}",
                category=cls_name,
                prompt=prompt,
                tools=tools,
                expected={"decision": expected_dec, "tool": expected_tool},
                expected_decision=expected_dec,
                metadata={"evaluation_class": cls_name, "benchmark_revision": "5e61a684b01e7e45217996c56891ebfe2c29bc96"},
            )
            tasks.append(task)
        if limit is not None:
            tasks = tasks[:limit]
        return tasks

    def evaluate_task(self, task: BenchmarkTask, generation: GenerationResult) -> NormalizedTaskResult:
        parsed = parse_qwen_native_output(generation.text)
        expected_dec = task.expected_decision
        success = False
        failure_category: str | None = None

        tool_calls = [
            {"name": c.name, "arguments": c.arguments, "id": c.call_id} for c in parsed.calls
        ]

        if parsed.status != "RAW_VALID":
            failure_category = ToolFailureCategory.MALFORMED_TOOL_CALL.value
        elif expected_dec == "CALL":
            if parsed.decision != "CALL":
                failure_category = ToolFailureCategory.MISSED_TOOL.value
            else:
                expected_tool = task.expected.get("tool") if isinstance(task.expected, dict) else None
                if expected_tool and parsed.calls[0].name != expected_tool:
                    failure_category = ToolFailureCategory.WRONG_TOOL.value
                else:
                    success = True
        elif expected_dec == "CLARIFY":
            if parsed.decision == "CALL":
                failure_category = ToolFailureCategory.UNNECESSARY_TOOL.value
            elif parsed.decision != "CLARIFY" and "clarif" not in generation.text.lower():
                failure_category = ToolFailureCategory.FAILED_CLARIFICATION.value
            else:
                success = True
        elif expected_dec == "ANSWER":
            if parsed.decision == "CALL":
                failure_category = ToolFailureCategory.UNNECESSARY_TOOL.value
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


def parse_acebench_output(text: str) -> dict[str, Any]:
    parsed = parse_qwen_native_output(text)
    return {"decision": parsed.decision, "calls": [c.name for c in parsed.calls]}
