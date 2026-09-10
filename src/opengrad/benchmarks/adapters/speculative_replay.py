"""Speculative Decoding Public Benchmark Replay Adapter."""

from __future__ import annotations

from typing import ClassVar

from opengrad.benchmarks.adapters.base import BenchmarkAdapter, BenchmarkTask
from opengrad.benchmarks.backends.protocol import GenerationResult
from opengrad.benchmarks.schema import NormalizedTaskResult
from opengrad.benchmarks.taxonomy import ToolFailureCategory
from opengrad.formatting.parser import parse_qwen_native_output


class SpeculativeReplayAdapter(BenchmarkAdapter):
    benchmark_id = "speculative-replay"
    name = "Speculative Decoding Public Benchmark Replay"

    REPLAY_BENCHMARKS: ClassVar[list[str]] = ["bfcl_replay", "tau3_replay", "acebench_replay"]

    def load_tasks(self, split: str = "bfcl_replay", limit: int | None = None) -> list[BenchmarkTask]:
        target_split = split if split in self.REPLAY_BENCHMARKS else "bfcl_replay"
        tasks: list[BenchmarkTask] = []
        specs = [
            ("spec_replay_01", "get_stock_price", "Fetch current price for MSFT.", "CALL"),
            ("spec_replay_02", "query_database", "Query customer records from database.", "CALL"),
            ("spec_replay_03", "lookup_policy", "Explain company refund policy.", "CALL"),
        ]
        for i, (task_id, tool_name, prompt, expected_dec) in enumerate(specs):
            task = BenchmarkTask(
                task_id=f"{target_split}_{task_id}_{i+1:03d}",
                category=target_split,
                prompt=f"Replay task [{target_split}]: {prompt}",
                tools=[
                    {
                        "name": tool_name,
                        "description": "Replay evaluation tool.",
                        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                    }
                ],
                expected={"tool": tool_name, "decision": expected_dec},
                expected_decision=expected_dec,
                metadata={"split": target_split, "benchmark_revision": "speculative-replay-v1"},
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

        task_meta = {
            **task.metadata,
            "ttft": generation.ttft,
            "latency": generation.latency,
        }
        if generation.speculative_metadata:
            task_meta["speculative"] = generation.speculative_metadata

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
            metadata=task_meta,
        )
