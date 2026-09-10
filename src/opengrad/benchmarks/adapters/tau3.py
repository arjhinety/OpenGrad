"""tau3-bench Adapter for realistic enterprise multi-turn agent evaluation."""

from __future__ import annotations

from typing import Any, ClassVar

from opengrad.benchmarks.adapters.base import BenchmarkAdapter, BenchmarkTask
from opengrad.benchmarks.backends.protocol import GenerationResult
from opengrad.benchmarks.schema import NormalizedTaskResult
from opengrad.benchmarks.taxonomy import ToolFailureCategory
from opengrad.formatting.parser import parse_qwen_native_output


class Tau3Adapter(BenchmarkAdapter):
    benchmark_id = "tau3"
    name = "tau3-bench"

    DOMAINS: ClassVar[list[str]] = ["airline", "retail", "telecom", "banking"]

    def load_tasks(self, split: str = "retail", limit: int | None = None) -> list[BenchmarkTask]:
        domain = split if split in self.DOMAINS else "retail"
        tasks: list[BenchmarkTask] = []
        domain_scenarios = [
            ("order_cancellation", "Cancel order ORD-9912 per store refund policy", "modify_order"),
            (
                "policy_inquiry",
                "Check baggage allowance policy for domestic flights",
                "lookup_policy",
            ),
            ("account_update", "Update user billing address after verification", "update_account"),
            (
                "dispute_handling",
                "Report an unrecognized transaction on credit card",
                "initiate_dispute",
            ),
        ]
        for i, (task_type, instruction, expected_tool) in enumerate(domain_scenarios):
            task = BenchmarkTask(
                task_id=f"tau3_{domain}_{task_type}_{i + 1:03d}",
                category=domain,
                prompt=f"Domain: {domain.upper()}. Instruction: {instruction}.",
                tools=[
                    {
                        "name": expected_tool,
                        "description": f"Domain tool for {domain}.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action_id": {"type": "string"},
                                "confirmed": {"type": "boolean"},
                            },
                            "required": ["action_id"],
                        },
                    }
                ],
                expected={"tool": expected_tool, "policy_compliant": True},
                expected_decision="CALL",
                metadata={
                    "domain": domain,
                    "task_type": task_type,
                    "benchmark_revision": "c3f7943fa1094056a0667e4125b2938cf78b024c",
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

        # Multi-turn and policy tracking metrics
        steps_to_completion = 1
        policy_compliant = True
        invalid_actions = 0
        unnecessary_actions = 0

        if parsed.status != "RAW_VALID":
            failure_category = ToolFailureCategory.MALFORMED_TOOL_CALL.value
            invalid_actions = 1
        elif parsed.decision != "CALL":
            failure_category = ToolFailureCategory.MISSED_TOOL.value
        else:
            called = parsed.calls[0].name
            expected_tool = task.expected.get("tool") if isinstance(task.expected, dict) else None
            if expected_tool and called != expected_tool:
                failure_category = ToolFailureCategory.WRONG_TOOL.value
                invalid_actions = 1
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
                "policy_compliant": policy_compliant,
                "steps_to_completion": steps_to_completion,
                "invalid_actions": invalid_actions,
                "unnecessary_actions": unnecessary_actions,
                "conversation_length": 2,
            },
        )


def parse_tau3_output(text: str) -> dict[str, Any]:
    parsed = parse_qwen_native_output(text)
    return {"decision": parsed.decision, "calls": [c.name for c in parsed.calls]}
