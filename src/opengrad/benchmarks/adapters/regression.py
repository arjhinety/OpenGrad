"""Secondary regression adapters: GSM8K and ARC-Challenge."""

from __future__ import annotations

import re
from typing import Any

from opengrad.benchmarks.adapters.base import BenchmarkAdapter, BenchmarkTask
from opengrad.benchmarks.backends.protocol import GenerationResult
from opengrad.benchmarks.schema import NormalizedTaskResult
from opengrad.benchmarks.taxonomy import ToolFailureCategory


class GSM8KAdapter(BenchmarkAdapter):
    benchmark_id = "gsm8k"
    name = "GSM8K"

    def load_tasks(self, split: str = "test", limit: int | None = None) -> list[BenchmarkTask]:
        tasks: list[BenchmarkTask] = []
        specs = [
            ("gsm8k_01", "Janet buys 3 packs of golf balls for $12 each. How much change does she get from $50?", 14),
            ("gsm8k_02", "A bakery bakes 120 loaves of bread. In the morning they sell 45 loaves, in the afternoon 50 loaves. How many are left?", 25),
        ]
        for task_id, question, expected_num in specs:
            task = BenchmarkTask(
                task_id=task_id,
                category="math_word_problem",
                prompt=f"Question: {question}\nGive step-by-step reasoning and conclude with '#### <answer>'.",
                expected={"answer": expected_num},
                expected_decision="ANSWER",
                metadata={"benchmark_revision": "b79e5cc092b3ef7753e1a6c0b396b27d42cf3ab8"},
            )
            tasks.append(task)
        if limit is not None:
            tasks = tasks[:limit]
        return tasks

    def evaluate_task(self, task: BenchmarkTask, generation: GenerationResult) -> NormalizedTaskResult:
        text = generation.text.strip()
        expected_ans = int(task.expected.get("answer", 0))

        # Extract number after #### or last number
        match = re.search(r"####\s*(-?\d+)", text)
        if not match:
            match = re.search(r"(-?\d+)\b", text)

        extracted = int(match.group(1)) if match else None
        success = extracted == expected_ans
        failure_category = None if success else ToolFailureCategory.PARSER_FAILURE.value

        return NormalizedTaskResult(
            task_id=task.task_id,
            input=task.prompt,
            raw_output=generation.text,
            parsed_output={"extracted_answer": extracted, "expected_answer": expected_ans},
            expected=task.expected,
            score=1.0 if success else 0.0,
            success=success,
            failure_category=failure_category,
            latency=generation.latency,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            metadata=task.metadata,
        )


class ARCChallengeAdapter(BenchmarkAdapter):
    benchmark_id = "arc-challenge"
    name = "ARC-Challenge"

    def load_tasks(self, split: str = "test", limit: int | None = None) -> list[BenchmarkTask]:
        tasks: list[BenchmarkTask] = []
        specs = [
            ("arc_01", "Which process converts liquid water into water vapor?\n(A) Condensation\n(B) Evaporation\n(C) Precipitation\n(D) Freezing", "B"),
            ("arc_02", "Which organ is primarily responsible for filtering blood in humans?\n(A) Heart\n(B) Lungs\n(C) Kidneys\n(D) Stomach", "C"),
        ]
        for task_id, question, expected_letter in specs:
            task = BenchmarkTask(
                task_id=task_id,
                category="science_qa",
                prompt=f"Question: {question}\nAnswer with the option letter directly.",
                expected={"letter": expected_letter},
                expected_decision="ANSWER",
                metadata={"benchmark_revision": "main"},
            )
            tasks.append(task)
        if limit is not None:
            tasks = tasks[:limit]
        return tasks

    def evaluate_task(self, task: BenchmarkTask, generation: GenerationResult) -> NormalizedTaskResult:
        text = generation.text.strip().upper()
        expected_letter = str(task.expected.get("letter", "A"))
        match = re.search(r"\b([A-D])\b", text)
        extracted = match.group(1) if match else text[:1]
        success = extracted == expected_letter
        failure_category = None if success else ToolFailureCategory.PARSER_FAILURE.value

        return NormalizedTaskResult(
            task_id=task.task_id,
            input=task.prompt,
            raw_output=generation.text,
            parsed_output={"predicted_letter": extracted, "expected_letter": expected_letter},
            expected=task.expected,
            score=1.0 if success else 0.0,
            success=success,
            failure_category=failure_category,
            latency=generation.latency,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            metadata=task.metadata,
        )


def parse_gsm8k_output(text: str) -> dict[str, Any]:
    match = re.search(r"####\s*(-?\d+)", text)
    return {"answer": int(match.group(1)) if match else None}


def parse_arc_output(text: str) -> dict[str, Any]:
    match = re.search(r"\b([A-D])\b", text.upper())
    return {"letter": match.group(1) if match else None}
