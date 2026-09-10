"""MMLU-Pro Adapter for hard general capability retention benchmarking."""

from __future__ import annotations

import re
from typing import Any, ClassVar

from opengrad.benchmarks.adapters.base import BenchmarkAdapter, BenchmarkTask
from opengrad.benchmarks.backends.protocol import GenerationResult
from opengrad.benchmarks.schema import NormalizedTaskResult
from opengrad.benchmarks.taxonomy import ToolFailureCategory


class MMLUProAdapter(BenchmarkAdapter):
    benchmark_id = "mmlu-pro"
    name = "MMLU-Pro"

    SUBJECTS: ClassVar[list[str]] = [
        "computer_science",
        "math",
        "physics",
        "chemistry",
        "biology",
        "economics",
        "engineering",
        "philosophy",
    ]

    def load_tasks(self, split: str = "test", limit: int | None = None) -> list[BenchmarkTask]:
        tasks: list[BenchmarkTask] = []
        specs = [
            (
                "computer_science",
                "What is the amortized time complexity of inserting into a dynamic array that doubles in size?\n(A) O(1) (B) O(N) (C) O(log N) (D) O(N^2) (E) O(sqrt N) (F) O(1/N) (G) O(N log N) (H) None of the above (I) Undefined (J) O(2^N)",
                "A",
            ),
            (
                "math",
                "What is the derivative of f(x) = e^(2x)?\n(A) e^(2x) (B) 2e^(2x) (C) 2x e^(2x) (D) 4e^(2x) (E) 1/2 e^(2x) (F) ln(2x) (G) 0 (H) None of the above (I) 2e^x (J) e^x",
                "B",
            ),
            (
                "physics",
                "Which law states that induced electromotive force is proportional to the negative rate of change of magnetic flux?\n(A) Ampere's Law (B) Gauss's Law (C) Faraday's Law (D) Coulomb's Law (E) Ohm's Law (F) Snell's Law (G) Hooke's Law (H) Joule's Law (I) Kepler's Law (J) Newton's Law",
                "C",
            ),
            (
                "economics",
                "What happens to the equilibrium price when demand shifts right and supply remains constant?\n(A) Decreases (B) Remains unchanged (C) Increases (D) Drops to zero (E) Fluctuates randomly (F) Indeterminate (G) Supply shifts right (H) Inverse relation (I) Negative infinity (J) None of the above",
                "C",
            ),
        ]
        for i, (subject, question, answer) in enumerate(specs):
            task = BenchmarkTask(
                task_id=f"mmlu_pro_{subject}_{i + 1:03d}",
                category=subject,
                prompt=f"Subject: {subject}.\nQuestion: {question}\nAnswer with the correct option letter directly.",
                expected={"answer": answer},
                expected_decision="ANSWER",
                metadata={
                    "subject": subject,
                    "benchmark_revision": "5d15a5eb0092f6b5b5c90b6ef6928e1fc1b1b117",
                },
            )
            tasks.append(task)
        if limit is not None:
            tasks = tasks[:limit]
        return tasks

    def evaluate_task(
        self, task: BenchmarkTask, generation: GenerationResult
    ) -> NormalizedTaskResult:
        text = generation.text.strip().upper()
        expected_ans = str(task.expected.get("answer", "A"))

        # Extract predicted option (A-J)
        match = re.search(r"\b([A-J])\b", text)
        extracted = match.group(1) if match else text[:1]
        success = extracted == expected_ans
        failure_category = None if success else ToolFailureCategory.PARSER_FAILURE.value

        return NormalizedTaskResult(
            task_id=task.task_id,
            input=task.prompt,
            raw_output=generation.text,
            parsed_output={"predicted_letter": extracted, "expected_letter": expected_ans},
            expected=task.expected,
            score=1.0 if success else 0.0,
            success=success,
            failure_category=failure_category,
            latency=generation.latency,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            metadata=task.metadata,
        )


def parse_mmlu_pro_output(text: str) -> dict[str, Any]:
    match = re.search(r"\b([A-J])\b", text.upper())
    return {"letter": match.group(1) if match else None}
