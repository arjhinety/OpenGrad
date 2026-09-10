"""Verification of quality parity and output equivalence under speculative decoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from opengrad.benchmarks.schema import NormalizedTaskResult


@dataclass
class ToolCallEquivalence:
    tool_selected_match: bool
    arguments_match: bool
    call_order_match: bool
    call_count_match: bool
    final_answer_match: bool
    ar_invalid_call: bool
    spec_invalid_call: bool

    @property
    def is_equivalent(self) -> bool:
        return (
            self.tool_selected_match
            and self.arguments_match
            and self.call_order_match
            and self.call_count_match
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_selected_match": self.tool_selected_match,
            "arguments_match": self.arguments_match,
            "call_order_match": self.call_order_match,
            "call_count_match": self.call_count_match,
            "final_answer_match": self.final_answer_match,
            "is_equivalent": self.is_equivalent,
            "ar_invalid_call": self.ar_invalid_call,
            "spec_invalid_call": self.spec_invalid_call,
        }


@dataclass
class QualityParitySummary:
    total_evaluated: int
    exact_token_match_count: int
    exact_token_match_rate: float
    tool_equivalence_count: int
    tool_equivalence_rate: float
    ar_benchmark_score: float
    spec_benchmark_score: float
    score_delta: float
    malformed_output_rate_ar: float
    malformed_output_rate_spec: float
    parity_maintained: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_evaluated": self.total_evaluated,
            "exact_token_match_count": self.exact_token_match_count,
            "exact_token_match_rate": round(self.exact_token_match_rate, 4),
            "tool_equivalence_count": self.tool_equivalence_count,
            "tool_equivalence_rate": round(self.tool_equivalence_rate, 4),
            "ar_benchmark_score": round(self.ar_benchmark_score, 2),
            "spec_benchmark_score": round(self.spec_benchmark_score, 2),
            "score_delta": round(self.score_delta, 2),
            "malformed_output_rate_ar": round(self.malformed_output_rate_ar, 4),
            "malformed_output_rate_spec": round(self.malformed_output_rate_spec, 4),
            "parity_maintained": self.parity_maintained,
            "notes": self.notes,
        }


def check_tool_call_equivalence(
    ar_task: NormalizedTaskResult, spec_task: NormalizedTaskResult
) -> ToolCallEquivalence:
    """Compare tool calling outputs between AR and Speculative decoding."""
    ar_calls = ar_task.tool_calls
    spec_calls = spec_task.tool_calls

    count_match = len(ar_calls) == len(spec_calls)
    if not ar_calls and not spec_calls:
        # Both emitted no tool calls
        ans_match = (
            str(ar_task.parsed_output).strip() == str(spec_task.parsed_output).strip()
            if ar_task.parsed_output and spec_task.parsed_output
            else True
        )
        return ToolCallEquivalence(
            tool_selected_match=True,
            arguments_match=True,
            call_order_match=True,
            call_count_match=True,
            final_answer_match=ans_match,
            ar_invalid_call=not ar_task.success,
            spec_invalid_call=not spec_task.success,
        )

    if not count_match:
        return ToolCallEquivalence(
            tool_selected_match=False,
            arguments_match=False,
            call_order_match=False,
            call_count_match=False,
            final_answer_match=False,
            ar_invalid_call=not ar_task.success,
            spec_invalid_call=not spec_task.success,
        )

    tools_match = all(a.get("name") == s.get("name") for a, s in zip(ar_calls, spec_calls))
    args_match = all(a.get("arguments") == s.get("arguments") for a, s in zip(ar_calls, spec_calls))

    return ToolCallEquivalence(
        tool_selected_match=tools_match,
        arguments_match=args_match,
        call_order_match=tools_match,
        call_count_match=True,
        final_answer_match=ar_task.parsed_output == spec_task.parsed_output,
        ar_invalid_call=not ar_task.success,
        spec_invalid_call=not spec_task.success,
    )


def evaluate_quality_parity(
    ar_tasks: list[NormalizedTaskResult],
    spec_tasks: list[NormalizedTaskResult],
    tolerance_score_drop: float = 0.5,
) -> QualityParitySummary:
    """Evaluate quality preservation between autoregressive and speculative runs."""
    total = min(len(ar_tasks), len(spec_tasks))
    if total == 0:
        return QualityParitySummary(
            total_evaluated=0,
            exact_token_match_count=0,
            exact_token_match_rate=0.0,
            tool_equivalence_count=0,
            tool_equivalence_rate=0.0,
            ar_benchmark_score=0.0,
            spec_benchmark_score=0.0,
            score_delta=0.0,
            malformed_output_rate_ar=0.0,
            malformed_output_rate_spec=0.0,
            parity_maintained=True,
        )

    exact_matches = 0
    tool_eq_matches = 0
    ar_score_sum = 0.0
    spec_score_sum = 0.0
    ar_malformed = 0
    spec_malformed = 0

    for ar_t, spec_t in zip(ar_tasks[:total], spec_tasks[:total]):
        ar_score_sum += ar_t.score
        spec_score_sum += spec_t.score

        if ar_t.raw_output == spec_t.raw_output:
            exact_matches += 1

        eq = check_tool_call_equivalence(ar_t, spec_t)
        if eq.is_equivalent:
            tool_eq_matches += 1

        if ar_t.failure_category in {"malformed_tool_call", "parser_failure"}:
            ar_malformed += 1
        if spec_t.failure_category in {"malformed_tool_call", "parser_failure"}:
            spec_malformed += 1

    ar_score = round(ar_score_sum / total * 100.0, 2)
    spec_score = round(spec_score_sum / total * 100.0, 2)
    delta = round(spec_score - ar_score, 2)

    notes = []
    parity_maintained = True
    if delta < -tolerance_score_drop:
        parity_maintained = False
        notes.append(
            f"Quality regression detected: {delta:.2f} pts drop exceeds tolerance of -{tolerance_score_drop:.2f} pts."
        )

    if spec_malformed > ar_malformed:
        notes.append(
            f"Speculative decoding increased malformed outputs ({spec_malformed} vs {ar_malformed})."
        )

    return QualityParitySummary(
        total_evaluated=total,
        exact_token_match_count=exact_matches,
        exact_token_match_rate=round(exact_matches / total, 4),
        tool_equivalence_count=tool_eq_matches,
        tool_equivalence_rate=round(tool_eq_matches / total, 4),
        ar_benchmark_score=ar_score,
        spec_benchmark_score=spec_score,
        score_delta=delta,
        malformed_output_rate_ar=round(ar_malformed / total, 4),
        malformed_output_rate_spec=round(spec_malformed / total, 4),
        parity_maintained=parity_maintained,
        notes=notes,
    )
