"""Unified OpenGrad Failure Taxonomy for tool-calling and agentic evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ToolFailureCategory(str, Enum):
    WRONG_TOOL = "wrong_tool"
    MISSED_TOOL = "missed_tool"
    UNNECESSARY_TOOL = "unnecessary_tool"
    MALFORMED_TOOL_CALL = "malformed_tool_call"
    MALFORMED_ARGUMENTS = "malformed_arguments"
    MISSING_ARGUMENT = "missing_argument"
    WRONG_ARGUMENT = "wrong_argument"
    HALLUCINATED_ARGUMENT = "hallucinated_argument"
    DUPLICATE_CALL = "duplicate_call"
    WRONG_CALL_ORDER = "wrong_call_order"
    FAILED_PARALLELIZATION = "failed_parallelization"
    PREMATURE_FINAL_ANSWER = "premature_final_answer"
    FAILED_CLARIFICATION = "failed_clarification"
    OVER_CLARIFICATION = "over_clarification"
    UNSAFE_ACTION = "unsafe_action"
    TOOL_LOOP = "tool_loop"
    CONTEXT_LOSS = "context_loss"
    PARSER_FAILURE = "parser_failure"
    EVALUATOR_FAILURE = "evaluator_failure"
    RUNTIME_FAILURE = "runtime_failure"


class AgentFailureCategory(str, Enum):
    PLANNING_FAILURE = "planning_failure"
    OBSERVATION_IGNORED = "observation_ignored"
    ENVIRONMENT_STATE_ERROR = "environment_state_error"
    RECOVERY_FAILURE = "recovery_failure"
    REPEATED_ACTION = "repeated_action"
    MAX_STEPS = "max_steps"
    INVALID_STATE_TRANSITION = "invalid_state_transition"


FAILURE_DESCRIPTIONS: dict[str, str] = {
    # Tool calling
    ToolFailureCategory.WRONG_TOOL.value: "Selected a semantically incorrect tool.",
    ToolFailureCategory.MISSED_TOOL.value: "Failed to invoke a required tool when expected.",
    ToolFailureCategory.UNNECESSARY_TOOL.value: "Invoked a tool when direct answer or no action was required.",
    ToolFailureCategory.MALFORMED_TOOL_CALL.value: "Tool call markup or JSON syntax is malformed.",
    ToolFailureCategory.MALFORMED_ARGUMENTS.value: "Tool arguments violate schema type or structure constraints.",
    ToolFailureCategory.MISSING_ARGUMENT.value: "Omitted one or more required tool arguments.",
    ToolFailureCategory.WRONG_ARGUMENT.value: "Provided an incorrect value for an argument.",
    ToolFailureCategory.HALLUCINATED_ARGUMENT.value: "Invented parameters or arguments not present in schema or prompt.",
    ToolFailureCategory.DUPLICATE_CALL.value: "Issued identical redundant tool calls without state change.",
    ToolFailureCategory.WRONG_CALL_ORDER.value: "Violated required execution order or causal dependency between calls.",
    ToolFailureCategory.FAILED_PARALLELIZATION.value: "Serialized calls that should have been parallel, or vice versa.",
    ToolFailureCategory.PREMATURE_FINAL_ANSWER.value: "Emitted a final response before receiving essential tool observations.",
    ToolFailureCategory.FAILED_CLARIFICATION.value: "Acted prematurely when a request was ambiguous or incomplete.",
    ToolFailureCategory.OVER_CLARIFICATION.value: "Requested user clarification when prompt contained sufficient information.",
    ToolFailureCategory.UNSAFE_ACTION.value: "Attempted an unsafe, unauthorized, or destructive action.",
    ToolFailureCategory.TOOL_LOOP.value: "Fell into an infinite or repetitive tool execution loop.",
    ToolFailureCategory.CONTEXT_LOSS.value: "Lost critical state or variables across multi-turn context.",
    ToolFailureCategory.PARSER_FAILURE.value: "Output could not be parsed into native or structured function call.",
    ToolFailureCategory.EVALUATOR_FAILURE.value: "Evaluator encountered an unhandled exception or malformed test case.",
    ToolFailureCategory.RUNTIME_FAILURE.value: "Model or inference backend encountered an execution runtime exception.",
    # Agent behavior
    AgentFailureCategory.PLANNING_FAILURE.value: "Constructed an invalid or contradictory action sequence.",
    AgentFailureCategory.OBSERVATION_IGNORED.value: "Ignored return value or observation from a previous tool execution.",
    AgentFailureCategory.ENVIRONMENT_STATE_ERROR.value: "Assumed an incorrect state of the external environment.",
    AgentFailureCategory.RECOVERY_FAILURE.value: "Failed to recover following an error observation from environment.",
    AgentFailureCategory.REPEATED_ACTION.value: "Repeated the exact same failing action multiple times.",
    AgentFailureCategory.MAX_STEPS.value: "Exceeded maximum allowed execution steps without finishing task.",
    AgentFailureCategory.INVALID_STATE_TRANSITION.value: "Attempted an action that is illegal from current state.",
}

# Mapping from legacy error_taxonomy.yaml codes and benchmark codes
TAXONOMY_MAP: dict[str, str] = {
    # Legacy codes from registry/error_taxonomy.yaml
    "UNDER_CALL": ToolFailureCategory.MISSED_TOOL.value,
    "OVER_CALL": ToolFailureCategory.UNNECESSARY_TOOL.value,
    "WRONG_TOOL": ToolFailureCategory.WRONG_TOOL.value,
    "INVALID_ARGUMENTS": ToolFailureCategory.MALFORMED_ARGUMENTS.value,
    "UNGROUNDED_ARGUMENTS": ToolFailureCategory.HALLUCINATED_ARGUMENT.value,
    "MISSING_CLARIFICATION": ToolFailureCategory.FAILED_CLARIFICATION.value,
    "FALSE_UNSUPPORTED": ToolFailureCategory.OVER_CLARIFICATION.value,
    "UNSUPPORTED_TOOL_CALL": ToolFailureCategory.WRONG_TOOL.value,
    "BAD_DEPENDENCY_ORDER": ToolFailureCategory.WRONG_CALL_ORDER.value,
    "BAD_PARALLELIZATION": ToolFailureCategory.FAILED_PARALLELIZATION.value,
    "OBSERVATION_IGNORED": AgentFailureCategory.OBSERVATION_IGNORED.value,
    "PREMATURE_STOP": ToolFailureCategory.PREMATURE_FINAL_ANSWER.value,
    "FAILURE_RECOVERY_ERROR": AgentFailureCategory.RECOVERY_FAILURE.value,
    "MULTI_TURN_STATE_ERROR": ToolFailureCategory.CONTEXT_LOSS.value,
    "FORMAT_ERROR": ToolFailureCategory.PARSER_FAILURE.value,
    # Short taxonomy codes E01-E20
    "E01": ToolFailureCategory.UNNECESSARY_TOOL.value,
    "E02": ToolFailureCategory.MISSED_TOOL.value,
    "E03": ToolFailureCategory.WRONG_TOOL.value,
    "E04": ToolFailureCategory.MISSING_ARGUMENT.value,
    "E05": ToolFailureCategory.HALLUCINATED_ARGUMENT.value,
    "E06": ToolFailureCategory.WRONG_ARGUMENT.value,
    "E07": ToolFailureCategory.WRONG_ARGUMENT.value,
    "E08": ToolFailureCategory.FAILED_CLARIFICATION.value,
    "E09": ToolFailureCategory.OVER_CLARIFICATION.value,
    "E10": ToolFailureCategory.MALFORMED_TOOL_CALL.value,
    "E11": ToolFailureCategory.WRONG_TOOL.value,
    "E12": ToolFailureCategory.FAILED_PARALLELIZATION.value,
    "E13": ToolFailureCategory.WRONG_CALL_ORDER.value,
    "E14": AgentFailureCategory.OBSERVATION_IGNORED.value,
    "E15": ToolFailureCategory.HALLUCINATED_ARGUMENT.value,
    "E16": ToolFailureCategory.TOOL_LOOP.value,
    "E17": ToolFailureCategory.PREMATURE_FINAL_ANSWER.value,
    "E18": ToolFailureCategory.PARSER_FAILURE.value,
    "E19": ToolFailureCategory.PARSER_FAILURE.value,
    "E20": ToolFailureCategory.CONTEXT_LOSS.value,
}


@dataclass(frozen=True)
class FailureRecord:
    category: str
    detail: str
    benchmark_code: str | None = None
    benchmark: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "description": FAILURE_DESCRIPTIONS.get(self.category, "Custom failure code"),
            "detail": self.detail,
            "benchmark_code": self.benchmark_code,
            "benchmark": self.benchmark,
        }


def normalize_failure_code(code: str, benchmark: str | None = None) -> str:
    """Map any benchmark or legacy failure code into the canonical OpenGrad failure taxonomy."""
    code_norm = code.strip().upper()
    if code_norm in TAXONOMY_MAP:
        return TAXONOMY_MAP[code_norm]

    # Check case-insensitive direct match against valid enum values
    code_lower = code.strip().lower()
    for cat in ToolFailureCategory:
        if cat.value == code_lower:
            return cat.value
    for cat_agent in AgentFailureCategory:
        if cat_agent.value == code_lower:
            return cat_agent.value

    # Benchmark-specific prefix preservation
    if benchmark:
        return f"{benchmark.lower()}:{code_lower}"
    return code_lower
