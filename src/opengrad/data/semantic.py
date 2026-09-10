"""Strict, source-agnostic validation for training tool trajectories.

The general IR validator intentionally accepts prompt-only records.  This module is
an explicit training gate: it validates the call/result graph, effective JSON
Schema, and ordering without silently repairing malformed records.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from opengrad.data.canonical import ToolConversation
from opengrad.data.schema import (
    ArgumentValidationError,
    SchemaValidationError,
    effective_schema,
    validate_arguments,
)


@dataclass(frozen=True)
class SemanticIssue:
    code: str
    message: str
    message_index: int | None = None


class SemanticValidationError(ValueError):
    def __init__(self, issues: list[SemanticIssue]) -> None:
        self.issues = issues
        super().__init__("; ".join(f"{item.code}: {item.message}" for item in issues))


def _issue(code: str, message: str, index: int | None = None) -> SemanticIssue:
    return SemanticIssue(code, message, index)


def _content_is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def validate_training_trajectory(example: ToolConversation) -> list[SemanticIssue]:
    """Return precise semantic failures for a strict complete-target SFT trajectory."""
    issues: list[SemanticIssue] = []
    if not example.messages:
        issues.append(_issue("EMPTY_CONVERSATION", "conversation has no messages"))
        return issues

    schemas: dict[str, dict[str, Any]] = {}
    for index, tool in enumerate(example.tools):
        try:
            name = tool.get("name") if isinstance(tool, dict) else None
            if not isinstance(name, str) or not name:
                issues.append(_issue("TOOL_NAME_REQUIRED", f"tool {index} has no nonempty name"))
                continue
            if name in schemas:
                issues.append(_issue("DUPLICATE_TOOL_NAME", name))
                continue
            schemas[name] = effective_schema(tool)
        except SchemaValidationError as exc:
            issues.append(_issue(exc.reason_code, str(exc), index))

    declared = set(schemas)
    calls: dict[str, tuple[str, int]] = {}
    results: dict[str, int] = {}
    # Current complete-target SFT policy is FIFO. Parallel/interleaved calls
    # need a separately versioned trajectory policy and are rejected here.
    pending: list[str] = []
    seen_assistant_target = False
    previous_role: str | None = None

    for index, message in enumerate(example.messages):
        if not isinstance(message, dict):
            issues.append(_issue("INVALID_MESSAGE", "message is not an object", index))
            continue
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            issues.append(_issue("UNSUPPORTED_ROLE", str(role), index))
            continue
        if role in {"system", "user", "assistant"} and "content" not in message:
            issues.append(_issue("MESSAGE_CONTENT_REQUIRED", role, index))
        if role in {"system", "user"} and _content_is_empty(message.get("content")):
            issues.append(_issue("EMPTY_MESSAGE", role, index))
        if role == "assistant":
            calls_value = message.get("tool_calls", [])
            if not isinstance(calls_value, list):
                issues.append(_issue("TOOL_CALLS_NOT_LIST", "tool_calls must be a list", index))
                calls_value = []
            has_calls = bool(calls_value)
            if pending:
                issues.append(
                    _issue(
                        "INVALID_MESSAGE_SEQUENCE",
                        "assistant message while tool result is pending; FIFO complete-target policy",
                        index,
                    )
                )
            if not has_calls and _content_is_empty(message.get("content")):
                issues.append(_issue("EMPTY_ASSISTANT_OUTPUT", "assistant target is empty", index))
            if has_calls or not _content_is_empty(message.get("content")):
                seen_assistant_target = True
            if has_calls:
                for call in calls_value:
                    if not isinstance(call, dict):
                        issues.append(_issue("INVALID_TOOL_CALL", "call is not an object", index))
                        continue
                    name = call.get("name")
                    if not isinstance(name, str) or not name:
                        issues.append(_issue("TOOL_NAME_REQUIRED", "call name is empty", index))
                        continue
                    if name not in declared:
                        issues.append(_issue("UNDECLARED_TOOL", name, index))
                    call_id = call.get("id", call.get("call_id"))
                    if not isinstance(call_id, str) or not call_id.strip():
                        issues.append(_issue("INVALID_TOOL_CALL_ID", "call ID is required", index))
                        continue
                    if call_id in calls:
                        issues.append(_issue("DUPLICATE_TOOL_CALL_ID", call_id, index))
                    else:
                        calls[call_id] = (name, index)
                        pending.append(call_id)
                    arguments = call.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(
                                arguments,
                                parse_constant=lambda value: (_ for _ in ()).throw(
                                    ValueError(value)
                                ),
                            )
                        except (ValueError, json.JSONDecodeError):
                            issues.append(_issue("MALFORMED_TOOL_ARGUMENTS", call_id, index))
                            continue
                    if name in schemas:
                        try:
                            validate_arguments(schemas[name], arguments)
                        except ArgumentValidationError as exc:
                            issues.append(_issue(exc.reason_code, str(exc), index))

        elif role == "tool":
            call_id = message.get("tool_call_id", message.get("call_id"))
            if not isinstance(call_id, str) or not call_id.strip():
                issues.append(_issue("INVALID_TOOL_CALL_ID", "result ID is required", index))
            elif call_id not in calls:
                issues.append(_issue("ORPHAN_TOOL_RESULT", call_id, index))
            elif call_id in results:
                issues.append(_issue("DUPLICATE_TOOL_RESULT", call_id, index))
            else:
                if not pending or pending[0] != call_id:
                    issues.append(
                        _issue(
                            "INVALID_MESSAGE_SEQUENCE",
                            "tool result violates FIFO call order",
                            index,
                        )
                    )
                else:
                    results[call_id] = index
                    pending.pop(0)
            tool_name = message.get("name")
            if tool_name is not None and call_id in calls and tool_name != calls[call_id][0]:
                issues.append(_issue("TOOL_RESULT_NAME_MISMATCH", str(call_id), index))
            if previous_role not in {"assistant", "tool"}:
                issues.append(
                    _issue("INVALID_MESSAGE_SEQUENCE", "tool result has no preceding call", index)
                )
        elif pending:
            issues.append(
                _issue(
                    "INVALID_MESSAGE_SEQUENCE", "message occurs before required tool result", index
                )
            )
        previous_role = role

    for call_id in sorted(calls):
        if call_id not in results:
            issues.append(_issue("MISSING_TOOL_RESULT", call_id))
    if not seen_assistant_target:
        issues.append(_issue("NO_USEFUL_TARGET", "conversation has no assistant target"))
    return issues


def validate_training_trajectory_or_raise(example: ToolConversation) -> None:
    issues = validate_training_trajectory(example)
    if issues:
        raise SemanticValidationError(issues)


def audit_records(records: list[ToolConversation]) -> dict[str, Any]:
    """Audit records without dropping them; output is stable and machine-readable."""
    reason_counts: Counter[str] = Counter()
    invalid = 0
    for example in records:
        issues = validate_training_trajectory(example)
        if issues:
            invalid += 1
            reason_counts.update(issue.code for issue in issues)
    return {
        "records": len(records),
        "valid": len(records) - invalid,
        "invalid": invalid,
        "reason_counts": dict(sorted(reason_counts.items())),
    }
