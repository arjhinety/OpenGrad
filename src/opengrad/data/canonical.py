from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from opengrad.data.behavior import validate_behavior

SCHEMA_VERSION = "tool_use_ir_v1"
ROLES = {"system", "user", "assistant", "tool"}


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def semantic_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ToolConversation:
    id: str
    source: str
    tools: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def schema_version(self) -> str:
        return SCHEMA_VERSION

    def validate(self) -> None:
        if not self.id or not self.source:
            raise ValueError("id and source are required")
        if not isinstance(self.tools, list) or not isinstance(self.messages, list):
            raise TypeError("tools and messages must be lists")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be an object")
        if not self.metadata.get("split"):
            raise ValueError("metadata.split is required for provenance")
        behavior = self.metadata.get("behavior")
        if behavior is not None:
            if not isinstance(behavior, dict):
                raise TypeError("metadata.behavior must be an object")
            validate_behavior(
                behavior.get("decision", ""),
                behavior.get("capabilities", []),
                behavior.get("confidence", "known"),
            )
        names: set[str] = set()
        for tool in self.tools:
            if (
                not isinstance(tool, dict)
                or not isinstance(tool.get("name"), str)
                or not tool["name"]
            ):
                raise TypeError("each tool needs a name")
            if tool["name"] in names:
                raise ValueError("duplicate tool name")
            names.add(tool["name"])
        seen_calls: set[str] = set()
        known_calls: set[str] = set()
        for index, message in enumerate(self.messages):
            if not isinstance(message, dict) or message.get("role") not in ROLES:
                raise ValueError(f"message {index} has invalid role")
            role = message["role"]
            if role in {"system", "user", "assistant"} and "content" not in message:
                raise ValueError(f"message {index} needs content")
            if role == "assistant":
                calls = message.get("tool_calls", [])
                if not isinstance(calls, list):
                    raise ValueError("tool_calls must be a list")
                for call in calls:
                    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
                        raise TypeError("tool call needs name")
                    if call["name"] not in names:
                        raise ValueError(f"unknown tool: {call['name']}")
                    if not isinstance(call.get("arguments", {}), dict):
                        raise TypeError("tool arguments must be an object")
                    call_id = call.get("id", call.get("call_id"))
                    if call_id:
                        if call_id in seen_calls:
                            raise ValueError("duplicate tool call id")
                        seen_calls.add(str(call_id))
                        known_calls.add(str(call_id))
            if role == "tool":
                call_id = message.get("tool_call_id")
                if not isinstance(call_id, str):
                    raise ValueError("tool message needs tool_call_id")
                if known_calls and call_id not in known_calls:
                    raise ValueError(f"tool result has unknown call id: {call_id}")

    def validate_training_semantics(self) -> None:
        """Validate the stricter trajectory contract used at the training boundary."""
        try:
            self.validate()
        except ValueError as exc:
            if str(exc).startswith("unknown tool:"):
                raise ValueError(
                    f"SEM_UNDECLARED_TOOL: unknown tool: {str(exc).split(':', 1)[1].strip()}"
                ) from exc
            raise
        from opengrad.data.semantic import validate_training_trajectory

        issues = validate_training_trajectory(self)
        if issues:
            issue = issues[0]
            aliases = {
                "INVALID_TOOL_CALL_ID": "SEM_CALL_ID_REQUIRED",
                "UNDECLARED_TOOL": "SEM_UNDECLARED_TOOL",
                "DUPLICATE_TOOL_CALL_ID": "SEM_DUPLICATE_CALL_ID",
                "ORPHAN_TOOL_RESULT": "SEM_ORPHAN_RESULT",
                "MISSING_TOOL_RESULT": "SEM_UNRESOLVED_CALL",
                "INVALID_MESSAGE_SEQUENCE": "SEM_RESULT_ORDER",
            }
            code = aliases.get(
                issue.code,
                "SEM_ARGUMENT_INVALID" if issue.code.startswith("ARG_") else f"SEM_{issue.code}",
            )
            detail = (
                f"unknown tool: {issue.message}"
                if issue.code == "UNDECLARED_TOOL"
                else issue.message
            )
            raise ValueError(f"{code}: {detail}")


@dataclass(frozen=True)
class CanonicalSFTExample:
    conversation: ToolConversation

    def validate(self) -> None:
        if self.conversation.metadata.get("split") in {
            "preference",
            "mcq",
            "mcq_test",
            "llm_judge",
            "llm_judge_test",
        }:
            raise ValueError("non-SFT split cannot be used as SFT")
        self.conversation.validate_training_semantics()


@dataclass(frozen=True)
class CanonicalPreferenceExample:
    example_id: str
    source: dict[str, Any]
    tools: list[dict[str, Any]]
    context: list[dict[str, Any]]
    chosen: dict[str, Any]
    rejected: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.example_id or not isinstance(self.context, list):
            raise ValueError("preference identity and context are required")
        if self.chosen == self.rejected:
            raise ValueError("chosen and rejected responses must differ")
        for response in (self.chosen, self.rejected):
            if response.get("role") != "assistant" or "content" not in response:
                raise ValueError("preference responses must be assistant messages")


@dataclass(frozen=True)
class CanonicalEvaluationExample:
    example_id: str
    source: dict[str, Any]
    question: str
    tools: list[dict[str, Any]]
    expected_decision: str
    candidates: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.example_id or not self.question:
            raise ValueError("evaluation identity and question are required")
        if self.metadata.get("eligibility") != "evaluation_only":
            raise ValueError("evaluation example must be evaluation_only")


def canonical_dict(c: ToolConversation) -> dict[str, Any]:
    c.validate()
    return {
        "schema_version": SCHEMA_VERSION,
        "id": c.id,
        "source": c.source,
        "tools": c.tools,
        "messages": c.messages,
        "metadata": c.metadata,
        "canonical_hash": semantic_hash({"tools": c.tools, "messages": c.messages}),
    }
