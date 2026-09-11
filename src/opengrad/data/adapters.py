from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from opengrad.data.canonical import ToolConversation
from opengrad.data.schema import normalize_tool
from opengrad.data.supervision import (
    SupervisionAssignment,
    SupervisionKind,
    supervision_block,
)
from opengrad.data.xlam_types import normalize_xlam_tools

# Per-row adapter version written into canonical metadata. It feeds `canonical_hash`, so changing
# it changes every fingerprint derived from re-materialized rows. The per-artifact value lives in
# each materialization manifest and is a separate field (see the v2 dataset card, which documents
# that the two differ).
ADAPTER_VERSION = "1.0.0"


def _json(value: Any, field: str) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"INVALID_{field.upper()}_JSON") from exc
    return value


def _tool(value: Any) -> list[dict[str, Any]]:
    value = _json(value, "tools")
    if value is None:
        return []
    if isinstance(value, dict):
        value = value.get("tools", value.get("functions", [value]))
    if not isinstance(value, list):
        raise TypeError("tools must be a list")
    result = []
    seen: dict[str, dict[str, Any]] = {}
    for item in value:
        if isinstance(item, str):
            item = _json(item, "tool")
        if not isinstance(item, dict):
            raise TypeError("tool definition must be an object")
        if "function" in item and isinstance(item["function"], dict):
            item = item["function"]
        normalized = dict(item)
        name = normalized.get("name")
        if (
            isinstance(name, str)
            and name in seen
            and json.dumps(seen[name], sort_keys=True, ensure_ascii=False)
            == json.dumps(normalized, sort_keys=True, ensure_ascii=False)
        ):
            continue
        if isinstance(name, str):
            seen[name] = normalized
        result.append(normalized)
    return result


@lru_cache(maxsize=4096)
def _embedded_tools(text: str) -> list[dict[str, Any]]:
    """Extract tool JSON only from known catalogue regions."""
    decoder = json.JSONDecoder()
    blocks = re.findall(r"<tools>(.*?)</tools>", text, re.DOTALL)
    if blocks:
        values: list[dict[str, Any]] = []
        for block in blocks:
            for line in block.splitlines():
                try:
                    value = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict) and isinstance(value.get("name"), str):
                    values.append(value)
        return _tool(values)
    # ToolACE embeds a complete JSON array in the system prompt.
    for match in re.finditer(r"\[", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if (
            isinstance(value, list)
            and value
            and all(isinstance(item, dict) and isinstance(item.get("name"), str) for item in value)
        ):
            return _tool(value)
    # Glaive embeds one JSON object after its prose header.
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("name"), str):
            return [value]
    return []


def _base(
    record: dict[str, Any],
    source: str,
    split: str,
    tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    *,
    adapter: str,
    status: str = "VALID",
    supervision_kind: SupervisionKind = SupervisionKind.COMPLETE_TRAJECTORY,
    supervision_assignment: SupervisionAssignment = SupervisionAssignment.SOURCE_ADAPTER,
    supervision_note: str = "",
    **extra: Any,
) -> ToolConversation:
    tools = [normalize_tool(tool) for tool in tools]
    raw_hash = hashlib.sha256(
        json.dumps(record, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()
    upstream_id = str(
        record.get("id", record.get("example_id", record.get("uid", "og_" + raw_hash[:16])))
    )
    metadata: dict[str, Any] = {
        "split": split,
        "source": {
            "dataset_id": source,
            "upstream_id": upstream_id,
            "revision": record.get("source_revision"),
            "original_split": split,
        },
        "source_revision": record.get("source_revision"),
        "raw_record_hash": raw_hash,
        "adapter": adapter,
        "adapter_version": ADAPTER_VERSION,
        "parse_status": status,
        "contamination_status": record.get("contamination_status", "UNASSESSED"),
        "source_fields": sorted(record),
        "source_features": extra,
        "tool_context": {"tool_count": len(tools)},
        # What this record supervises, decided by the adapter from the upstream shape and
        # carried in the canonical record independently of the source name.
        "supervision": supervision_block(
            supervision_kind,
            assignment=supervision_assignment,
            adapter=adapter,
            adapter_version=ADAPTER_VERSION,
            note=supervision_note,
        ),
    }
    source_metadata = record.get("metadata")
    if isinstance(source_metadata, dict) and "eligibility" in source_metadata:
        # Preserve eligibility provenance at the canonical boundary; training
        # selection must not depend on a split-name heuristic alone.
        metadata["eligibility"] = source_metadata["eligibility"]
    if any(m.get("role") == "tool" for m in messages):
        metadata["behavior"] = {
            "decision": "CALL",
            "capabilities": ["consume_tool_result"],
            "confidence": "derived",
        }
    elif any(m.get("tool_calls") for m in messages):
        metadata["behavior"] = {
            "decision": "CALL",
            "capabilities": ["single_tool_selection", "argument_grounding"],
            "confidence": "derived",
        }
    else:
        metadata["behavior"] = {"decision": "ANSWER", "capabilities": [], "confidence": "derived"}
    return ToolConversation(upstream_id, source, tools, messages, metadata)


def _generic(record: dict[str, Any], source: str, split: str) -> ToolConversation:
    messages = record.get("messages", record.get("conversation"))
    if not isinstance(messages, list):
        raise TypeError("missing messages/conversation list")
    return _base(
        record,
        source,
        split,
        _tool(record.get("tools", record.get("functions", []))),
        list(messages),
        adapter="generic",
        # The source-agnostic messages path. Its callers supply whole conversations whose calls
        # are answered, so it declares the complete-trajectory contract rather than inheriting it.
        supervision_kind=SupervisionKind.COMPLETE_TRAJECTORY,
    )


def adapt(record: dict[str, Any], source: str, split: str = "fixture") -> ToolConversation:
    if not isinstance(record, dict) or not record.get("id"):
        raise ValueError("record id is required")
    c = _generic(record, source, split)
    c.validate()
    return c


def adapt_xlam(record: dict[str, Any], split: str = "train") -> ToolConversation:
    if "messages" in record and "query" not in record:
        return adapt(record, "xlam-function-calling-60k", split)
    if "query" not in record or not isinstance(record["query"], str):
        raise TypeError("missing query")
    tools = _tool(record.get("tools", []))
    # xLAM stores `parameters` as a property-definition map, not as JSON Schema. Wrapping it
    # is a source-contract transformation, so it happens here at the source boundary rather
    # than in the generic schema layer, which must keep rejecting ambiguous bare dicts.
    tools, repair = normalize_xlam_tools(tools)
    answers = _json(record.get("answers", []), "answers")
    if isinstance(answers, dict):
        answers = [answers]
    if not isinstance(answers, list):
        raise TypeError("answers must be a list")
    calls = []
    final = None
    for i, answer in enumerate(answers):
        if not isinstance(answer, dict):
            raise TypeError("answer must be an object")
        name = answer.get("name", answer.get("function", answer.get("tool_name")))
        if name:
            args = answer.get("arguments", answer.get("parameters", {}))
            args = _json(args, "arguments")
            if not isinstance(args, dict):
                raise TypeError("xLAM arguments must be an object")
            calls.append({"id": f"call_{i:04d}", "name": name, "arguments": args})
        elif answer.get("content") is not None:
            final = str(answer["content"])
    messages: list[dict[str, Any]] = [{"role": "user", "content": record["query"]}]
    messages.append({"role": "assistant", "content": final, "tool_calls": calls})
    c = _base(
        record,
        "xlam-function-calling-60k",
        split,
        tools,
        messages,
        adapter="xlam_function_calling_60k_v2",
        source_format="query/tools/answers, parameters as a property-definition map",
        # xLAM/APIGen is a next-tool-call prediction corpus: `query + tools -> answers`, where an
        # answer names the call to make. It structurally contains no tool-result turn, so the
        # terminal call is the supervised target rather than an unresolved trajectory. Verified
        # against the upstream dataset card at the pinned revision and against the retained
        # derivative: every gold call's argument keys are parameter names, and 0 of 59,370
        # records contain a tool result.
        supervision_kind=SupervisionKind.CALL_PREDICTION,
        supervision_assignment=SupervisionAssignment.UPSTREAM_DECLARED,
        supervision_note=(
            "upstream card documents answers as the call to make; the corpus has no tool-result turn"
        ),
        xlam_schema_normalization=repair.as_dict(),
    )
    c.validate()
    return c


def _when_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    raw = record.get("messages", record.get("conversation", record.get("text")))
    if isinstance(raw, list):
        return list(raw)
    if not isinstance(raw, str):
        raise TypeError("When2Call conversation is missing")
    out = []
    for part in re.split(r"(?=<TOOLCALL>|</TOOLCALL>)", raw):
        if not part.strip():
            continue
        if "<TOOLCALL>" in part:
            body = part.split("<TOOLCALL>", 1)[1].split("</TOOLCALL>", 1)[0].strip()
            value = _json(body, "toolcall")
            if isinstance(value, dict):
                out.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_0000",
                                "name": value.get("name", value.get("function", "")),
                                "arguments": value.get("arguments", {}),
                            }
                        ],
                    }
                )
        else:
            out.append({"role": "assistant", "content": part.strip()})
    return out


def adapt_when2call(record: dict[str, Any], split: str = "train_sft") -> ToolConversation:
    if split in {"mcq", "test", "llm_judge", "mcq_test", "llm_judge_test"}:
        raise ValueError("evaluation rows cannot be normalized as training conversations")
    if "messages" in record and "conversation" not in record and record.get("id"):
        return adapt(record, "when2call", split)
    messages = _when_messages(record)
    if "prompt" in record and not any(m.get("role") == "user" for m in messages):
        messages.insert(0, {"role": "user", "content": str(record["prompt"])})
    tools = _tool(record.get("tools", []))
    c = _base(
        record,
        "when2call",
        split,
        tools,
        messages,
        adapter="when2call_v1",
        source_format="<TOOLCALL>",
        # Measured: 0 of 14,829 canonical When2Call records carry a tool result or a structured
        # tool call; every one ends in an assistant response. Its supervised behaviour is the
        # call/answer/clarify/unsupported *decision* expressed as prose, which the complete
        # trajectory contract validates as-is (there are no calls to resolve).
        supervision_kind=SupervisionKind.COMPLETE_TRAJECTORY,
    )
    c.validate()
    return c


def _toolace_call(text: str) -> dict[str, Any]:
    calls = _toolace_calls(text)
    if len(calls) != 1:
        raise ValueError("ambiguous ToolACE call marker")
    return calls[0]


def _toolace_calls(text: str) -> list[dict[str, Any]]:
    """Parse ToolACE's Python-like calls without evaluating source text."""
    match = re.search(r"\[(?:Function\s+)?(.*)\]", text, re.DOTALL)
    if not match:
        raise ValueError("ToolACE call marker not found")
    body = match.group(1).strip()
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    seen_open = False
    for index, char in enumerate(body):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
            if char == "(":
                seen_open = True
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0 and seen_open:
            parts.append(body[start:index].strip())
            start = index + 1
    parts.append(body[start:].strip())
    result: list[dict[str, Any]] = []
    try:
        for part in parts:
            opening = part.rfind("(")
            if opening <= 0 or not part.endswith(")"):
                raise ValueError("malformed call")
            name = part[:opening].strip()
            args_text = part[opening + 1 : -1].strip()
            arguments: dict[str, Any] = {}
            arg_parts: list[str] = []
            arg_start = 0
            arg_depth = 0
            arg_quote: str | None = None
            arg_escape = False
            for arg_index, arg_char in enumerate(args_text):
                if arg_quote:
                    if arg_escape:
                        arg_escape = False
                    elif arg_char == "\\":
                        arg_escape = True
                    elif arg_char == arg_quote:
                        arg_quote = None
                elif arg_char in {"'", '"'}:
                    arg_quote = arg_char
                elif arg_char in "[{(":
                    arg_depth += 1
                elif arg_char in "]})":
                    arg_depth -= 1
                elif arg_char == "," and arg_depth == 0:
                    arg_parts.append(args_text[arg_start:arg_index].strip())
                    arg_start = arg_index + 1
            if args_text:
                arg_parts.append(args_text[arg_start:].strip())
            for arg_part in arg_parts:
                if "=" not in arg_part:
                    raise ValueError("positional arguments are unsupported")
                key, raw_value = (piece.strip() for piece in arg_part.split("=", 1))
                if not re.fullmatch(r"[A-Za-z_$][\w$.-]*", key):
                    raise ValueError("invalid argument name")
                try:
                    arguments[key] = ast.literal_eval(raw_value)
                except (ValueError, SyntaxError):
                    arguments[key] = json.loads(raw_value)
            result.append({"id": "", "name": name, "arguments": arguments})
    except (SyntaxError, ValueError, TypeError, MemoryError) as exc:
        raise ValueError("INVALID_ARGUMENT_SYNTAX") from exc
    if not result:
        raise ValueError("empty ToolACE call list")
    return result


def adapt_toolace(record: dict[str, Any], split: str = "train") -> ToolConversation:
    if "messages" in record and "conversations" not in record:
        return adapt(record, "toolace", split)
    conv = record.get("conversations")
    if conv is None and "from" in record and "value" in record:
        conv = [{"from": record["from"], "value": record["value"]}]
    if not isinstance(conv, list):
        raise TypeError("ToolACE conversations must be a list")
    tools = _tool(record.get("tools", []))
    system = record.get("system", "")
    if isinstance(system, str) and not tools:
        tools = list(_embedded_tools(system))
        for block in re.findall(r"<tool>(.*?)</tool>", system, re.DOTALL):
            try:
                tools.extend(_tool(block))
            except (TypeError, ValueError):
                pass
    messages = []
    last_call_id: str | None = None
    next_call_id = 0
    for item in conv:
        if not isinstance(item, dict):
            raise TypeError("ToolACE conversation item must be an object")
        role = {
            "human": "user",
            "user": "user",
            "gpt": "assistant",
            "assistant": "assistant",
            "tool": "tool",
        }.get(str(item.get("from", item.get("role"))))
        if not role:
            raise ValueError("unknown ToolACE role")
        content = item.get("value", item.get("content", ""))
        msg = {"role": role, "content": content}
        if (
            role == "assistant"
            and isinstance(content, str)
            and content.lstrip().startswith("[")
            and "(" in content
        ):
            calls = _toolace_calls(content)
            for call in calls:
                call["id"] = f"call_{next_call_id:04d}"
                next_call_id += 1
            last_call_id = calls[-1]["id"]
            msg["tool_calls"] = calls
            msg["content"] = content.split("[Function", 1)[0].strip() or None
        if role == "tool" and last_call_id:
            msg["tool_call_id"] = last_call_id
        messages.append(msg)
    if system:
        messages.insert(0, {"role": "system", "content": system})
    c = _base(
        record,
        "toolace",
        split,
        tools,
        messages,
        adapter="toolace_v1",
        source_format="system/conversations from/value",
        # Declared COMPLETE_TRAJECTORY, which is the status quo and is deliberately unchanged.
        # Measured: 9,187 records carry a tool call and only 793 carry a tool result, so 8,394
        # end on an unresolved call. Whether those are truncated trajectories or intended
        # next-call supervision is NOT established by the available bytes, and reclassifying
        # them on structure alone would be guessing. Recorded as an open question in
        # reports/SUPERVISION_CONTRACT_REPORT.md; the yield report counts them explicitly.
        supervision_kind=SupervisionKind.COMPLETE_TRAJECTORY,
    )
    c.validate()
    return c


def _tagged_messages(
    record: dict[str, Any],
    source: str,
    split: str,
    adapter: str,
    *,
    supervision_kind: SupervisionKind = SupervisionKind.COMPLETE_TRAJECTORY,
) -> ToolConversation:
    raw = record.get("messages", record.get("chat"))
    if not isinstance(raw, list):
        raise TypeError("message list is required")
    tools = _tool(record.get("tools", []))
    messages = []
    next_call_id = 0
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError("message must be an object")
        role = item.get("role", item.get("from"))
        text = item.get("content", item.get("value", ""))
        msg = {"role": role, "content": text}
        if role == "assistant" and isinstance(text, str):
            calls = []
            for body in re.findall(r"<call>(.*?)</call>", text, re.DOTALL):
                parsed = _json(body.strip(), "call")
                if isinstance(parsed, list):
                    calls.extend(parsed)
                elif isinstance(parsed, dict):
                    calls.append(parsed)
                else:
                    raise TypeError("call payload must be an object or array")
            if any(
                not isinstance(call, dict) or not isinstance(call.get("name"), str)
                for call in calls
            ):
                raise ValueError("call payload needs a name")
            if calls:
                msg["tool_calls"] = [
                    {
                        "id": f"call_{next_call_id + i:04d}",
                        "name": x["name"],
                        "arguments": x.get("arguments", {}),
                    }
                    for i, x in enumerate(calls)
                ]
                next_call_id += len(calls)
                msg["content"] = (
                    re.sub(r"<call>.*?</call>", "", text, flags=re.DOTALL).strip() or None
                )
        messages.append(msg)
    call_ids = [
        str(call.get("id"))
        for message in messages
        if message.get("role") == "assistant"
        for call in (message.get("tool_calls") or [])
        if call.get("id")
    ]
    observation_index = 0
    for message in messages:
        if message.get("role") == "tool" and "tool_call_id" not in message:
            if observation_index >= len(call_ids):
                raise ValueError("tool result has no preceding call")
            message["tool_call_id"] = call_ids[observation_index]
            observation_index += 1
    if not tools:
        system_text = next(
            (str(m.get("content", "")) for m in messages if m.get("role") == "system"), ""
        )
        tools = _embedded_tools(system_text)
    c = _base(
        record,
        source,
        split,
        tools,
        messages,
        adapter=adapter,
        source_format="messages",
        # Shared by BUTTON and Glaive.
        #
        # Measured, and not the same for both. Glaive: 0 of 99,794 records have an unresolved
        # call, so every call is answered and the trajectory is complete. BUTTON: 5,582 of 7,941
        # are fully resolved, but 2,359 emit calls that no tool message ever answers -- an
        # assistant turn asks for three tools and the conversation returns results for only some
        # of them. Those are invalid under *both* contracts (the terminal turn is prose, so there
        # is no call-prediction target either), which is correct: the trajectory is genuinely
        # incomplete and no contract should rescue it. They stay quarantined.
        supervision_kind=supervision_kind,
    )
    c.validate()
    return c


def adapt_button(record: dict[str, Any], split: str = "train") -> ToolConversation:
    if (
        "messages" in record
        and "chat" not in record
        and not any("<call>" in str(message.get("content", "")) for message in record["messages"])
    ):
        return adapt(record, "button", split)
    c = _tagged_messages(record, "button", split, "button_instruct_v1")
    c.metadata["source_features"]["source_reasoning_present"] = any(
        "<think>" in str(m.get("content", "")) for m in c.messages
    )
    c.metadata["source_features"]["source_reasoning_policy"] = "QUARANTINED"
    return c


def adapt_looptool(record: dict[str, Any], split: str = "train") -> ToolConversation:
    if "messages" in record and "input" not in record:
        return adapt(record, "looptool-23k", split)
    instruction = record.get("instruction", "")
    history = record.get("input", record.get("dialogue", []))
    output = record.get("output")
    if not isinstance(history, (list, str)):
        raise TypeError("LoopTool input must be dialogue history")
    if isinstance(history, list):
        messages = list(history)
    else:
        chunks = re.findall(r"<\|im_start\|>(\w+)\s*(.*?)<\|im_end\|>", history, re.DOTALL)
        messages = [
            {"role": "tool" if "<tool_response>" in text else role, "content": text.strip()}
            for role, text in chunks
        ] or [{"role": "user", "content": history}]
    if output is not None:
        output_text = str(output)
        calls: list[dict[str, Any]] = []
        for body in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", output_text, re.DOTALL):
            value = _json(body, "tool_call")
            if isinstance(value, dict) and value.get("name"):
                calls.append(
                    {
                        "id": f"call_{len(calls):04d}",
                        "name": value["name"],
                        "arguments": _json(value.get("arguments", {}), "arguments"),
                    }
                )
        if not calls and "[" in output_text and "]" in output_text:
            calls = _toolace_calls(output_text)
            for index, call in enumerate(calls):
                call["id"] = f"call_{index:04d}"
        message: dict[str, Any] = {"role": "assistant", "content": output_text}
        if calls:
            message["tool_calls"] = calls
            message["content"] = (
                re.sub(r"<tool_call>.*?</tool_call>", "", output_text, flags=re.DOTALL).strip()
                or None
            )
        messages.append(message)
    tools = _tool(record.get("tools", []))
    if not tools and isinstance(instruction, str):
        tools = _embedded_tools(instruction)
    call_ids = [
        call["id"]
        for message in messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls", [])
    ]
    for index, message in enumerate(messages):
        if message.get("role") == "tool" and call_ids and "tool_call_id" not in message:
            message["tool_call_id"] = call_ids[min(index, len(call_ids) - 1)]
    synthetic = {
        "id": record.get("id", record.get("example_id", "unknown")),
        "source_revision": record.get("source_revision"),
    }
    c = _base(
        {**record, **synthetic},
        "looptool-23k",
        split,
        tools,
        messages,
        adapter="looptool_23k_v1",
        source_format="instruction/input/output",
        derived_from="ToolACE (reported upstream lineage)",
        # Same open question as ToolACE, and the same decision not to guess: 20,192 of 20,827
        # records carry a call, 14,448 carry a result, and 20,192 end on an assistant call, so
        # the source contains both shapes. Left as COMPLETE_TRAJECTORY — the stricter contract —
        # until upstream evidence establishes the intent.
        supervision_kind=SupervisionKind.COMPLETE_TRAJECTORY,
    )
    c.metadata["system_instruction"] = instruction
    c.validate()
    return c


def adapt_when2call_preference(record: dict[str, Any], split: str = "train_pref") -> Any:
    from opengrad.data.canonical import CanonicalPreferenceExample

    chosen = record.get("chosen_response")
    rejected = record.get("rejected_response")
    if not isinstance(chosen, dict) or not isinstance(rejected, dict):
        raise TypeError("When2Call preference responses are required")
    result = CanonicalPreferenceExample(
        str(record.get("uuid", record.get("id", "unknown"))),
        {"dataset_id": "when2call", "split": split, "revision": record.get("source_revision")},
        _tool(record.get("tools", [])),
        list(record.get("messages", [])),
        chosen,
        rejected,
        {"eligibility": "preference_only", "source_fields": sorted(record)},
    )
    result.validate()
    return result


def adapt_when2call_evaluation(record: dict[str, Any], split: str = "mcq") -> Any:
    from opengrad.data.canonical import CanonicalEvaluationExample

    result = CanonicalEvaluationExample(
        str(record.get("uuid", record.get("id", "unknown"))),
        {"dataset_id": "when2call", "split": split, "revision": record.get("source_revision")},
        str(record.get("question", "")),
        _tool(record.get("tools", [])),
        str(record.get("correct_answer", "UNKNOWN")),
        record.get("answers", {}) if isinstance(record.get("answers", {}), dict) else {},
        {"eligibility": "evaluation_only", "source_fields": sorted(record)},
    )
    result.validate()
    return result


def adapt_glaive(record: dict[str, Any], split: str = "train") -> ToolConversation:
    if isinstance(record.get("messages"), list) and "chat" not in record:
        return adapt(record, "glaive-function-calling-v2", split)
    chat = record.get("chat", record.get("messages"))
    system = record.get("system", "")
    if not isinstance(chat, str):
        return _tagged_messages(record, "glaive-function-calling-v2", split, "glaive_v2")
    parts = re.split(r"(?=USER:|ASSISTANT:|FUNCTION RESPONSE:)", chat)
    messages = []
    pending = 0
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("USER:"):
            messages.append({"role": "user", "content": part[5:].strip()})
        elif part.startswith("ASSISTANT:"):
            text = part[9:].strip()
            calls = []
            for body in re.findall(r"<functioncall>\s*(.*?)\s*</functioncall>", text, re.DOTALL):
                value = _json(body, "functioncall")
                if not isinstance(value, dict) or not value.get("name"):
                    raise ValueError("malformed Glaive function call")
                args = value.get("arguments", {})
                args = _json(args, "arguments")
                if not isinstance(args, dict):
                    raise TypeError("Glaive arguments must be an object")
                calls.append(
                    {"id": f"call_{pending:04d}", "name": value["name"], "arguments": args}
                )
                pending += 1
            messages.append(
                {
                    "role": "assistant",
                    "content": re.sub(
                        r"<functioncall>.*?</functioncall>", "", text, flags=re.DOTALL
                    ).strip()
                    or None,
                    "tool_calls": calls,
                }
            )
        elif part.startswith("FUNCTION RESPONSE:"):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{max(0, pending - 1):04d}",
                    "content": part[18:].strip(),
                }
            )
    c = _base(
        record,
        "glaive-function-calling-v2",
        split,
        _tool(record.get("tools", [])) or _embedded_tools(str(system)),
        messages,
        adapter="glaive_function_calling_v2_v1",
        source_format="system/chat delimiters",
    )
    c.metadata["system"] = system
    c.validate()
    return c


GLAIVE_CALL_MARKER = "<functioncall>"
GLAIVE_CALL_TERMINATOR = "</functioncall>"
GLAIVE_TEXT_ARTIFACTS = ("<|endoftext|>", "<|endoftext|>\n")
# In this revision the call body is a JSON-shaped object whose `arguments` value is a
# single-quoted string holding more JSON. It is valid neither as JSON nor as a Python literal
# (`true`/`false` are lower case), so it is read by converting the single-quoted strings to JSON
# strings and then parsing normally.
GLAIVE_SINGLE_QUOTED = re.compile(r"'([^']*)'")


def _glaive_balanced_body(text: str, start: int) -> tuple[str, int] | None:
    """Read the brace-balanced object starting at ``start``, respecting both quote styles.

    The block has no closing tag to stop at, so its extent has to come from the structure of the
    value itself. Brace counting that ignores quoted text is what keeps a brace inside an
    argument value from ending the object early.
    """
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    quote: str | None = None
    index = start
    while index < len(text):
        char = text[index]
        if quote is not None:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = None
        elif char in {'"', "'"}:
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1], index + 1
        index += 1
    return None


def parse_glaive_body(body: str) -> Any:
    """Parse a Glaive call body, accepting this revision's single-quoted string values."""
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    # Convert `'...'` to a JSON string literal. The captured text is re-encoded with json.dumps,
    # so any double quotes inside it are escaped rather than terminating the string.
    repaired = GLAIVE_SINGLE_QUOTED.sub(lambda match: json.dumps(match.group(1)), body)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed Glaive function call: {exc}") from exc


def extract_glaive_calls(
    text: str, pending: int
) -> tuple[list[dict[str, Any]], int, list[tuple[int, int]]]:
    """Extract Glaive function calls, tolerating an unterminated block.

    The ``v1`` adapter requires ``</functioncall>`` to close the block, but in this revision of
    the upstream data the closing tag is absent from every block: 67,481 assistant turns contain
    an opening ``<functioncall>`` and none of them contain a closing one. ``v1`` therefore parses
    zero calls, leaves the raw marker in the assistant content, and still emits a ``tool`` message
    with a synthesised id -- producing a trajectory that the canonical validator correctly rejects
    as an orphaned tool result. That one interaction cost 51,034 training records.

    The body is also not plain JSON: ``arguments`` is a single-quoted string containing more
    JSON, and booleans inside it are lower case, so it is valid neither as JSON nor as a Python
    literal. :func:`parse_glaive_body` converts those strings and then parses normally.

    Together those two changes recover 66,467 of the 67,481 unparsed turns (98.5%) across the
    released corpus, and 49,846 of the 50,851 affected records. Whether a recovered record then
    passes the remaining schema and argument gates is a separate matter that only a rebuild can
    measure; this function reports only that the call was read.

    Returns the calls, the updated call counter, and the character spans of the markers so the
    caller can strip them from the assistant content.
    """
    decoder = json.JSONDecoder()
    calls: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = text.find(GLAIVE_CALL_MARKER, cursor)
        if start == -1:
            break
        body_start = start + len(GLAIVE_CALL_MARKER)
        position = body_start
        while position < len(text) and text[position].isspace():
            position += 1
        balanced = _glaive_balanced_body(text, position)
        if balanced is not None:
            body, end = balanced
            value = parse_glaive_body(body)
        else:
            try:
                value, end = decoder.raw_decode(text, position)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"malformed Glaive function call at offset {start}: {exc}"
                ) from exc
        if not isinstance(value, dict) or not value.get("name"):
            raise ValueError("malformed Glaive function call")
        arguments = _json(value.get("arguments", {}), "arguments")
        if not isinstance(arguments, dict):
            raise TypeError("Glaive arguments must be an object")
        calls.append({"id": f"call_{pending:04d}", "name": value["name"], "arguments": arguments})
        pending += 1

        terminated = text.find(GLAIVE_CALL_TERMINATOR, end)
        block_end = (
            terminated + len(GLAIVE_CALL_TERMINATOR)
            if terminated != -1 and text[end:terminated].strip() == ""
            else end
        )
        spans.append((start, block_end))
        cursor = block_end
    return calls, pending, spans


def _glaive_parts(record: dict[str, Any]) -> tuple[str, list[str]] | None:
    chat = record.get("chat", record.get("messages"))
    if not isinstance(chat, str):
        return None
    return str(record.get("system", "")), re.split(r"(?=USER:|ASSISTANT:|FUNCTION RESPONSE:)", chat)


def adapt_glaive_v2(record: dict[str, Any], split: str = "train") -> ToolConversation:
    """Glaive adapter that survives this revision's unterminated function-call blocks.

    Registered separately rather than replacing :func:`adapt_glaive` so that the pinned v1
    release stays reproducible from the code that produced it. Selecting this adapter is a new
    corpus version with its own manifest hash, not an edit to v1.
    """
    if isinstance(record.get("messages"), list) and "chat" not in record:
        return adapt(record, "glaive-function-calling-v2", split)
    parts_result = _glaive_parts(record)
    if parts_result is None:
        return _tagged_messages(record, "glaive-function-calling-v2", split, "glaive_v2")
    system, parts = parts_result
    messages: list[dict[str, Any]] = []
    pending = 0
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("USER:"):
            messages.append({"role": "user", "content": part[5:].strip()})
        elif part.startswith("ASSISTANT:"):
            text = part[9:].strip()
            calls, pending, spans = extract_glaive_calls(text, pending)
            remainder = text
            for start, end in reversed(spans):
                remainder = remainder[:start] + remainder[end:]
            for artifact in GLAIVE_TEXT_ARTIFACTS:
                remainder = remainder.replace(artifact, " ")
            # The upstream delimiter leaves stray colons and whitespace at the start of these
            # turns; keeping them would put punctuation at the front of the assistant target.
            remainder = re.sub(r"^[\s:]+", "", remainder).strip()
            messages.append(
                {"role": "assistant", "content": remainder or None, "tool_calls": calls}
            )
        elif part.startswith("FUNCTION RESPONSE:"):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{max(0, pending - 1):04d}",
                    "content": part[18:].strip(),
                }
            )
    c = _base(
        record,
        "glaive-function-calling-v2",
        split,
        _tool(record.get("tools", [])) or _embedded_tools(str(system)),
        messages,
        adapter="glaive_function_calling_v2_v2",
        source_format="system/chat delimiters, unterminated functioncall blocks",
        # Glaive's calls are answered by `<functioncall>`/function-response turns, so its
        # resolved calls form real trajectories. Records with no calls end in assistant prose,
        # which the complete-trajectory contract accepts as the terminal response.
        supervision_kind=SupervisionKind.COMPLETE_TRAJECTORY,
    )
    c.metadata["system"] = system
    c.validate()
    return c


ADAPTERS: dict[str, Callable[[dict[str, Any], str], ToolConversation]] = {
    "xlam": adapt_xlam,
    "when2call": adapt_when2call,
    "toolace": adapt_toolace,
    "button": adapt_button,
    "looptool": adapt_looptool,
    "glaive": adapt_glaive,
    "glaive_v2": adapt_glaive_v2,
}
