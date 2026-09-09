import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedCall:
    name: str
    arguments: dict[str, Any]
    call_id: str | None = None
    state: str = "RAW_VALID"


@dataclass(frozen=True)
class ParsedNativeOutput:
    """The lossless boundary between generated text and evaluation IR."""

    decision: str
    calls: list[ParsedCall]
    content: str
    status: str = "RAW_VALID"
    errors: list[str] | None = None
    truncated: bool = False


_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _call_values(body: str) -> list[Any]:
    """Decode one native Qwen payload without repairing malformed JSON."""
    value = json.loads(body.strip())
    return value if isinstance(value, list) else [value]


def parse_qwen_native_output(raw: str, *, truncated: bool = False) -> ParsedNativeOutput:
    """Parse Qwen's native ``<tool_call>{...}</tool_call>`` protocol.

    The parser is deliberately strict: a malformed payload makes the complete
    tool-call emission a format error rather than silently producing a repaired
    call. Text outside tags is retained as the assistant's final content.
    """
    if not isinstance(raw, str):
        raise TypeError("generated output must be a string")
    matches = list(_TOOL_CALL.finditer(raw))
    content = _THINK.sub("", _TOOL_CALL.sub("", raw)).strip()
    if not matches:
        if "<tool_call" in raw.casefold() or "</tool_call>" in raw.casefold():
            return ParsedNativeOutput(
                "ANSWER", [], content, "FORMAT_ERROR", ["UNCLOSED_TOOL_CALL"], truncated
            )
        lowered = content.casefold()
        if not content and truncated:
            return ParsedNativeOutput(
                "ANSWER", [], content, "FORMAT_ERROR", ["EOS_WITHOUT_OUTPUT"], True
            )
        if any(
            token in lowered
            for token in ("cannot help", "can't help", "unable to", "not supported", "unsupported")
        ):
            decision = "UNSUPPORTED"
        elif "?" in content or any(
            token in lowered for token in ("could you clarify", "please provide", "which one")
        ):
            decision = "CLARIFY"
        else:
            decision = "ANSWER"
        return ParsedNativeOutput(decision, [], content, "RAW_VALID", [], truncated)

    calls: list[ParsedCall] = []
    errors: list[str] = []
    for index, match in enumerate(matches):
        try:
            values = _call_values(match.group(1))
        except (json.JSONDecodeError, TypeError) as exc:
            errors.append(
                f"MALFORMED_TOOL_CALL[{index}]: {exc.msg if isinstance(exc, json.JSONDecodeError) else exc}"
            )
            continue
        for value in values:
            if (
                not isinstance(value, dict)
                or not isinstance(value.get("name"), str)
                or not value["name"]
            ):
                errors.append(f"INVALID_TOOL_CALL[{index}]: missing name")
                continue
            arguments = value.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    errors.append(f"MALFORMED_ARGUMENTS[{index}]: {exc.msg}")
                    continue
            if not isinstance(arguments, dict):
                errors.append(f"INVALID_ARGUMENTS[{index}]: arguments must be an object")
                continue
            calls.append(ParsedCall(value["name"], arguments, value.get("id")))
    if errors or not calls:
        if not errors:
            errors.append("EMPTY_TOOL_CALL")
        return ParsedNativeOutput("ANSWER", [], content, "FORMAT_ERROR", errors, truncated)
    return ParsedNativeOutput("CALL", calls, content, "RAW_VALID", [], truncated)


def parse_calls(value: Any) -> list[ParsedCall]:
    if not isinstance(value, list):
        value = [value]
    result = []
    for item in value:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError as exc:
                raise ValueError("INVALID: malformed JSON") from exc
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise TypeError("INVALID: missing call name")
        args = item.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError as exc:
                raise ValueError("INVALID: malformed arguments") from exc
        if not isinstance(args, dict):
            raise TypeError("INVALID: arguments must be object")
        result.append(ParsedCall(item["name"], args, item.get("id"), "RAW_VALID"))
    return result


def parse_native_tool_calls(raw: str) -> list[ParsedCall]:
    """Compatibility helper returning only valid native calls."""
    parsed = parse_qwen_native_output(raw)
    if parsed.status != "RAW_VALID" or parsed.decision != "CALL":
        raise ValueError("INVALID: native Qwen tool-call output")
    return parsed.calls
