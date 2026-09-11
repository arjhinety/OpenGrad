"""OpenWeights On-Device Tool-Calling Benchmark Adapter.

Evaluates small-model tool calling under the exact lightweight prompt format
used by the OpenWeights Android application (no explicit 2000-token system prompts).
Supports both CallFormat.BARE and CallFormat.TAGGED arms.
"""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar

from opengrad.benchmarks.adapters.base import BenchmarkAdapter, BenchmarkTask
from opengrad.benchmarks.backends.protocol import GenerationResult
from opengrad.benchmarks.schema import NormalizedTaskResult
from opengrad.benchmarks.taxonomy import ToolFailureCategory

# The 18 standard on-device tools available in OpenWeights
OPENWEIGHTS_TOOLS: list[dict[str, Any]] = [
    {
        "name": "web_search",
        "description": "Search the web for fresh information or external facts.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": "Fetch web page content from a URL.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents from the shared workspace folder.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write text content to a file in the workspace folder.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_dir",
        "description": "List directory entries in the workspace folder.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "run_script",
        "description": "Run a sandboxed JavaScript script.",
        "parameters": {
            "type": "object",
            "properties": {"script": {"type": "string"}},
            "required": ["script"],
        },
    },
    {
        "name": "remember",
        "description": "Save an important fact or preference to long-term memory.",
        "parameters": {
            "type": "object",
            "properties": {"fact": {"type": "string"}},
            "required": ["fact"],
        },
    },
    {
        "name": "recall",
        "description": "Recall memories relevant to a query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]


def render_openweights_prompt(
    tools: list[dict[str, Any]], user_prompt: str, format_mode: str = "bare"
) -> str:
    """Render the exact lightweight OpenWeights system instruction (under 150 tokens)."""
    tool_lines = []
    for t in tools:
        params_compact = json.dumps(t.get("parameters", {}), separators=(",", ":"))
        tool_lines.append(f"- {t['name']}: {t.get('description', '')} Arguments: {params_compact}")
    tools_str = "\n".join(tool_lines)

    if format_mode == "tagged":
        format_instruction = (
            "To use one, reply with only this and nothing else:\n"
            '<tool_call>{"name": "tool_name", "arguments": {"argument": "value"}}</tool_call>\n'
            "Do not explain that you are going to use it. Just send the tags. "
            "If no tool is needed, answer normally and send no tags."
        )
    else:
        # BARE format (default in OpenWeights)
        format_instruction = (
            "To use one, reply with only this and nothing else:\n"
            '{"tool": "name", "arguments": {"argument": "value"}}\n'
            "Do not explain that you are going to use it. Just send the object. "
            "If no tool is needed, answer normally and send no object."
        )

    system_msg = f"You can use these tools:\n{tools_str}\n\n{format_instruction}"
    return f"{system_msg}\n\nUser: {user_prompt}\nAssistant:"


def _call_from_json_object(obj: Any) -> tuple[str, str | None, dict[str, Any]]:
    """Build a CALL triple from a parsed JSON call object.

    Both JSON shapes name the tool with a string, so a non-string value is reported as "no name"
    rather than leaking through the declared ``str | None`` return. The two JSON branches below
    share this so they cannot drift apart on malformed input.
    """
    candidate = obj.get("name") or obj.get("tool")
    name = candidate if isinstance(candidate, str) else None
    args = obj.get("arguments", {})
    return "CALL", name, args if isinstance(args, dict) else {}


def parse_openweights_reply(reply: str) -> tuple[str, str | None, dict[str, Any]]:
    """Parse reply using the OpenWeights ToolPrompting parser logic.

    Accepts:
    1. {"tool": "...", "arguments": {...}}          (CallFormat.BARE)
    2. {"name": "...", "arguments": {...}}          (CallFormat.BARE variant)
    3. <tool_call>{"name": "...", "arguments": {...}}</tool_call>   (CallFormat.TAGGED)
    4. <tool_call><function=n><parameter=k>v</parameter></function></tool_call>

    Shape 4 is the model's *native* template output, not one of the two arms this benchmark
    asks for. It is accepted anyway because OpenWeights prefers a model's own template when
    that template carries tools — and Qwen3.5-2B's does, mandating this XML form — so a model
    that has learned the native shape will emit it here no matter what the arm's prompt says.
    Scoring a correct call as FORMAT_ERROR would misreport the model rather than the format.
    """
    cleaned = reply.strip()

    # Native XML tool call first: it is unambiguous, and its inner digits and braces would
    # otherwise be picked up by the BARE JSON scan below.
    xml_calls = _parse_native_xml(cleaned)
    if xml_calls is not None:
        name, arguments = xml_calls[0]
        return "CALL", name, arguments

    # Check for TAGGED format
    tagged_match = re.search(r"<tool_call>(.*?)</tool_call>", cleaned, re.DOTALL)
    if tagged_match:
        try:
            return _call_from_json_object(json.loads(tagged_match.group(1)))
        except (json.JSONDecodeError, TypeError):
            return "FORMAT_ERROR", None, {}

    # Check for BARE format
    bare_match = re.search(r"\{[\s\S]*\}", cleaned)
    if bare_match:
        try:
            obj = json.loads(bare_match.group(0))
            if isinstance(obj, dict) and ("tool" in obj or "name" in obj):
                return _call_from_json_object(obj)
        except (json.JSONDecodeError, TypeError):
            pass

    return "ANSWER", None, {}


def _parse_native_xml(reply: str) -> list[tuple[str, dict[str, Any]]] | None:
    """Decode ``<function=...><parameter=...>`` calls, or None when the reply has none."""
    from opengrad.formatting.parser import parse_qwen_native_output

    if "<function=" not in reply.casefold():
        return None
    parsed = parse_qwen_native_output(reply)
    if parsed.status != "RAW_VALID" or parsed.decision != "CALL" or not parsed.calls:
        return None
    return [(call.name, call.arguments) for call in parsed.calls]


class OpenWeightsAdapter(BenchmarkAdapter):
    benchmark_id = "openweights"
    name = "OpenWeights On-Device Tool Calling Benchmark"

    FORMAT_ARMS: ClassVar[list[str]] = ["bare", "tagged"]

    def load_tasks(self, split: str = "bare", limit: int | None = None) -> list[BenchmarkTask]:
        format_mode = split if split in self.FORMAT_ARMS else "bare"
        tasks: list[BenchmarkTask] = []
        specs = [
            (
                "ow_search_needed",
                "What is the latest release version of the llama.cpp mobile runtime?",
                "CALL",
                "web_search",
            ),
            (
                "ow_fetch_needed",
                "Fetch the content of https://alpharomercoma.github.io/openweights/latency.html.",
                "CALL",
                "fetch_url",
            ),
            (
                "ow_read_needed",
                "Read the notes from file 'experiment_notes.txt'.",
                "CALL",
                "read_file",
            ),
            ("ow_write_needed", "Save the summary text to 'summary.md'.", "CALL", "write_file"),
            (
                "ow_remember_needed",
                "Remember that I prefer bfloat16 precision over float16.",
                "CALL",
                "remember",
            ),
            ("ow_recall_needed", "What precision did I say I prefer?", "CALL", "recall"),
            ("ow_answer_direct", "What is the capital city of Japan?", "ANSWER", None),
            ("ow_math_direct", "Calculate 15 * 14.", "ANSWER", None),
        ]
        for task_id, prompt, expected_dec, expected_tool in specs:
            task = BenchmarkTask(
                task_id=f"{task_id}_{format_mode}",
                category=format_mode,
                prompt=prompt,
                tools=OPENWEIGHTS_TOOLS,
                expected={"decision": expected_dec, "tool": expected_tool, "format": format_mode},
                expected_decision=expected_dec,
                metadata={
                    "format_arm": format_mode,
                    "target_tool": expected_tool,
                    "system_prompt_type": "openweights_minimal",
                },
            )
            tasks.append(task)
        if limit is not None:
            tasks = tasks[:limit]
        return tasks

    def render_prompt(self, task: BenchmarkTask, renderer_name: str = "openweights_bare") -> str:
        format_mode = task.metadata.get("format_arm", "bare")
        return render_openweights_prompt(task.tools, task.prompt, format_mode=format_mode)

    def evaluate_task(
        self, task: BenchmarkTask, generation: GenerationResult
    ) -> NormalizedTaskResult:
        decision, tool_name, args = parse_openweights_reply(generation.text)
        expected_dec = task.expected_decision
        expected_tool = task.expected.get("tool") if isinstance(task.expected, dict) else None

        success = False
        failure_category = None
        tool_calls = [{"name": tool_name, "arguments": args}] if tool_name else []

        if decision == "FORMAT_ERROR":
            failure_category = ToolFailureCategory.MALFORMED_TOOL_CALL.value
        elif expected_dec == "CALL":
            if decision != "CALL" or not tool_name:
                failure_category = ToolFailureCategory.MISSED_TOOL.value
            elif expected_tool and tool_name != expected_tool:
                failure_category = ToolFailureCategory.WRONG_TOOL.value
            else:
                success = True
        elif expected_dec == "ANSWER":
            if decision == "CALL":
                failure_category = ToolFailureCategory.UNNECESSARY_TOOL.value
            else:
                success = True

        return NormalizedTaskResult(
            task_id=task.task_id,
            input=task.prompt,
            raw_output=generation.text,
            parsed_output={"decision": decision, "tool": tool_name, "arguments": args},
            expected=task.expected,
            score=1.0 if success else 0.0,
            success=success,
            failure_category=failure_category,
            latency=generation.latency,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            tool_calls=tool_calls,
            metadata=task.metadata,
        )
