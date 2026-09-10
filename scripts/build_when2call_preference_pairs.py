#!/usr/bin/env python3
"""Build DPO preference pairs from the pinned When2Call preference split.

Why this exists: `configs/experiments/m1_dpo.yaml` pins `when2call_pref_v1` at the When2Call
*dataset* revision, and no preference artifact was ever materialized, so there was nothing to
train DPO on. The split itself is real and is published at that revision
(`train/when2call_train_pref.jsonl`, 9000 rows, disjoint from the `test` split that B0 measures).
This script turns it into pairs in the format the policy and the evaluator actually use.

The conversion is renderer-based on purpose. The upstream response content uses the
`<TOOLCALL>[{...}]` convention, but the policy is scored on Qwen's `<tool_call>` markup, so the
pair has to be rendered through the same pinned renderer that produced the SFT targets and the
evaluation prompts. Prompts and completions are extracted with the same assistant-span logic the
SFT trainer uses, so the completion text is exactly what the model is expected to generate.

Records whose tool schemas the canonical contract rejects are skipped and counted, never repaired.

Usage:
    python scripts/build_when2call_preference_pairs.py [--max-records N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opengrad.data.canonical import ToolConversation
from opengrad.data.renderers import Qwen35_2BRenderer
from opengrad.training.sft_data import assistant_spans_with_offsets

MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
DEFAULT_INPUT = ROOT / "data/raw/when2call/train/when2call_train_pref.jsonl"
DEFAULT_OUTPUT = ROOT / "data/processed/when2call-preference-pairs-v1.jsonl"

TOOLCALL_BLOCK = re.compile(r"<TOOLCALL>\s*(.*?)\s*(?:</TOOLCALL>|$)", re.DOTALL)


def response_messages(response: dict) -> list[dict]:
    """Canonical assistant messages for one chosen/rejected response.

    The content may interleave `<TOOLCALL>` blocks with prose. Each block holds a JSON array of
    calls, unlike the single-object form the SFT adapter reads, so it is handled explicitly.
    """
    content = str(response.get("content") or "")
    if "<TOOLCALL>" not in content:
        return [{"role": "assistant", "content": content or None, "tool_calls": []}]
    messages: list[dict] = []
    cursor = 0
    for match in TOOLCALL_BLOCK.finditer(content):
        prose = content[cursor : match.start()].strip()
        if prose:
            messages.append({"role": "assistant", "content": prose, "tool_calls": []})
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            parsed = None
        calls = (
            parsed if isinstance(parsed, list) else ([parsed] if isinstance(parsed, dict) else [])
        )
        tool_calls = []
        for index, call in enumerate(calls):
            if not isinstance(call, dict) or not call.get("name"):
                continue
            arguments = call.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            tool_calls.append(
                {"id": f"call_{index:04d}", "name": call["name"], "arguments": arguments or {}}
            )
        messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
        cursor = match.end()
    tail = content[cursor:].strip()
    if tail:
        messages.append({"role": "assistant", "content": tail, "tool_calls": []})
    return messages


def completion_text(
    renderer: Qwen35_2BRenderer, tokenizer, prefix: list[dict], tools: list[dict], response: dict
) -> str:
    """Prompt text and completion text for one response, in the pinned renderer's format."""
    conversation = ToolConversation(
        id="preference",
        source="when2call-preference",
        tools=tools,
        messages=list(prefix) + response_messages(response),
        metadata={"split": "train"},
    )
    rendered = renderer.render_sft(conversation)
    encoded = tokenizer(rendered.text, add_special_tokens=False, return_offsets_mapping=True)
    tokens = list(encoded["input_ids"])
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    messages = _rendered_messages(renderer, conversation)
    spans, error = assistant_spans_with_offsets(tokenizer, messages, rendered.text, tokens, offsets)
    if error is not None or not spans:
        raise ValueError(error or "no assistant span")
    # The response is the last assistant turn, so its span is the completion.
    start_token, end_token = spans[-1]
    start_char = offsets[start_token][0]
    end_char = offsets[end_token - 1][1]
    return rendered.text[:start_char], rendered.text[start_char:end_char]


def _rendered_messages(renderer: Qwen35_2BRenderer, conversation: ToolConversation) -> list[dict]:
    from opengrad.data.renderers import _qwen_messages

    return _qwen_messages(conversation)


def normalize_tools(raw: Any) -> list[dict]:
    """Parse the declared tools, which this revision stores as JSON strings in a list."""
    tools = raw or []
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            return []
    parsed: list[dict] = []
    for item in tools:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                continue
        if isinstance(item, dict):
            parsed.append(item)
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"preference split not found: {args.input}", file=sys.stderr)
        return 1
    records = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    if args.max_records:
        records = records[: args.max_records]

    renderer = Qwen35_2BRenderer(revision=MODEL_REVISION, enable_thinking=False)
    tokenizer = renderer._load()

    pairs: list[dict] = []
    skipped = Counter()
    for record in records:
        tools = normalize_tools(record.get("tools"))
        prefix = record.get("messages") or []
        try:
            prompt, chosen = completion_text(
                renderer, tokenizer, prefix, tools, record["chosen_response"]
            )
        except Exception as exc:  # noqa: BLE001 - unrenderable records are counted, not repaired
            skipped[f"{type(exc).__name__}: {str(exc)[:70]}"] += 1
            continue
        try:
            _, rejected = completion_text(
                renderer, tokenizer, prefix, tools, record["rejected_response"]
            )
        except Exception as exc:  # noqa: BLE001
            skipped[f"rejected {type(exc).__name__}: {str(exc)[:60]}"] += 1
            continue
        if not chosen.strip() or not rejected.strip() or chosen == rejected:
            skipped["empty or identical completion"] += 1
            continue
        pairs.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "preference_source": "when2call_train_pref@pinned",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False, sort_keys=True) + "\n")

    from opengrad.training.dpo_runner import preference_dataset_identity

    identity = preference_dataset_identity(args.output)
    report = {
        "schema_version": 1,
        "prompt_format": "rendered",
        "source": str(args.input.relative_to(ROOT))
        if args.input.is_relative_to(ROOT)
        else str(args.input),
        "output": str(args.output.relative_to(ROOT))
        if args.output.is_relative_to(ROOT)
        else str(args.output),
        "records_in": len(records),
        "pairs_out": len(pairs),
        "skipped": dict(skipped.most_common(10)),
        "identity": identity,
    }
    (args.output.parent / "when2call-preference-pairs-v1.report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(
        f"\nAdd to a DPO config:\n"
        f"  datasets:\n"
        f"    preference_path: {report['output']}\n"
        f"    preference_prompt_format: rendered\n"
        f"    hashes:\n"
        f"      preference: {identity['sha256']}\n"
    )
    return 0 if pairs else 1


if __name__ == "__main__":
    raise SystemExit(main())
