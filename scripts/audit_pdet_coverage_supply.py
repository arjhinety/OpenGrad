"""Feasibility audit for the P-DET-COVERAGE-v1 preregistration draft. READ-ONLY; NOT A DRAW.

Counts, per candidate source, how many records fall in each preregistered stratum
(docs/research/study-002/30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md, section 7). Nothing is ranked,
selected or copied, and no record text is written: the output holds counts only.

The counts are computed over the ``normalization-v1`` artifacts, which are a **PROXY**. They are not the
representation the future classifier will receive (canonical-v3 does not exist; section 5). They
answer one question only: could each stratum plausibly be supplied? They may not be used to sample.
Study 001's #80 is why that distinction is recorded in the output itself (01-LESSONS-FROM-STUDY-001.md, L4).

    python scripts/audit_pdet_coverage_supply.py        # writes reports/pdet-coverage/pdet-coverage-v1.supply-proxy.json
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from opengrad.evaluation.capability import detect_refusal

ROOT = Path(__file__).resolve().parents[1]
PROXY = ROOT / "data" / "processed" / "normalization-v1"
OUT = ROOT / "reports" / "pdet-coverage" / "pdet-coverage-v1.supply-proxy.json"
SOURCES = ("glaive", "toolace", "xlam", "looptool", "button")
LAYER_B_SOURCES = ("glaive", "toolace", "looptool")

# ── the preregistered stratum predicates (draft section 7.2) ─────────────────────────────────────
QUESTION_CUES = ("could you", "can you please", "please provide", "please specify", "please tell me",
                 "which ", "what is the", "do you have", "would you like")  # as P-DET-v1 (pdet.py)
HEDGE_CUES = ("i can't", "i cannot", "i'm unable", "i am unable", "i don't know", "i'm not able",
              "i do not have")  # as P-DET-v1 (pdet.py)
TEXTUAL_CALL = re.compile(r'\{\s*"name"\s*:|<functioncall>|<tool_call>|<TOOLCALL>|"arguments"\s*:', re.IGNORECASE)
INVOCATION_TALK = re.compile(
    r"\b(i('ll| will| can)? (use|call|run|invoke)|let me (use|call|check|run)|using the \w+ (function|tool|api))\b",
    re.IGNORECASE,
)
GENERIC_TOOL_WORD = re.compile(r"\b(function|functions|tool|tools|api|apis)\b", re.IGNORECASE)
STRATA = ("X", "M", "R", "Q", "P1", "P2")


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def proxy_clean(text: str | None) -> str:
    """Approximate ``adapt_glaive_v2``'s cleanup on v1 text: strip leading ':'/whitespace and end tokens."""
    text = (text or "").replace("<|endoftext|>", " ")
    return re.sub(r"^[\s:]+", "", text).strip()


def offered_names(tools: list[dict[str, Any]]) -> set[str]:
    names = set()
    for tool in tools or []:
        spec = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = spec.get("name") if isinstance(spec, dict) else None
        if name and len(name) >= 4:
            names.add(str(name).casefold())
    return names


def stratum(response: str, tools: list[dict[str, Any]]) -> str:
    """First match in priority order X, M, R, Q, P1/P2. A sampling stratum, never a label."""
    low = response.casefold()
    if TEXTUAL_CALL.search(response):
        return "X"
    if any(name in low for name in offered_names(tools)) or INVOCATION_TALK.search(response):
        return "M"
    if detect_refusal(response).is_refusal or any(cue in low for cue in HEDGE_CUES):
        return "R"
    if "?" in response or any(cue in low for cue in QUESTION_CUES):
        return "Q"
    return "P1" if tools else "P2"


def skeleton(response: str, tools: list[dict[str, Any]]) -> str:
    text = response.casefold()
    for name in offered_names(tools):
        text = text.replace(name, "<tool>")
    text = re.sub(r"\"[^\"]*\"|'[^']*'", "<q>", text)
    text = re.sub(r"\d+(\.\d+)?", "<n>", text)
    return re.sub(r"\s+", " ", text).strip()


def uses_tool(messages: list[dict[str, Any]]) -> bool:
    return any(m.get("role") == "tool" or m.get("tool_calls") for m in messages) or any(
        "<functioncall>" in (m.get("content") or "") for m in messages if m.get("role") == "assistant"
    )


def audit_source(source: str) -> dict[str, Any]:
    directory = PROXY / source
    shards = sorted(directory.glob("*.parquet"))
    manifest_path = directory / "manifest.json"
    if not shards and (PROXY / f"{source}.jsonl").is_file():
        rows = [json.loads(line) for line in (PROXY / f"{source}.jsonl").read_text(encoding="utf-8").splitlines() if line]
        batches = [(r.get("messages"), r.get("tools")) for r in rows]
        manifest_path = PROXY / f"{source}.manifest.json"
    else:
        batches = []
        for shard in shards:
            table = pq.read_table(shard, columns=["messages", "tools"])
            batches.extend(zip(table.column("messages").to_pylist(), table.column("tools").to_pylist()))
    shape = collections.Counter()
    layer_b = {s: collections.Counter() for s in STRATA}
    distinct = {s: set() for s in STRATA}
    skeletons = {s: set() for s in STRATA}
    generic_word_in_single = 0
    for raw_messages, raw_tools in batches:
        messages = _json(raw_messages) or []
        tools = _json(raw_tools) or []
        if not messages:
            continue
        shape["records"] += 1
        last = messages[-1]
        if last.get("tool_calls"):
            shape["final_turn_structured_call"] += 1
        if any(m.get("tool_calls") for m in messages):
            shape["records_with_structured_call"] += 1
        if uses_tool(messages):
            shape["uses_a_tool"] += 1
            if not last.get("tool_calls") and last.get("role") == "assistant":
                shape["final_turn_prose_after_tool_use"] += 1
            continue
        roles = [m.get("role") for m in messages if m.get("role") != "system"]
        if roles != ["user", "assistant"]:
            shape["tool_free_multi_turn"] += 1
            continue
        shape["tool_free_single_exchange"] += 1
        shape["tool_free_single_exchange_tools_offered" if tools else "tool_free_single_exchange_no_tools"] += 1
        response = proxy_clean(last.get("content"))
        if GENERIC_TOOL_WORD.search(response):
            generic_word_in_single += 1
        if source not in LAYER_B_SOURCES:
            continue
        s = stratum(response, tools)
        layer_b[s]["records"] += 1
        distinct[s].add(re.sub(r"\s+", " ", response.casefold()).strip())
        skeletons[s].add(skeleton(response, tools))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    config = manifest.get("config") or {}
    return {
        "artifact": str(directory.relative_to(ROOT)).replace("\\", "/"),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.is_file() else None,
        "source_sha256": config.get("source_sha256"),
        "adapter_version_recorded": config.get("adapter_version"),
        "shape": dict(shape),
        "single_exchange_responses_with_generic_tool_word": generic_word_in_single,
        "layer_b_strata": {
            s: {"records": layer_b[s]["records"], "distinct_responses": len(distinct[s]), "skeletons": len(skeletons[s])}
            for s in STRATA
        }
        if source in LAYER_B_SOURCES
        else None,
    }


def main() -> int:
    report = {
        "artifact_kind": "PDET_COVERAGE_SUPPLY_PROXY",
        "status": "PROXY_NOT_FOR_SAMPLING",
        "statement": (
            "Counts over normalization-v1, which is not the representation the future classifier will "
            "receive. Feasibility evidence for the P-DET-COVERAGE-v1 draft only; sampling must read the "
            "canonical-v3 post-adapter representation and is BLOCKED until it exists."
        ),
        "preregistration": "docs/research/study-002/30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md",
        "unit": "single-exchange, tool-free records (one user turn, one assistant turn, no tool use)",
        "strata_priority": list(STRATA),
        "sources": {source: audit_source(source) for source in SOURCES},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    for source, data in report["sources"].items():
        print(source, data["shape"])
        if data["layer_b_strata"]:
            print("   ", {s: v["records"] for s, v in data["layer_b_strata"].items()})
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
