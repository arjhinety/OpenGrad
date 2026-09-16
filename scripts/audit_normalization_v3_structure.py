"""Structural audit of the normalization-v3 pre-classifier artifact. READ-ONLY; COUNTS ONLY.

Per source it reports what the representation *is*: acceptance and rejection by reason, exchange shape,
where the final assistant turn sits relative to tool use, residual call syntax and special tokens, tool
availability, trajectory validity, eligibility under prose-decision-input-v1, and a per-row check that
every call the raw source expresses in its own syntax became a structured call.

It reads **structure and syntax only**. No refusal regex, question cue, direct-answer heuristic or other
decision logic runs here; the sampling strata of the P-DET-COVERAGE draft are measured separately, by
``scripts/audit_pdet_coverage_supply_v3.py``, and labelled as supply analysis.

    python scripts/audit_normalization_v3_structure.py   # writes reports/normalization-v3/normalization-v3.structural-audit.json
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from opengrad.data import versions
from opengrad.data.classifier_input import EXCLUSION_ORDER, HeldoutIndex, eligibility
from opengrad.data.normalization_v3 import (
    OUTPUT_DIR,
    iter_rows,
    load_source_manifest,
    source_specs,
    structure_of,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / OUTPUT_DIR
OUT = ROOT / "reports" / "normalization-v3" / "normalization-v3.structural-audit.json"

# ── call syntax, per source convention (syntax only) ───────────────────────────────────────────────
MARKERS = {
    "glaive_functioncall": re.compile(r"</?functioncall>", re.IGNORECASE),
    "when2call_toolcall": re.compile(r"</?TOOLCALL>"),
    "tool_call_tag": re.compile(r"</?tool_call>", re.IGNORECASE),
    "json_name_object": re.compile(r'\{\s*"name"\s*:'),
    "json_arguments_key": re.compile(r'"arguments"\s*:'),
    "bracket_call_at_start": re.compile(r"^\s*\[\s*[A-Za-z_][\w .\-]*\("),
}
#: What the per-row conservation check compares, per source, with the unit pinned on both sides.
CONSERVATION_FIELD = {
    "glaive": "structured_calls",
    "xlam": "structured_calls",
    "when2call": "structured_calls",
    "toolace": "assistant_turns_with_structured_call",
}
CONSERVATION_UNIT = {
    "glaive": "calls per accepted row: raw <functioncall> markers vs structured tool_calls entries",
    "xlam": "calls per accepted row: raw named `answers` entries vs structured tool_calls entries",
    "when2call": "calls per accepted row: raw <TOOLCALL> markers vs structured tool_calls entries",
    "toolace": (
        "call-bearing assistant turns per accepted row: raw turns starting '[' with '(' vs turns with "
        "tool_calls (one bracket turn may hold several calls, so calls are not the unit)"
    ),
}
SPECIAL_TOKENS = ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|eot_id|>", "</s>", "[INST]")
DECODER = json.JSONDecoder()


def _parses_as_call(text: str) -> bool:
    """Does some JSON object with a string ``name`` start at a ``{`` in the text?"""
    for match in re.finditer(r"\{", text):
        try:
            value, _ = DECODER.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("name"), str):
            return True
    return False


def _raw_call_counts(name: str, raw_path: Path) -> list[int]:
    """Calls each raw row expresses in its source's own syntax, by raw row index."""
    table = pq.read_table(raw_path)
    if name == "glaive":
        return [chat.count("<functioncall>") for chat in table.column("chat").to_pylist()]
    if name == "xlam":
        counts = []
        for answers in table.column("answers").to_pylist():
            value = json.loads(answers) if isinstance(answers, str) else answers
            value = [value] if isinstance(value, dict) else (value or [])
            counts.append(
                sum(
                    1
                    for item in value
                    if isinstance(item, dict)
                    and (item.get("name") or item.get("function") or item.get("tool_name"))
                )
            )
        return counts
    if name == "toolace":
        counts = []
        for conversation in table.column("conversations").to_pylist():
            counts.append(
                sum(
                    1
                    for turn in conversation or []
                    if turn.get("from") in {"assistant", "gpt"}
                    and isinstance(turn.get("value"), str)
                    and turn["value"].lstrip().startswith("[")
                    and "(" in turn["value"]
                )
            )
        return counts
    if name == "when2call":
        return [
            sum(str(message.get("content") or "").count("<TOOLCALL>") for message in messages or [])
            for messages in table.column("messages").to_pylist()
        ]
    raise ValueError(name)


def _nest(counts: Counter[str]) -> dict[str, dict[str, int]]:
    """``kind|assignment|field`` counts as ``{"kind|assignment": {field: n}}``."""
    nested: dict[str, dict[str, int]] = {}
    for key, value in sorted(counts.items()):
        group, _, field = key.rpartition("|")
        nested.setdefault(group, {})[field] = value
    return nested


def audit_source(spec: Any, heldout: HeldoutIndex) -> dict[str, Any]:
    directory = ARTIFACT / spec.name
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    ledger = [
        json.loads(line)
        for line in (directory / "dispositions.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    raw_calls = _raw_call_counts(spec.name, spec.raw_path)

    shape: Counter[str] = Counter()
    final: Counter[str] = Counter()
    tools: Counter[str] = Counter()
    markers_any: Counter[str] = Counter()
    markers_final: Counter[str] = Counter()
    textual_calls: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    trajectory: Counter[str] = Counter()
    reasons_primary: Counter[str] = Counter()
    reasons_all: Counter[str] = Counter()
    eligible_tools: Counter[str] = Counter()
    invariants: Counter[str] = Counter()
    conservation: Counter[str] = Counter()
    call_turns: Counter[str] = Counter()
    supervision: Counter[str] = Counter()

    for record in iter_rows(ARTIFACT, spec.name):
        metadata = record["metadata"]
        structure = metadata["structure"]
        messages = record["messages"]
        invariants["rows"] += 1
        invariants["behavior_label_present"] += "behavior" in metadata
        invariants["adapter_version_not_authoritative"] += (
            metadata.get("adapter_version") != versions.ADAPTER_VERSION
            or metadata["supervision"].get("adapter_version") != versions.ADAPTER_VERSION
        )
        invariants["adapter_key_not_manifest"] += metadata.get("adapter_key") != spec.adapter_key
        recomputed = structure_of(messages, record["tools"])
        invariants["structure_block_disagrees_with_messages"] += any(
            structure[key] != value for key, value in recomputed.items()
        )

        shape[structure["exchange_shape"]] += 1
        shape["leading_system_message"] += structure["leading_system_message"]
        shape["system_text_in_metadata"] += bool(metadata.get("system"))
        tools["tools_available" if structure["tools_available"] else "no_tools"] += 1
        for code in structure["trajectory_issue_codes"]:
            trajectory[code] += 1
        trajectory["records_with_any_issue"] += bool(structure["trajectory_issue_codes"])
        block = metadata["supervision"]
        kind = f"{block['kind']}|{block['assignment']}"
        supervision[f"{kind}|records"] += 1
        supervision[f"{kind}|trajectory_valid"] += not structure["trajectory_issue_codes"]
        supervision[f"{kind}|final_assistant_structured_call"] += bool(
            structure["final_assistant_structured_call"]
        )
        for code in structure["trajectory_issue_codes"]:
            supervision[f"{kind}|issue:{code}"] += 1

        last = messages[-1] if messages else {}
        any_call = structure["structured_calls"] > 0
        any_result = structure["tool_result_messages"] > 0
        if structure["final_assistant_structured_call"]:
            final["final_assistant_structured_call"] += 1
        elif last.get("role") == "assistant" and structure["final_assistant_prose"]:
            if structure["tool_result_before_final"]:
                final["final_prose_after_tool_result"] += 1
            elif any_call or any_result:
                final["final_prose_after_unresolved_call"] += 1
            else:
                final["final_prose_no_prior_tool_use"] += 1
        else:
            final[f"final_other:{last.get('role')}"] += 1

        for index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            text = str(message.get("content") or "")
            is_final = index == len(messages) - 1
            if message.get("tool_calls"):
                # A structured call turn: its text should not repeat the call.
                call_turns["structured_call_turns"] += 1
                call_turns["structured_call_turns_with_nonempty_content"] += bool(text.strip())
                call_turns["structured_call_turns_whose_content_starts_with_bracket"] += (
                    text.lstrip().startswith("[")
                )
                call_turns["structured_call_turns_whose_content_also_holds_call_text"] += any(
                    pattern.search(text) for pattern in MARKERS.values()
                )
                continue
            for family, pattern in MARKERS.items():
                if pattern.search(text):
                    markers_any[family] += 1
                    if is_final:
                        markers_final[family] += 1
            if MARKERS["json_name_object"].search(text) or MARKERS["glaive_functioncall"].search(
                text
            ):
                textual_calls["parseable" if _parses_as_call(text) else "unparseable"] += 1
            for token in SPECIAL_TOKENS:
                tokens[token] += token in text
            tokens["leading_colon"] += text.startswith(":")

        observed = structure[CONSERVATION_FIELD[spec.name]]
        conservation[
            "raw_equal_structured"
            if raw_calls[metadata["source"]["raw_row_index"]] == observed
            else "raw_differ_from_structured"
        ] += 1

        result = eligibility(record, heldout)
        if result.eligible:
            reasons_primary["ELIGIBLE"] += 1
            eligible_tools["tools_available" if record["tools"] else "no_tools"] += 1
        else:
            reasons_primary[str(result.primary_reason)] += 1
            for reason in result.reasons:
                reasons_all[reason] += 1

    rejected_with_raw_calls = sum(
        1
        for entry in ledger
        if entry["disposition"] == "rejected" and raw_calls[entry["raw_row_index"]]
    )
    return {
        "manifest_sha256": hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest(),
        "adapter": {
            "key": spec.adapter_key,
            "function": spec.adapter_function,
            "row_label": spec.row_label,
        },
        "counts": manifest["counts"],
        "rejected_by_reason": manifest["rejected_by_reason"],
        "exchange_shape": dict(sorted(shape.items())),
        "final_turn": dict(sorted(final.items())),
        "tools": dict(sorted(tools.items())),
        "trajectory_issues": dict(sorted(trajectory.items())),
        "supervision_kind_assignment": _nest(supervision),
        "residual_call_markers_prose_turns": dict(sorted(markers_any.items())),
        "residual_call_markers_final_prose_turn": dict(sorted(markers_final.items())),
        "residual_textual_calls_prose_turns_by_parseability": dict(sorted(textual_calls.items())),
        "residual_special_tokens_prose_turns": {k: v for k, v in sorted(tokens.items()) if v},
        "structured_call_turns": dict(sorted(call_turns.items())),
        "call_conservation": {
            "unit": CONSERVATION_UNIT[spec.name],
            **dict(sorted(conservation.items())),
            "raw_rows_with_calls": sum(1 for count in raw_calls if count),
            "rejected_rows_with_raw_calls": rejected_with_raw_calls,
        },
        "eligibility_primary": dict(sorted(reasons_primary.items())),
        "eligibility_all_reasons": {reason: reasons_all[reason] for reason in EXCLUSION_ORDER},
        "eligible_by_tools": dict(sorted(eligible_tools.items())),
        "invariants": dict(sorted(invariants.items())),
    }


def main() -> int:
    top = json.loads((ARTIFACT / "manifest.json").read_text(encoding="utf-8"))
    heldout = HeldoutIndex.load(ROOT)
    specs = source_specs(load_source_manifest(ROOT))
    report = {
        "artifact_kind": "NORMALIZATION_V3_STRUCTURAL_AUDIT",
        "statement": (
            "Structural facts about the normalization-v3 pre-classifier artifact. Counts only; no record "
            "text. No decision heuristic (refusal, question, direct-answer cue) was run."
        ),
        "artifact_fingerprint": top["fingerprint"],
        "artifact_manifest_sha256": hashlib.sha256(
            (ARTIFACT / "manifest.json").read_bytes()
        ).hexdigest(),
        "classifier_input_contract": versions.CLASSIFIER_INPUT_CONTRACT_VERSION,
        "heldout_inputs": [list(item) for item in heldout.inputs],
        "units": {
            "counts, exchange_shape, final_turn, tools, trajectory_issues, supervision_kind_assignment, eligibility_*": "records",
            "residual_*, structured_call_turns": "assistant turns (prose turns: no tool_calls)",
            "call_conservation": "accepted records",
        },
        "sources": {spec.name: audit_source(spec, heldout) for spec in specs},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    for name, data in report["sources"].items():
        print(name, data["counts"], data["final_turn"], data["eligibility_primary"])
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
