"""Counts-only dry run of the proposed ``first reply`` unit (draft amendment 36; 35 §6, owner decision 2026-09-17).

35 §6 found that direct answers sit mostly in the first reply of multi-turn conversations, which
``prose-decision-input-v1`` excludes. Draft 36 proposes ``prose-decision-input-v2``, whose unit is the **first
assistant reply** of a record: a prose turn, with no structured call, directly after exactly one user turn (a
leading system message allowed), whatever follows it. A v1 single exchange is the special case with nothing after.

Before anything is adopted or drawn, this audit counts what that unit would supply for a new population:

1. **Unit rules (as drafted in 36 §2).** Excluded, in order: held-out records (any turn, as v1); records whose
   parse status is not ``VALID``, whose system or user content is not text, or that fail the training-trajectory
   gate anywhere (v1's malformation checks, except the two about the *final* message, which a first reply does not
   depend on); records whose first assistant turn is not a prose reply to exactly one user turn.
2. **Population exclusions (30 §9 and 35 §2):** sentinel prompts, the QAD recovery set, P-DET-v1, every
   P-DET-COVERAGE-v1 item and every item of the three classifier development and check sets, by identity,
   normalized prompt and normalized response.
3. **Dedup** as 30 §9: raw hash, response, one per user prompt, one per response skeleton per stratum.

Reported per source (Glaive and ToolACE, the layer B sources of 30 §6), stratum (30 §7.2), whether tools are
offered and whether the reply is a single exchange or has a continuation. **Sizing only:** the frozen classifier v1's
predictions are counted per stratum and source, to estimate how quotas translate into DIRECT, CLARIFY and
UNSUPPORTED items. They never select an item. Nothing prints or writes item text, ids, tools or rationales.

    python scripts/audit_first_reply_supply.py            # print
    python scripts/audit_first_reply_supply.py --write    # also write reports/pdet-coverage-v2/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_pdet_coverage_v2_supply as supply_audit

from opengrad.data.classifier_input import (
    ClassifierFeatures,
    HeldoutIndex,
    _heldout_hits,
    _malformations,
    _require_v3,
)
from opengrad.data.decision_classifier import CLASSIFIER_VERSION, LABELS, classify
from opengrad.data.normalization_v3 import OUTPUT_DIR as NORMALIZATION_V3_DIR
from opengrad.data.normalization_v3 import iter_rows
from opengrad.verification import classifier_devcheck as devcheck
from opengrad.verification import classifier_devset as devset
from opengrad.verification import pdet_coverage as coverage
from opengrad.verification import prose_classifier_oneshot as oneshot

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("reports/pdet-coverage-v2/first-reply.supply-dry-run.json")
SOURCES = ("glaive", "toolace")
FINAL_MESSAGE_CHECKS = ("final_role:", "final_assistant_empty")
SINGLE = "single_exchange"
CONTINUED = "has_continuation"


def first_reply(record: Mapping[str, Any], heldout: HeldoutIndex) -> tuple[str, dict[str, Any] | None]:
    """The record's disposition under the drafted unit and, when it has one, its first-reply unit."""
    metadata = _require_v3(record)
    if _heldout_hits(record, metadata, heldout):
        return "EVALUATION_ONLY_OR_HELDOUT", None
    if [p for p in _malformations(record, metadata) if not p.startswith(FINAL_MESSAGE_CHECKS)]:
        return "MALFORMED_OR_UNRENDERABLE", None
    body = [m for m in record["messages"] if m.get("role") != "system"]
    if len(body) < 2 or body[0].get("role") != "user" or body[1].get("role") != "assistant":
        return "FIRST_TURN_NOT_ONE_USER_THEN_ASSISTANT", None
    reply = body[1]
    if reply.get("tool_calls"):
        return "FIRST_REPLY_IS_STRUCTURAL_CALL", None
    if not (isinstance(reply.get("content"), str) and reply["content"].strip()):
        return "FIRST_REPLY_EMPTY", None
    rest = body[2:]
    unit = {
        **coverage._base_fields(record),
        "layer": coverage.LAYER_B,
        "user_message": body[0]["content"],
        "assistant_response": reply["content"],
        "tools": [dict(tool) for tool in record["tools"]],
        "unit_kind": SINGLE if not rest else CONTINUED,
        "continuation_uses_tools": any(m.get("tool_calls") or m.get("role") == "tool" for m in rest),
    }
    unit["stratum"] = coverage.stratum(unit["assistant_response"], unit["tools"])
    return "FIRST_REPLY_UNIT", unit


def audit(root: Path = ROOT) -> dict[str, Any]:
    oneshot.check_frozen(root)
    input_record = coverage.check_input(root)
    heldout = HeldoutIndex.load(root)
    dispositions: dict[str, Counter[str]] = {}
    units: list[dict[str, Any]] = []
    for source in SOURCES:
        counts = dispositions.setdefault(source, Counter())
        for record in iter_rows(root / NORMALIZATION_V3_DIR, source):
            disposition, unit = first_reply(record, heldout)
            counts[disposition] += 1
            if unit is not None:
                counts[f"{disposition}:{unit['unit_kind']}"] += 1
                units.append(unit)

    kept, exclusion_stats = coverage.apply_draw_exclusions(units, coverage.load_draw_exclusions(root))
    coverage_exclusion = devset.load_coverage_exclusion(root)
    used_exclusion = devcheck.load_devset_exclusion(root, supply_audit.USED_SETS)
    not_coverage = [unit for unit in kept if not coverage_exclusion.hits(unit)]
    unused = [unit for unit in not_coverage if not used_exclusion.hits(unit)]
    survivors, dedup_stats = devset.deduplicate(unused)

    supply: Counter[tuple[str, str, str, str]] = Counter()
    predicted: Counter[tuple[str, str, str, str]] = Counter()
    continuation: Counter[tuple[str, bool]] = Counter()
    for unit in survivors:
        tools = "tools_offered" if unit["tools"] else "no_tools"
        supply[(unit["source_name"], unit["stratum"], tools, unit["unit_kind"])] += 1
        if unit["unit_kind"] == CONTINUED:
            continuation[(unit["source_name"], unit["continuation_uses_tools"])] += 1
        label = classify(
            ClassifierFeatures(
                user_message=unit["user_message"],
                assistant_response=unit["assistant_response"],
                tools=tuple(unit["tools"]),
                structured_call_present=False,
            )
        ).label
        predicted[(unit["source_name"], unit["stratum"], unit["unit_kind"], label)] += 1

    kinds = (SINGLE, CONTINUED)
    return {
        "artifact_kind": "FIRST_REPLY_UNIT_SUPPLY_DRY_RUN",
        "authorization": "docs/research/study-002/35-OWNER-DECISIONS-AFTER-CLASSIFIER-V1-TEST.md §6; draft 36",
        "status": "COUNTS ONLY. A dry run of a drafted unit; nothing adopted or drawn. Predictions are for sizing only.",
        "input": input_record,
        "dispositions_by_source": {s: dict(sorted(c.items())) for s, c in sorted(dispositions.items())},
        "counts": {
            "first_reply_units": len(units),
            "construction_exclusions": exclusion_stats[coverage.LAYER_B],
            "pdet_coverage_v1_overlap_removed": len(kept) - len(not_coverage),
            "development_and_check_set_overlap_removed": len(not_coverage) - len(unused),
            "dedup": dedup_stats,
            "after_dedup": len(survivors),
        },
        "supply_after_dedup": {
            source: {
                name: {
                    tools: {kind: supply[(source, name, tools, kind)] for kind in kinds}
                    for tools in ("tools_offered", "no_tools")
                }
                for name in coverage.STRATA
            }
            for source in SOURCES
        },
        "continuations_after_dedup": {
            source: {"uses_tools_later": continuation[(source, True)], "prose_only_later": continuation[(source, False)]}
            for source in SOURCES
        },
        "sizing_only_frozen_classifier_v1": {
            "version": CLASSIFIER_VERSION,
            "source_sha256_lf": oneshot.FROZEN_SOURCE_SHA256_LF,
            "predictions_after_dedup": {
                source: {
                    name: {kind: {label: predicted[(source, name, kind, label)] for label in LABELS} for kind in kinds}
                    for name in coverage.STRATA
                }
                for source in SOURCES
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    text = json.dumps(audit(args.root), indent=2, sort_keys=True) + "\n"
    if args.write:
        path = args.root / OUTPUT
        if path.exists():
            raise SystemExit(f"{OUTPUT.as_posix()} already exists; it is never overwritten")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
