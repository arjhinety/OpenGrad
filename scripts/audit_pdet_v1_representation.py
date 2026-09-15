"""P-DET-v1 against normalization-v3: a representation-mismatch AUDIT, not a migration. READ-ONLY.

P-DET-v1 (581 items, frozen) was drawn from ``normalization-v1/when2call-sft``. The future classifier will
receive normalization-v3 records through the prose-decision-input-v1 contract. This script measures,
item by item, whether what an annotator saw in P-DET-v1 is what the classifier would receive:

* is the item's raw record present in normalization-v3 (matched by ``raw_record_hash``), or was it
  rejected or deduplicated there;
* is the user message identical, the assistant response identical, and are the tools identical or
  different only by the documented schema translation;
* is the v3 record eligible under the contract, and if not, why.

Nothing in P-DET-v1 or ``reports/pdet/`` is read for writing or changed. The output holds ids and
difference categories only, no text.

    python scripts/audit_pdet_v1_representation.py   # writes reports/normalization-v3/pdet-v1-representation-audit.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from opengrad.data.canonical import stable_json
from opengrad.data.classifier_input import HeldoutIndex, eligibility, normalize_prompt
from opengrad.data.normalization_v3 import OUTPUT_DIR, iter_rows
from opengrad.data.schema import normalize_tool
from opengrad.data.source_schema import translate_source_tools

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / OUTPUT_DIR
POPULATION = ROOT / "reports" / "pdet" / "pdet-v1.population.jsonl"
MANIFEST = ROOT / "reports" / "pdet" / "pdet-v1.manifest.json"
OUT = ROOT / "reports" / "normalization-v3" / "pdet-v1-representation-audit.json"


def _tools_relation(v1_tools: list[dict[str, Any]], v3_tools: list[dict[str, Any]]) -> str:
    if stable_json(v1_tools) == stable_json(v3_tools):
        return "identical"
    translated, _changes, quarantine, _detail = translate_source_tools(v1_tools)
    if quarantine is None:
        try:
            if stable_json([normalize_tool(tool) for tool in translated]) == stable_json(v3_tools):
                return "schema_translation_only"
        except (TypeError, ValueError):
            pass
    if [tool.get("name") for tool in v1_tools] == [tool.get("name") for tool in v3_tools]:
        return "same_names_other_schema_difference"
    return "different_tool_set"


def main() -> int:
    items = [
        json.loads(line) for line in POPULATION.read_text(encoding="utf-8").splitlines() if line
    ]
    heldout = HeldoutIndex.load(ROOT)
    by_hash: dict[str, dict[str, Any]] = {}
    by_index: dict[int, dict[str, Any]] = {}
    for record in iter_rows(ARTIFACT, "when2call"):
        by_hash[record["metadata"]["raw_record_hash"]] = record
        by_index[record["metadata"]["source"]["raw_row_index"]] = record
    ledger = {
        entry["raw_record_hash"]: entry
        for entry in (
            json.loads(line)
            for line in (ARTIFACT / "when2call" / "dispositions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        )
    }

    summary: Counter[str] = Counter()
    differing: list[dict[str, Any]] = []
    for item in items:
        raw_hash = item["raw_record_hash"]
        record = by_hash.get(raw_hash)
        presence = "present"
        if record is None:
            entry = ledger.get(raw_hash)
            if entry is None:
                presence = "absent"
            elif entry["disposition"] == "duplicate":
                presence = "deduplicated_to_survivor"
                record = by_index.get(entry["duplicate_of_raw_row_index"])
            else:
                presence = f"rejected:{entry['reason']}"
        summary[f"presence:{presence}"] += 1
        if record is None:
            differing.append({"pdet_id": item["pdet_id"], "presence": presence})
            continue
        users = [m["content"] for m in record["messages"] if m.get("role") == "user"]
        final = record["messages"][-1]
        user_same = users == [item["prompt"]]
        response_same = (
            final.get("role") == "assistant" and final.get("content") == item["response"]
        )
        tools = _tools_relation(item["tools"], record["tools"])
        result = eligibility(record, heldout)
        facts = {
            "user_message": "identical"
            if user_same
            else (
                "normalized_equal"
                if [normalize_prompt(u) for u in users] == [normalize_prompt(item["prompt"])]
                else "different"
            ),
            "assistant_response": "identical" if response_same else "different",
            "tools": tools,
            "upstream_id": "identical" if record["id"] == item["upstream_id"] else "different",
            "canonical_hash": "identical"
            if record["canonical_hash"] == item["canonical_hash"]
            else "different",
            "eligibility": "ELIGIBLE" if result.eligible else "+".join(result.reasons),
        }
        for key, value in facts.items():
            summary[f"{key}:{value}"] += 1
        equivalent = (
            presence == "present"
            and user_same
            and response_same
            and tools in {"identical", "schema_translation_only"}
            and result.eligible
        )
        summary["structurally_equivalent" if equivalent else "not_structurally_equivalent"] += 1
        if not equivalent:
            differing.append({"pdet_id": item["pdet_id"], "presence": presence, **facts})

    report = {
        "artifact_kind": "PDET_V1_REPRESENTATION_AUDIT",
        "statement": (
            "An audit, not a migration. P-DET-v1 is unchanged. Each frozen item is matched by raw_record_hash "
            "to normalization-v3 When2Call and compared on what the classifier input contract makes visible."
        ),
        "definition_structurally_equivalent": (
            "present in normalization-v3 (not rejected or deduplicated), user message and assistant response "
            "byte-identical, tools identical or different only by source-schema-normalization-v1, and eligible "
            "under prose-decision-input-v1"
        ),
        "pdet_v1_population_sha256": hashlib.sha256(POPULATION.read_bytes()).hexdigest(),
        "pdet_v1_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "normalization_v3_fingerprint": json.loads(
            (ARTIFACT / "manifest.json").read_text(encoding="utf-8")
        )["fingerprint"],
        "unit": "P-DET-v1 items",
        "items": len(items),
        "summary": dict(sorted(summary.items())),
        "not_structurally_equivalent": differing,
        "metadata_differences_not_visible_to_the_classifier": {
            "split": "normalization-v1 recorded train_sft (the HF config); normalization-v3 records train (the registry-allowlisted split)",
            "adapter_version": "v1 rows 1.0.0 (record) / 1.0.2 (manifest); v3 rows 2.0.0",
            "behavior": "v1 rows carry a message-shape ANSWER default; v3 rows carry none",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    for key, value in sorted(summary.items()):
        print(f"{key:60} {value}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
