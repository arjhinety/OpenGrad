"""What reading ToolACE's call-final records as CALL_PREDICTION changed in normalization-v3. COUNTS ONLY.

Adapter version 2.2.0 (`adapt_toolace_v3`) changes only the declared supervision of ToolACE records;
their messages, tools and membership are those of `adapt_toolace_v2`. So the state before the change is
computed, not remembered: every accepted row is validated once under the supervision it now declares and
once under `COMPLETE_TRAJECTORY`, which is what `adapt_toolace_v2` declared for every record.

Every record that is still quarantined gets exactly one category, chosen from its structure:

* `call_final_after_tool_result` — the final call follows a tool result, a shape the rule does not admit;
* `call_final_result_count_mismatch` — an earlier call turn has fewer tool results than calls;
* `call_final_other` — any other call-final record the rule did not admit;
* `prose_final_result_count_mismatch` — a prose-final record whose call turn has fewer results than calls;
* `empty_assistant_turn` — an assistant turn with neither content nor calls (quarantined under any contract);
* `other` — anything else (expected 0).

No record text is written or printed: ToolACE also supplies P-DET-COVERAGE-v1, whose annotator is blind.

    python scripts/report_toolace_call_prediction.py   # writes reports/normalization-v3/toolace-call-prediction.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from opengrad.data import versions
from opengrad.data.adapters import TOOLACE_CALL_PREDICTION_EVIDENCE, toolace_call_prediction_shape
from opengrad.data.canonical import ToolConversation
from opengrad.data.normalization_v3 import OUTPUT_DIR, iter_rows
from opengrad.data.semantic import validate_training_trajectory
from opengrad.data.supervision import (
    SUPERVISION_METADATA_KEY,
    SupervisionAssignment,
    SupervisionKind,
    supervision_block,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / OUTPUT_DIR
OUT = ROOT / "reports" / "normalization-v3" / "toolace-call-prediction.json"


def issue_codes(row: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    conversation = ToolConversation(
        row["id"], row["source"], row["tools"], row["messages"], metadata
    )
    return sorted({issue.code for issue in validate_training_trajectory(conversation)})


def as_complete_trajectory(metadata: dict[str, Any]) -> dict[str, Any]:
    block = supervision_block(
        SupervisionKind.COMPLETE_TRAJECTORY,
        assignment=SupervisionAssignment.SOURCE_ADAPTER,
        adapter="toolace_v2",
        adapter_version=versions.ADAPTER_VERSION,
    )
    return {**metadata, SUPERVISION_METADATA_KEY: block}


def result_count_mismatch(messages: list[dict[str, Any]]) -> bool:
    turns = [message for message in messages if message.get("role") != "system"]
    for index, turn in enumerate(turns):
        calls = turn.get("tool_calls") if turn.get("role") == "assistant" else None
        if not calls or index == len(turns) - 1:
            continue
        answered = 0
        while (
            index + 1 + answered < len(turns) and turns[index + 1 + answered].get("role") == "tool"
        ):
            answered += 1
        if answered < len(calls):
            return True
    return False


def category(messages: list[dict[str, Any]], issues: list[str]) -> str:
    turns = [message for message in messages if message.get("role") != "system"]
    final = turns[-1] if turns else {}
    if "EMPTY_ASSISTANT_OUTPUT" in issues:
        return "empty_assistant_turn"
    if final.get("role") == "assistant" and final.get("tool_calls"):
        if len(turns) > 1 and turns[-2].get("role") == "tool":
            return "call_final_after_tool_result"
        if result_count_mismatch(messages):
            return "call_final_result_count_mismatch"
        return "call_final_other"
    if result_count_mismatch(messages):
        return "prose_final_result_count_mismatch"
    return "other"


def measure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    before: Counter[str] = Counter()
    after: Counter[str] = Counter()
    residual: Counter[str] = Counter()
    residual_issues: dict[str, Counter[str]] = {}
    for row in rows:
        metadata = row["metadata"]
        kind = metadata[SUPERVISION_METADATA_KEY]["kind"]
        final = row["messages"][-1] if row["messages"] else {}
        call_final = final.get("role") == "assistant" and bool(final.get("tool_calls"))
        old = issue_codes(row, as_complete_trajectory(metadata))
        new = issue_codes(row, metadata)
        if new != metadata["structure"]["trajectory_issue_codes"]:
            raise SystemExit(f"stored trajectory issues disagree with a recomputation: {row['id']}")
        shape = toolace_call_prediction_shape(row["messages"])
        if shape != (kind == SupervisionKind.CALL_PREDICTION.value):
            raise SystemExit(f"declared supervision disagrees with the rule: {row['id']}")

        before["accepted"] += 1
        before["trajectory_valid"] += not old
        before["quarantined"] += bool(old)
        before["quarantined_call_final"] += bool(old) and call_final
        after["accepted"] += 1
        after[f"declared_{kind}"] += 1
        after["trajectory_valid"] += not new
        after[f"trajectory_valid_{kind}"] += not new
        after["quarantined"] += bool(new)
        after["recovered_call_prediction"] += bool(old) and not new
        after["newly_quarantined"] += not old and bool(new)
        if new:
            name = category(row["messages"], new)
            residual[f"{kind}|{name}"] += 1
            residual_issues.setdefault(f"{kind}|{name}", Counter()).update(new)
    return {
        "before_complete_trajectory_for_every_record": dict(sorted(before.items())),
        "after_declared_supervision": dict(sorted(after.items())),
        "residual_quarantine_by_category": {
            key: {
                "records": residual[key],
                "issue_codes": dict(sorted(residual_issues[key].items())),
            }
            for key in sorted(residual)
        },
    }


def main() -> int:
    manifest = json.loads((ARTIFACT / "manifest.json").read_text(encoding="utf-8"))
    toolace = json.loads((ARTIFACT / "toolace" / "manifest.json").read_text(encoding="utf-8"))
    result = measure(list(iter_rows(ARTIFACT, "toolace")))
    report = {
        "artifact_kind": "TOOLACE_CALL_PREDICTION_RECLASSIFICATION",
        "statement": (
            "ToolACE call-final records of the validated shape declare CALL_PREDICTION from adapter version "
            f"{versions.ADAPTER_VERSION}. This is an OpenGrad corpus-level structural inference, not an "
            "upstream ToolACE annotation. Canonical-v2 is unchanged. Counts only; no record text."
        ),
        "evidence": TOOLACE_CALL_PREDICTION_EVIDENCE,
        "normalization_v3_fingerprint": manifest["fingerprint"],
        "adapter_version": versions.ADAPTER_VERSION,
        "source_counts": toolace["counts"],
        "rejected_by_reason": toolace["rejected_by_reason"],
        "result": result,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
