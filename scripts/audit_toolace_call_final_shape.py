"""Do ToolACE's call-final rows look like deliberate next-call targets or like truncated trajectories?
READ-ONLY; COUNTS ONLY.

`reports/SUPERVISION_CONTRACT_REPORT.md` §4 left ToolACE's call-final rows unclassified: a conversation that
ends on an unanswered call is consistent with next-call supervision and with a trajectory whose results
were lost, and resolving it "needs upstream evidence — a documented format, a maintainer statement, or a
construction pattern". This script measures the construction pattern on the raw upstream rows (the pinned
parquet, read through the canonical-v3 source manifest):

* **where rows end** — on an assistant call, assistant prose, a tool result or a user turn. Lost results
  would leave some rows ending on a tool or user turn;
* **whether earlier calls are resolved** — in a call-final row, how many earlier calls and tool results
  precede the final call;
* **exchange shape** — how many user turns a call-final row has;
* **prefix structure** — whether any row is a proper prefix of another row (overlapping cuts of one dialog);
* **subgroups the blanket reading would hide** — what the final call follows (a user turn or a tool result),
  whether the final turn is only the call, and whether each earlier call turn has as many tool results as
  calls (calls counted by the adapter-independent reader of `opengrad.verification.call_fidelity`).

`adapters.toolace_call_prediction_shape` admits only the shape these counts validate.

No record text is written or printed: ToolACE also supplies P-DET-COVERAGE-v1, whose annotator is blind.

    python scripts/audit_toolace_call_final_shape.py   # writes reports/normalization-v3/toolace-call-final-shape.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from opengrad.data.normalization_v3 import (
    file_sha256,
    iter_raw,
    load_source_manifest,
    source_specs,
)
from opengrad.verification import call_fidelity

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "normalization-v3" / "toolace-call-final-shape.json"
ASSISTANT = {"assistant", "gpt"}
TOOL = {"tool", "observation", "function"}
USER = {"user", "human"}


def is_call(turn: dict[str, Any]) -> bool:
    value = turn.get("value")
    return (
        turn.get("from") in ASSISTANT
        and isinstance(value, str)
        and value.lstrip().startswith("[")
        and "(" in value
    )


def ending(turns: list[dict[str, Any]]) -> str:
    if not turns:
        return "empty"
    last = turns[-1]
    if is_call(last):
        return "assistant_call"
    if last.get("from") in ASSISTANT:
        return "assistant_prose"
    if last.get("from") in TOOL:
        return "tool_result"
    if last.get("from") in USER:
        return "user"
    return f"other:{last.get('from')}"


def independent_call_count(value: str) -> int | None:
    """Calls in one call turn by the adapter-independent readers; ``None`` when neither can read it."""
    try:
        return len(call_fidelity.toolace_raw_calls([{"from": "assistant", "value": value}]))
    except call_fidelity.RawCallError:
        return None


def conversation_key(system: Any, turns: list[dict[str, Any]]) -> str:
    payload = [system, [[turn.get("from"), turn.get("value")] for turn in turns]]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


def measure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    endings: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    call_final: Counter[str] = Counter()
    unresolved_earlier = 0
    final_follows: Counter[str] = Counter()
    final_only_call = 0
    earlier_turn_results: Counter[str] = Counter()
    prose_final_with_calls = 0
    keys: Counter[str] = Counter()
    full_keys = set()
    prefix_keys = set()
    for row in rows:
        turns = list(row.get("conversations") or [])
        full_keys.add(conversation_key(row.get("system"), turns))
        keys[conversation_key(row.get("system"), turns)] += 1
        for end in range(1, len(turns)):
            prefix_keys.add(conversation_key(row.get("system"), turns[:end]))
        endings[ending(turns)] += 1
        roles.update(str(turn.get("from")) for turn in turns)
        if ending(turns) == "assistant_prose" and any(is_call(turn) for turn in turns):
            prose_final_with_calls += 1
        if ending(turns) != "assistant_call":
            continue
        previous = turns[-2].get("from") if len(turns) > 1 else None
        final_follows[
            "user" if previous in USER else "tool_result" if previous in TOOL else str(previous)
        ] += 1
        value = str(turns[-1].get("value"))
        final_only_call += value.strip().startswith("[") and value.strip().endswith("]")
        for index, turn in enumerate(turns[:-1]):
            if not is_call(turn):
                continue
            results = 0
            while (
                index + 1 + results < len(turns) and turns[index + 1 + results].get("from") in TOOL
            ):
                results += 1
            calls = independent_call_count(str(turn.get("value")))
            earlier_turn_results[
                "unreadable"
                if calls is None
                else "results_equal_calls"
                if results == calls
                else "results_fewer_than_calls"
                if results < calls
                else "results_more_than_calls"
            ] += 1
        earlier_calls = sum(is_call(turn) for turn in turns[:-1])
        earlier_results = sum(turn.get("from") in TOOL for turn in turns[:-1])
        user_turns = sum(turn.get("from") in USER for turn in turns)
        unresolved_earlier += earlier_results < earlier_calls
        call_final[
            f"user_turns={min(user_turns, 3)}|earlier_calls={min(earlier_calls, 3)}|"
            f"earlier_results={min(earlier_results, 3)}"
        ] += 1
    rows_that_prefix_another = sum(
        conversation_key(row.get("system"), list(row.get("conversations") or [])) in prefix_keys
        for row in rows
    )
    return {
        "rows": len(rows),
        "turn_roles": dict(sorted(roles.items())),
        "row_endings": dict(sorted(endings.items())),
        "call_final_rows": {
            "total": endings["assistant_call"],
            "single_exchange_user_then_call": call_final.get(
                "user_turns=1|earlier_calls=0|earlier_results=0", 0
            ),
            "with_an_earlier_call_lacking_a_result": unresolved_earlier,
            "final_call_follows": dict(sorted(final_follows.items())),
            "final_turn_is_only_the_call_block": final_only_call,
            "earlier_call_turns_by_result_count": dict(sorted(earlier_turn_results.items())),
            "by_shape": dict(sorted(call_final.items())),
        },
        "distinct_conversations": len(full_keys),
        "rows_with_an_exact_duplicate": sum(count for count in keys.values() if count > 1),
        "prose_final_rows_containing_calls": prose_final_with_calls,
        "rows_that_are_a_proper_prefix_of_another_row": rows_that_prefix_another,
    }


def main() -> int:
    spec = {entry.name: entry for entry in source_specs(load_source_manifest(ROOT), ROOT)}[
        "toolace"
    ]
    observed = file_sha256(spec.raw_path)
    if observed != spec.raw_sha256:
        raise SystemExit(f"raw ToolACE bytes are not the pinned ones: {observed}")
    result = measure(list(iter_raw(spec.raw_path)))
    report = {
        "artifact_kind": "TOOLACE_CALL_FINAL_SHAPE_AUDIT",
        "status": "COUNTS_ONLY_NOT_A_RECLASSIFICATION",
        "question": "reports/SUPERVISION_CONTRACT_REPORT.md §4: are ToolACE's call-final rows next-call "
        "targets or truncated trajectories?",
        "raw_artifact": {
            "path": spec.raw_path.relative_to(ROOT).as_posix(),
            "sha256": observed,
            "upstream_revision": spec.upstream_revision,
        },
        "result": result,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
