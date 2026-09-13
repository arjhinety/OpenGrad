#!/usr/bin/env python3
"""Grade one stage's run of the 7-case OpenWeights sentinel, using OpenWeights' own graders.

This is a REGRESSION SMOKE TEST, not a capability benchmark. Seven cases cannot support a claim
about general capability; what they can do is show, cheaply, whether the answer->refuse transition
is present at a given stage. The statistically meaningful evidence is IFEval/GSM8K/MMLU-Pro.

The cases and graders are the ones transcribed verbatim from OpenWeights' `ParitySuite.kt`. They
are not edited after seeing any model's output -- a sentinel that moves when results are
inconvenient measures nothing.

Usage:
    python scripts/score_sentinel.py <generations_sentinel.jsonl> --stage M0_SFT --out <out.json>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opengrad.evaluation.capability import detect_refusal
from opengrad.formatting.parser import parse_qwen_native_output

SPEC = ROOT / "results/benchmarks/openweights_parity_cases_v1.json"
FLAGS = {"i": re.IGNORECASE, "": 0}


def grade(case: dict, raw: str) -> tuple[bool, str]:
    g = case["grader"]
    if g["type"] == "regex":
        pattern = re.compile(g["pattern"], FLAGS.get(g.get("flags", ""), 0))
        return bool(pattern.search(raw)), f"regex /{g['pattern']}/{g.get('flags','')}"
    if g["type"] == "tool_call":
        parsed = parse_qwen_native_output(raw)
        ok = any(
            c.name == g["name"]
            and g["arguments_must_contain"] in json.dumps(c.arguments).lower()
            for c in parsed.calls
        )
        return ok, f"tool_call name={g['name']} args contain {g['arguments_must_contain']!r}"
    raise ValueError(f"unknown grader: {g['type']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("generations")
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    gens = {}
    for line in Path(args.generations).read_text(encoding="utf-8").split("\n"):
        if line.strip():
            g = json.loads(line)
            gens[g.get("id") or g["example_id"]] = g

    rows = []
    for case in spec["cases"]:
        gen = gens.get(case["id"])
        if gen is None:
            rows.append({"id": case["id"], "status": "missing"})
            continue
        raw = gen.get("raw") or gen.get("output") or ""
        ok, rule = grade(case, raw)
        parsed = parse_qwen_native_output(raw)
        refusal = detect_refusal(raw)
        rows.append({
            "id": case["id"],
            "status": "pass" if ok else "fail",
            "grader": rule,
            "tool_case": bool(case.get("tools")),
            "refusal": refusal.is_refusal,
            "refusal_pattern": refusal.pattern,
            "failure_mode": (None if ok else "REFUSAL" if refusal.is_refusal else "WRONG_CONTENT"),
            "raw": raw,
            "parsed_decision": parsed.decision,
            "parser_status": parsed.status,
            "calls": [{"name": c.name, "arguments": c.arguments} for c in parsed.calls],
        })

    passed = sum(1 for r in rows if r["status"] == "pass")
    failed = sum(1 for r in rows if r["status"] == "fail")
    tool_rows = [r for r in rows if r.get("tool_case")]
    nontool = [r for r in rows if not r.get("tool_case") and r["status"] != "missing"]

    payload = {
        "suite": spec["suite"],
        "role": "SENTINEL / REGRESSION SMOKE TEST -- not a general-capability benchmark",
        "stage": args.stage,
        "pass": passed,
        "fail": failed,
        "total": len(rows),
        "tool_cases": {"pass": sum(1 for r in tool_rows if r["status"] == "pass"),
                       "total": len(tool_rows)},
        "non_tool_cases": {"pass": sum(1 for r in nontool if r["status"] == "pass"),
                           "total": len(nontool)},
        "refusal_failures": sum(1 for r in rows if r.get("failure_mode") == "REFUSAL"),
        "wrong_content_failures": sum(1 for r in rows if r.get("failure_mode") == "WRONG_CONTENT"),
        "cases": rows,
        "generation_metadata": next(iter(gens.values())).get("generation_metadata") if gens else None,
        "statistical_caveat": (
            "7 cases. A one-case change is 14 percentage points. Direction here is a signal to "
            "check against the large benchmarks, never a result on its own."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"{args.stage}  sentinel {passed}/7  (tool {payload['tool_cases']['pass']}/"
          f"{payload['tool_cases']['total']}, general {payload['non_tool_cases']['pass']}/"
          f"{payload['non_tool_cases']['total']})  refusal-failures {payload['refusal_failures']}")
    for r in rows:
        mode = f" [{r['failure_mode']}]" if r.get("failure_mode") else ""
        print(f"    {r['id']:<22} {r['status']:<5}{mode:<18} {(r.get('raw') or '')[:58]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
