#!/usr/bin/env python3
"""Grade the OpenGrad run of the OpenWeights ParitySuite, using OpenWeights' own graders.

The graders are transcribed verbatim from `ParitySuite.kt` into
`results/benchmarks/openweights_parity_cases_v1.json`; this applies them unchanged so the grade is
OpenWeights' grade, not a re-interpretation. The comparison table then places OpenGrad beside the
device results OpenWeights actually recorded.

What is and is not comparable is stated on every row: capability grades are comparable because the
cases and graders are identical; latency and memory are not, because OpenWeights ran quantized
models on a phone and this ran BF16 on an H200.

Usage:
    python scripts/grade_openweights_parity.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opengrad.formatting.parser import parse_qwen_native_output

SPEC = ROOT / "results/benchmarks/openweights_parity_cases_v1.json"
RUN = ROOT / "results/benchmarks/h200/capability_run.json"
OUT_JSON = ROOT / "results/benchmarks/openweights_parity_opengrad_v1.json"
OUT_MD = ROOT / "reports/OPENWEIGHTS_TRANSFER_EVALUATION.md"

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
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    run = json.loads(RUN.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in run["parity"]["results"]}
    env = run["parity"]["environment"]

    rows = []
    for case in spec["cases"]:
        result = by_id.get(case["id"])
        if result is None:
            rows.append({"id": case["id"], "status": "missing"})
            continue
        raw = result["raw"]
        ok, rule = grade(case, raw)
        parsed = parse_qwen_native_output(raw)
        rows.append({
            "id": case["id"],
            "status": "pass" if ok else "fail",
            "grader": rule,
            "tool_case": bool(case.get("tools")),
            "raw": raw,
            "parsed_decision": parsed.decision,
            "parser_status": parsed.status,
            "calls": [{"name": c.name, "arguments": c.arguments} for c in parsed.calls],
            "turns": result["turns"],
        })

    passed = sum(1 for r in rows if r["status"] == "pass")
    failed = sum(1 for r in rows if r["status"] == "fail")
    tool_rows = [r for r in rows if r.get("tool_case")]
    nontool_rows = [r for r in rows if not r.get("tool_case") and r["status"] != "missing"]

    payload = {
        "suite": spec["suite"],
        "model": "OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2",
        "engine": "vLLM 0.29.0 (H200, BF16)",
        "checkpoint_sha256": env["weights_sha256"],
        "chat_template_sha256": env["chat_template_sha256"],
        "pass": passed,
        "fail": failed,
        "total": len(rows),
        "tool_cases": {
            "pass": sum(1 for r in tool_rows if r["status"] == "pass"),
            "total": len(tool_rows),
        },
        "non_tool_cases": {
            "pass": sum(1 for r in nontool_rows if r["status"] == "pass"),
            "total": len(nontool_rows),
        },
        "cases": rows,
        "recorded_device_results": spec["recorded_device_results"],
        "comparability_caveats": spec["comparability_caveats"],
        "template_caveat": spec["template_caveat"],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # -- leaderboard -------------------------------------------------------------------------
    board = [("OpenGrad-Qwen3.5-2B (BF16, vLLM/H200)", "vLLM", passed, failed, 0, True)]
    for name, rec in spec["recorded_device_results"].items():
        if name == "note":
            continue
        board.append((name, rec["engine"].replace("Engine", ""), rec["pass"], rec["fail"],
                      rec["skipped"], False))
    board.sort(key=lambda r: (-r[2], r[0]))

    board_md = "\n".join(
        f"| {'**' + n + '**' if mine else n} | {e} | {'**' + str(p) + '/7**' if mine else f'{p}/7'} | {f} | {s} |"
        for n, e, p, f, s, mine in board
    )
    case_md = "\n".join(
        f"| `{r['id']}` | {'tool' if r.get('tool_case') else 'general'} | "
        f"{'✅ pass' if r['status'] == 'pass' else '❌ fail'} | "
        f"`{(r.get('raw') or '')[:70].strip()}…` |"
        for r in rows
    )

    body = f"""# OpenWeights transfer evaluation — OpenGrad Qwen3.5-2B

OpenGrad run against **OpenWeights' own ParitySuite**, using OpenWeights' cases and graders
transcribed verbatim from `ParitySuite.kt`. The comparison targets are the results OpenWeights
actually recorded on-device, not re-runs.

## Headline

**{passed}/7 pass.** Both tool cases pass ({payload['tool_cases']['pass']}/{payload['tool_cases']['total']});
{payload['non_tool_cases']['pass']}/{payload['non_tool_cases']['total']} general-capability cases pass.

The two failures share one mode: the model **refuses a task it is capable of**, replying
"Apologies, but I'm unable to…". That is an over-refusal / alignment-tax signature, not a
tool-policy failure.

## Per case

| case | kind | result | output |
|---|---|---|---|
{case_md}

## Against the models OpenWeights measured

| model | engine | pass | fail | skipped |
|---|---|---|---|---|
{board_md}

**OpenGrad sits mid-pack and below Qwen3-1.7B**, a smaller model that scores 7/7 on both llama.cpp
and ExecuTorch. The gap is entirely in general capability, not tool use.

## What this does and does not show

- **Tool-use transfers.** `tool-call` emits the right function with the right argument, and
  `tool-result` correctly reads back the injected result (`31`). The trained tool policy survives
  into a realistic multi-turn agent shape.
- **General capability regressed.** `multi-step-change` (100 − 7×12 = 16) and `format-constraint`
  (a three-element JSON array) are well within a 2B model's ability; the model declines both. This
  is consistent with the frozen confirmatory partition containing **no ANSWER examples**, so
  nothing in the primary evaluation could have detected it.
- **Not a device result.** This is BF16 on an H200 via vLLM. OpenWeights' rows are quantized models
  on a phone. Capability grades are comparable because the cases and graders are identical;
  **latency, memory and throughput are not.**

## Caveats

{chr(10).join('- ' + c for c in spec["comparability_caveats"])}
- {spec['template_caveat']}

## Provenance

| | |
|---|---|
| checkpoint sha256 | `{env['weights_sha256']}` |
| chat template sha256 | `{env['chat_template_sha256']}` |
| engine | vLLM {env['vllm']}, torch {env['torch']}, CUDA {env['cuda']} |
| GPU | {env['gpu_name']} |
| sampling | greedy, `temperature 0.0`, `top_k 1`, `top_p 1.0`, seed {env['seed']} |
| max tokens | {spec['params']['max_tokens']} |
| raw results | `results/benchmarks/openweights_parity_opengrad_v1.json` |
"""
    OUT_MD.write_text(body, encoding="utf-8")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    print(f"\nOpenGrad: {passed}/7 pass  (tool {payload['tool_cases']['pass']}/{payload['tool_cases']['total']}, "
          f"general {payload['non_tool_cases']['pass']}/{payload['non_tool_cases']['total']})")
    for r in rows:
        print(f"  {r['id']:<22} {r['status']:<5} {'[tool]' if r.get('tool_case') else '      '} "
              f"{(r.get('raw') or '')[:60]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
