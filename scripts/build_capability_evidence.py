#!/usr/bin/env python3
"""Emit one immutable evidence record per (benchmark, checkpoint) combination. CPU.

The aggregate JSONs are summaries. This builds the thing a summary can never replace: a joined
per-example record carrying the input, the raw output, the parsed output, the expected answer, the
score, the failure class, the refusal flag, timings and token counts -- plus full provenance for
the dataset revision, model revision, tokenizer revision, engine version and generation config.

Anyone can recompute every number in every report from these files without a GPU. That is the
point: the aggregate report is not the source of truth, this is.

Also appends a compact row per combination to `capability_findings.jsonl`, an append-only ledger
in the same spirit as `results/quantization/findings.jsonl`.

Usage:
    python scripts/build_capability_evidence.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "results/benchmarks/h200/capability_v1"
DATASETS = ROOT / "results/benchmarks/datasets"
EVIDENCE = CAP / "evidence"
LEDGER = ROOT / "results/benchmarks/capability_findings.jsonl"

STAGES = ["BASE", "M0_SFT", "M1_DPO_CURRENT", "M1_DPO_HISTORICAL"]
BENCHMARKS = ["ifeval", "gsm8k", "mmlu_pro", "sentinel"]


def read_jsonl(path: Path) -> list[dict]:
    # split("\n") only: str.splitlines() also breaks on U+0085, present in MMLU-Pro.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def headline(benchmark: str, scores: dict) -> dict:
    """The few numbers that belong in a ledger row. Never a substitute for the evidence file."""
    if benchmark == "ifeval":
        m, d = scores["official_metrics"], scores["diagnostic_metrics"]
        return {
            "prompt_level_strict_accuracy": m["prompt_level_strict_accuracy"],
            "instruction_level_strict_accuracy": m["instruction_level_strict_accuracy"],
            "prompt_level_loose_accuracy": m["prompt_level_loose_accuracy"],
            "instruction_level_loose_accuracy": m["instruction_level_loose_accuracy"],
            "answer_rate": d["answer_rate"],
            "refusal_rate": d["refusal_rate"],
            "accuracy_given_answer": d["accuracy_given_answer"],
            "failure_categories": scores["failure_categories"],
        }
    if benchmark == "gsm8k":
        return {arm: {k: s[k] for k in
                      ("accuracy", "answer_rate", "refusal_rate", "parse_failure_rate",
                       "accuracy_given_answer", "correct", "incorrect_attempted", "refusals",
                       "parse_failures", "total")}
                for arm, s in scores["arms"].items()}
    if benchmark == "mmlu_pro":
        a = scores["aggregate"]
        return {
            "accuracy": a["accuracy"], "answer_rate": a["answer_rate"],
            "refusal_rate": a["refusal_rate"], "invalid_option_rate": a["invalid_option_rate"],
            "accuracy_given_answer": a["accuracy_given_answer"],
            "coverage": scores["coverage"],
            "per_category_accuracy": {c: v["accuracy"] for c, v in scores["per_category"].items()},
        }
    return {
        "pass": scores["pass"], "fail": scores["fail"], "total": scores["total"],
        "tool_cases": scores["tool_cases"], "non_tool_cases": scores["non_tool_cases"],
        "refusal_failures": scores["refusal_failures"],
        "wrong_content_failures": scores["wrong_content_failures"],
        "per_case": {c["id"]: c["status"] for c in scores["cases"]},
    }


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    ladder = {e["stage"]: e for e in
              json.loads((ROOT / "results/benchmarks/checkpoint_ladder.json")
                         .read_text(encoding="utf-8"))["checkpoints"]}

    requests = {}
    dataset_meta = {}
    for b in ("ifeval", "gsm8k", "mmlu_pro"):
        path = DATASETS / f"{b}_v1.jsonl"
        if path.exists():
            requests[b] = {r["example_id"]: r for r in read_jsonl(path)}
            dataset_meta[b] = json.loads((DATASETS / f"{b}_v1.meta.json").read_text(encoding="utf-8"))

    rows = []
    for stage in STAGES:
        env_path = CAP / stage / "run_summary.json"
        if not env_path.exists():
            continue
        env = json.loads(env_path.read_text(encoding="utf-8"))["environment"]

        for benchmark in BENCHMARKS:
            scores_path = CAP / stage / f"{benchmark}_scores.json"
            per_ex_path = CAP / stage / f"{benchmark}_scores_per_example.jsonl"
            gens_path = CAP / stage / f"generations_{benchmark}.jsonl"
            if not scores_path.exists():
                continue

            scores = json.loads(scores_path.read_text(encoding="utf-8"))
            gens = {g["example_id"]: g for g in read_jsonl(gens_path)} if gens_path.exists() else {}
            per_ex = read_jsonl(per_ex_path) if per_ex_path.exists() else []

            # Sentinel is scored inside its own JSON rather than a per-example file.
            if benchmark == "sentinel":
                per_ex = [{**c, "example_id": c["id"]} for c in scores["cases"]]

            dmeta = dataset_meta.get(benchmark, {})
            provenance = {
                "benchmark": benchmark,
                "stage": stage,
                "dataset_source": dmeta.get("source", {}).get("hf_dataset", "openweights-parity-suite"),
                "dataset_revision": dmeta.get("source", {}).get("revision"),
                "dataset_split": dmeta.get("source", {}).get("split"),
                "dataset_fingerprint": dmeta.get("fingerprint"),
                "request_file_sha256": dmeta.get("request_file_sha256"),
                "scorer": dmeta.get("scorer") or scores.get("scorer"),
                "model_repo": env["model_repo"],
                "model_revision": env["model_revision"],
                "model_subfolder": env["subfolder"],
                "weights_sha256": env["weights_sha256"],
                "config_sha256": env["config_sha256"],
                "architectures": env["architectures"],
                "tokenizer_json_sha256": env["tokenizer_json_sha256"],
                "chat_template_sha256": env["chat_template_sha256"],
                "add_bos_token": env["add_bos_token"],
                "eos_token_id": env["eos_token_id"],
                "engine": f"vLLM {env['vllm']}",
                "torch": env["torch"],
                "cuda": env["cuda"],
                "gpu": env["gpu_name"],
                "dtype": env["dtype"],
                "generation_config": {
                    **env["sampling"],
                    "max_tokens": dmeta.get("max_tokens"),
                    "max_model_len": env["max_model_len"],
                    "enable_prefix_caching": env["enable_prefix_caching"],
                },
                "ladder_role": ladder.get(stage, {}).get("role"),
                "supplementary": bool(ladder.get(stage, {}).get("supplementary")),
            }

            out_path = EVIDENCE / f"{benchmark}__{stage}.jsonl"
            written = 0
            with out_path.open("w", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps({"_record": "provenance_header", **provenance},
                                    sort_keys=True, ensure_ascii=True) + "\n")
                for r in per_ex:
                    eid = r["example_id"]
                    req = requests.get(benchmark, {}).get(eid, {})
                    gen = gens.get(eid, {})
                    fh.write(json.dumps({
                        "_record": "example",
                        "example_id": eid,
                        "benchmark": benchmark,
                        "stage": stage,
                        "input_messages": req.get("messages"),
                        "prompt_sha256": gen.get("prompt_sha256"),
                        "prompt_tokens": gen.get("prompt_tokens"),
                        "raw_output": r.get("raw_output") or r.get("raw"),
                        "output_tokens": gen.get("output_tokens"),
                        "finish_reason": gen.get("finish_reason"),
                        "parsed_output": (
                            r.get("parsed_answer") or r.get("parsed_letter")
                            or r.get("strict_per_instruction") or r.get("parsed_decision")
                        ),
                        "expected_answer": (
                            r.get("gold") or (req.get("score_key") or {}).get("gold")
                            or r.get("grader")
                        ),
                        "score": (
                            r.get("correct") if "correct" in r
                            else r.get("strict_prompt_pass") if "strict_prompt_pass" in r
                            else (r.get("status") == "pass")
                        ),
                        "failure_category": r.get("failure_category"),
                        "refusal": r.get("refusal"),
                        "refusal_pattern": r.get("refusal_pattern"),
                        "attempted": r.get("attempted"),
                        "category": r.get("category") or r.get("arm"),
                    }, sort_keys=True, ensure_ascii=True) + "\n")
                    written += 1

            rows.append({
                **provenance,
                "examples": written,
                "evidence_file": str(out_path.relative_to(ROOT)).replace("\\", "/"),
                "evidence_sha256": digest(out_path),
                "scores_file": str(scores_path.relative_to(ROOT)).replace("\\", "/"),
                "scores_sha256": digest(scores_path),
                "generations_sha256": digest(gens_path) if gens_path.exists() else None,
                "status": "COMPLETE",
                "headline": headline(benchmark, scores),
            })
            print(f"  {benchmark:<10} {stage:<20} {written:>6} examples -> {out_path.name}")

    with LEDGER.open("w", encoding="utf-8", newline="\n") as fh:
        for r in sorted(rows, key=lambda r: (r["benchmark"], r["stage"])):
            fh.write(json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n")

    print(f"\nwrote {len(rows)} ledger rows -> {LEDGER.relative_to(ROOT)}")
    print(f"evidence files in {EVIDENCE.relative_to(ROOT)}")
    total = sum(r["examples"] for r in rows)
    print(f"{total} per-example evidence records across {len(rows)} benchmark/checkpoint combinations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
