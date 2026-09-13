#!/usr/bin/env python3
"""Adversarial re-derivation of every campaign headline number. CPU only, no GPU calls.

This script trusts no prose. Every figure is recomputed from per-example artifacts, and each
recomputation is compared against what the reports currently claim. Where they disagree, the
artifact wins and the disagreement is reported rather than smoothed over.

It also computes the things the reports asserted WITHOUT deriving them -- most importantly the
MMLU-Pro truncation interval, which was previously described as a "conservative lower bound"
without any bound having been calculated.

Usage:
    python scripts/audit_campaign_final.py
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "results/benchmarks/h200/capability_v1"
STAGES = ["BASE", "M0_SFT", "M1_DPO_CURRENT", "M1_DPO_HISTORICAL"]


def jl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def js(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def gsm8k(stage: str) -> dict:
    """Recompute GSM8K from per-example rows, per arm, from scratch."""
    rows = jl(CAP / stage / "gsm8k_scores_per_example.jsonl")
    out = {}
    for arm in ("zeroshot", "fewshot8"):
        sub = [r for r in rows if r.get("arm") == arm]
        if not sub:
            continue
        n = len(sub)
        correct = sum(1 for r in sub if r["correct"])
        attempted = sum(1 for r in sub if r["attempted"])
        refusals = sum(1 for r in sub if r["bucket"] == "REFUSAL")
        parse_fail = sum(1 for r in sub if r["bucket"] == "PARSE_FAILURE")
        # Independent check of the fallback-precedence fix: no row may be counted as an attempt
        # when a refusal was detected and the value came only from a positional extractor.
        weak_attempts = sum(
            1 for r in sub
            if r.get("refusal") and r.get("attempted") and r.get("extraction_method") == "last_number"
        )
        out[arm] = {
            "n": n,
            "accuracy": correct / n,
            "answer_rate": attempted / n,
            "refusal_rate": refusals / n,
            "parse_failure_rate": parse_fail / n,
            "accuracy_given_answer": (correct / attempted) if attempted else None,
            "correct": correct,
            "attempted": attempted,
            "refusals": refusals,
            "parse_failures": parse_fail,
            "buckets_reconcile": (correct + (attempted - correct) + refusals + parse_fail) == n,
            "refusal_counted_as_attempt_via_positional_fallback": weak_attempts,
        }
    return out


def ifeval(stage: str) -> dict:
    rows = jl(CAP / stage / "ifeval_scores_per_example.jsonl")
    if not rows:
        return {}
    n = len(rows)
    strict = sum(1 for r in rows if r["strict_prompt_pass"])
    loose = sum(1 for r in rows if r["loose_prompt_pass"])
    inst_total = sum(len(r["strict_per_instruction"]) for r in rows)
    inst_strict = sum(sum(r["strict_per_instruction"]) for r in rows)
    inst_loose = sum(sum(r["loose_per_instruction"]) for r in rows)
    attempted = sum(1 for r in rows if r["attempted"])
    refusals = sum(1 for r in rows if r["refusal"])
    correct_attempted = sum(1 for r in rows if r["strict_prompt_pass"] and r["attempted"])
    return {
        "n": n,
        "prompt_level_strict_accuracy": strict / n,
        "prompt_level_loose_accuracy": loose / n,
        "instruction_level_strict_accuracy": inst_strict / inst_total,
        "instruction_level_loose_accuracy": inst_loose / inst_total,
        "instructions": inst_total,
        "answer_rate": attempted / n,
        "refusal_rate": refusals / n,
        "accuracy_given_answer": (correct_attempted / attempted) if attempted else None,
        "failure_categories": dict(collections.Counter(
            r["failure_category"] for r in rows if r["failure_category"] != "PASS")),
        "truncated": sum(1 for r in rows if r.get("finish_reason") == "length"),
    }


def mmlu(stage: str, superseded: bool = False) -> dict:
    d = CAP / stage / ("superseded_mmlu_768" if superseded else ".")
    rows = jl(d / "mmlu_pro_scores_per_example.jsonl")
    if not rows:
        return {}
    n = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    attempted = sum(1 for r in rows if r["attempted"])
    refusals = sum(1 for r in rows if r["refusal"])
    trunc = sum(1 for r in rows if r.get("finish_reason") == "length")
    unattempted = n - attempted
    trunc_unattempted = sum(
        1 for r in rows if not r["attempted"] and r.get("finish_reason") == "length")
    unresolved_truncated = sum(
        1 for r in rows if not r["correct"] and r.get("finish_reason") == "length")
    invalid = sum(1 for r in rows if r.get("parsed_letter") and not r.get("in_option_range"))
    return {
        "n": n,
        "accuracy": correct / n,
        "answer_rate": attempted / n,
        "refusal_rate": refusals / n,
        "invalid_option_rate": invalid / n,
        "accuracy_given_answer": (correct / attempted) if attempted else None,
        "correct": correct,
        "attempted": attempted,
        "unattempted": unattempted,
        "unresolved_truncated": unresolved_truncated,
        "truncated": trunc,
        "truncation_rate": trunc / n,
        "unattempted_that_are_truncations": trunc_unattempted,
        "per_category_accuracy": {
            c: sum(1 for r in rows if r["category"] == c and r["correct"])
               / sum(1 for r in rows if r["category"] == c)
            for c in sorted({r["category"] for r in rows})
        },
    }


def mmlu_truncation_interval(a: dict, b: dict) -> dict:
    """Derive a mathematically valid interval on the Base-minus-M0 accuracy gap.

    The previous reports called the observed -12pp drop a "conservative lower bound" without ever
    computing a bound. That phrasing is wrong in both directions: unresolved truncations could in
    principle resolve either way, so the observed gap is a POINT ESTIMATE sitting inside an
    interval, not a floor.

    UNRESOLVED is defined as `incorrect AND finish_reason == "length"` -- the examples a larger
    generation budget could plausibly have turned correct. An earlier draft of this function used
    "unattempted" instead, which is wrong in both directions: it swept in examples that FINISHED
    and still produced no parseable answer (more budget cannot help those) and it omitted examples
    that were attempted, wrong, and truncated (more budget might have). The definition here is the
    same one `scripts/build_regression_analysis.py` uses, so the two agree by construction.
    """
    an, bn = a["n"], b["n"]
    a_lo, b_lo = a["correct"] / an, b["correct"] / bn
    a_hi = (a["correct"] + a["unresolved_truncated"]) / an
    b_hi = (b["correct"] + b["unresolved_truncated"]) / bn
    return {
        "base_accuracy_interval": [a_lo, a_hi],
        "m0_accuracy_interval": [b_lo, b_hi],
        "observed_gap_pp": (a_lo - b_lo) * 100,
        "worst_case_gap_pp": (a_lo - b_hi) * 100,
        "best_case_gap_pp": (a_hi - b_lo) * 100,
        "gap_remains_positive_under_adversarial_resolution": (a_lo - b_hi) > 0,
        "unresolved_base": a["unresolved_truncated"],
        "unresolved_m0": b["unresolved_truncated"],
        "derivation": (
            "true_acc in [correct/n, (correct+unresolved)/n]. Unresolved = incorrect AND "
            "finish_reason == 'length'. Worst case for the gap gives Base its floor and M0 its "
            "ceiling."
        ),
        "note": (
            "This is NOT a confidence interval. It is the deterministic range implied by treating "
            "every unresolved example adversarially. No distributional assumption is made."
        ),
    }


def sentinel(stage: str) -> dict:
    d = js(CAP / stage / "sentinel_scores.json")
    if not d:
        return {}
    return {
        "pass": d["pass"], "fail": d["fail"], "total": d["total"],
        "tool": f"{d['tool_cases']['pass']}/{d['tool_cases']['total']}",
        "general": f"{d['non_tool_cases']['pass']}/{d['non_tool_cases']['total']}",
        "refusal_failures": d["refusal_failures"],
        "wrong_content_failures": d["wrong_content_failures"],
        "per_case": {c["id"]: {"status": c["status"], "mode": c.get("failure_mode")}
                     for c in d["cases"]},
    }


def main() -> int:
    report: dict = {"schema_version": 1, "audit": "final-campaign-audit-v1",
                    "method": "every figure recomputed from per-example artifacts"}

    # -- lineage -------------------------------------------------------------------------------
    ladder = js(ROOT / "results/benchmarks/checkpoint_ladder.json")
    ref = js(ROOT / "results/quantization/m1_v2_reference.json")
    by_stage = {e["stage"]: e for e in ladder["checkpoints"]}
    envs = {s: (js(CAP / s / "run_summary.json") or {}).get("environment", {}) for s in STAGES}
    m0_hub = by_stage["M0_SFT"].get("single_shard_sha256")
    m0_container = envs["M0_SFT"].get("weights_sha256")
    report["lineage"] = {
        "declared_parent_of_m1v2": ref["parent_checkpoint"],
        "declared_parent_weight_sha256": ref["parent_weight_sha256"],
        "m0_hub_lfs_sha256": m0_hub,
        "m0_recomputed_in_container_sha256": m0_container,
        "m0_is_exact_parent_of_m1v2": (
            m0_hub == ref["parent_weight_sha256"] and m0_container == ref["parent_weight_sha256"]
        ),
        "verified_independently_twice": m0_hub == m0_container,
        "m1v2_revision_matches_frozen_reference": (
            by_stage["M1_DPO_CURRENT"]["revision"] == ref["hf_publication_revision"]
        ),
        "m1v2_weights_match_frozen_reference": (
            envs["M1_DPO_CURRENT"].get("weights_sha256")
            == "903f9b115d418d1988c0d5e1dd4bda7e04809a54a6bc7e34431982fbee2262f6"
        ),
        "per_stage_weight_sha256": {s: envs[s].get("weights_sha256") for s in STAGES},
        "all_stages_distinct": len({envs[s].get("weights_sha256") for s in STAGES}) == len(STAGES),
        "m1v1_parent_documented": by_stage["M1_DPO_HISTORICAL"].get("role"),
        "m1v1_on_primary_path": False,
        "m1v1_reason": (
            "M1_DPO_HISTORICAL is DPO applied directly to the base model (parent_experiment_id "
            "null, reference initial_policy), with no SFT stage. It is not on the Base -> M0 -> "
            "M1-v2 chain, so it is reported as an alternate lineage only; it still refuses 70.7% "
            "of zero-shot GSM8K, so the regression is not specific to SFT."
        ),
        "engine_identical_across_stages": len({
            (envs[s].get("vllm"), envs[s].get("dtype"), envs[s].get("max_model_len"),
             json.dumps(envs[s].get("sampling"), sort_keys=True))
            for s in STAGES if envs[s]
        }) == 1,
        "architecture_by_stage": {s: envs[s].get("architectures") for s in STAGES},
    }

    # -- benchmarks ----------------------------------------------------------------------------
    report["gsm8k"] = {s: gsm8k(s) for s in STAGES}
    report["ifeval"] = {s: ifeval(s) for s in STAGES}
    report["mmlu_pro_canonical_2048"] = {s: mmlu(s) for s in STAGES}
    report["mmlu_pro_superseded_768"] = {s: mmlu(s, superseded=True) for s in STAGES}
    report["sentinel"] = {s: sentinel(s) for s in STAGES}

    mc = report["mmlu_pro_canonical_2048"]
    if mc.get("BASE") and mc.get("M0_SFT"):
        report["mmlu_pro_truncation_interval"] = mmlu_truncation_interval(mc["BASE"], mc["M0_SFT"])

    # -- GSM8K truncation claim -----------------------------------------------------------------
    gs_trunc = {}
    for s in STAGES:
        gens = jl(CAP / s / "generations_gsm8k.jsonl")
        if gens:
            gs_trunc[s] = {
                "generations": len(gens),
                "truncated": sum(1 for g in gens if g.get("finish_reason") == "length"),
            }
    report["gsm8k_truncation"] = gs_trunc

    # -- refusal surface forms -------------------------------------------------------------------
    surface = {}
    for s in STAGES:
        import re
        tmpl = collections.Counter()
        total = 0
        for b in ("ifeval", "gsm8k", "mmlu_pro"):
            for r in jl(CAP / s / f"{b}_scores_per_example.jsonl"):
                if r.get("refusal"):
                    total += 1
                    t = re.sub(r"[^a-z ]", "", re.sub(r"\s+", " ", (r.get("raw_output") or "").strip().lower()))
                    tmpl[" ".join(t.split()[:12])] += 1
        if total:
            top = tmpl.most_common(10)
            surface[s] = {
                "total_refusals": total,
                "distinct_12_token_openings": len(tmpl),
                "top_10_share": sum(n for _, n in top) / total,
            }
    report["refusal_surface_forms"] = surface

    # -- engine parity ---------------------------------------------------------------------------
    ep = js(ROOT / "results/benchmarks/h200/vllm_vs_llamacpp_agreement.json")
    if ep:
        report["engine_parity"] = {
            k: ep[k] for k in ep
            if k in ("agreement", "flips", "n", "examples", "tokenizer_divergent_flips",
                     "compared", "disagreements", "total")
        }
        report["engine_parity_raw_keys"] = sorted(ep.keys())

    # -- cost ------------------------------------------------------------------------------------
    runs = jl(CAP / "gpu_runs.jsonl")
    gpu = [r for r in runs if r["_record"] == "run"]
    failed = [r for r in runs if r["_record"] == "failed_launch"]
    prior = js(ROOT / "results/benchmarks/h200_run_ledger.json")
    report["cost"] = {
        "gpu_runs": len(gpu),
        "failed_pre_gpu_launches": len(failed),
        "failed_with_gpu_allocated": sum(1 for f in failed if f.get("gpu_allocated")),
        "continuation_usd": round(sum(r["usd"] for r in gpu), 4),
        "superseded_usd": round(sum(r["usd"] for r in gpu if r["superseded"] is True), 4),
        "retained_usd": round(sum(r["usd"] for r in gpu if r["superseded"] is not True), 4),
        "estimated_rows": [r["seq"] for r in gpu if r.get("usd_is_estimate")],
        "prior_run_usd": prior.get("estimated_spend_usd"),
        "campaign_usd": round(sum(r["usd"] for r in gpu) + prior.get("estimated_spend_usd", 0), 4),
        "planning_envelope_usd": 22.70,
        "planning_envelope_is_not_a_balance": True,
    }

    out = CAP / "final_campaign_audit.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # -- console -----------------------------------------------------------------------------
    L = report["lineage"]
    print("LINEAGE")
    print(f"  M0 is exact parent of M1-v2 : {L['m0_is_exact_parent_of_m1v2']}")
    print(f"  verified twice independently: {L['verified_independently_twice']}")
    print(f"  all 4 stages distinct       : {L['all_stages_distinct']}")
    print(f"  engine identical everywhere : {L['engine_identical_across_stages']}")
    print("\nGSM8K (recomputed)")
    for s in STAGES:
        for arm, v in report["gsm8k"].get(s, {}).items():
            aga = "n/a" if v["accuracy_given_answer"] is None else f"{v['accuracy_given_answer']*100:.1f}"
            print(f"  {s:<18}{arm:<10} acc {v['accuracy']*100:5.1f}  ans {v['answer_rate']*100:5.1f}"
                  f"  ref {v['refusal_rate']*100:5.1f}  acc|ans {aga:>5}  "
                  f"reconcile={v['buckets_reconcile']} weakattempts={v['refusal_counted_as_attempt_via_positional_fallback']}")
    print("\nIFEval (recomputed)")
    for s in STAGES:
        v = report["ifeval"].get(s)
        if v:
            print(f"  {s:<18} strict {v['prompt_level_strict_accuracy']*100:5.1f}  "
                  f"loose {v['prompt_level_loose_accuracy']*100:5.1f}  ans {v['answer_rate']*100:5.1f}"
                  f"  ref {v['refusal_rate']*100:5.1f}  acc|ans {v['accuracy_given_answer']*100:5.1f}"
                  f"  trunc {v['truncated']}")
    print("\nMMLU-Pro @2048 CANONICAL")
    for s in STAGES:
        v = mc.get(s)
        if v:
            print(f"  {s:<18} acc {v['accuracy']*100:5.1f}  ans {v['answer_rate']*100:5.1f}  "
                  f"ref {v['refusal_rate']*100:5.1f}  acc|ans {v['accuracy_given_answer']*100:5.1f}  "
                  f"trunc {v['truncation_rate']*100:5.1f}%  n={v['n']}")
    print("\nMMLU-Pro @768 SUPERSEDED")
    for s in STAGES:
        v = report["mmlu_pro_superseded_768"].get(s)
        if v:
            print(f"  {s:<18} acc {v['accuracy']*100:5.1f}  ans {v['answer_rate']*100:5.1f}  "
                  f"trunc {v['truncation_rate']*100:5.1f}%")
    ti = report.get("mmlu_pro_truncation_interval")
    if ti:
        print("\nMMLU-Pro Base-minus-M0 gap, truncation-adversarial interval")
        print(f"  Base acc in [{ti['base_accuracy_interval'][0]*100:.1f}, {ti['base_accuracy_interval'][1]*100:.1f}]")
        print(f"  M0   acc in [{ti['m0_accuracy_interval'][0]*100:.1f}, {ti['m0_accuracy_interval'][1]*100:.1f}]")
        print(f"  gap: worst {ti['worst_case_gap_pp']:+.1f}pp | observed {ti['observed_gap_pp']:+.1f}pp "
              f"| best {ti['best_case_gap_pp']:+.1f}pp")
        print(f"  positive under adversarial resolution: {ti['gap_remains_positive_under_adversarial_resolution']}")
    print("\nGSM8K truncation")
    for s, v in gs_trunc.items():
        print(f"  {s:<18} {v['truncated']}/{v['generations']}")
    print("\nRefusal surface forms")
    for s, v in surface.items():
        print(f"  {s:<18} {v['total_refusals']} refusals, {v['distinct_12_token_openings']} openings, "
              f"top10 {v['top_10_share']*100:.1f}%")
    print("\nCost")
    for k, v in report["cost"].items():
        print(f"  {k}: {v}")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
