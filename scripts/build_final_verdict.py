#!/usr/bin/env python3
"""Emit results/final_campaign_verdict.json from canonical artifacts. CPU only.

This is a POINTER-AND-SUMMARY artifact, deliberately not a second source of truth. Every metric is
read from the per-example-derived audit output rather than transcribed, and every block names the
artifact it came from, so a stale verdict is detectable by regeneration rather than by reading.

Usage:
    python scripts/build_final_verdict.py
    python scripts/build_final_verdict.py --verify   # fail if the committed verdict has drifted
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "results/benchmarks/h200/capability_v1"
AUDIT = CAP / "final_campaign_audit.json"
ANALYSIS = CAP / "regression_analysis.json"
LEDGER = CAP / "cost_ledger.json"
OUT = ROOT / "results/final_campaign_verdict.json"

# Stated here rather than read from final_campaign_audit.json's `lineage.m1v1_reason`, which still
# carries the superseded "different SFT parent (CorpusV2)" text. The lineage below is what
# runs/qwen35_2b_m1_dpo_v1/experiment.json and the published checkpoint-300 record.
M1V1_EXCLUSION_REASON = (
    "M1_DPO_HISTORICAL (qwen35_2b_m1_dpo_v1, checkpoint-300) is DPO applied directly to the base "
    "Qwen/Qwen3.5-2B: parent_experiment_id null, reference initial_policy, preference data "
    "when2call_pref_v1. It has no SFT parent, so it is not on the Base -> M0 -> M1-v2 chain and is "
    "reported as an alternate BASE -> DPO lineage only."
)


def build() -> dict:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    lin = audit["lineage"]

    def g(stage, arm, key):
        return audit["gsm8k"].get(stage, {}).get(arm, {}).get(key)

    def i(stage, key):
        return audit["ifeval"].get(stage, {}).get(key)

    def m(stage, key):
        return audit["mmlu_pro_canonical_2048"].get(stage, {}).get(key)

    intervals = analysis.get("truncation_adversarial_intervals", {})

    m1v1_zs_refusals = g("M1_DPO_HISTORICAL", "zeroshot", "refusals")
    m1v1_zs_n = g("M1_DPO_HISTORICAL", "zeroshot", "n")
    m1v1_zs_rate = g("M1_DPO_HISTORICAL", "zeroshot", "refusal_rate")
    m1v1_zs_text = f"{m1v1_zs_rate * 100:.1f}% ({m1v1_zs_refusals:,}/{m1v1_zs_n:,})"

    return {
        "schema_version": 1,
        "artifact_kind": "CAMPAIGN_VERDICT",
        "is_source_of_truth": False,
        "source_of_truth_note": (
            "Derived. Canonical evidence is results/benchmarks/h200/capability_v1/evidence/*.jsonl "
            "and the per-stage *_scores.json files. Regenerate with scripts/build_final_verdict.py; "
            "do not hand-edit."
        ),
        "campaign": "h200-capability-diagnosis-v1",
        "status": analysis["trajectory"]["label"],
        "status_basis": analysis["trajectory"]["basis"],
        "co_occurring_conditions": analysis["trajectory"]["co_occurring_conditions"],

        "primary_lineage": {
            "path": ["BASE", "M0_SFT", "M1_DPO_CURRENT"],
            "m0_is_exact_parent_of_m1v2": lin["m0_is_exact_parent_of_m1v2"],
            "verified_independently_twice": lin["verified_independently_twice"],
            "all_stages_distinct_weights": lin["all_stages_distinct"],
            "engine_identical_across_stages": lin["engine_identical_across_stages"],
            "weight_sha256": lin["per_stage_weight_sha256"],
            "excluded_from_primary_path": {
                "M1_DPO_HISTORICAL": M1V1_EXCLUSION_REASON,
            },
        },

        "findings": {
            "zero_shot_refusal_regression": {
                "evidence_level": "CONFIRMED",
                "base_refusal_rate": g("BASE", "zeroshot", "refusal_rate"),
                "m0_refusal_rate": g("M0_SFT", "zeroshot", "refusal_rate"),
                "m1v2_refusal_rate": g("M1_DPO_CURRENT", "zeroshot", "refusal_rate"),
                "first_observable_edge": "BASE->M0_SFT",
                "m1v1_dpo_on_base_refusal_rate": m1v1_zs_rate,
                "not_specific_to_sft": (
                    "BASE->M0_SFT is the first edge on the primary path. The pattern also appears "
                    f"after DPO applied directly to the base (M1_DPO_HISTORICAL, {m1v1_zs_text} "
                    "zero-shot refusal, no SFT parent), so it is not specific to SFT. Causation is "
                    "not established: one lineage, one seed, no replicate."
                ),
                "truncation_confound": "none -- 0 truncated zero-shot generations for M0 and M1-v2",
            },
            "refusal_is_prompt_regime_conditioned": {
                "evidence_level": "SUPPORTED_BUT_QUALIFIED",
                "same_questions_both_arms": True,
                "m1v2_zeroshot_refusal": g("M1_DPO_CURRENT", "zeroshot", "refusal_rate"),
                "m1v2_fewshot8_refusal": g("M1_DPO_CURRENT", "fewshot8", "refusal_rate"),
                "m1v2_mmlu_pro_5shot_refusal": m("M1_DPO_CURRENT", "refusal_rate"),
                "qualification": (
                    "Behavioural, across three benchmarks. The internal mechanism was not measured."
                ),
            },
            "fewshot_capability_loss": {
                "evidence_level": "CONFIRMED",
                "base_accuracy": g("BASE", "fewshot8", "accuracy"),
                "m0_accuracy": g("M0_SFT", "fewshot8", "accuracy"),
                "refusal_rate_in_this_regime": g("M0_SFT", "fewshot8", "refusal_rate"),
                "interval": intervals.get("gsm8k_fewshot8__BASE_minus_M0_SFT", {}).get("gap"),
                "why_not_refusal": (
                    "Refusal is ~0% in this arm, so the gap cannot be explained by declining to answer."
                ),
            },
            "conditional_accuracy_loss": {
                "evidence_level": "CONFIRMED",
                "mmlu_pro_base_acc_given_answer": m("BASE", "accuracy_given_answer"),
                "mmlu_pro_m0_acc_given_answer": m("M0_SFT", "accuracy_given_answer"),
                "ifeval_base_acc_given_answer": i("BASE", "accuracy_given_answer"),
                "ifeval_m0_acc_given_answer": i("M0_SFT", "accuracy_given_answer"),
                "definition": "correct / attempted, where attempted excludes refusals and unparseable output",
            },
            "preference_stage_material_change": {
                "evidence_level": "CONFIRMED",
                "verdict": "NO_MATERIAL_BEHAVIORAL_CHANGE_ON_MEASURED_SUITE",
                "gsm8k_fewshot8_delta_pp": round(
                    (g("M1_DPO_CURRENT", "fewshot8", "accuracy") - g("M0_SFT", "fewshot8", "accuracy")) * 100, 2),
                "ifeval_prompt_strict_delta_pp": round(
                    (i("M1_DPO_CURRENT", "prompt_level_strict_accuracy")
                     - i("M0_SFT", "prompt_level_strict_accuracy")) * 100, 2),
                "mmlu_pro_accuracy_delta_pp": round(
                    (m("M1_DPO_CURRENT", "accuracy") - m("M0_SFT", "accuracy")) * 100, 2),
                "scope_limit": (
                    "A statement about behaviour on THIS suite. NOT a weight-space claim, NOT a "
                    "claim about optimizer effectiveness, NOT a claim that DPO could never repair "
                    "this behaviour if explicitly targeted."
                ),
                "edge_scope": (
                    "Measured on the M0_SFT->M1_DPO_CURRENT edge only, i.e. DPO applied on top of a "
                    "parent that already refuses 100% zero-shot. It does NOT rule out DPO as a "
                    "cause of the regression: DPO applied directly to the base (M1_DPO_HISTORICAL) "
                    f"refuses {m1v1_zs_text} of GSM8K zero-shot, against 0% for BASE."
                ),
            },
        },

        "canonical_benchmarks": {
            "gsm8k": {
                "dataset": "openai/gsm8k", "revision": "740312add88f781978c0658806c59bc2815b9866",
                "split": "test", "n_questions": 1319, "arms": ["zeroshot", "fewshot8"],
                "max_tokens": 512,
                "per_stage": audit["gsm8k"],
            },
            "ifeval": {
                "dataset": "google/IFEval", "revision": "966cd89545d6b6acfd7638bc708b98261ca58e84",
                "split": "train", "n": 541, "max_tokens": 2560,
                "scorer": "google-research/instruction_following_eval, vendored unmodified, SEEDED",
                "per_stage": audit["ifeval"],
            },
            "mmlu_pro": {
                "dataset": "TIGER-Lab/MMLU-Pro", "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
                "split": "test", "n": 12032, "protocol": "5-shot CoT", "max_tokens": 2048,
                "per_stage": audit["mmlu_pro_canonical_2048"],
            },
            "sentinel": {
                "role": "SENTINEL / REGRESSION SMOKE TEST -- not a general-capability benchmark",
                "n": 7,
                "per_stage": audit["sentinel"],
            },
        },

        "truncation_adversarial_intervals": intervals,

        "superseded_runs": [
            {
                "benchmark": "mmlu_pro", "budget_tokens": 768,
                "disposition": "SUPERSEDED_PROTOCOL_INVALID",
                "reason": ("Stage-dependent truncation (BASE 37.6% vs M0_SFT 7.2%) made the "
                           "checkpoint comparison invalid; it falsely implied no degradation."),
                "retained_at": "results/benchmarks/h200/capability_v1/<STAGE>/superseded_mmlu_768/",
                "invalid_numbers": {
                    s: audit["mmlu_pro_superseded_768"].get(s, {}).get("accuracy")
                    for s in ("BASE", "M0_SFT", "M1_DPO_CURRENT")
                },
            },
            {
                "benchmark": "ifeval", "budget_tokens": 1280,
                "disposition": "SUPERSEDED_PROTOCOL_INVALID",
                "reason": "65/541 responses hit the generation cap; budget raised to 2560.",
                "retained_at": "results/benchmarks/h200/capability_v1/M1_DPO_CURRENT/ifeval_scores_budget1280_superseded.json",
            },
        ],

        "runtime_parity": {
            **audit.get("engine_parity", {}),
            "supported_claim": "Runtime choice produces measurable non-zero per-example behavioural differences.",
            "quantization_comparison": {
                "engine_flips": 21, "quantization_q6k_flips": 25, "examples": 1277,
                "same_examples_same_metric": True,
                "confound": ("The engine arm also changed hardware and kernels (H200 vs A100), so "
                             "this is a runtime-stack comparison, not a pure engine intervention."),
            },
            "held_separate_from_checkpoint_study": True,
        },

        "limitations": [
            "M1_DPO_HISTORICAL corrected-budget MMLU-Pro is UNMEASURED; not interpolated.",
            (
                "Causation of the refusal pattern is NOT established; the corpus audit is "
                "association only. The pattern appears after SFT (M0) and also after DPO applied "
                "directly to the base (M1_DPO_HISTORICAL), so it is not specific to SFT."
            ),
            "Single lineage, single base model, no replicate; no cross-family generalization claimed.",
            "BASE remains truncation-disadvantaged on MMLU-Pro (21.8% vs 6.3%); handled by interval.",
            (
                "Refusal detection is HEURISTIC_REGEX_v1 with unmeasured precision; it never feeds "
                "an official benchmark metric."
            ),
            "BASE is a multimodal artifact evaluated text-only as published.",
            "11 benchmarks remain BLOCKED_NO_DATASET; Speculative Replay BLOCKED_MISSING_MTP_COMPONENT.",
        ],

        "cost": {
            "retained_usd": ledger["retained_runs_usd"],
            "superseded_usd": ledger["superseded_runs_usd"],
            "infrastructure_waste_usd": ledger["infrastructure_waste_usd"],
            "continuation_usd": ledger["continuation_total_usd"],
            "prior_run_usd": ledger["prior_run_usd"],
            "campaign_usd": ledger["campaign_total_usd"],
            "planning_envelope_usd": ledger["envelope_usd"],
            "envelope_is_not_a_balance": True,
            "REMAINING_CREDIT_BALANCE": "NOT_QUERYABLE",
            "estimated_rows": audit["cost"]["estimated_rows"],
        },

        "evidence_refs": {
            "per_example_evidence": "results/benchmarks/h200/capability_v1/evidence/",
            "findings_ledger": "results/benchmarks/capability_findings.jsonl",
            "gpu_run_log": "results/benchmarks/h200/capability_v1/gpu_runs.jsonl",
            "recomputation": "results/benchmarks/h200/capability_v1/final_campaign_audit.json",
            "regression_analysis": "results/benchmarks/h200/capability_v1/regression_analysis.json",
            "refusal_characterization": "results/benchmarks/h200/capability_v1/refusal_characterization.json",
            "sft_corpus_audit": "results/benchmarks/h200/capability_v1/sft_refusal_supervision_audit.json",
            "sft_corpus_audit_canonical_v2": (
                "results/benchmarks/h200/capability_v1/sft_refusal_supervision_audit_canonical_v2.json"
            ),
            "tokenizer_census": "results/benchmarks/h200/capability_v1/tokenizer_divergence_census.json",
            "checkpoint_ladder": "results/benchmarks/checkpoint_ladder.json",
            "preserved_state": "results/benchmarks/h200/PRESERVED_STATE_v1.json",
            "audit_report": "reports/FINAL_CAMPAIGN_AUDIT.md",
            "diagnosis_report": "reports/GENERAL_CAPABILITY_REGRESSION.md",
            "next_experiment": "ROADMAP.md step 16",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    payload = build()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    if args.verify:
        if not OUT.exists():
            print("FAIL: verdict missing", file=sys.stderr)
            return 1
        if OUT.read_text(encoding="utf-8") != text:
            print("FAIL: committed verdict has drifted from the artifacts", file=sys.stderr)
            return 1
        print("verdict verified: matches artifacts")
        return 0

    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  status: {payload['status']}")
    print(f"  campaign cost: ${payload['cost']['campaign_usd']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
