#!/usr/bin/env python3
"""Cross-stage regression analysis: where does the behaviour change, and what kind of change is it?

Consumes whatever scored results exist and produces the transition matrix, the stage deltas, the
answer-rate vs conditional-accuracy split, and the tool-policy/general-capability Pareto view. It
never invents a row: a stage with no result for a benchmark is reported as absent, and a delta is
computed only where both endpoints exist.

The trajectory label (PURE_IMPROVEMENT / TRADEOFF / OVER_ALIGNMENT / GENERAL_DEGRADATION /
CHECKPOINT_ANOMALY / INSUFFICIENT_EVIDENCE) is derived from thresholds declared in LABEL_RULES
below, which are written here rather than chosen after seeing the numbers. If the evidence does
not satisfy any rule, the label is INSUFFICIENT_EVIDENCE -- not the closest-looking one.

Usage:
    python scripts/build_regression_analysis.py
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT /"results/benchmarks/h200/capability_v1"
OUT_JSON = CAP / "regression_analysis.json"

STAGE_ORDER = ["BASE", "M0_SFT", "M1_DPO_CURRENT", "M1_DPO_HISTORICAL"]
PRIMARY_PATH = ["BASE", "M0_SFT", "M1_DPO_CURRENT"]

# Declared before the numbers were available. "Material" is 5 percentage points: smaller than that
# on a 541-prompt or 1319-prompt set is inside the range where a handful of examples moves it.
MATERIAL_PP = 0.05

LABEL_RULES = {
    "PURE_IMPROVEMENT": "no general-capability metric falls materially, and tool policy improves",
    "TRADEOFF": "tool policy improves materially AND at least one general metric falls materially, "
                "with accuracy_given_answer broadly held",
    "OVER_ALIGNMENT": "answer_rate falls materially while accuracy_given_answer is broadly held -- "
                      "the ability is intact and is not being used",
    "GENERAL_DEGRADATION": "accuracy_given_answer falls materially -- the ability itself is worse",
    "CHECKPOINT_ANOMALY": "one stage is out of line with both its neighbours",
    "INSUFFICIENT_EVIDENCE": "the measured stages do not satisfy any rule above",
}

# Tool-policy numbers already measured, from frozen artifacts. Only the current stage has a
# confirmatory-partition measurement; the others are recorded as absent rather than guessed.
TOOL_POLICY_SOURCES = {
    "M1_DPO_CURRENT": ROOT / "results/benchmarks/h200/score_vllm-bf16_confirmatory.json",
}


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _jl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def _interval(stage: str, benchmark: str, arm: str | None = None) -> dict | None:
    """Accuracy interval implied by treating every unresolved example adversarially.

    An example is UNRESOLVED if it did not yield a scoreable correct answer AND its generation hit
    the token cap -- i.e. the model may have been on its way to the right answer. True accuracy is
    then in [correct/n, (correct + unresolved)/n]. This is a deterministic range, not a confidence
    interval, and makes no distributional assumption.
    """
    rows = _jl(CAP / stage / f"{benchmark}_scores_per_example.jsonl")
    if arm:
        rows = [r for r in rows if r.get("arm") == arm]
    if not rows:
        return None
    gens = {g["example_id"]: g for g in _jl(CAP / stage / f"generations_{benchmark}.jsonl")}
    n = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    unresolved = sum(
        1 for r in rows
        if not r["correct"] and gens.get(r["example_id"], {}).get("finish_reason") == "length"
    )
    return {"n": n, "correct": correct, "unresolved_truncated": unresolved,
            "low": correct / n, "high": (correct + unresolved) / n}


def _gap(a: dict, b: dict) -> dict:
    return {
        "observed_pp": (a["low"] - b["low"]) * 100,
        "worst_case_pp": (a["low"] - b["high"]) * 100,
        "best_case_pp": (a["high"] - b["low"]) * 100,
        "positive_under_adversarial_resolution": (a["low"] - b["high"]) > 0,
    }


def _truncation_intervals() -> dict:
    out: dict = {
        "method": (
            "true_accuracy in [correct/n, (correct+unresolved)/n], where unresolved = incorrect "
            "AND finish_reason == 'length'. Deterministic range under adversarial resolution of "
            "truncated generations; NOT a confidence interval."
        ),
        "why": (
            "Truncation rates differ by stage, so a raw difference could in principle be a "
            "generation-budget artifact. The interval shows whether the comparison survives the "
            "worst admissible resolution of those examples."
        ),
    }
    pairs = [
        ("mmlu_pro", None, "BASE", "M0_SFT"),
        ("gsm8k", "fewshot8", "BASE", "M0_SFT"),
        ("gsm8k", "zeroshot", "BASE", "M0_SFT"),
    ]
    for bench, arm, a_stage, b_stage in pairs:
        a, b = _interval(a_stage, bench, arm), _interval(b_stage, bench, arm)
        if not (a and b):
            continue
        key = f"{bench}" + (f"_{arm}" if arm else "") + f"__{a_stage}_minus_{b_stage}"
        out[key] = {a_stage: a, b_stage: b, "gap": _gap(a, b)}
    return out


def pct(x) -> str:
    return "-" if x is None else f"{x * 100:.1f}"


def delta(a, b):
    return None if (a is None or b is None) else b - a


def collect() -> dict:
    stages = {}
    for stage in STAGE_ORDER:
        row = {"stage": stage}

        ife = load(CAP / f"{stage}/ifeval_scores.json")
        if ife:
            row["ifeval"] = {
                "prompt_strict": ife["official_metrics"]["prompt_level_strict_accuracy"],
                "inst_strict": ife["official_metrics"]["instruction_level_strict_accuracy"],
                "prompt_loose": ife["official_metrics"]["prompt_level_loose_accuracy"],
                "inst_loose": ife["official_metrics"]["instruction_level_loose_accuracy"],
                "answer_rate": ife["diagnostic_metrics"]["answer_rate"],
                "refusal_rate": ife["diagnostic_metrics"]["refusal_rate"],
                "accuracy_given_answer": ife["diagnostic_metrics"]["accuracy_given_answer"],
                "failure_categories": ife["failure_categories"],
            }

        gsm = load(CAP / f"{stage}/gsm8k_scores.json")
        if gsm:
            row["gsm8k"] = {
                arm: {
                    "accuracy": s["accuracy"],
                    "answer_rate": s["answer_rate"],
                    "refusal_rate": s["refusal_rate"],
                    "parse_failure_rate": s["parse_failure_rate"],
                    "accuracy_given_answer": s["accuracy_given_answer"],
                    "correct": s["correct"],
                    "incorrect_attempted": s["incorrect_attempted"],
                    "refusals": s["refusals"],
                    "parse_failures": s["parse_failures"],
                }
                for arm, s in gsm["arms"].items()
            }

        mmlu = load(CAP / f"{stage}/mmlu_pro_scores.json")
        if mmlu:
            row["mmlu_pro"] = {
                "coverage": mmlu["coverage"],
                "accuracy": mmlu["aggregate"]["accuracy"],
                "answer_rate": mmlu["aggregate"]["answer_rate"],
                "refusal_rate": mmlu["aggregate"]["refusal_rate"],
                "invalid_option_rate": mmlu["aggregate"]["invalid_option_rate"],
                "accuracy_given_answer": mmlu["aggregate"]["accuracy_given_answer"],
                "per_category": mmlu["per_category"],
            }

        sent = load(CAP / f"{stage}/sentinel_scores.json")
        if sent:
            row["sentinel"] = {
                "pass": sent["pass"], "fail": sent["fail"], "total": sent["total"],
                "tool_pass": sent["tool_cases"]["pass"],
                "tool_total": sent["tool_cases"]["total"],
                "general_pass": sent["non_tool_cases"]["pass"],
                "general_total": sent["non_tool_cases"]["total"],
                "per_case": {c["id"]: c["status"] for c in sent["cases"]},
            }

        tp_path = TOOL_POLICY_SOURCES.get(stage)
        if tp_path and tp_path.exists():
            tp = json.loads(tp_path.read_text(encoding="utf-8"))
            metrics = tp.get("metrics") or tp.get("quantization_loss_vs_llamacpp_bf16", {}).get("metrics")
            if metrics:
                row["tool_policy"] = {
                    k: metrics[k] for k in
                    ("call_f1", "call_precision", "call_recall", "over_call_rate",
                     "clarification_accuracy", "unsupported_accuracy", "parse_valid_rate")
                    if k in metrics
                }
                row["tool_policy_source"] = str(tp_path.relative_to(ROOT)).replace("\\", "/")

        if len(row) > 1:
            stages[stage] = row
    return stages


def build_deltas(stages: dict) -> dict:
    deltas = {}
    for a, b in pairwise(PRIMARY_PATH):
        if a not in stages or b not in stages:
            deltas[f"{a}->{b}"] = {"status": "ABSENT",
                                   "missing": [s for s in (a, b) if s not in stages]}
            continue
        sa, sb = stages[a], stages[b]
        edge = {"status": "MEASURED"}
        if "ifeval" in sa and "ifeval" in sb:
            edge["ifeval"] = {
                k: delta(sa["ifeval"][k], sb["ifeval"][k])
                for k in ("prompt_strict", "inst_strict", "prompt_loose", "inst_loose",
                          "answer_rate", "refusal_rate", "accuracy_given_answer")
            }
        for arm in ("zeroshot", "fewshot8"):
            if "gsm8k" in sa and "gsm8k" in sb and arm in sa["gsm8k"] and arm in sb["gsm8k"]:
                edge[f"gsm8k_{arm}"] = {
                    k: delta(sa["gsm8k"][arm][k], sb["gsm8k"][arm][k])
                    for k in ("accuracy", "answer_rate", "refusal_rate", "accuracy_given_answer")
                }
        if "mmlu_pro" in sa and "mmlu_pro" in sb:
            edge["mmlu_pro"] = {
                k: delta(sa["mmlu_pro"][k], sb["mmlu_pro"][k])
                for k in ("accuracy", "answer_rate", "refusal_rate", "accuracy_given_answer")
            }
        if "sentinel" in sa and "sentinel" in sb:
            edge["sentinel"] = {
                "pass_delta": sb["sentinel"]["pass"] - sa["sentinel"]["pass"],
                "newly_failing": sorted(
                    cid for cid, st in sb["sentinel"]["per_case"].items()
                    if st == "fail" and sa["sentinel"]["per_case"].get(cid) == "pass"
                ),
                "newly_passing": sorted(
                    cid for cid, st in sb["sentinel"]["per_case"].items()
                    if st == "pass" and sa["sentinel"]["per_case"].get(cid) == "fail"
                ),
            }
        deltas[f"{a}->{b}"] = edge
    return deltas


def first_observable(deltas: dict) -> dict:
    """Name the first edge on which each signal becomes materially visible. Phrased as an
    observation about ordering, never as a causal claim."""
    signals = {}
    for name, getter in [
        ("refusal_rises", lambda e: max(
            [v for v in (
                (e.get("ifeval") or {}).get("refusal_rate"),
                (e.get("gsm8k_zeroshot") or {}).get("refusal_rate"),
                (e.get("mmlu_pro") or {}).get("refusal_rate"),
            ) if v is not None] or [None])),
        ("answer_rate_falls", lambda e: min(
            [v for v in (
                (e.get("ifeval") or {}).get("answer_rate"),
                (e.get("gsm8k_zeroshot") or {}).get("answer_rate"),
                (e.get("mmlu_pro") or {}).get("answer_rate"),
            ) if v is not None] or [None])),
        ("instruction_following_falls", lambda e: (e.get("ifeval") or {}).get("prompt_strict")),
        ("math_accuracy_falls", lambda e: (e.get("gsm8k_zeroshot") or {}).get("accuracy")),
        ("conditional_math_accuracy_falls",
         lambda e: (e.get("gsm8k_zeroshot") or {}).get("accuracy_given_answer")),
        # The 8-shot arm is the only place conditional math capability stays measurable once
        # zero-shot answer rate hits zero: with no attempts there is no conditional accuracy to
        # compare, so the zero-shot signal above goes silent precisely when refusal is total.
        ("fewshot_math_accuracy_falls", lambda e: (e.get("gsm8k_fewshot8") or {}).get("accuracy")),
        ("conditional_instruction_accuracy_falls",
         lambda e: (e.get("ifeval") or {}).get("accuracy_given_answer")),
    ]:
        hit = None
        for edge_name, edge in deltas.items():
            if edge.get("status") != "MEASURED":
                continue
            v = getter(edge)
            if v is None:
                continue
            rising = name == "refusal_rises"
            if (v >= MATERIAL_PP) if rising else (v <= -MATERIAL_PP):
                hit = {"edge": edge_name, "magnitude_pp": round(v * 100, 2)}
                break
        signals[name] = hit or {"edge": None, "note": "not materially observable on any measured edge"}
    return signals


def label_trajectory(stages: dict, deltas: dict) -> dict:
    """Apply LABEL_RULES. Returns INSUFFICIENT_EVIDENCE unless a rule is actually satisfied."""
    measured = [e for e in deltas.values() if e.get("status") == "MEASURED"]
    if not measured:
        return {"label": "INSUFFICIENT_EVIDENCE", "basis": "no stage pair measured",
                "rules": LABEL_RULES}

    ans_drops, cond_drops = [], []
    for e in measured:
        for block in ("ifeval", "gsm8k_zeroshot", "mmlu_pro"):
            b = e.get(block) or {}
            if b.get("answer_rate") is not None:
                ans_drops.append(b["answer_rate"])
            if b.get("accuracy_given_answer") is not None:
                cond_drops.append(b["accuracy_given_answer"])

    worst_answer = min(ans_drops) if ans_drops else None
    worst_conditional = min(cond_drops) if cond_drops else None

    if worst_conditional is not None and worst_conditional <= -MATERIAL_PP:
        label = "GENERAL_DEGRADATION"
        basis = (f"accuracy_given_answer falls by {abs(worst_conditional)*100:.1f}pp on at least "
                 f"one measured edge -- the ability itself is worse, not merely unused")
    elif (worst_answer is not None and worst_answer <= -MATERIAL_PP
          and (worst_conditional is None or worst_conditional > -MATERIAL_PP)):
        label = "OVER_ALIGNMENT"
        basis = (f"answer_rate falls by {abs(worst_answer)*100:.1f}pp while accuracy_given_answer "
                 f"holds (worst {(worst_conditional or 0)*100:+.1f}pp) -- capability retained, "
                 f"not exercised")
    elif worst_answer is not None and worst_answer > -MATERIAL_PP:
        label = "PURE_IMPROVEMENT"
        basis = "no general-capability metric falls materially on any measured edge"
    else:
        label = "INSUFFICIENT_EVIDENCE"
        basis = "measured edges do not satisfy any declared rule"

    # The rules are a precedence ordering, so they yield ONE primary label. The data can satisfy
    # several conditions at once, and reporting only the winner would hide half the finding --
    # "capability degraded" and "the model stopped answering" are both true here and have
    # different remediations. Co-occurring conditions are therefore recorded explicitly.
    co_occurring = []
    if worst_answer is not None and worst_answer <= -MATERIAL_PP:
        co_occurring.append({
            "condition": "ANSWER_RATE_COLLAPSE",
            "detail": f"answer_rate falls by {abs(worst_answer)*100:.1f}pp on a measured edge",
        })
    if worst_conditional is not None and worst_conditional <= -MATERIAL_PP:
        co_occurring.append({
            "condition": "CONDITIONAL_ACCURACY_LOSS",
            "detail": (f"accuracy_given_answer falls by {abs(worst_conditional)*100:.1f}pp on a "
                       f"measured edge, i.e. worse even when it does attempt"),
        })
    fewshot_drops = [
        (e.get("gsm8k_fewshot8") or {}).get("accuracy") for e in measured
        if (e.get("gsm8k_fewshot8") or {}).get("accuracy") is not None
    ]
    if fewshot_drops and min(fewshot_drops) <= -MATERIAL_PP:
        co_occurring.append({
            "condition": "FEWSHOT_CAPABILITY_LOSS",
            "detail": (f"GSM8K 8-shot accuracy falls by {abs(min(fewshot_drops))*100:.1f}pp on a "
                       f"measured edge. The 8-shot arm has a near-zero refusal rate, so this drop "
                       f"is NOT explained by refusal -- it is lost ability."),
        })

    return {
        "label": label,
        "basis": basis,
        "co_occurring_conditions": co_occurring,
        "co_occurring_note": (
            "More than one condition holding is the normal case, not a contradiction. A single "
            "label is reported only because the rules were pre-declared as a precedence ordering."
        ),
        "material_threshold_pp": MATERIAL_PP * 100,
        "worst_answer_rate_delta_pp": None if worst_answer is None else round(worst_answer * 100, 2),
        "worst_conditional_accuracy_delta_pp":
            None if worst_conditional is None else round(worst_conditional * 100, 2),
        "rules": LABEL_RULES,
        "caveat": (
            "This is an ordering statement about measured stages, not a causal one. A label says "
            "the pattern first becomes observable after a stage; it does not establish that the "
            "stage caused it, because no stage was re-run with a controlled intervention."
        ),
    }


def main() -> int:
    stages = collect()
    if not stages:
        print("no scored capability results found under results/benchmarks/h200/capability_v1")
        return 1

    deltas = build_deltas(stages)
    payload = {
        "schema_version": 1,
        "run": "h200-capability-diagnosis-v1",
        "primary_path": PRIMARY_PATH,
        "stages_measured": sorted(stages),
        "stages_absent": [s for s in STAGE_ORDER if s not in stages],
        "transition_matrix": stages,
        "stage_deltas": deltas,
        "first_observable": first_observable(deltas),
        "trajectory": label_trajectory(stages, deltas),
        # Residual truncation, after the 768 -> 2048 correction. Earlier drafts described the
        # observed gap as a "conservative lower bound" WITHOUT deriving any bound. That phrasing
        # was wrong: an unresolved example could in principle resolve either way, so the observed
        # gap is a point estimate inside an interval, not a floor. The interval below is derived
        # from exact counts by scripts/audit_campaign_final.py.
        "truncation_adversarial_intervals": _truncation_intervals(),
        "engine_caveat": (
            "Engine effects are NOT folded in here. The vLLM<->llama.cpp agreement of 0.9836 "
            "(21 per-example flips) is a separate measurement on a different example set and a "
            "different metric, and is recorded in results/benchmarks/h200/"
            "vllm_vs_llamacpp_agreement.json. Every stage in this table was run on the same "
            "engine and version, so engine choice cannot explain a stage delta."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {OUT_JSON.relative_to(ROOT)}\n")
    hdr = f"{'stage':<20}{'IFEval str':>11}{'GSM8K 0s':>10}{'acc|ans':>9}{'ans rate':>10}{'refusal':>9}{'MMLU-Pro':>10}{'sentinel':>10}"
    print(hdr)
    print("-" * len(hdr))
    for s in STAGE_ORDER:
        r = stages.get(s)
        if not r:
            continue
        ife = r.get("ifeval", {})
        g = (r.get("gsm8k") or {}).get("zeroshot", {})
        m = r.get("mmlu_pro", {})
        sen = r.get("sentinel", {})
        print(f"{s:<20}{pct(ife.get('prompt_strict')):>11}{pct(g.get('accuracy')):>10}"
              f"{pct(g.get('accuracy_given_answer')):>9}{pct(g.get('answer_rate')):>10}"
              f"{pct(g.get('refusal_rate')):>9}{pct(m.get('accuracy')):>10}"
              f"{(str(sen.get('pass')) + '/7') if sen else '-':>10}")

    print("\nfirst observable:")
    for k, v in payload["first_observable"].items():
        print(f"  {k:<38} {v.get('edge') or v.get('note')}"
              + (f"  ({v['magnitude_pp']:+.1f}pp)" if v.get("magnitude_pp") is not None else ""))
    t = payload["trajectory"]
    print(f"\ntrajectory: {t['label']}\n  {t['basis']}")
    for c in t.get("co_occurring_conditions", []):
        print(f"  + {c['condition']}: {c['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
