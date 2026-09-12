#!/usr/bin/env python3
"""Characterise WHAT triggers refusal, per stage, from the per-example evidence. CPU only.

Knowing that a checkpoint refuses is not enough to act on. This separates the trigger conditions,
because they have different implications:

  * Refusal rate by BENCHMARK -- GSM8K 0-shot, IFEval and MMLU-Pro are three different prompt
    shapes. A model that refuses one and not the others has a targeted behaviour, not a global one.
  * Refusal rate by SHOT COUNT -- GSM8K runs 0-shot and 8-shot over the SAME questions, so the
    comparison isolates the effect of exemplar presence with content held constant. This is the
    single most diagnostic contrast available.
  * Refusal rate by IFEval INSTRUCTION TYPE -- shows whether a particular constraint provokes it.
  * The refusal TEXT ITSELF, clustered -- a small set of templates repeated verbatim indicates a
    learned surface form rather than a reasoned decision.

Everything here is descriptive. No claim is made about which training data caused it, because that
was not measured.

Usage:
    python scripts/characterize_refusal.py
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "results/benchmarks/h200/capability_v1"
DATASETS = ROOT / "results/benchmarks/datasets"
OUT = CAP / "refusal_characterization.json"

STAGES = ["BASE", "M0_SFT", "M1_DPO_CURRENT", "M1_DPO_HISTORICAL"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def normalise_refusal(text: str) -> str:
    """Collapse a refusal to its template so verbatim repetition is countable."""
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    t = re.sub(r"[^a-z ]", "", t)
    return " ".join(t.split()[:12])


def main() -> int:
    ifeval_req = {r["example_id"]: r for r in read_jsonl(DATASETS / "ifeval_v1.jsonl")}

    report = {}
    for stage in STAGES:
        stage_dir = CAP / stage
        if not stage_dir.exists():
            continue
        entry: dict = {"by_benchmark": {}}

        # -- by benchmark and, for GSM8K, by shot count ---------------------------------------
        for bench in ("ifeval", "gsm8k", "mmlu_pro"):
            path = stage_dir / f"{bench}_scores_per_example.jsonl"
            if not path.exists():
                continue
            rows = read_jsonl(path)
            if bench == "gsm8k":
                for arm in ("zeroshot", "fewshot8"):
                    sub = [r for r in rows if r.get("arm") == arm]
                    if sub:
                        entry["by_benchmark"][f"gsm8k_{arm}"] = {
                            "n": len(sub),
                            "refusals": sum(1 for r in sub if r["refusal"]),
                            "refusal_rate": sum(1 for r in sub if r["refusal"]) / len(sub),
                        }
            else:
                entry["by_benchmark"][bench] = {
                    "n": len(rows),
                    "refusals": sum(1 for r in rows if r["refusal"]),
                    "refusal_rate": sum(1 for r in rows if r["refusal"]) / len(rows),
                }

        # -- the exemplar contrast, content held constant ---------------------------------------
        z = entry["by_benchmark"].get("gsm8k_zeroshot")
        f = entry["by_benchmark"].get("gsm8k_fewshot8")
        if z and f:
            entry["exemplar_contrast"] = {
                "zeroshot_refusal_rate": z["refusal_rate"],
                "fewshot8_refusal_rate": f["refusal_rate"],
                "delta": f["refusal_rate"] - z["refusal_rate"],
                "design_note": (
                    "Both arms cover the SAME 1319 questions. Only the presence of 8 worked "
                    "exemplars differs, so the gap isolates exemplar presence from question "
                    "content. MMLU-Pro is 5-shot by protocol and belongs on the exemplar-present "
                    "side of this contrast."
                ),
            }

        # -- IFEval: which constraint types provoke it ------------------------------------------
        ife_path = stage_dir / "ifeval_scores_per_example.jsonl"
        if ife_path.exists():
            rows = read_jsonl(ife_path)
            refused, total = collections.Counter(), collections.Counter()
            for r in rows:
                for i in ifeval_req[r["example_id"]]["score_key"]["instruction_id_list"]:
                    total[i] += 1
                    if r["refusal"]:
                        refused[i] += 1
            by_type = {
                i: {"refused": refused[i], "total": total[i], "rate": refused[i] / total[i]}
                for i in sorted(total) if total[i] >= 15
            }
            entry["ifeval_refusal_by_instruction_type"] = dict(
                sorted(by_type.items(), key=lambda kv: -kv[1]["rate"])
            )

        # -- refusal surface forms ---------------------------------------------------------------
        templates = collections.Counter()
        examples: dict[str, str] = {}
        for bench in ("ifeval", "gsm8k", "mmlu_pro"):
            path = stage_dir / f"{bench}_scores_per_example.jsonl"
            if not path.exists():
                continue
            for r in read_jsonl(path):
                if r["refusal"]:
                    key = normalise_refusal(r.get("raw_output", ""))
                    templates[key] += 1
                    examples.setdefault(key, (r.get("raw_output") or "")[:200])
        total_ref = sum(templates.values())
        top = templates.most_common(10)
        entry["refusal_surface_forms"] = {
            "total_refusals": total_ref,
            "distinct_templates_12_token_prefix": len(templates),
            "top_10_share": (sum(n for _, n in top) / total_ref) if total_ref else None,
            "top_templates": [
                {"count": n, "share": n / total_ref, "example": examples[k]} for k, n in top
            ],
        }
        entry["refusal_pattern_counts"] = dict(collections.Counter(
            r["refusal_pattern"]
            for bench in ("ifeval", "gsm8k", "mmlu_pro")
            if (stage_dir / f"{bench}_scores_per_example.jsonl").exists()
            for r in read_jsonl(stage_dir / f"{bench}_scores_per_example.jsonl")
            if r["refusal"]
        ))
        report[stage] = entry

    payload = {
        "schema_version": 1,
        "run": "h200-capability-diagnosis-v1",
        "purpose": "Describe the conditions under which each checkpoint refuses.",
        "scope_limit": (
            "Descriptive only. No claim is made here about which training data produced the "
            "behaviour; no attribution experiment was run."
        ),
        "refusal_detection": "HEURISTIC_REGEX_v1 (opengrad.evaluation.capability)",
        "stages": report,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {OUT.relative_to(ROOT)}\n")
    hdr = f"{'stage':<20}{'GSM8K 0-shot':>14}{'GSM8K 8-shot':>14}{'MMLU-Pro 5s':>13}{'IFEval 0-shot':>15}"
    print(hdr)
    print("-" * len(hdr))
    for stage, e in report.items():
        b = e["by_benchmark"]
        def rate(k):
            return f"{b[k]['refusal_rate']*100:.1f}%" if k in b else "-"
        print(f"{stage:<20}{rate('gsm8k_zeroshot'):>14}{rate('gsm8k_fewshot8'):>14}"
              f"{rate('mmlu_pro'):>13}{rate('ifeval'):>15}")
    for stage, e in report.items():
        sf = e.get("refusal_surface_forms", {})
        if sf.get("total_refusals"):
            print(f"\n{stage}: {sf['total_refusals']} refusals, "
                  f"{sf['distinct_templates_12_token_prefix']} distinct 12-token openings, "
                  f"top-10 cover {sf['top_10_share']*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
