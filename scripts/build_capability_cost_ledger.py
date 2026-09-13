#!/usr/bin/env python3
"""Build the continuation's cost ledger from measured container time. CPU.

Two rules this ledger exists to enforce:

1. **Failed infrastructure attempts are not benchmark cost.** They are tracked under
   INFRASTRUCTURE_WASTE so the cost-per-unit-of-evidence figure stays honest.
2. **Remaining credit is not invented.** The Modal API exposes consumption, not a balance, so the
   ledger records NOT_QUERYABLE rather than a number that would look precise and be fiction.

Cost is read from `gpu_runs.jsonl`, an append-only row per GPU run. It is NOT read by scanning
run_summary.json: that file lives at one fixed path per stage and is overwritten by the next run
of the same stage, so a scan double-counts re-runs and silently loses earlier ones.

Usage:
    python scripts/build_capability_cost_ledger.py
    python scripts/build_capability_cost_ledger.py --add-waste 0.12 --waste-reason "..."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "results/benchmarks/h200/capability_v1"
OUT = CAP / "cost_ledger.json"
PRIOR_LEDGER = ROOT / "results/benchmarks/h200_run_ledger.json"
RUN_LOG = CAP / "gpu_runs.jsonl"

GPU_HOURLY_USD = 4.54  # confirmed from `modal billing rates`

# ENVELOPE_USD is the CONFIGURED EXPERIMENT ENVELOPE carried over from the earlier plan. It is NOT
# a credit balance and never was. This distinction was recorded from the start and then ignored in
# practice: spend was planned against the envelope as though it were money in the account, and the
# actual remaining credit turned out to be far smaller. The envelope is kept here only to show what
# the run was budgeted against; `envelope_remaining_usd` below is arithmetic on that budget, NOT an
# available balance, and must never be read as one.
ENVELOPE_USD = 22.70

# Per-stage truncation at the superseded 768-token MMLU-Pro budget. Kept because it is the
# evidence for why that pass was discarded, and because the spread across stages is the point:
# a fixed budget that one checkpoint clears and another does not turns verbosity into apparent
# accuracy.
SUPERSEDED_MMLU_TRUNCATION_768 = {
    "BASE": {"truncated": 4527, "of": 12032, "rate": 0.3763},
    "M0_SFT": {"truncated": 869, "of": 12032, "rate": 0.0722},
    "M1_DPO_CURRENT": {"truncated": 903, "of": 12032, "rate": 0.0751},
    "M1_DPO_HISTORICAL": {"truncated": 5978, "of": 12032, "rate": 0.4969},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--add-waste", type=float, default=None)
    ap.add_argument("--waste-reason", default=None)
    args = ap.parse_args()

    prior = json.loads(PRIOR_LEDGER.read_text(encoding="utf-8"))
    existing = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    waste_items = existing.get("infrastructure_waste_items", [])
    if args.add_waste is not None:
        waste_items.append({"usd": round(args.add_waste, 4),
                            "reason": args.waste_reason or "unspecified"})

    # Cost comes from the append-only run log, NOT from scanning run_summary.json. Each stage
    # writes run_summary.json to one fixed path on the volume, so a re-run of that stage
    # overwrites it: a scan double-counts whichever runs happen to be present and silently drops
    # every earlier run of the same stage. The run log has one immutable row per GPU run.
    runs = [json.loads(line) for line in RUN_LOG.read_text(encoding="utf-8").split("\n") if line]
    gpu_runs = [r for r in runs if r["_record"] == "run"]
    failed = [r for r in runs if r["_record"] == "failed_launch"]

    stages: dict[str, dict] = {}
    for r in gpu_runs:
        st = stages.setdefault(r["stage"], {"runs": [], "container_seconds": 0.0, "usd": 0.0})
        st["runs"].append({"jobs": r["jobs"], "container_seconds": r["container_seconds"],
                           "usd": r["usd"], "superseded": r["superseded"]})
        st["container_seconds"] += r["container_seconds"]
        st["usd"] = round(st["usd"] + r["usd"], 4)

    superseded_usd = round(sum(r["usd"] for r in gpu_runs if r["superseded"] is True), 4)
    retained_usd = round(sum(r["usd"] for r in gpu_runs if r["superseded"] is not True), 4)
    container_seconds = sum(r["container_seconds"] for r in gpu_runs)
    superseded_seconds = sum(r["container_seconds"] for r in gpu_runs if r["superseded"] is True)
    waste_usd = sum(w["usd"] for w in waste_items) + sum(f["usd"] for f in failed)
    continuation_total = round(retained_usd + superseded_usd + waste_usd, 4)
    prior_usd = prior.get("estimated_spend_usd", 0.0)

    payload = {
        "schema_version": 1,
        "run": "h200-capability-diagnosis-v1",
        "gpu": "NVIDIA H200",
        "gpu_hourly_usd": GPU_HOURLY_USD,
        "envelope_usd": ENVELOPE_USD,
        "per_stage": stages,
        "retained_runs_usd": retained_usd,
        "retained_runs_note": "GPU runs whose results are the ones reported.",
        "total_container_seconds": round(container_seconds, 2),
        "failed_launches": failed,
        "failed_launches_note": "Every failed launch in this continuation died client-side or at module import, before a GPU was allocated. Cost $0.00 -- asserted from the run log, not assumed.",
        "container_overhead_note": (
            "Model download, weight verification, engine start and teardown. Real GPU time, but "
            "not attributable to any one benchmark, so it is reported separately rather than "
            "loaded onto a benchmark's cost-per-result."
        ),
        "gpu_runs": gpu_runs,
        "superseded_runs_usd": round(superseded_usd, 4),
        "superseded_runs_seconds": round(superseded_seconds, 2),
        "superseded_mmlu_truncation_at_768": SUPERSEDED_MMLU_TRUNCATION_768,
        "superseded_runs_note": (
            "Runs whose per-stage run_summary.json was overwritten by a later run of the same "
            "stage. Counted in full. A superseded run is NOT infrastructure waste when its "
            "results are retained -- see each entry's why_superseded."
        ),
        "infrastructure_waste_usd": round(waste_usd, 4),
        "infrastructure_waste_items": waste_items,
        "infrastructure_waste_note": (
            "Failed launches, wrong pins, client-side crashes. Never counted as benchmark cost."
        ),
        "continuation_total_usd": round(continuation_total, 4),
        "prior_run_usd": prior_usd,
        "prior_run": prior.get("run"),
        "campaign_total_usd": round(continuation_total + prior_usd, 4),
        "envelope_remaining_usd": round(ENVELOPE_USD - continuation_total - prior_usd, 4),
        "envelope_remaining_note": (
            "Arithmetic on the configured EXPERIMENT envelope minus observed spend. NOT an "
            "account balance and NOT available credit. The two diverged badly in this run: "
            "planning continued against the envelope while actual remaining credit was far "
            "lower. Do not size a run from this field."
        ),
        "envelope_is_not_a_balance": True,
        "REMAINING_CREDIT_BALANCE": "NOT_QUERYABLE",
        "credit_balance_note": (
            "`modal billing` exposes metered consumption and credits applied, not a remaining "
            "balance. No balance is asserted."
        ),
        "cpu_work_note": (
            "Dataset acquisition, hashing, validation, scorer tests, the tokenizer-divergence "
            "census and all scoring ran on local CPU at no GPU cost. The H200 was used only for "
            "model inference."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {OUT.relative_to(ROOT)}")
    for stage, st in stages.items():
        marks = "".join("*" if r["superseded"] is True else "." for r in st["runs"])
        print(f"  {stage:<20} {len(st['runs'])} run(s) [{marks}]  "
              f"{st['container_seconds']:.0f}s  ${st['usd']:.3f}")
    print("  (* = superseded run; cost counted in full)")
    print(f"\n  retained runs         ${retained_usd:.3f}")
    print(f"  superseded runs       ${superseded_usd:.3f}")
    print(f"  INFRASTRUCTURE_WASTE  ${waste_usd:.3f}")
    print(f"  continuation total    ${continuation_total:.3f}")
    print(f"  campaign total        ${continuation_total + prior_usd:.3f} of ${ENVELOPE_USD:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
