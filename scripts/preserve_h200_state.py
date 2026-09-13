#!/usr/bin/env python3
"""Snapshot and verify the completed H200 run before the capability continuation writes anything.

The completed run had no per-file digest manifest -- its ledger recorded environment hashes only.
This takes the digests now, so from this point forward any edit to a completed artifact is
detectable. `--verify` re-checks them and fails loudly on drift.

The continuation writes into `results/benchmarks/h200/capability_v1/`, a separate namespace. No
file recorded here is ever opened for writing by the continuation.

Usage:
    python scripts/preserve_h200_state.py
    python scripts/preserve_h200_state.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/benchmarks/h200/PRESERVED_STATE_v1.json"
CONTINUATION_DIR = ROOT / "results/benchmarks/h200/capability_v1"

# Everything the completed run produced, plus the frozen inputs it depended on.
PROTECTED = [
    ("run_ledger", "results/benchmarks/h200_run_ledger.json"),
    ("phase0_validation", "results/benchmarks/h200/validate_run.json"),
    ("capability_run", "results/benchmarks/h200/capability_run.json"),
    ("performance_run", "results/benchmarks/h200/perf_run.json"),
    ("frozen_1277_scores", "results/benchmarks/h200/score_vllm-bf16_confirmatory.json"),
    ("frozen_1277_predictions", "results/benchmarks/h200/predictions_vllm-bf16_confirmatory.json"),
    ("engine_agreement", "results/benchmarks/h200/vllm_vs_llamacpp_agreement.json"),
    ("openweights_spec", "results/benchmarks/openweights_parity_cases_v1.json"),
    ("openweights_results", "results/benchmarks/openweights_parity_opengrad_v1.json"),
    ("frozen_reference", "results/quantization/m1_v2_reference.json"),
    ("preservation_gate", "results/quantization/quantization_preservation_v1.json"),
    ("frozen_eval_set", "results/quantization/frozen_behavioral_eval_v1.jsonl"),
    ("ptq_closure", "manifests/quantization/ptq_phase_closure_v1.json"),
    ("findings_ledger", "results/quantization/findings.jsonl"),
    ("report_h200", "reports/H200_BENCHMARK_RUN.md"),
    ("report_openweights", "reports/OPENWEIGHTS_TRANSFER_EVALUATION.md"),
    ("report_engine_parity", "reports/QUANTIZATION_ENGINE_PARITY.md"),
    ("report_ptq_eval", "reports/QUANTIZATION_PTQ_EVALUATION.md"),
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect() -> dict:
    entries = {}
    for key, rel in PROTECTED:
        path = ROOT / rel
        if not path.exists():
            entries[key] = {"path": rel, "present": False}
            continue
        entries[key] = {
            "path": rel,
            "present": True,
            "sha256": digest(path),
            "bytes": path.stat().st_size,
        }
    return entries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    current = collect()
    missing = [k for k, v in current.items() if not v["present"]]

    if args.verify:
        if not MANIFEST.exists():
            print("FAIL: no preserved-state manifest; run without --verify first", file=sys.stderr)
            return 1
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        prior = manifest["artifacts"]
        # Reports the continuation was explicitly asked to extend are allowed to change, but only
        # if the extension was declared here first. Data artifacts are never in this set.
        allowed = set(manifest.get("append_only_updates", {}))
        drift, vanished = [], []
        for key, rec in prior.items():
            if not rec["present"] or key in allowed:
                continue
            now = current.get(key, {"present": False})
            if not now["present"]:
                vanished.append(rec["path"])
            elif now["sha256"] != rec["sha256"]:
                drift.append(rec["path"])
        if drift or vanished:
            for p in vanished:
                print(f"FAIL vanished: {p}", file=sys.stderr)
            for p in drift:
                print(f"FAIL modified: {p}", file=sys.stderr)
            return 1
        print(f"verified {sum(1 for v in prior.values() if v['present'])} preserved artifacts: no drift")
        return 0

    if missing:
        print(f"WARNING: {len(missing)} protected artifact(s) absent: {missing}")

    CONTINUATION_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "preserved_run": "h200-credit-constrained-v1",
        "continuation_run": "h200-capability-diagnosis-v1",
        "continuation_namespace": str(CONTINUATION_DIR.relative_to(ROOT)).replace("\\", "/"),
        "policy": (
            "Files listed here are immutable for the continuation. The continuation may append "
            "metadata elsewhere and may publish explicitly versioned corrected reports, but it "
            "never rewrites a completed result in place."
        ),
        "artifacts": current,
        "absent": missing,
        "append_only_updates": {
            "report_h200": {
                "reason": (
                    "The continuation was explicitly asked to update reports/H200_BENCHMARK_RUN.md. "
                    "It is extended with a continuation section that ADDS the capability-diagnosis "
                    "results; no existing figure, table or conclusion in it is rewritten. The "
                    "original digest is recorded above so the extension is auditable."
                ),
                "original_sha256": current.get("report_h200", {}).get("sha256"),
            },
        },
        "append_only_policy": (
            "Only keys listed in append_only_updates may change hash under --verify, and only "
            "reports may appear there. Every data artifact -- predictions, scores, ledgers, "
            "hashes, frozen inputs -- is strictly immutable."
        ),
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    present = sum(1 for v in current.values() if v["present"])
    print(f"wrote {MANIFEST.relative_to(ROOT)}")
    print(f"preserved {present}/{len(PROTECTED)} artifacts")
    print(f"continuation namespace: {payload['continuation_namespace']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
