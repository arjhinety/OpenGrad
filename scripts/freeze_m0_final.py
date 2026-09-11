#!/usr/bin/env python3
"""Freeze the M0-final result set: hash every artifact the reported numbers rest on.

The reports make numeric claims. This manifest records which files those claims rest on and
hashes them, so a reader can check that a number in a report still corresponds to the artifact
that produced it instead of trusting that nothing has been edited since. It is the same idea as
the release manifest pinning its shards, applied to a result instead of a corpus.

Deliberately *not* included: checkpoint weights and per-example predictions. Those are gigabytes
and megabytes respectively and stay on disk unversioned; their identity is recorded here by hash
so they can still be checked if they are present, without requiring them to be.

Usage:
    python scripts/freeze_m0_final.py            # write the manifest
    python scripts/freeze_m0_final.py --verify   # re-check every recorded hash
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = "reports/data/m0-final-freeze.json"
EXPERIMENT = "m0_sft_canonical_v2_final"

# Every artifact a claim in the M0-final reports rests on. Grouped by what it establishes, because
# a flat list of hashes does not tell a reader which claim would be unsupported if one drifted.
EVIDENCE: dict[str, list[str]] = {
    "corpus": [
        "configs/releases/toolpolicy_canonical_v2_final.yaml",
        ".release/hf/toolpolicy-canonical-v2-final/release-manifest.json",
        "reports/data/canonical-v2-final-yield.json",
    ],
    "contamination": [
        "reports/data/behavioral-heldout-v2-contamination--toolpolicy-canonical-v2-final.json",
    ],
    "evaluation": [
        "reports/evaluation/behavioral-heldout-v2.manifest.json",
        "reports/evaluation/behavioral-heldout-v2-partition.json",
        "reports/baselines/qwen35_2b_baseline/metrics.json",
        "reports/baselines/qwen35_2b_baseline/predictions.jsonl",
    ],
    "config": [
        "configs/experiments/m0_sft_canonical_v2_final.yaml",
        "docs/evaluation/CHECKPOINT_SELECTION_RULE.md",
        "reports/data/m0-final-launch-record.json",
    ],
    "training": [
        f"runs/{EXPERIMENT}/experiment.json",
        f"runs/{EXPERIMENT}/environment.json",
        f"runs/{EXPERIMENT}/resolved_config.yaml",
        f"runs/{EXPERIMENT}/dataset_manifest.json",
        f"runs/{EXPERIMENT}/overflow_report.json",
        f"runs/{EXPERIMENT}/rendering_report.json",
        f"runs/{EXPERIMENT}/metrics/train_log.jsonl",
        f"runs/{EXPERIMENT}/ledger.jsonl",
        f"runs/{EXPERIMENT}/events.jsonl",
    ],
    "selection": [
        f"runs/{EXPERIMENT}/eval/dev/curve.json",
        f"runs/{EXPERIMENT}/eval/dev/selection--dev.json",
        f"runs/{EXPERIMENT}/eval/dev/checkpoint-600/metrics.json",
        f"runs/{EXPERIMENT}/eval/dev/checkpoint-1200/metrics.json",
        f"runs/{EXPERIMENT}/eval/dev/checkpoint-1800/metrics.json",
        f"runs/{EXPERIMENT}/eval/dev/checkpoint-2400/metrics.json",
    ],
    "confirmatory": [
        f"runs/{EXPERIMENT}/eval/confirmatory/curve.json",
        f"runs/{EXPERIMENT}/eval/confirmatory/checkpoint-1800/metrics.json",
    ],
    "decision": [
        "runs/checkpoint_registry.json",
        "runs/central_ledger.jsonl",
        "results/registry.jsonl",
    ],
    "reports": [
        "reports/M0_CANONICAL_V2_FINAL_EXECUTION_REPORT.md",
        "reports/M0_CANONICAL_V2_FINAL_EVALUATION.md",
    ],
}

# Present on disk but intentionally not version-controlled.
UNVERSIONED: dict[str, list[str]] = {
    "weights": [
        f"runs/{EXPERIMENT}/checkpoints/checkpoint-600/model.safetensors",
        f"runs/{EXPERIMENT}/checkpoints/checkpoint-1200/model.safetensors",
        f"runs/{EXPERIMENT}/checkpoints/checkpoint-1800/model.safetensors",
        f"runs/{EXPERIMENT}/checkpoints/checkpoint-2400/model.safetensors",
    ],
    "predictions": [
        f"runs/{EXPERIMENT}/eval/dev/checkpoint-1800/predictions.jsonl",
        f"runs/{EXPERIMENT}/eval/confirmatory/checkpoint-1800/predictions.jsonl",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build() -> int:
    evidence: dict[str, dict[str, dict]] = {}
    missing: list[str] = []
    for group, paths in EVIDENCE.items():
        entries: dict[str, dict] = {}
        for relative in paths:
            path = ROOT / relative
            if not path.is_file():
                missing.append(relative)
                continue
            entries[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        evidence[group] = dict(sorted(entries.items()))

    unversioned: dict[str, dict[str, dict]] = {}
    for group, paths in UNVERSIONED.items():
        entries = {}
        for relative in paths:
            path = ROOT / relative
            if path.is_file():
                entries[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        unversioned[group] = dict(sorted(entries.items()))

    record = json.loads((ROOT / f"runs/{EXPERIMENT}/experiment.json").read_text())
    manifest = {
        "schema_version": 1,
        "artifact_kind": "M0_FINAL_RESULT_FREEZE",
        "experiment_id": EXPERIMENT,
        "experiment_status": record.get("status"),
        # The commit the hashes were computed at, not the commit that contains this file -- a
        # manifest cannot name its own commit. Compare against the parent of the commit that
        # introduced it.
        "hashes_computed_at_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=False
        ).stdout.strip(),
        "role": (
            "Pins the artifacts behind the numbers reported for the definitive M0 SFT, so a claim "
            "can be checked against the file that produced it rather than trusted."
        ),
        "evidence": evidence,
        "unversioned_by_design": unversioned,
        "reproduce": {
            "selection": (
                "python scripts/evaluate_sft_checkpoints.py m0_sft_canonical_v2_final "
                "--partition reports/evaluation/behavioral-heldout-v2-partition.json "
                "--partition-side dev"
            ),
            "selection_rule": "python scripts/select_checkpoint.py --side dev",
            "confirmatory": (
                "python scripts/evaluate_sft_checkpoints.py m0_sft_canonical_v2_final "
                "--partition reports/evaluation/behavioral-heldout-v2-partition.json "
                "--partition-side confirmatory --checkpoint-ids 1800"
            ),
        },
    }
    (ROOT / OUT).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    total = sum(len(v) for v in evidence.values())
    print(f"frozen {total} evidence artifacts across {len(evidence)} groups")
    for group, entries in evidence.items():
        print(f"  {group:<14}{len(entries):>3}")
    print("  unversioned: " + ", ".join(f"{k} ({len(v)})" for k, v in unversioned.items()))
    if missing:
        print("\nMISSING (not present on disk):")
        for relative in missing:
            print(f"  - {relative}")
        return 1
    print(f"\nwrote {OUT}")
    return 0


def verify() -> int:
    manifest_path = ROOT / OUT
    if not manifest_path.is_file():
        raise SystemExit(f"no freeze manifest at {OUT}; build it first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    drift: list[str] = []
    checked = 0
    for entries in manifest["evidence"].values():
        for relative, expected in entries.items():
            path = ROOT / relative
            if not path.is_file():
                drift.append(f"missing: {relative}")
                continue
            checked += 1
            actual = sha256(path)
            if actual != expected["sha256"]:
                drift.append(
                    f"changed: {relative}\n    frozen {expected['sha256']}\n    actual {actual}"
                )

    optional = 0
    for entries in manifest["unversioned_by_design"].values():
        for relative, expected in entries.items():
            path = ROOT / relative
            if path.is_file():
                optional += 1
                if sha256(path) != expected["sha256"]:
                    drift.append(f"unversioned artifact changed: {relative}")

    print(f"hashes computed at commit : {manifest['hashes_computed_at_commit']}")
    print(f"evidence      : {checked} checked")
    print(f"unversioned   : {optional} present and checked")
    print()
    if drift:
        print("VERIFY FAILED")
        for item in drift:
            print(f"  - {item}")
        return 1
    print("VERIFY PASSED — every recorded hash matches")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    return verify() if args.verify else build()


if __name__ == "__main__":
    raise SystemExit(main())
