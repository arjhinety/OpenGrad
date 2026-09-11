#!/usr/bin/env python3
"""Freeze the paired minus-xLAM result set: hash every artifact both arms' numbers rest on.

The two arms are one experiment and must be reported together, so this refuses to write a
manifest unless both are present and complete. Pinning only one arm's evidence would let a
partial result be presented as the experiment's conclusion.

Usage:
    python scripts/freeze_minus_xlam_ablation.py            # write the manifest
    python scripts/freeze_minus_xlam_ablation.py --verify   # re-check every recorded hash
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = "reports/data/m0-v2-minus-xlam-freeze.json"

# Each arm is its own evidence set; the shared inputs appear once.
ARMS = {
    "m0_v2_final_minus_xlam_fixed_compute": {
        "steps": [600, 1200, 1800, 2400],
        "selected": 1200,
    },
    "m0_v2_final_minus_xlam_matched_exposure": {
        "steps": [530, 1060, 1590, 2119],
        "selected": 1060,
    },
}
SHARED = {
    "corpus": [
        "configs/releases/toolpolicy_canonical_v2_final.yaml",
        ".release/hf/toolpolicy-canonical-v2-final/release-manifest.json",
        "reports/data/canonical-v2-final-yield.json",
        "reports/data/canonical-v2-final-minus-xlam-yield.json",
        "reports/data/canonical-v2-ablation-mass.json",
        ".release/hf/toolpolicy-canonical-v2-minus-xlam/release-manifest.json",
    ],
    "evaluation": [
        "reports/evaluation/behavioral-heldout-v2.manifest.json",
        "reports/evaluation/behavioral-heldout-v2-partition.json",
        "reports/baselines/qwen35_2b_baseline/metrics.json",
        "reports/baselines/qwen35_2b_baseline/predictions.jsonl",
    ],
    "design": [
        "configs/experiments/m0_v2_final_minus_xlam_fixed_compute.yaml",
        "configs/experiments/m0_v2_final_minus_xlam_matched_exposure.yaml",
        "docs/evaluation/CHECKPOINT_SELECTION_RULE.md",
        "reports/M0_V2_FINAL_ABLATION_DESIGN.md",
    ],
    "decision": [
        "runs/checkpoint_registry.json",
        "runs/central_ledger.jsonl",
        "results/registry.jsonl",
    ],
    "reports": [
        "reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EXECUTION.md",
        "reports/M0_V2_FINAL_MINUS_XLAM_ABLATION_EVALUATION.md",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entries(paths: list[str]) -> tuple[dict[str, dict], list[str]]:
    entries: dict[str, dict] = {}
    missing: list[str] = []
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        entries[relative] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return dict(sorted(entries.items())), missing


def arm_paths(run: str, steps: list[int], selected: int) -> dict[str, list[str]]:
    return {
        "run": [
            f"runs/{run}/experiment.json",
            f"runs/{run}/environment.json",
            f"runs/{run}/resolved_config.yaml",
            f"runs/{run}/dataset_manifest.json",
            f"runs/{run}/rendering_report.json",
            f"runs/{run}/training_metadata.json",
            f"runs/{run}/metrics/train_log.jsonl",
            f"runs/{run}/ledger.jsonl",
            f"runs/{run}/events.jsonl",
            f"reports/data/{run}-readiness.json",
        ],
        "selection": [
            f"runs/{run}/eval/dev/curve.json",
            f"runs/{run}/eval/dev/selection--dev.json",
            *[f"runs/{run}/eval/dev/checkpoint-{s}/metrics.json" for s in steps],
        ],
        "confirmatory": [
            f"runs/{run}/eval/confirmatory/curve.json",
            f"runs/{run}/eval/confirmatory/checkpoint-{selected}/metrics.json",
        ],
    }


def build() -> int:
    evidence: dict[str, dict[str, dict]] = {}
    missing: list[str] = []
    for group, paths in SHARED.items():
        entries, absent = _entries(paths)
        evidence[group] = entries
        missing.extend(absent)
    unversioned: dict[str, dict[str, dict]] = {}
    for run, spec in ARMS.items():
        for group, paths in arm_paths(run, spec["steps"], spec["selected"]).items():
            entries, absent = _entries(paths)
            evidence.setdefault(run, {}).update(entries)
            missing.extend(absent)
        weights = []
        for step in spec["steps"]:
            weights.append(f"runs/{run}/checkpoints/checkpoint-{step}/model.safetensors")
        weight_entries, _ = _entries(weights)
        unversioned[run] = weight_entries

    if missing:
        # Requiring both arms is the point: a one-arm manifest could be presented as the result.
        print("MISSING (both arms and all shared inputs must exist before freezing):")
        for relative in missing:
            print(f"  - {relative}")
        return 1

    manifest = {
        "schema_version": 1,
        "artifact_kind": "MINUS_XLAM_PAIRED_RESULT_FREEZE",
        "treatment": "remove xLAM together with the corpus's CALL_PREDICTION supervision channel",
        "attribution_limit": (
            "Source identity and supervision type are perfectly aligned in Canonical-v2, so this "
            "pins a joint-removal result and supports no xLAM-specific causal claim."
        ),
        "arms": {
            name: {"steps": spec["steps"], "selected": spec["selected"]}
            for name, spec in ARMS.items()
        },
        "hashes_computed_at_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=False
        ).stdout.strip(),
        "role": (
            "Pins the evidence behind both arms of the paired joint-removal experiment together, "
            "so neither arm can be reported as the conclusion alone."
        ),
        "evidence": evidence,
        "unversioned_by_design": unversioned,
        "reproduce": {
            "fixed_dev": (
                "python scripts/evaluate_sft_checkpoints.py m0_v2_final_minus_xlam_fixed_compute "
                "--partition reports/evaluation/behavioral-heldout-v2-partition.json --partition-side dev"
            ),
            "matched_dev": (
                "python scripts/evaluate_sft_checkpoints.py m0_v2_final_minus_xlam_matched_exposure "
                "--partition reports/evaluation/behavioral-heldout-v2-partition.json --partition-side dev"
            ),
            "select": "python scripts/select_checkpoint.py --run runs/<arm> --side dev",
            "confirmatory": (
                "python scripts/evaluate_sft_checkpoints.py <arm> "
                "--partition reports/evaluation/behavioral-heldout-v2-partition.json "
                "--partition-side confirmatory --checkpoint-ids <selected>"
            ),
        },
    }
    (ROOT / OUT).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    total = sum(len(v) for v in evidence.values())
    print(f"frozen {total} evidence artifacts")
    for group, entries in evidence.items():
        print(f"  {group:<52}{len(entries):>3}")
    print("  unversioned weights: " + ", ".join(f"{k} ({len(v)})" for k, v in unversioned.items()))
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
            if sha256(path) != expected["sha256"]:
                drift.append(f"changed: {relative}")
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
