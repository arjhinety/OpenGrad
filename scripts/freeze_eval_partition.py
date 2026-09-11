#!/usr/bin/env python3
"""Freeze the DEV / confirmatory partition of the behavioural evaluation population.

Why this exists
---------------
`behavioral-heldout-v2` is not pristine test evidence. The partial-v2 experiment selected its
checkpoint 1200 by taking the maximum over four checkpoints scored on this same population, so a
peak value reported from it is the maximum over several draws rather than an unbiased measurement
of one model. The upstream test population has also influenced earlier OpenGrad work.

This does not block training, but it does mean the choice of checkpoint and the measurement of
that checkpoint must not come from the same examples. The population is therefore split, before
training starts, into:

  DEV           checkpoint selection only
  CONFIRMATORY  evaluated exactly once, on the selected checkpoint only

The confirmatory partition is a **pre-registered internal** partition. It is not an untouched
external benchmark, and the wording matters: the broader upstream population has already
participated in earlier work, so a clean-external-test claim would be false.

Determinism
-----------
The split is a pure function of (example_id, expected_decision, seed): every example is sorted by
a keyed SHA-256 of `seed || example_id`, and the confirmatory partition takes the last-ranked
example of each class until it reaches the target size. Re-running this produces byte-identical
partitions, and the script records fingerprints so a later run can prove it.

Stratification is by `expected_decision`, because the population has three classes of very
different size and an unstratified split can silently starve a class that the whole experiment
cares about — CLARIFY and UNSUPPORTED behaviour is exactly what the partial-v2 result improved.

Usage:
    python scripts/freeze_eval_partition.py
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opengrad.evaluation.runner import load_evaluation_examples  # noqa: E402

MANIFEST = "reports/evaluation/behavioral-heldout-v2.manifest.json"
OUTPUT = "reports/evaluation/behavioral-heldout-v2-partition.json"

# Fixed and documented. Chosen once, not searched: trying seeds until the class balance looks
# attractive would make the partition a fitted artefact rather than a pre-registered one.
SPLIT_SEED = "opengrad-m0-final-v1"
CONFIRMATORY_FRACTION = 0.35


def ranking_key(seed: str, example_id: str) -> str:
    """Deterministic, order-independent rank for one example."""
    return hashlib.sha256(f"{seed}\x00{example_id}".encode("utf-8")).hexdigest()


def fingerprint(example_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for example_id in sorted(example_ids):
        digest.update(example_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT)
    args = parser.parse_args()

    examples = load_evaluation_examples(ROOT, ROOT / MANIFEST)
    by_class: dict[str, list[str]] = collections.defaultdict(list)
    for example in examples:
        by_class[str(example.expected_decision)].append(str(example.example_id))

    confirmatory: list[str] = []
    dev: list[str] = []
    per_class: dict[str, dict[str, int]] = {}
    for label in sorted(by_class):
        ids = sorted(by_class[label], key=lambda item: ranking_key(SPLIT_SEED, item))
        # Largest rank first into confirmatory, so the choice does not depend on input ordering.
        take = round(len(ids) * CONFIRMATORY_FRACTION)
        # Every class keeps at least one example on each side where its size permits, so a rare
        # class cannot vanish from either partition.
        if 0 < take < len(ids) or len(ids) >= 2:
            take = max(1, min(take, len(ids) - 1))
        confirmatory.extend(ids[:take])
        dev.extend(ids[take:])
        per_class[label] = {"total": len(ids), "dev": len(ids) - take, "confirmatory": take}

    payload = {
        "schema_version": 1,
        "artifact_kind": "EVAL_PARTITION",
        "partition_id": "behavioral-heldout-v2-dev-confirmatory",
        "evaluation_manifest": MANIFEST,
        "evaluation_manifest_sha256": hashlib.sha256((ROOT / MANIFEST).read_bytes()).hexdigest(),
        "population_size": len(examples),
        "algorithm": "stratified_by_expected_decision__sha256(seed||example_id)_descending",
        "seed": SPLIT_SEED,
        "confirmatory_fraction": CONFIRMATORY_FRACTION,
        "dev": {
            "count": len(dev),
            "fingerprint": fingerprint(dev),
        },
        "confirmatory": {
            "count": len(confirmatory),
            "fingerprint": fingerprint(confirmatory),
        },
        "per_class": per_class,
        "roles": {
            "dev": (
                "Checkpoint selection only. Every scheduled checkpoint may be evaluated here, "
                "and the selection rule is applied to these results."
            ),
            "confirmatory": (
                "Pre-registered internal confirmatory partition. Evaluated exactly once, on the "
                "checkpoint selected on DEV. Never used to choose a checkpoint, change steps, "
                "change the learning rate, change thresholds, change the source mixture, or "
                "select generation settings."
            ),
        },
        "not_a_clean_external_benchmark": (
            "The wider upstream evaluation population has already participated in earlier "
            "OpenGrad work, including the partial-v2 checkpoint selection. This partition "
            "pre-registers which examples are held back from selection; it does not make the "
            "underlying population pristine, and results from it must be described as a "
            "pre-registered internal confirmatory evaluation."
        ),
        "example_ids": {"dev": sorted(dev), "confirmatory": sorted(confirmatory)},
    }
    out = ROOT / args.output
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"population    : {len(examples)}")
    print(f"dev           : {len(dev)}  {payload['dev']['fingerprint'][:16]}")
    print(f"confirmatory  : {len(confirmatory)}  {payload['confirmatory']['fingerprint'][:16]}")
    for label, counts in per_class.items():
        print(f"  {label:<12} total={counts['total']:<5} dev={counts['dev']:<5} conf={counts['confirmatory']}")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
