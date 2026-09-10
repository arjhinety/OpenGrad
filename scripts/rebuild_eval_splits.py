"""Rebuild the frozen held-out evaluation splits and re-freeze their content hashes.

The withheld When2Call evaluation splits are not redistributable training data, so
they are intentionally absent from the canonical release and from Git. This script
recreates them from the pinned upstream revision, materializes them with the
repository materializer, and (with ``--write-manifest``) records the resulting
content hashes plus provenance in the frozen evaluation manifest.

Why this exists: the manifest's original ``content_hash`` values were authored by
hand before ``materialize_parquet`` computed content hashes at all, so they could
never match a real artifact and the verification gate could never pass. Re-freezing
replaces those placeholders with hashes computed from the real pinned data and
records exactly how they were produced.

Usage:
    python scripts/rebuild_eval_splits.py                 # materialize + print provenance
    python scripts/rebuild_eval_splits.py --write-manifest  # also re-freeze the manifest

`--write-manifest` edits a frozen evidence contract. It only replaces
``content_hash`` and adds ``hash_provenance``; every other field is preserved.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY = "nvidia/When2Call"
REVISION = "0582f7749df63a96fdc3070932e83e72396ace53"

MANIFEST_RELATIVE = Path("reports/evaluation/behavioral-heldout-v2.manifest.json")

RE_FREEZE_REASON = (
    "The previous content_hash was not produced by any repository code path; "
    "content_hash support was added to opengrad.data.materialize after this "
    "manifest was frozen. Re-frozen from the real pinned materialization."
)


@dataclass(frozen=True)
class SplitSpec:
    split: str
    manifest_id: str
    directory: str
    upstream_file: str


SPLITS = (
    SplitSpec("mcq", "when2call-mcq", "when2call-mcq", "test/when2call_test_mcq.jsonl"),
    SplitSpec(
        "llm_judge",
        "when2call-llm-judge",
        "when2call-llm-judge",
        "test/when2call_test_llm_judge.jsonl",
    ),
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _jsonl_to_parquet(jsonl: Path, destination: Path) -> Path:
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_jsonl(jsonl)
    pq.write_table(pa.Table.from_pylist(rows), destination)
    return destination


def _fetch(root: Path, spec: SplitSpec) -> Path:
    from huggingface_hub import hf_hub_download

    target = root / "data/raw/when2call/test" / Path(spec.upstream_file).name
    if target.exists():
        return target
    downloaded = hf_hub_download(
        repo_id=REPOSITORY,
        filename=spec.upstream_file,
        revision=REVISION,
        repo_type="dataset",
        local_dir=str(root / "data/raw/when2call"),
    )
    return Path(downloaded)


def _materialize(root: Path, spec: SplitSpec, parquet: Path) -> dict[str, Any]:
    sys.path.insert(0, str(root / "src"))
    from opengrad.data.materialize import materialize_parquet

    output = root / "data/processed/normalization-v1" / spec.directory
    if output.exists():
        shutil.rmtree(output)
    return materialize_parquet(
        parquet, output, dataset="when2call", split=spec.split, mode="evaluation"
    )


def _provenance(root: Path, spec: SplitSpec, written: int) -> dict[str, Any]:
    command = (
        "opengrad-data materialize-parquet --dataset when2call "
        f"--input data/raw/when2call/parquet/{spec.directory}.parquet "
        f"--output data/processed/normalization-v1/{spec.directory} "
        f"--split {spec.split} --mode evaluation"
    )
    return {
        "computed_by": "opengrad.data.materialize.materialize_parquet",
        "source_repository": REPOSITORY,
        "source_revision": REVISION,
        "source_files": [spec.upstream_file],
        "materialized_path": f"data/processed/normalization-v1/{spec.directory}",
        "command": command,
        "reason": RE_FREEZE_REASON,
        "re_frozen_at": datetime.now(UTC).isoformat(),
        "superseded_content_hash": None,
    }


def _write_manifest(root: Path, results: dict[str, Any]) -> Path:
    manifest_path = root / MANIFEST_RELATIVE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for split in manifest["splits"]:
        result = results.get(split["id"])
        if result is None:
            continue
        existing = split.get("hash_provenance") or {}
        # Preserve the first-recorded superseded value and timestamp so re-running
        # this script is idempotent rather than rewriting its own history.
        provenance = {
            **result["provenance"],
            "re_frozen_at": existing.get("re_frozen_at", result["provenance"]["re_frozen_at"]),
            "superseded_content_hash": existing.get(
                "superseded_content_hash", split["content_hash"]
            ),
        }
        split["hash_provenance"] = provenance
        split["content_hash"] = result["content_hash"]
        split["items"] = result["written"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _sync_experiment_dataset_hash(root, manifest_path)
    return manifest_path


def _sync_experiment_dataset_hash(root: Path, manifest_path: Path) -> None:
    """Keep the baseline experiment's dataset_hash bound to the manifest bytes.

    ``opengrad-validate`` requires this equality, so a re-freeze that changed the
    manifest would otherwise break registry validation. Rewriting the single
    scalar line preserves the rest of the hand-formatted config exactly.
    """
    import hashlib
    import re

    experiment = root / "configs/experiments/tool_calling/qwen35_2b_baseline.yaml"
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    updated, count = re.subn(
        r"(?m)^dataset_hash:\s*[0-9a-f]{64}\s*$",
        f"dataset_hash: {digest}",
        experiment.read_text(encoding="utf-8"),
    )
    if count != 1:
        raise SystemExit(f"expected exactly one dataset_hash line in {experiment}, found {count}")
    experiment.write_text(updated, encoding="utf-8")
    print(f"  synced dataset_hash in {experiment.relative_to(root)} -> {digest[:16]}…")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="re-freeze content_hash and hash_provenance in the frozen manifest",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    manifest = json.loads((root / MANIFEST_RELATIVE).read_text(encoding="utf-8"))
    expected = {split["id"]: split["items"] for split in manifest["splits"]}

    parquet_dir = root / "data/raw/when2call/parquet"
    results: dict[str, Any] = {}
    for spec in SPLITS:
        jsonl = _fetch(root, spec)
        parquet = _jsonl_to_parquet(jsonl, parquet_dir / f"{spec.directory}.parquet")
        outcome = _materialize(root, spec, parquet)
        written = outcome["manifest"]["counts"]["written"]
        want = expected.get(spec.manifest_id)
        status = "OK" if want == written else f"MISMATCH (manifest declares {want})"
        print(f"{spec.manifest_id:22s} rows={written:5d} {status}")
        print(f"  content_hash = {outcome['manifest']['content_hash']}")
        if want != written:
            raise SystemExit(f"row count mismatch for {spec.manifest_id}: {written} != {want}")
        results[spec.manifest_id] = {
            "written": written,
            "content_hash": outcome["manifest"]["content_hash"],
            "provenance": _provenance(root, spec, written),
        }

    if args.write_manifest:
        path = _write_manifest(root, results)
        print(f"\nre-froze {path.relative_to(root)}")
    else:
        print("\n(dry run; pass --write-manifest to re-freeze)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
