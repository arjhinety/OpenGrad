#!/usr/bin/env python3
"""Build the minus-xLAM view of the frozen Canonical-v2 corpus.

A source ablation is a selection, not a rebuild. Shards in the frozen corpus are written per
source, so removing xLAM is the removal of that source's shards and nothing else: every retained
shard is copied byte for byte and re-hashed against the parent manifest to prove it. No record is
parsed, rewritten, re-encoded or re-sharded, so the retained records cannot drift from the corpus
the reference run trained on.

The derived manifest records the parent it came from, so the published view is verifiable against
the published corpus rather than merely asserted to match it.

Usage:
    python scripts/build_minus_xlam_release.py                  # build
    python scripts/build_minus_xlam_release.py --verify         # re-verify an existing build
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PARENT = ".release/hf/toolpolicy-canonical-v2-final"
OUTPUT = ".release/hf/toolpolicy-canonical-v2-minus-xlam"
EXCLUDED = "xlam-function-calling-60k"
PARENT_HUB = "arrochi112/OpenGrad-ToolPolicy-Canonical-v2"
PARENT_HUB_REVISION = "66470c07ed0a79941f49a5cf67c1b3b1a7d8196e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_fingerprint(path: Path) -> str:
    """The release identity: the sha256 of the manifest file itself."""
    return sha256(path)


def load_parent() -> tuple[dict, Path]:
    parent_dir = ROOT / PARENT
    manifest_path = parent_dir / "release-manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"parent manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8")), manifest_path


def build() -> int:
    parent, parent_manifest_path = load_parent()
    parent_fingerprint = manifest_fingerprint(parent_manifest_path)

    kept_sources = [s for s in parent["sources"] if s["source"] != EXCLUDED]
    dropped_sources = [s for s in parent["sources"] if s["source"] == EXCLUDED]
    if not dropped_sources:
        raise SystemExit(f"{EXCLUDED!r} is not a source of the parent corpus")

    kept_files = [name for s in kept_sources for name in s["output_shards"]]
    dropped_files = [name for s in dropped_sources for name in s["output_shards"]]
    parent_shards = {s["file"]: s for s in parent["output_shards"]}

    output = ROOT / OUTPUT
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    copied: list[dict] = []
    for name in sorted(kept_files):
        source_path = ROOT / PARENT / name
        if not source_path.is_file():
            raise SystemExit(f"parent shard missing: {source_path}")
        target = output / name
        shutil.copyfile(source_path, target)
        expected = parent_shards[name]["sha256"]
        actual = sha256(target)
        if actual != expected:
            raise SystemExit(f"byte-identity check failed for {name}: {actual} != {expected}")
        copied.append({"file": name, "sha256": actual, "bytes": target.stat().st_size})
        if len(copied) % 25 == 0:
            print(f"  copied {len(copied)}/{len(kept_files)} shards", flush=True)

    record_count = sum(int(s["records"]) for s in kept_sources)

    derived = dict(parent)
    derived["release_name"] = f"{parent['release_name']} — minus xLAM view"
    derived["release_version"] = "v2.1.0-minus-xlam"
    derived["hub_repository"] = "arrochi112/OpenGrad-ToolPolicy-Canonical-v2-minus-xlam"
    derived["sources"] = kept_sources
    derived["record_count"] = record_count
    derived["output_shards"] = copied
    # The records were built by the parent at its build commit, so the derived view pins the same
    # commit rather than the commit of the script that selected them. This keeps the derived
    # manifest's identity stable when it is regenerated.
    derived["derived_view"] = {
        "kind": "SOURCE_ABLATION_VIEW",
        "method": (
            "byte-identical shard selection; no record was parsed, rewritten, re-encoded or "
            "re-sharded"
        ),
        "parent": {
            "release_name": parent["release_name"],
            "release_version": parent["release_version"],
            "manifest_fingerprint": parent_fingerprint,
            "hub_repository": PARENT_HUB,
            "hub_revision": PARENT_HUB_REVISION,
            "record_count": parent["record_count"],
        },
        "excluded_source": EXCLUDED,
        "excluded_record_count": sum(int(s["records"]) for s in dropped_sources),
        "excluded_shards": [
            {
                "file": name,
                "sha256": parent_shards[name]["sha256"],
                "bytes": parent_shards[name]["bytes"],
            }
            for name in sorted(dropped_files)
        ],
        "retained_shard_count": len(kept_files),
        "excluded_shard_count": len(dropped_files),
        "why_published": (
            "This view is the exact training input of the xLAM source ablation "
            "(m0_v2_final_minus_xlam_fixed_compute and m0_v2_final_minus_xlam_matched_exposure). "
            "It is published so the ablation's input can be inspected independently of the runs "
            "and independently of the project's tooling."
        ),
        "not_a_result": (
            "A corpus view carries no result. Whether including xLAM helps is measured by the "
            "ablation runs against the frozen evaluation partitions, not by this dataset."
        ),
    }

    manifest_path = output / "release-manifest.json"
    manifest_path.write_text(json.dumps(derived, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Attribution files are inherited verbatim rather than regenerated. Over-including a licence
    # is not an error; dropping one is.
    for extra in ("CITATIONS.bib", "source-licenses.md"):
        source_file = ROOT / PARENT / extra
        if source_file.is_file():
            shutil.copyfile(source_file, output / extra)

    print()
    print(f"parent fingerprint : {parent_fingerprint}")
    print(
        f"excluded source    : {EXCLUDED} ({derived['derived_view']['excluded_record_count']:,} records, {len(dropped_files)} shards)"
    )
    print(f"retained sources   : {[s['source'] for s in kept_sources]}")
    print(f"retained shards    : {len(kept_files)}")
    print(f"record count       : {parent['record_count']:,} -> {record_count:,}")
    print(f"derived fingerprint: {manifest_fingerprint(manifest_path)}")
    print(f"wrote {manifest_path.relative_to(ROOT)}")
    return 0


def verify() -> int:
    """Re-check an existing build against the parent, independently of how it was made."""
    parent, parent_manifest_path = load_parent()
    output = ROOT / OUTPUT
    manifest_path = output / "release-manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"no derived manifest at {manifest_path}; build first")

    derived = json.loads(manifest_path.read_text(encoding="utf-8"))
    parent_shards = {s["file"]: s for s in parent["output_shards"]}
    failures: list[str] = []

    print(f"parent fingerprint : {manifest_fingerprint(parent_manifest_path)}")
    print(f"parent revision    : {PARENT_HUB_REVISION}")
    print()

    # 1. Every retained shard must be byte-identical to the parent's.
    for entry in derived["output_shards"]:
        name = entry["file"]
        target = output / name
        if not target.is_file():
            failures.append(f"missing shard {name}")
            continue
        if sha256(target) != parent_shards[name]["sha256"]:
            failures.append(f"shard {name} differs from parent")
    print(f"byte-identity    : {len(derived['output_shards'])} shards checked against parent")

    # 2. No shard from the excluded source may be present, by name or by recorded content.
    present = {p.name for p in output.glob("*.parquet")}
    leaked = sorted(present - {e["file"] for e in derived["output_shards"]})
    if leaked:
        failures.append(f"unexpected shards present: {leaked}")
    print(f"shard set        : {len(present)} present, {len(derived['output_shards'])} declared")

    # 3. Independent content check: read the source column of every shard. The parent's shards are
    #    per-source, but the check belongs on the data rather than on the file names.
    import pyarrow.parquet as pq

    sources: dict[str, int] = {}
    rows = 0
    for name in sorted(present):
        table = pq.read_table(output / name, columns=["source_dataset"])
        column = table.column("source_dataset").to_pylist()
        rows += len(column)
        for value in column:
            sources[str(value)] = sources.get(str(value), 0) + 1
    print(f"content check    : {rows:,} rows read across {len(present)} shards")
    for source, count in sorted(sources.items()):
        marker = "EXCLUDED SOURCE PRESENT" if source == EXCLUDED else ""
        print(f"                   {source:<32}{count:>8,} {marker}")
        if source == EXCLUDED:
            failures.append(f"excluded source {EXCLUDED} found in {count} rows")
    if rows != derived["record_count"]:
        failures.append(f"row count {rows:,} != manifest record_count {derived['record_count']:,}")

    print()
    if failures:
        print("VERIFY FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("VERIFY PASSED")
    print(f"  fingerprint : {manifest_fingerprint(manifest_path)}")
    print(f"  records     : {rows:,}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="re-verify an existing build")
    args = parser.parse_args()
    return verify() if args.verify else build()


if __name__ == "__main__":
    raise SystemExit(main())
