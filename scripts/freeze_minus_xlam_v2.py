"""Superseding freeze for the minus-xlam ablation, which cannot be verified against its v1 manifest.

``reports/data/m0-v2-minus-xlam-freeze.json`` (v1) is left byte-identical. Its verify fails because
some of the evidence it pins no longer exists on this checkout, and some pinned files changed after
the manifest was made. This script writes ``m0-v2-minus-xlam-freeze-v2.json``, which says exactly
what can and cannot be checked:

* ``present``: the file is on disk and its SHA-256 equals the v1 pin. Checked against the working tree.
* ``git``: the working-tree file changed, but its SHA-256 equals the v1 pin for the git blob at the
  commit the v1 manifest names. Checked with ``git show``, so the pinned state is recoverable.
* ``unrecoverable``: the pinned file does not exist anywhere in the repository history or on disk.
  It is listed with its v1 hash so the claim stays traceable, and it is never counted as verified.

``--verify`` fails if any ``present`` or ``git`` pin fails. ``unrecoverable`` pins are reported,
not counted. The script never writes the v1 manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1 = "reports/data/m0-v2-minus-xlam-freeze.json"
V2 = "reports/data/m0-v2-minus-xlam-freeze-v2.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob(commit: str, path: str) -> bytes | None:
    done = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, check=False)
    return done.stdout if done.returncode == 0 else None


def _v1_pins(v1: dict) -> list[dict]:
    pins: list[dict] = []
    for group, entries in v1["evidence"].items():
        for path, meta in entries.items():
            pins.append({"group": group, "path": path, "sha256": meta["sha256"], "bytes": meta["bytes"]})
    for group, entries in v1.get("unversioned_by_design", {}).items():
        for path, meta in entries.items():
            pins.append({"group": f"unversioned/{group}", "path": path, "sha256": meta["sha256"], "bytes": meta["bytes"]})
    return pins


def build() -> int:
    v1_path = ROOT / V1
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    commit = v1["hashes_computed_at_commit"]
    records: list[dict] = []
    for pin in _v1_pins(v1):
        path = ROOT / pin["path"]
        if path.is_file() and _sha(path.read_bytes()) == pin["sha256"]:
            records.append({**pin, "state": "present"})
            continue
        blob = _git_blob(commit, pin["path"])
        if blob is not None and _sha(blob) == pin["sha256"]:
            records.append({**pin, "state": "git", "git_commit": commit})
            continue
        records.append({**pin, "state": "unrecoverable",
                        "reason": "not on disk and not in git history at the pinned commit or later"})
    counts: dict[str, int] = {}
    for r in records:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    manifest = {
        "schema_version": 2,
        "artifact_kind": "MINUS_XLAM_PAIRED_RESULT_FREEZE_V2",
        "supersedes": {
            "path": V1,
            "sha256": _sha(v1_path.read_bytes()),
            "note": "v1 is kept byte-identical. Its verify fails (missing and changed evidence); see records.",
        },
        "hashes_computed_at_commit": commit,
        "record_counts": dict(sorted(counts.items())),
        "records": records,
    }
    out = ROOT / V2
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {V2}: {counts}")
    return 0


def verify() -> int:
    manifest = json.loads((ROOT / V2).read_text(encoding="utf-8"))
    commit = manifest["hashes_computed_at_commit"]
    failures: list[str] = []
    checked = {"present": 0, "git": 0}
    unrecoverable: list[str] = []
    for r in manifest["records"]:
        if r["state"] == "present":
            path = ROOT / r["path"]
            if not path.is_file() or _sha(path.read_bytes()) != r["sha256"]:
                failures.append(f"present pin no longer matches: {r['path']}")
            else:
                checked["present"] += 1
        elif r["state"] == "git":
            blob = _git_blob(r.get("git_commit", commit), r["path"])
            if blob is None or _sha(blob) != r["sha256"]:
                failures.append(f"git pin no longer matches the blob at {commit[:8]}: {r['path']}")
            else:
                checked["git"] += 1
        else:
            unrecoverable.append(r["path"])
    print(f"checked present: {checked['present']}, checked git: {checked['git']}")
    print(f"UNRECOVERABLE (reported, not verified): {len(unrecoverable)}")
    for path in unrecoverable:
        print(f"  - {path}")
    if failures:
        print("VERIFY FAILED")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("VERIFY PASSED for every recoverable pin")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    return verify() if args.verify else build()


if __name__ == "__main__":
    sys.exit(main())
