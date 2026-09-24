#!/usr/bin/env python3
"""Snapshot and verify the completed H200 run before the capability continuation writes anything.

The completed run had no per-file digest manifest -- its ledger recorded environment hashes only.
`PRESERVED_STATE_v1.json` took the digests on 2026-09-12, so from that point any edit to a completed
artifact is detectable.

**v1 cannot be verified off the author's machine.** Its digests were taken over the Windows working
tree (CRLF), not over the committed blobs, so a clone -- and CI, which runs on Linux -- sees LF bytes
and reports every text artifact as modified (`reports/ERRATA.md` §6). v1 is kept byte-for-byte as the
historical record. `PRESERVED_STATE_v2.json` (`reports/ERRATA.md` §21) is its successor:

* it pins the **committed blob** of every tracked artifact, read with
  :func:`opengrad.registry.provenance.read_evidence_bytes`, so the digest is the same on every platform;
* it proves continuity with v1: each v2 blob, rendered with LF or with CRLF line endings, reproduces
  the v1 digest, so the committed bytes are the bytes v1 preserved;
* the two gitignored artifacts v1 pinned are recorded ``in_clone: false`` with the reason, and are
  checked against their v1 digest only where a working copy exists;
* ``reports/H200_BENCHMARK_RUN.md`` stays append-only: its original section is pinned as an LF prefix
  of the committed blob.

Usage:
    python scripts/preserve_h200_state.py --verify        # v2, platform-independent (CI runs this)
    python scripts/preserve_h200_state.py --verify-v1     # v1, the author's Windows working tree only
    python scripts/preserve_h200_state.py --write-v2      # once; refuses to overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from opengrad.hashing import sha256_bytes as sha256
from opengrad.hashing import sha256_file as digest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/benchmarks/h200/PRESERVED_STATE_v1.json"
MANIFEST_V2 = ROOT / "results/benchmarks/h200/PRESERVED_STATE_v2.json"
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

APPEND_ONLY = "report_h200"


def committed(relative: str) -> tuple[bytes | None, str | None]:
    """The committed blob of a tracked path, or why there is none."""
    from opengrad.registry.provenance import read_evidence_bytes

    return read_evidence_bytes(ROOT, relative)


def rendering_of(blob: bytes, pinned: str) -> str | None:
    """Which line-ending rendering of `blob` reproduces a v1 digest: "lf", "crlf", or None."""
    if sha256(blob) == pinned:
        return "lf"
    if sha256(blob.replace(b"\n", b"\r\n")) == pinned:
        return "crlf"
    return None


def original_prefix(blob: bytes, pinned_crlf: str) -> int | None:
    """Length of the LF prefix of `blob` whose CRLF rendering has the v1 `original_sha256`."""
    for end in range(len(blob) + 1):
        if end < len(blob) and blob[end - 1 : end] != b"\n":
            continue
        if sha256(blob[:end].replace(b"\n", b"\r\n")) == pinned_crlf:
            return end
    return None


# -- v2 ------------------------------------------------------------------------------------------


def build_v2() -> dict:
    v1 = json.loads(MANIFEST.read_text(encoding="utf-8"))
    artifacts: dict[str, dict] = {}
    for key, rel in PROTECTED:
        prior = v1["artifacts"][key]
        blob, why = committed(rel)
        if blob is None:
            artifacts[key] = {
                "path": rel,
                "in_clone": False,
                "reason": why,
                "v1_sha256": prior.get("sha256"),
                "note": "gitignored bulk artifact; verified against v1 only where a working copy exists",
            }
            continue
        entry = {"path": rel, "in_clone": True, "sha256": sha256(blob), "bytes": len(blob)}
        if key == APPEND_ONLY:
            original = v1["append_only_updates"][key]["original_sha256"]
            end = original_prefix(blob, original)
            if end is None:
                raise SystemExit(
                    f"{rel}: no prefix of the committed blob reproduces v1's original_sha256"
                )
            entry.pop("sha256")
            entry.pop("bytes")
            entry["original_prefix_bytes"] = end
            entry["original_prefix_sha256"] = sha256(blob[:end])
            entry["v1_original_sha256"] = original
        else:
            rendering = rendering_of(blob, prior["sha256"])
            if rendering is None:
                raise SystemExit(
                    f"{rel}: committed blob does not reproduce the v1 digest in LF or CRLF"
                )
            entry["v1_sha256"] = prior["sha256"]
            entry["v1_rendering"] = rendering
        artifacts[key] = entry
    return {
        "schema_version": 2,
        "supersedes": "results/benchmarks/h200/PRESERVED_STATE_v1.json",
        "preserved_run": v1["preserved_run"],
        "continuation_run": v1["continuation_run"],
        "continuation_namespace": v1["continuation_namespace"],
        "digest_basis": (
            "sha256 of the committed git blob (git cat-file blob HEAD:<path>), not of a working tree, so "
            "the digest is the same on every platform. v1 hashed the author's CRLF working tree."
        ),
        "policy": v1["policy"],
        "append_only_policy": v1["append_only_policy"],
        "append_only_updates": {
            APPEND_ONLY: {
                "reason": v1["append_only_updates"][APPEND_ONLY]["reason"],
                "rule": "the committed blob must begin with the pinned original prefix",
            }
        },
        "artifacts": artifacts,
    }


def verify_v2(manifest_path: Path = MANIFEST_V2) -> tuple[bool, list[str]]:
    """Check every v2 pin against the committed blobs. Returns (ok, report lines)."""
    if not manifest_path.exists():
        return False, [f"FAIL: {manifest_path.relative_to(ROOT).as_posix()} does not exist"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lines: list[str] = []
    failures = checked = local = 0
    for key, record in sorted(manifest["artifacts"].items()):
        rel = record["path"]
        if not record["in_clone"]:
            path = ROOT / rel
            if path.exists() and record.get("v1_sha256"):
                local += 1
                ok = digest(path) == record["v1_sha256"]
                failures += not ok
                lines.append(f"{'ok  ' if ok else 'FAIL'} {rel} (local working copy against v1)")
            else:
                lines.append(f"skip {rel}: not in a clone ({record.get('reason')})")
            continue
        blob, why = committed(rel)
        checked += 1
        if blob is None:
            failures += 1
            lines.append(f"FAIL vanished: {rel} ({why})")
        elif key == APPEND_ONLY:
            end = record["original_prefix_bytes"]
            ok = sha256(blob[:end]) == record["original_prefix_sha256"]
            failures += not ok
            lines.append(f"{'ok  ' if ok else 'FAIL modified original section:'} {rel}")
        else:
            ok = sha256(blob) == record["sha256"]
            failures += not ok
            lines.append(f"{'ok  ' if ok else 'FAIL modified:'} {rel}")
    if checked == 0:
        failures += 1
        lines.append(
            "FAIL: the manifest pins no tracked artifact; a verifier that checked nothing proves nothing"
        )
    lines.append(
        f"checked {checked} committed artifacts, {local} local-only, {failures} failure(s)"
    )
    return failures == 0, lines


# -- v1 (historical) -----------------------------------------------------------------------------


def verify_v1() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    allowed = set(manifest.get("append_only_updates", {}))
    drift, vanished = [], []
    for key, rec in manifest["artifacts"].items():
        if not rec["present"] or key in allowed:
            continue
        path = ROOT / rec["path"]
        if not path.exists():
            vanished.append(rec["path"])
        elif digest(path) != rec["sha256"]:
            drift.append(rec["path"])
    for p in vanished:
        print(f"FAIL vanished: {p}", file=sys.stderr)
    for p in drift:
        print(f"FAIL modified: {p}", file=sys.stderr)
    if drift or vanished:
        print(
            "v1 pins CRLF working-tree bytes; use --verify (v2) off the author's machine",
            file=sys.stderr,
        )
        return 1
    print(
        f"verified {sum(1 for v in manifest['artifacts'].values() if v['present'])} artifacts against v1"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--verify", action="store_true", help="verify PRESERVED_STATE_v2 (committed blobs)"
    )
    group.add_argument(
        "--verify-v1", action="store_true", help="verify v1 against this working tree"
    )
    group.add_argument("--write-v2", action="store_true", help="write PRESERVED_STATE_v2 once")
    args = ap.parse_args(argv)

    if args.verify_v1:
        return verify_v1()
    if args.write_v2:
        if MANIFEST_V2.exists():
            print(
                f"FAIL: {MANIFEST_V2.relative_to(ROOT).as_posix()} exists and is immutable",
                file=sys.stderr,
            )
            return 1
        payload = build_v2()
        with MANIFEST_V2.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"wrote {MANIFEST_V2.relative_to(ROOT).as_posix()}")
        return 0
    ok, lines = verify_v2()
    stream = sys.stdout if ok else sys.stderr
    for line in lines:
        print(line, file=stream)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
