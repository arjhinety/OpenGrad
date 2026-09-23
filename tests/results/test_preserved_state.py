"""Frozen-artifact pins are verified in CI, on the committed blobs (GUARDRAILS G7, `reports/ERRATA.md` §21).

Before this test, nothing ran `scripts/preserve_h200_state.py --verify`, and on any LF checkout it
failed: v1 pinned the author's CRLF working tree. `PRESERVED_STATE_v2.json` pins the committed blobs
and proves they are the bytes v1 preserved. The second test covers every tracked `.sha256` sidecar,
which until now only per-artifact tests checked.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _script():
    spec = importlib.util.spec_from_file_location(
        "preserve_h200_state", ROOT / "scripts/preserve_h200_state.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _blob(relative: str) -> bytes:
    return subprocess.run(
        ["git", "cat-file", "blob", f"HEAD:{relative}"], cwd=ROOT, capture_output=True, check=True
    ).stdout


def test_the_preserved_h200_state_verifies_on_committed_blobs() -> None:
    ok, lines = _script().verify_v2()
    assert ok, "\n".join(lines)
    assert lines[-1].startswith("checked 16 committed artifacts"), lines[-1]


def test_v2_is_continuous_with_v1() -> None:
    """Every v2 pin reproduces its v1 digest under one line-ending rendering: 14 CRLF, 1 LF (ERRATA §6)."""
    v1 = json.loads(
        (ROOT / "results/benchmarks/h200/PRESERVED_STATE_v1.json").read_text(encoding="utf-8")
    )
    v2 = json.loads(
        (ROOT / "results/benchmarks/h200/PRESERVED_STATE_v2.json").read_text(encoding="utf-8")
    )
    assert set(v1["artifacts"]) == set(v2["artifacts"])
    renderings = [
        record.get("v1_rendering")
        for record in v2["artifacts"].values()
        if record.get("v1_rendering")
    ]
    assert (renderings.count("crlf"), renderings.count("lf")) == (14, 1)
    assert sum(1 for record in v2["artifacts"].values() if not record["in_clone"]) == 2
    for record in v2["artifacts"].values():
        if record.get("v1_rendering"):
            assert record["v1_sha256"] == v1["artifacts"][_key(v1, record["path"])]["sha256"]


def _key(manifest: dict, path: str) -> str:
    return next(key for key, record in manifest["artifacts"].items() if record["path"] == path)


def test_a_modified_artifact_fails_verification(tmp_path) -> None:
    module = _script()
    manifest = json.loads(
        (ROOT / "results/benchmarks/h200/PRESERVED_STATE_v2.json").read_text(encoding="utf-8")
    )
    manifest["artifacts"]["findings_ledger"]["sha256"] = "0" * 64
    tampered = tmp_path / "PRESERVED_STATE_v2.json"
    tampered.write_text(json.dumps(manifest), encoding="utf-8")
    ok, lines = module.verify_v2(tampered)
    assert not ok
    assert any("FAIL modified: results/quantization/findings.jsonl" in line for line in lines)


def test_an_empty_manifest_fails_rather_than_passes(tmp_path) -> None:
    module = _script()
    empty = tmp_path / "PRESERVED_STATE_v2.json"
    empty.write_text(json.dumps({"artifacts": {}}), encoding="utf-8")
    ok, lines = module.verify_v2(empty)
    assert not ok
    assert any("checked nothing" in line for line in lines)


def test_every_tracked_sha256_sidecar_matches_its_committed_file() -> None:
    sidecars = subprocess.run(
        ["git", "ls-files", "*.sha256"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    assert len(sidecars) >= 43
    for sidecar in sidecars:
        target = sidecar[: -len(".sha256")]
        recorded = _blob(sidecar).decode().split()
        assert recorded, sidecar
        if len(recorded) == 2:
            assert recorded[1] == Path(target).name, sidecar
        assert hashlib.sha256(_blob(target)).hexdigest() == recorded[0], sidecar
