"""The freeze builder must not erase evidence it previously pinned.

Two defects, found on 2026-10-22 when M1-v2's registry status was restated:

* a rebuild on a checkout where a pinned weights or predictions file is absent recorded it as
  nothing, silently, and wrote a manifest that no longer pinned it;
* a rebuild with missing evidence wrote the manifest first and only then reported the failure.

Both are tested here against a scratch repository, so the real manifests are never touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def _module(tmp_path: Path, monkeypatch, evidence: dict, unversioned: dict):
    import freeze_m0_final

    scratch = tmp_path / "repo"
    (scratch / "runs" / freeze_m0_final.EXPERIMENT).mkdir(parents=True)
    (scratch / "runs" / freeze_m0_final.EXPERIMENT / "experiment.json").write_text(
        json.dumps({"status": "REJECTED"}), encoding="utf-8"
    )
    monkeypatch.setattr(freeze_m0_final, "ROOT", scratch)
    monkeypatch.setattr(freeze_m0_final, "EVIDENCE", evidence)
    monkeypatch.setattr(freeze_m0_final, "UNVERSIONED", unversioned)
    return freeze_m0_final, scratch


def test_build_refuses_to_drop_a_pinned_unversioned_file(tmp_path: Path, monkeypatch, capsys) -> None:
    module, scratch = _module(tmp_path, monkeypatch, evidence={}, unversioned={"weights": ["runs/x/model.safetensors"]})
    existing = scratch / module.OUT
    existing.parent.mkdir(parents=True)
    pinned = {"unversioned_by_design": {"weights": {"runs/x/model.safetensors": {"sha256": "0" * 64, "bytes": 1}}}}
    existing.write_text(json.dumps(pinned), encoding="utf-8")
    before = existing.read_bytes()

    assert module.build() == 1
    assert existing.read_bytes() == before, "the manifest must not be rewritten"
    assert "REFUSED" in capsys.readouterr().out


def test_build_refuses_missing_evidence_before_writing_anything(tmp_path: Path, monkeypatch, capsys) -> None:
    module, scratch = _module(tmp_path, monkeypatch, evidence={"g": ["does/not/exist.json"]}, unversioned={})
    out = scratch / module.OUT

    assert module.build() == 1
    assert not out.exists(), "a manifest must not be written when evidence is missing"
    assert "no manifest written" in capsys.readouterr().out


def test_build_writes_when_nothing_is_dropped(tmp_path: Path, monkeypatch) -> None:
    module, scratch = _module(tmp_path, monkeypatch, evidence={}, unversioned={})
    out = scratch / module.OUT
    out.parent.mkdir(parents=True, exist_ok=True)

    assert module.build() == 0
    assert json.loads(out.read_text(encoding="utf-8"))["unversioned_by_design"] == {}
