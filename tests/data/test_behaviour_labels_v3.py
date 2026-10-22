"""labels-v2 verify (D9, O3): the real artifact passes, and a tampered copy cannot pass.

Verify must catch an incomplete artifact: a source missing from the labels, or stored totals that are not the sum
of the rows. The real artifact is checked when it has been built; the tampered copies are always checked.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from opengrad.data import behaviour_labels_v3 as bl3

ROOT = Path(__file__).resolve().parents[2]
LABELS_BUILT = (ROOT / bl3.OUTPUT_DIR / "manifest.json").is_file()


def _copy_artifact(tmp_path: Path) -> Path:
    """A minimal root with the two frozen sources, the corpus manifest and the labels-v2 output."""
    for module in (bl3.CLASSIFIER_MODULE, bl3.RULES_MODULE, bl3.CORPUS_MANIFEST):
        target = tmp_path / module
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / module, target)
    shutil.copytree(ROOT / bl3.OUTPUT_DIR, tmp_path / bl3.OUTPUT_DIR)
    return tmp_path


@pytest.mark.skipif(
    not LABELS_BUILT, reason="labels-v2 not built: run opengrad.data.behaviour_labels_v3 --build"
)
def test_the_built_labels_v2_artifact_verifies() -> None:
    result = bl3.verify(ROOT)
    assert result["status"] == "PASS", result["problems"]
    assert result["totals"]["records"] == 181433
    assert result["totals"]["labels"]["DECLINE_WITH_CONTENT"] == 1222
    assert result["totals"]["weight_permitted"] == {}


@pytest.mark.skipif(
    not LABELS_BUILT, reason="labels-v2 not built: run opengrad.data.behaviour_labels_v3 --build"
)
def test_a_missing_source_fails_verify(tmp_path: Path) -> None:
    root = _copy_artifact(tmp_path)
    manifest_path = root / bl3.OUTPUT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dropped = min(manifest["sources"])
    (root / bl3.OUTPUT_DIR / manifest["sources"][dropped]["file"]).unlink()
    del manifest["sources"][dropped]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = bl3.verify(root)
    assert result["status"] == "FAIL"
    assert any(
        "not the sources of the normalization-v3 corpus" in problem
        for problem in result["problems"]
    )


@pytest.mark.skipif(
    not LABELS_BUILT, reason="labels-v2 not built: run opengrad.data.behaviour_labels_v3 --build"
)
def test_stored_totals_that_do_not_add_up_fail_verify(tmp_path: Path) -> None:
    root = _copy_artifact(tmp_path)
    manifest_path = root / bl3.OUTPUT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["totals"]["records"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = bl3.verify(root)
    assert result["status"] == "FAIL"
    assert any("stored totals" in problem for problem in result["problems"])
