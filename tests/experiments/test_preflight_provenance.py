"""Preflight must report real git provenance, and the SFT data gate must read the registry
the right way round.

Two defects are pinned here:

1. Preflight read `env["git"]["sha"]`, but `capture()` returns flat `git_sha`/`git_dirty`
   keys, so the check always printed "SHA: unknown" no matter the repository state -- a
   provenance field that silently carried no information.

2. It used the whole-tree dirty flag, which is True for any untracked file. Every run
   creates untracked outputs, so "clean tree" was unsatisfiable in practice. It now uses
   tracked-tree provenance, matching the baseline evidence contract.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from opengrad.env_capture import tracked_tree_provenance
from opengrad.experiments.preflight import run_experiment_preflight

ROOT = Path(__file__).parents[2]
M0_SFT = ROOT / "configs/experiments/m0_sft.yaml"


def _check(result, name):
    return next(check for check in result.checks if check.name == name)


def test_preflight_reports_the_real_commit_not_unknown():
    result = run_experiment_preflight(M0_SFT, root=ROOT)
    git = _check(result, "Git State")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert head[:10] in git.details, git.details
    assert "unknown" not in git.details, git.details


def test_preflight_git_state_reflects_only_tracked_files():
    result = run_experiment_preflight(M0_SFT, root=ROOT)
    git = _check(result, "Git State")
    provenance = tracked_tree_provenance(ROOT)
    assert (git.status == "PASS") is (provenance["dirty"] is False)


def test_untracked_files_do_not_make_provenance_dirty():
    probe = ROOT / "preflight-provenance-probe.txt"
    before = tracked_tree_provenance(ROOT)
    probe.write_text("probe\n", encoding="utf-8")
    try:
        assert tracked_tree_provenance(ROOT) == before
    finally:
        probe.unlink(missing_ok=True)


def test_provenance_fails_closed_outside_a_repository(tmp_path: Path):
    assert tracked_tree_provenance(tmp_path)["dirty"] is True


def _contract_with_registry(monkeypatch, entry: dict) -> tuple[bool, str]:
    from opengrad import readiness as readiness_module

    monkeypatch.setattr(readiness_module, "_read_dataset_registry", lambda root: {"canonical_v1": entry})
    return readiness_module._training_data_contract(
        ROOT,
        {"datasets": {"manifest_ids": ["canonical_v1"], "hashes": {"canonical_v1": "b" * 64}}},
    )


def _entry(**overrides) -> dict:
    base = {
        "intended_stages": ["future_sft"],
        "allowed_splits": ["train"],
        "forbidden_splits": ["evaluation", "heldout", "mcq_test"],
        "source_revision": {"value": "a" * 40},
        "processed_dataset_hash": {"value": "b" * 64},
    }
    base.update(overrides)
    return base


def test_sft_gate_accepts_a_corpus_that_forbids_evaluation_splits(monkeypatch):
    """Declaring a split forbidden is what makes a corpus safe; the gate had it inverted."""
    ok, detail = _contract_with_registry(monkeypatch, _entry())
    assert ok is True, detail


def test_sft_gate_rejects_a_corpus_that_allows_evaluation_splits(monkeypatch):
    ok, detail = _contract_with_registry(
        monkeypatch, _entry(allowed_splits=["train", "mcq_test"])
    )
    assert ok is False
    assert "cannot enter SFT" in detail


def test_sft_gate_rejects_an_evaluation_intended_corpus(monkeypatch):
    ok, _ = _contract_with_registry(monkeypatch, _entry(intended_stages=["evaluation"]))
    assert ok is False


def test_sft_gate_rejects_a_preference_intended_corpus(monkeypatch):
    ok, _ = _contract_with_registry(monkeypatch, _entry(intended_stages=["future_preference"]))
    assert ok is False


def test_sft_gate_requires_a_pinned_source_revision(monkeypatch):
    ok, detail = _contract_with_registry(monkeypatch, _entry(source_revision={"value": "not-a-sha"}))
    assert ok is False
    assert "source revision is not pinned" in detail


def test_sft_gate_requires_the_registry_hash_to_match_the_config(monkeypatch):
    ok, detail = _contract_with_registry(
        monkeypatch, _entry(processed_dataset_hash={"value": "c" * 64})
    )
    assert ok is False
    assert "does not match registry" in detail


def test_registered_canonical_corpus_satisfies_the_training_data_contract():
    """The real registry entry, not a stub, must pass the gate it exists to satisfy."""
    import yaml

    from opengrad.readiness import _training_data_contract

    config = yaml.safe_load(M0_SFT.read_text())
    ok, detail = _training_data_contract(ROOT, config)
    assert ok is True, detail


def test_canonical_entry_hash_matches_the_release_manifest():
    """The pinned hash must identify the artifact, not a remembered value."""
    import hashlib
    import json

    import yaml

    manifest = ROOT / ".release/hf/toolpolicy-canonical-v1/release-manifest.json"
    if not manifest.is_file():
        pytest.skip("the assembled release is not present in this checkout")
    registry = yaml.safe_load((ROOT / "registry/datasets.yaml").read_text())
    entry = next(item for item in registry["datasets"] if item["id"] == "canonical_v1")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert entry["checksum"] == digest
    assert entry["processed_dataset_hash"]["value"] == digest
    assert entry["source_revision"]["value"] == json.loads(manifest.read_text())["opengrad_git_commit"]
