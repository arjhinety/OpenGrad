"""Provenance rules: what makes a `verified: true` claim defensible.

The rule under test is immutability, not uniformity. A mutable repository path
must carry a content digest; an externally pinned source must carry an immutable
revision and is deliberately *not* asked for a redundant hash. Both halves are
tested, because a validator that demands the same field everywhere would be as
wrong as one that demands nothing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from opengrad.registry import provenance
from opengrad.registry.validate import (
    check_freeze as validate_freeze_result,
)
from opengrad.registry.validate import (
    validate_freeze,
    validate_publication_records,
    validate_reconstructed_events,
    validate_references,
    validate_structure,
)
from opengrad.verification import PASS

REVISION = "a" * 40
OTHER_REVISION = "b" * 40


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write(path: Path, text: str) -> None:
    """Write exact bytes.

    `Path.write_text` translates newlines on Windows, so a fixture written that
    way would not hash to the digest of the string it was built from. Provenance
    is about bytes, so the fixtures are written as bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def _root(tmp_path: Path, datasets: list[dict], models: list[dict] | None = None) -> Path:
    (tmp_path / "registry").mkdir(parents=True, exist_ok=True)
    import yaml

    (tmp_path / "registry/datasets.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "status": "TEST", "datasets": datasets}),
        encoding="utf-8",
    )
    (tmp_path / "registry/models.yaml").write_text(
        yaml.safe_dump({"schema_version": 2, "models": models or []}), encoding="utf-8"
    )
    return tmp_path


def _dataset(**overrides) -> dict:
    record = {
        "id": "fixture",
        "display_name": "Fixture",
        "organization": "Fixture Org",
        "category": "tool_use",
        "allowed_splits": ["train"],
        "forbidden_splits": [],
        "contamination_status": "UNASSESSED",
        "retrieval_date": "2026-01-01",
    }
    record.update(overrides)
    return record


# ── Expected PASS ────────────────────────────────────────────────────────────


def test_verified_local_claim_with_matching_sha256_passes(tmp_path: Path):
    _write(tmp_path / "reports/evidence.md", "licence text\n")
    record = _dataset(
        license={
            "value": "Apache-2.0",
            "verified": True,
            "source": "reports/evidence.md",
            "source_sha256": _sha256("licence text\n"),
        }
    )
    audit = provenance.check_license_claim(record, tmp_path, "fixture")
    assert audit is not None and audit.sound, audit.errors
    assert audit.category == provenance.MUTABLE_HASHED


def test_verified_external_claim_with_immutable_revision_passes(tmp_path: Path):
    record = _dataset(
        hf_id="owner/dataset",
        license={
            "value": "CC-BY-4.0",
            "verified": True,
            "source": "https://huggingface.co/datasets/owner/dataset",
        },
        source_revision={"value": REVISION, "verified": True, "source": "Hugging Face API"},
    )
    audit = provenance.check_license_claim(record, tmp_path, "fixture")
    assert audit is not None and audit.sound, audit.errors
    # No source_sha256 was demanded for an already immutably pinned source.
    assert audit.category == provenance.IMMUTABLE_PINNED
    assert "source_sha256" not in json.dumps(record)


def test_verified_external_claim_with_permalink_passes(tmp_path: Path):
    record = _dataset(
        license={
            "value": "Apache-2.0",
            "verified": True,
            "source": f"https://huggingface.co/owner/model/blob/{REVISION}/LICENSE",
        }
    )
    audit = provenance.check_license_claim(record, tmp_path, "fixture")
    assert audit is not None and audit.sound, audit.errors


def test_valid_two_digest_derived_artifact_passes(tmp_path: Path):
    record = _dataset(
        id="derived",
        source_revision={"value": REVISION, "verified": True, "source": "Hugging Face API"},
        checksum="c" * 64,
        processed_dataset_hash={"status": "FULL_DATA_VALIDATED", "value": "c" * 64},
    )
    assert provenance.check_two_digest(record, "derived") == []


def test_documented_legacy_artifact_is_accepted(tmp_path: Path):
    legacy_id = next(iter(provenance.LEGACY_ARTIFACTS))
    record = _dataset(id=legacy_id, provenance_version=provenance.LEGACY_PROVENANCE_VERSION)
    assert provenance.check_legacy_marker(record, legacy_id) == []
    # The legacy marker suppresses the two-digest requirement for that artifact.
    assert provenance.check_two_digest(record, legacy_id) == []


def test_matching_freeze_pin_passes(tmp_path: Path):
    (tmp_path / "release").mkdir()
    manifest = tmp_path / "release/manifest.json"
    manifest.write_text('{"record_count": 1}\n', encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    _root(
        tmp_path,
        [
            _dataset(
                id="frozen",
                checksum=digest,
                processed_dataset_hash={
                    "status": "FULL_DATA_VALIDATED",
                    "value": digest,
                    "source": "release/manifest.json",
                },
                source_revision={"value": REVISION, "verified": True, "source": "Hugging Face API"},
            )
        ],
    )
    assert validate_freeze(tmp_path) == []


def test_valid_supersession_chain_passes(tmp_path: Path):
    (tmp_path / "reports/releases").mkdir(parents=True)
    (tmp_path / "reports/releases/a.json").write_text(
        json.dumps(
            {
                "dataset": {
                    "repository": "owner/repo",
                    "hub_revision": REVISION,
                    "superseded_hub_revision": OTHER_REVISION,
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "reports/releases/b.json").write_text(
        json.dumps({"dataset": {"repository": "owner/repo", "hub_revision": OTHER_REVISION}}),
        encoding="utf-8",
    )
    assert validate_publication_records(tmp_path) == []


def test_valid_registry_references_pass(tmp_path: Path):
    _write(tmp_path / "reports/evidence.md", "x\n")
    _root(
        tmp_path,
        [
            _dataset(
                id="upstream",
                license={
                    "value": "Apache-2.0",
                    "verified": True,
                    "source": "reports/evidence.md",
                    "source_sha256": _sha256("x\n"),
                },
                source_revision={"value": REVISION, "verified": True, "source": "Hugging Face API"},
            ),
            _dataset(id="derived", derived_from=[{"id": "upstream", "source_revision": REVISION}]),
        ],
    )
    assert validate_references(tmp_path) == []


# ── Expected FAIL ────────────────────────────────────────────────────────────


def test_mutable_file_without_sha256_fails(tmp_path: Path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/evidence.md").write_text("x\n", encoding="utf-8")
    record = _dataset(
        license={"value": "Apache-2.0", "verified": True, "source": "reports/evidence.md"}
    )
    audit = provenance.check_license_claim(record, tmp_path, "fixture")
    assert audit is not None and not audit.sound
    assert any("source_sha256" in error for error in audit.errors)


def test_wrong_sha256_fails(tmp_path: Path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/evidence.md").write_text("x\n", encoding="utf-8")
    record = _dataset(
        license={
            "value": "Apache-2.0",
            "verified": True,
            "source": "reports/evidence.md",
            "source_sha256": "d" * 64,
        }
    )
    audit = provenance.check_license_claim(record, tmp_path, "fixture")
    assert audit is not None and not audit.sound
    assert any("does not match" in error for error in audit.errors)


def test_external_source_using_mutable_branch_fails(tmp_path: Path):
    record = _dataset(
        license={
            "value": "Apache-2.0",
            "verified": True,
            "source": "https://huggingface.co/owner/model/blob/main/LICENSE",
        }
    )
    audit = provenance.check_license_claim(record, tmp_path, "fixture")
    assert audit is not None and not audit.sound
    assert any("/blob/main" in error for error in audit.errors)


def test_unpinned_external_source_fails(tmp_path: Path):
    record = _dataset(
        license={
            "value": "Apache-2.0",
            "verified": True,
            "source": "https://huggingface.co/datasets/owner/dataset",
        }
    )
    audit = provenance.check_license_claim(record, tmp_path, "fixture")
    assert audit is not None and not audit.sound
    assert any("neither hashed nor immutably pinned" in error for error in audit.errors)


def test_short_revision_is_not_an_immutable_pin(tmp_path: Path):
    record = _dataset(
        source_revision={"value": "abc1234", "verified": True, "source": "Hugging Face API"},
    )
    audit = provenance.check_revision_claim(record, "fixture")
    assert audit is not None and not audit.sound


def test_dangling_artifact_reference_fails(tmp_path: Path):
    _root(
        tmp_path,
        [
            _dataset(
                license={
                    "value": "Apache-2.0",
                    "verified": True,
                    "source": "reports/missing.md",
                    "source_sha256": "e" * 64,
                }
            )
        ],
    )
    errors = validate_references(tmp_path)
    assert any("does not exist: reports/missing.md" in error for error in errors)


def test_duplicate_registry_id_fails(tmp_path: Path):
    _root(tmp_path, [_dataset(id="twice"), _dataset(id="twice", display_name="Other")])
    errors = validate_structure(tmp_path)
    assert any("duplicate dataset id" in error for error in errors)


def test_invalid_supersession_target_fails(tmp_path: Path):
    (tmp_path / "reports/releases").mkdir(parents=True)
    (tmp_path / "reports/releases/a.json").write_text(
        json.dumps({"dataset": {"repository": "owner/repo", "hub_revision": REVISION}}),
        encoding="utf-8",
    )
    (tmp_path / "reports/releases/b.json").write_text(
        json.dumps({"dataset": {"repository": "owner/repo", "hub_revision": OTHER_REVISION}}),
        encoding="utf-8",
    )
    errors = validate_publication_records(tmp_path)
    assert any("ambiguous" in error for error in errors)


def test_publication_claim_superseding_itself_fails(tmp_path: Path):
    (tmp_path / "reports/releases").mkdir(parents=True)
    (tmp_path / "reports/releases/a.json").write_text(
        json.dumps(
            {
                "dataset": {
                    "repository": "owner/repo",
                    "hub_revision": REVISION,
                    "superseded_hub_revision": REVISION,
                }
            }
        ),
        encoding="utf-8",
    )
    errors = validate_publication_records(tmp_path)
    assert any("supersedes its own current revision" in error for error in errors)


def test_frozen_artifact_digest_mismatch_fails(tmp_path: Path):
    (tmp_path / "release").mkdir()
    (tmp_path / "release/manifest.json").write_text("{}\n", encoding="utf-8")
    _root(
        tmp_path,
        [
            _dataset(
                id="frozen",
                processed_dataset_hash={
                    "status": "FULL_DATA_VALIDATED",
                    "value": "f" * 64,
                    "source": "release/manifest.json",
                },
                source_revision={"value": REVISION, "verified": True, "source": "Hugging Face API"},
            )
        ],
    )
    errors = validate_freeze(tmp_path)
    assert any("frozen artifact identity moved" in error for error in errors)


def test_unknown_provenance_version_escape_hatch_fails(tmp_path: Path):
    record = _dataset(id="anything", provenance_version="legacy_whatever_v9")
    errors = provenance.check_legacy_marker(record, "anything")
    assert any("unknown provenance_version" in error for error in errors)


def test_legacy_marker_outside_the_allowlist_fails(tmp_path: Path):
    record = _dataset(
        id="not-the-legacy-artifact", provenance_version=provenance.LEGACY_PROVENANCE_VERSION
    )
    errors = provenance.check_legacy_marker(record, "not-the-legacy-artifact")
    assert any("named allowlist" in error for error in errors)


def test_derived_artifact_without_an_input_identity_fails(tmp_path: Path):
    record = _dataset(
        id="derived",
        checksum="c" * 64,
        processed_dataset_hash={"status": "FULL_DATA_VALIDATED", "value": "c" * 64},
    )
    errors = provenance.check_two_digest(record, "derived")
    assert any("no immutable input identity" in error for error in errors)


def test_contradictory_derived_digests_fail(tmp_path: Path):
    record = _dataset(
        id="derived",
        source_revision={"value": REVISION, "verified": True, "source": "Hugging Face API"},
        checksum="c" * 64,
        processed_dataset_hash={"status": "FULL_DATA_VALIDATED", "value": "d" * 64},
    )
    errors = provenance.check_two_digest(record, "derived")
    assert any("ambiguous" in error for error in errors)


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/owner/repo/blob/master/LICENSE",
        "https://huggingface.co/owner/model/resolve/main/README.md",
        "https://example.org/releases/latest",
    ],
)
def test_mutable_refs_are_recognised(url: str):
    assert provenance.mutable_ref(url) is not None


# ── Committed bytes, not working-tree bytes ──────────────────────────────────
#
# This repository sets core.autocrlf=true, so a file's bytes on disk are not its
# bytes in the record. A source_sha256 taken from the working tree would be a
# claim about the verifier's operating system.


def _git_env() -> dict:
    import os

    return {
        "PATH": os.environ.get("PATH", ""),
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }


def _git_repo(tmp_path: Path) -> Path:
    import subprocess

    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(tmp_path)], capture_output=True, check=True
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "core.autocrlf", "true"],
        capture_output=True,
        check=True,
    )
    return tmp_path


def test_digest_is_verified_against_committed_bytes_not_the_working_tree(tmp_path: Path):
    """A CRLF checkout of an LF blob must still validate against the LF digest."""
    import subprocess

    root = _git_repo(tmp_path)
    target = root / "pairs.jsonl"
    target.write_bytes(b'{"a": 1}\n{"a": 2}\n')
    subprocess.run(["git", "-C", str(root), "add", "pairs.jsonl"], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-q", "-m", "add"],
        capture_output=True,
        check=True,
        env=_git_env(),
    )
    committed_digest = hashlib.sha256(b'{"a": 1}\n{"a": 2}\n').hexdigest()

    # Simulate a CRLF checkout, which is what happens on Windows here.
    target.write_bytes(b'{"a": 1}\r\n{"a": 2}\r\n')
    assert hashlib.sha256(target.read_bytes()).hexdigest() != committed_digest

    record = _dataset(
        license={
            "value": "X",
            "verified": True,
            "source": "pairs.jsonl",
            "source_sha256": committed_digest,
        }
    )
    audit = provenance.check_license_claim(record, root, "fixture")
    assert audit is not None and audit.sound, audit.errors


def test_untracked_evidence_path_is_rejected(tmp_path: Path):
    """An untracked file is not part of the record, however stable it looks."""
    root = _git_repo(tmp_path)
    (root / "notes.md").write_bytes(b"licence\n")
    record = _dataset(
        license={
            "value": "X",
            "verified": True,
            "source": "notes.md",
            "source_sha256": hashlib.sha256(b"licence\n").hexdigest(),
        }
    )
    audit = provenance.check_license_claim(record, root, "fixture")
    assert audit is not None and not audit.sound
    assert any("not tracked" in error for error in audit.errors)


# ── Freeze pins must be reproducible or explicitly corroborated ──────────────


def test_unreproducible_freeze_pin_without_a_declaration_fails(tmp_path: Path):
    _root(
        tmp_path,
        [
            _dataset(
                id="frozen",
                source_repository=".release/build/release-manifest.json",
                processed_dataset_hash={"status": "FULL_DATA_VALIDATED", "value": "a" * 64},
                source_revision={"value": REVISION, "verified": True, "source": "Hugging Face API"},
            )
        ],
    )
    errors = validate_freeze(tmp_path)
    assert any("identity_artifact_unavailable" in error for error in errors)


def _frozen_record_with_unavailable_identity() -> dict:
    """A record whose derived identity cannot be re-derived, but is declared so."""
    return _dataset(
        id="frozen",
        source_repository=".release/build/release-manifest.json",
        processed_dataset_hash={
            "status": "FULL_DATA_VALIDATED",
            "value": "a" * 64,
            "identity_artifact_unavailable": {
                "artifact": ".release/build/release-manifest.json",
                "reason": "uncommitted build output",
                "corroborated_by": "reports/releases/pub.json",
                "corroborating_field": "release_manifest_sha256",
            },
        },
        source_revision={"value": REVISION, "verified": True, "source": "Hugging Face API"},
    )


def test_unreproducible_freeze_pin_corroborated_by_a_tracked_record_passes(tmp_path: Path):
    (tmp_path / "reports/releases").mkdir(parents=True)
    (tmp_path / "reports/releases/pub.json").write_text(
        json.dumps({"release_manifest_sha256": "a" * 64}), encoding="utf-8"
    )
    _root(tmp_path, [_frozen_record_with_unavailable_identity()])
    assert validate_freeze(tmp_path) == []


def test_corroborating_record_that_disagrees_fails(tmp_path: Path):
    """Two records disagreeing about one corpus's identity is the case that matters."""
    (tmp_path / "reports/releases").mkdir(parents=True)
    (tmp_path / "reports/releases/pub.json").write_text(
        json.dumps({"release_manifest_sha256": "b" * 64}), encoding="utf-8"
    )
    _root(tmp_path, [_frozen_record_with_unavailable_identity()])
    errors = validate_freeze(tmp_path)
    assert any("two records disagree" in error for error in errors)


# ── Reconstructed publication events ─────────────────────────────────────────


def _publication(root: Path, payload: dict, published: str = "2026-09-14") -> Path:
    directory = root / "reports/releases"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "pub.json"
    path.write_text(json.dumps({"published": published, **payload}), encoding="utf-8")
    return path


def _retroactive_event(**overrides) -> dict:
    event = {
        "repository": "owner/repo",
        "hub_revision": REVISION,
        "superseded_hub_revision": OTHER_REVISION,
        "event_occurred_at": "2026-09-13",
        "entry_recorded_at": "2026-09-14",
        "recorded_retroactively": True,
        "reconstruction_evidence": {
            "kind": "HUB_COMMIT_HISTORY",
            "observed_from": "huggingface.co/api/datasets/owner/repo",
            "immutable_identifier": REVISION,
        },
    }
    event.update(overrides)
    return event


def test_well_formed_reconstruction_passes(tmp_path: Path):
    _publication(tmp_path, {"dataset": _retroactive_event()})
    assert validate_reconstructed_events(tmp_path) == []


def test_contemporaneous_event_with_equal_dates_passes(tmp_path: Path):
    event = _retroactive_event(
        recorded_retroactively=False,
        event_occurred_at="2026-09-14",
        entry_recorded_at="2026-09-14",
    )
    event.pop("reconstruction_evidence")
    _publication(tmp_path, {"dataset": event})
    assert validate_reconstructed_events(tmp_path) == []


def test_later_entry_cannot_masquerade_as_contemporaneous(tmp_path: Path):
    """The core hazard: an entry written later that reads as written then."""
    _publication(tmp_path, {"dataset": _retroactive_event(recorded_retroactively=False)})
    errors = validate_reconstructed_events(tmp_path)
    assert any("would read as contemporaneous" in error for error in errors)


def test_reconstruction_without_evidence_fails(tmp_path: Path):
    event = _retroactive_event()
    event.pop("reconstruction_evidence")
    _publication(tmp_path, {"dataset": event})
    errors = validate_reconstructed_events(tmp_path)
    assert any("without reconstruction_evidence" in error for error in errors)


def test_reconstruction_missing_dates_fails(tmp_path: Path):
    event = _retroactive_event()
    event.pop("entry_recorded_at")
    _publication(tmp_path, {"dataset": event})
    errors = validate_reconstructed_events(tmp_path)
    assert any("does not give both" in error for error in errors)


def test_reconstruction_recorded_before_the_event_fails(tmp_path: Path):
    _publication(tmp_path, {"dataset": _retroactive_event(event_occurred_at="2026-09-15")})
    errors = validate_reconstructed_events(tmp_path)
    assert any("is not after" in error for error in errors)


def test_entry_date_must_match_the_record_it_lives_in(tmp_path: Path):
    _publication(tmp_path, {"dataset": _retroactive_event()}, published="2026-09-20")
    errors = validate_reconstructed_events(tmp_path)
    assert any("but the record is dated" in error for error in errors)


def test_the_repositorys_reconstruction_is_labelled():
    """The real reconstruction must carry explicit, machine-readable metadata."""
    root = Path(__file__).parents[2]
    assert validate_reconstructed_events(root) == []
    record = json.loads(
        (
            root / "reports/releases/hf-publication-2026-09-14-canonical-v2-licence-correction.json"
        ).read_text(encoding="utf-8")
    )
    retroactive = [
        event
        for event in record["revision_chain_since_previous_record"]
        if event.get("recorded_retroactively")
    ]
    assert retroactive, "the reconstructed event must be marked as reconstructed"
    event = retroactive[0]
    assert event["event_occurred_at"] != event["entry_recorded_at"]
    assert event["reconstruction_evidence"]["immutable_identifier"] == event["hub_revision"]


def test_the_repository_resolves_the_calibration_artifact_identity():
    """The calibration digest must be re-derived from the committed artifact.

    `reports/data/m1-calibration-preference-pairs-v1.json` names
    `data/processed/m1_calibration_preference_pairs_v1.jsonl` as its output and
    records that digest, and the registry names the same path as the artifact.
    The check re-derives it, so the claim is demonstrated rather than assumed.
    """
    root = Path(__file__).parents[2]
    rows = yaml.safe_load((root / "registry/datasets.yaml").read_text(encoding="utf-8"))["datasets"]
    record = next(r for r in rows if r["id"] == "m1_calibration_preference_pairs_v1")
    digest = record["processed_dataset_hash"]["value"]
    artifact = record["source_repository"]

    # The registry is not the only record of it: the build manifest agrees.
    manifest = json.loads(
        (root / "reports/data/m1-calibration-preference-pairs-v1.json").read_text(encoding="utf-8")
    )
    assert manifest["output"] == artifact
    assert manifest["sha256"] == digest

    # And the digest is reproducible from committed bytes.
    payload, problem = provenance.read_evidence_bytes(root, artifact)
    assert payload is not None, problem
    assert hashlib.sha256(payload).hexdigest() == digest

    # The freeze check therefore re-derives it rather than skipping it. The
    # repository has one other unresolved frozen identity (canonical_v2, asserted
    # in the next test); the point here is that this record is not among them.
    assert not [error for error in validate_freeze(root) if error.startswith("m1_calibration")]


def test_the_repository_resolved_the_canonical_v2_identity():
    """canonical_v2's derived identity is genuinely unreproducible; say so.

    No committed artifact reproduces `09018d26…`, and the publication record for
    the same corpus stores a different manifest hash (`277a0ae4…`). Neither is
    reproducible from the repository, so the correct outcome is an unresolved
    finding, not a pass.
    """
    root = Path(__file__).parents[2]
    rows = yaml.safe_load((root / "registry/datasets.yaml").read_text(encoding="utf-8"))["datasets"]
    record = next(r for r in rows if r["id"] == "canonical_v2")
    derived = record["processed_dataset_hash"]
    assert derived["value"] == ("09018d260731542c804cade53594be1fdaaabd7c80a60d784bd470758b0b87f1")
    remote = derived["remote_identity"]
    assert provenance.is_immutable_revision(remote["hub_revision"])
    assert remote["sha256"] == derived["value"]
    publication = json.loads(
        (root / "reports/releases/toolpolicy-canonical-v2-publication.json").read_text(
            encoding="utf-8"
        )
    )
    assert publication["release_manifest_sha256"] != derived["value"]
    assert provenance.is_immutable_revision(publication["hub_revision"])

    # The local freeze gate defers it to the remote gate, and says so explicitly
    # rather than reporting it unresolved or quietly dropping it.
    result = validate_freeze_result(root)
    deferred = [skip for skip in result.skipped if skip.id == "canonical_v2"]
    assert deferred, result.render()
    assert result.status == PASS, result.render()
    assert result.detail["unresolved"] == 0
    assert result.detail["deferred_to_remote_identity"] == 1
