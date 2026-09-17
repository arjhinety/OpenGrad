"""The archived external model audit trails (reports/pdet-coverage/provenance/external-models) prove what they record.

Counts and hashes only: no item text, label or rationale is inspected.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from opengrad.annotation.provenance import ANNOTATION_ENTRY_FIELDS, verify_chain

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "reports" / "pdet-coverage" / "provenance" / "external-models"
TASKS = {"pdet-coverage-v1": 306, "pdet-coverage-v1-routing": 30}
ANNOTATORS = {
    "model-gemini": "model.gemini-3.8-flash-high",
    "model-gpt": "model.gpt-5.6-sol",
    "model-deepseek": "model.deepseek-v4.1-flash",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(scope="module", params=sorted(TASKS))
def trail(request: pytest.FixtureRequest) -> tuple[str, dict, dict[str, bytes]]:
    task = request.param
    stem = f"{task}.external-models.audit-trail"
    manifest_path = DIRECTORY / f"{stem}.manifest.json"
    if not manifest_path.is_file():
        pytest.skip(f"{task} external audit trail not present")
    data = manifest_path.read_bytes()
    assert (DIRECTORY / f"{stem}.manifest.json.sha256").read_text(encoding="utf-8").split()[0] == sha256(data)
    manifest = json.loads(data)
    archive = (DIRECTORY / manifest["archive"]["file"]).read_bytes()
    assert sha256(archive) == manifest["archive"]["sha256"]
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        members = {info.name: tar.extractfile(info).read() for info in tar.getmembers() if info.isfile()}  # type: ignore[union-attr]
    return task, manifest, members


def test_every_member_is_listed_and_no_raw_cli_stream_is_tracked(trail: tuple[str, dict, dict[str, bytes]]) -> None:
    _task, manifest, members = trail
    listed = {entry["path"]: entry["sha256"] for entry in manifest["archive"]["members"]}
    assert listed == {name: sha256(data) for name, data in members.items()}
    assert not any(name.endswith((".stdout.txt", ".stderr.txt")) for name in members)
    assert manifest["cli_streams"]["tracked"] is False


def test_each_annotator_labelled_every_item_through_an_intact_ingest_log(trail: tuple[str, dict, dict[str, bytes]]) -> None:
    task, manifest, members = trail
    expected = TASKS[task]
    assert {s["session_id"]: s["annotator_id"] for s in manifest["sessions"]} == ANNOTATORS
    for session in manifest["sessions"]:
        entries = [
            json.loads(line)
            for line in members[f"ingest/{session['session_id']}.history.jsonl"].decode("utf-8").splitlines()
        ]
        assert verify_chain(entries, ANNOTATION_ENTRY_FIELDS, session["session_id"]) == []
        assert len({entry["item_id"] for entry in entries}) == expected == session["labels_recorded"]
        assert session["files_created_in_isolated_dirs"] == 0
        assert all(entry["actor_id"] == session["annotator_id"] for entry in entries)
        recorded = [b for b in session["batches"] if b["outcome"] == "recorded"]
        assert sum(b["labels_recorded"] for b in recorded) == expected


def test_the_procedure_is_the_pinned_one(trail: tuple[str, dict, dict[str, bytes]]) -> None:
    _task, manifest, members = trail
    procedure = members["procedure/pdet-coverage-v1.model-procedure.md"]
    assert sha256(procedure) == manifest["procedure"]["sha256"] == "4d36a3f8d97ebd5b62987ee2c01118ac1fafb4b0625673f867288f986d84e7ab"
    for session in manifest["sessions"]:
        assert {b["procedure_sha256"] for b in session["batches"]} == {manifest["procedure"]["sha256"]}


def test_the_untracked_streams_match_their_recorded_hashes_when_present(trail: tuple[str, dict, dict[str, bytes]]) -> None:
    _task, manifest, _members = trail
    local = DIRECTORY / manifest["cli_streams"]["file"]
    if not local.is_file():
        pytest.skip("raw CLI streams are kept locally only")
    data = local.read_bytes()
    assert sha256(data) == manifest["cli_streams"]["sha256"]
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        found = {info.name: sha256(tar.extractfile(info).read()) for info in tar.getmembers() if info.isfile()}  # type: ignore[union-attr]
    assert found == {entry["path"]: entry["sha256"] for entry in manifest["cli_streams"]["members"]}
