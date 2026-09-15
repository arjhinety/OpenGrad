"""The archived `model-a` audit trail (reports/pdet/provenance/model-a) still proves what it records."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from opengrad.annotation.config import load_task_config
from opengrad.annotation.items import canonical_json
from opengrad.annotation.provenance import ANNOTATION_ENTRY_FIELDS, verify_chain

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "reports" / "pdet" / "provenance" / "model-a"
STEM = "pdet-v1.model-a.audit-trail"
MANIFEST = DIRECTORY / f"{STEM}.manifest.json"

pytestmark = pytest.mark.skipif(not MANIFEST.is_file(), reason="model-a audit trail not present")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(scope="module")
def manifest() -> dict:
    data = MANIFEST.read_bytes()
    assert (DIRECTORY / f"{STEM}.manifest.json.sha256").read_text(encoding="utf-8").split()[0] == sha256(data)
    return json.loads(data)


@pytest.fixture(scope="module")
def members(manifest: dict) -> dict[str, bytes]:
    archive = (DIRECTORY / manifest["archive"]["file"]).read_bytes()
    assert sha256(archive) == manifest["archive"]["sha256"]
    assert (DIRECTORY / f"{STEM}.tar.gz.sha256").read_text(encoding="utf-8").split()[0] == sha256(archive)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        return {info.name: tar.extractfile(info).read() for info in tar.getmembers() if info.isfile()}  # type: ignore[union-attr]


def test_every_member_is_the_one_the_manifest_names(manifest: dict, members: dict[str, bytes]) -> None:
    listed = {entry["path"]: entry for entry in manifest["archive"]["members"]}
    assert set(listed) == set(members)
    for name, data in members.items():
        assert sha256(data) == listed[name]["sha256"] and len(data) == listed[name]["bytes"], name
    assert manifest["annotator_id"] == "model.claude-opus-5" and manifest["model"] == "claude-opus-5"
    assert "not human gold" in manifest["status"]


def test_batches_are_complete_unchanged_and_audited(manifest: dict, members: dict[str, bytes]) -> None:
    config = load_task_config(ROOT / "configs" / "annotation" / "pdet-v1.yaml")
    (declared,) = config.model_annotators
    procedure = members["procedure/pdet-v1.model-procedure.md"]
    assert sha256(procedure) == declared.procedure_sha256 == manifest["procedure"]["sha256"]
    rows = manifest["batches"]
    assert [row["batch_id"] for row in rows] == [f"{n:02d}" for n in range(1, len(rows) + 1)] and len(rows) == 12
    seen: list[str] = []
    for row in rows:
        stem = f"model-a/batch-{row['batch_id']}"
        batch = json.loads(members[f"{stem}.json"])
        audit = json.loads(members[f"{stem}.audit.json"])
        raw = members[f"{stem}.answers.json"]
        answers = json.loads(raw)
        content = sha256(canonical_json({"items": batch["items"], "instructions": batch["instructions"]}).encode("utf-8"))
        assert content == batch["content_sha256"] == row["content_sha256"]
        assert batch["procedure_sha256"] == declared.procedure_sha256 == audit["procedure_sha256"]
        assert batch["annotator_id"] == "model.claude-opus-5" and audit["models"] == ["claude-opus-5"]
        assert audit["prompt_equals_procedure_plus_batch_line"] is True
        assert all(call["name"] == "Read" and call["file_path"].endswith(f"batch-{row['batch_id']}.md") for call in audit["tool_calls"])
        # The audit hashed the answer text before Windows text-mode newline translation.
        assert sha256(raw.replace(b"\r\n", b"\n")) == audit["answers_sha256"] == row["answers_lf_sha256"]
        assert sorted(answer["item_id"] for answer in answers) == sorted(batch["item_ids"])
        assert audit["transcript_sha256"] == row["transcript_sha256"] and row["transcript_matches_audit"]
        seen.extend(batch["item_ids"])
    assert len(seen) == len(set(seen)) == 570


def test_the_ingest_log_is_an_intact_chain_naming_each_batch(manifest: dict, members: dict[str, bytes]) -> None:
    entries = [json.loads(line) for line in members["ingest/model-a.history.jsonl"].decode("utf-8").splitlines()]
    assert verify_chain(entries, ANNOTATION_ENTRY_FIELDS, "model-a") == []
    assert len(entries) == manifest["ingest"]["history_entries"] == 570
    assert entries[-1]["entry_sha256"] == manifest["ingest"]["history_head_sha256"]
    by_batch = {row["batch_id"]: row for row in manifest["batches"]}
    for entry in entries:
        assert entry["actor_id"] == "model.claude-opus-5" and entry["action"] == "label"
        batch_id = entry["reason"].split(";")[0].removeprefix("model batch ")
        assert by_batch[batch_id]["content_sha256"] in entry["reason"]
        assert manifest["procedure"]["sha256"] in entry["reason"]
    assert manifest["ingest"]["entries_per_batch"] == {row["batch_id"]: row["answers"] for row in manifest["batches"]}


def test_the_untracked_transcripts_match_their_recorded_hashes_when_present(manifest: dict) -> None:
    local = DIRECTORY / manifest["transcripts"]["file"]
    assert manifest["transcripts"]["tracked"] is False
    if not local.is_file():
        pytest.skip("the verbatim transcripts are kept locally only")
    data = local.read_bytes()
    assert sha256(data) == manifest["transcripts"]["sha256"]
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        found = {info.name: sha256(tar.extractfile(info).read()) for info in tar.getmembers() if info.isfile()}  # type: ignore[union-attr]
    assert found == {entry["path"]: entry["sha256"] for entry in manifest["transcripts"]["members"]}
    for row in manifest["batches"]:
        assert found[f"transcripts/{row['transcript_file']}"] == row["transcript_sha256"]
