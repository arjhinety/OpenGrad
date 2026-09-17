"""The archived `model-dev` audit trail (reports/prose-classifier/dev/provenance/model-dev) proves what it records."""

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
DIRECTORY = ROOT / "reports" / "prose-classifier" / "dev" / "provenance" / "model-dev"
STEM = "prose-classifier-dev-v1.model-dev.audit-trail"
MANIFEST = DIRECTORY / f"{STEM}.manifest.json"

pytestmark = pytest.mark.skipif(not MANIFEST.is_file(), reason="model-dev audit trail not present")


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


def test_the_five_parts_cover_the_batch_once_and_were_audited(manifest: dict, members: dict[str, bytes]) -> None:
    config = load_task_config(ROOT / "configs" / "annotation" / "prose-classifier-dev-v1.yaml")
    (declared,) = config.model_annotators
    assert sha256(members["procedure/prose-classifier-dev-v1.model-procedure.md"]) == declared.procedure_sha256
    batch = json.loads(members["model-dev/batch-01.json"])
    content = sha256(canonical_json({"items": batch["items"], "instructions": batch["instructions"]}).encode("utf-8"))
    assert content == batch["content_sha256"] == manifest["batch"]["content_sha256"]
    assert batch["procedure_sha256"] == declared.procedure_sha256

    merged: list[dict] = []
    ranges = []
    for row in manifest["parts"]:
        answers = json.loads(members[f"model-dev/batch-01.part-{row['part']}.answers.json"])
        assert [a["item_id"] for a in answers] == batch["item_ids"][row["first_number"] - 1 : row["last_number"]]
        assert row["subagent_models"] == ["claude-opus-5"]
        assert row["prompt_equals_procedure_plus_batch_line"] and row["prompt_range_matches"]
        assert row["final_message_equals_part_answers"] and row["item_ids_match_range_in_order"]
        assert row["tool_calls"] and all(
            c["name"] == "Read" and c["file_path"].endswith("batch-01.md") for c in row["tool_calls"]
        )
        ranges.append((row["first_number"], row["last_number"]))
        merged.extend(answers)
    assert ranges == [(1, 50), (51, 100), (101, 150), (151, 200), (201, 250)]
    assert json.loads(members["model-dev/batch-01.answers.json"]) == merged
    assert [a["item_id"] for a in merged] == batch["item_ids"] and len(merged) == 250


def test_the_ingest_log_is_an_intact_chain_naming_the_batch(manifest: dict, members: dict[str, bytes]) -> None:
    entries = [json.loads(line) for line in members["ingest/model-dev.history.jsonl"].decode("utf-8").splitlines()]
    assert verify_chain(entries, ANNOTATION_ENTRY_FIELDS, "model-dev") == []
    assert len(entries) == manifest["ingest"]["history_entries"] == 250
    assert entries[-1]["entry_sha256"] == manifest["ingest"]["history_head_sha256"]
    for entry in entries:
        assert entry["actor_id"] == "model.claude-opus-5" and entry["action"] == "label"
        assert manifest["batch"]["content_sha256"] in entry["reason"]
        assert manifest["procedure"]["sha256"] in entry["reason"]


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
    for row in manifest["parts"]:
        assert found[f"transcripts/{row['transcript_file']}"] == row["transcript_sha256"]
