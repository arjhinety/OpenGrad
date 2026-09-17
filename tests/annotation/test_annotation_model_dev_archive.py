"""The archived model-label audit trails of the classifier's development sets prove what they record.

`model-dev` on prose-classifier-dev-v1 (33 §3) and `model-devcheck` on the held-out check set (33 §5a)."""

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
#: task, session, archive directory, expected item ranges
TRAILS = {
    "dev": (
        "prose-classifier-dev-v1",
        "model-dev",
        ROOT / "reports" / "prose-classifier" / "dev" / "provenance" / "model-dev",
        [(1, 50), (51, 100), (101, 150), (151, 200), (201, 250)],
    ),
    "devcheck": (
        "prose-classifier-devcheck-v1",
        "model-devcheck",
        ROOT / "reports" / "prose-classifier" / "devcheck" / "provenance" / "model-devcheck",
        [(1, 50), (51, 100), (101, 125)],
    ),
    "devcheck-v2": (
        "prose-classifier-devcheck-v2",
        "model-devcheck-v2",
        ROOT / "reports" / "prose-classifier" / "devcheck-v2" / "provenance" / "model-devcheck-v2",
        [(1, 50), (51, 100), (101, 125)],
    ),
    "dev-v2": (
        "prose-classifier-dev-v2",
        "model-dev-v2",
        ROOT / "reports" / "prose-classifier" / "dev-v2" / "provenance" / "model-dev-v2",
        [(1, 50), (51, 100), (101, 150), (151, 200), (201, 250), (251, 300)],
    ),
    "v2-devcheck-1": (
        "prose-classifier-v2-devcheck-1",
        "model-v2-devcheck-1",
        ROOT / "reports" / "prose-classifier" / "v2-devcheck-1" / "provenance" / "model-v2-devcheck-1",
        [(1, 50), (51, 100), (101, 150)],
    ),
    "v2-devcheck-2": (
        "prose-classifier-v2-devcheck-2",
        "model-v2-devcheck-2",
        ROOT / "reports" / "prose-classifier" / "v2-devcheck-2" / "provenance" / "model-v2-devcheck-2",
        [(1, 50), (51, 100), (101, 150)],
    ),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(scope="module", params=sorted(TRAILS))
def trail(request: pytest.FixtureRequest) -> tuple:
    task, session, directory, ranges = TRAILS[request.param]
    stem = f"{task}.{session}.audit-trail"
    if not (directory / f"{stem}.manifest.json").is_file():
        pytest.skip(f"{session} audit trail not present")
    return task, session, directory, ranges, stem


@pytest.fixture(scope="module")
def manifest(trail: tuple) -> dict:
    _task, _session, directory, _ranges, stem = trail
    data = (directory / f"{stem}.manifest.json").read_bytes()
    assert (directory / f"{stem}.manifest.json.sha256").read_text(encoding="utf-8").split()[0] == sha256(data)
    return json.loads(data)


@pytest.fixture(scope="module")
def members(trail: tuple, manifest: dict) -> dict[str, bytes]:
    _task, _session, directory, _ranges, stem = trail
    archive = (directory / manifest["archive"]["file"]).read_bytes()
    assert sha256(archive) == manifest["archive"]["sha256"]
    assert (directory / f"{stem}.tar.gz.sha256").read_text(encoding="utf-8").split()[0] == sha256(archive)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        return {info.name: tar.extractfile(info).read() for info in tar.getmembers() if info.isfile()}  # type: ignore[union-attr]


def test_every_member_is_the_one_the_manifest_names(manifest: dict, members: dict[str, bytes]) -> None:
    listed = {entry["path"]: entry for entry in manifest["archive"]["members"]}
    assert set(listed) == set(members)
    for name, data in members.items():
        assert sha256(data) == listed[name]["sha256"] and len(data) == listed[name]["bytes"], name
    assert manifest["annotator_id"] == "model.claude-opus-5" and manifest["model"] == "claude-opus-5"
    assert "not human gold" in manifest["status"]


def test_the_parts_cover_the_batch_once_and_were_audited(trail: tuple, manifest: dict, members: dict[str, bytes]) -> None:
    task, session, _directory, expected_ranges, _stem = trail
    config = load_task_config(ROOT / "configs" / "annotation" / f"{task}.yaml")
    (declared,) = config.model_annotators
    assert sha256(members[f"procedure/{Path(manifest['procedure']['path']).name}"]) == declared.procedure_sha256
    batch = json.loads(members[f"{session}/batch-01.json"])
    content = sha256(canonical_json({"items": batch["items"], "instructions": batch["instructions"]}).encode("utf-8"))
    assert content == batch["content_sha256"] == manifest["batch"]["content_sha256"]
    assert batch["procedure_sha256"] == declared.procedure_sha256

    merged: list[dict] = []
    ranges = []
    for row in manifest["parts"]:
        answers = json.loads(members[f"{session}/batch-01.part-{row['part']}.answers.json"])
        assert [a["item_id"] for a in answers] == batch["item_ids"][row["first_number"] - 1 : row["last_number"]]
        assert row["subagent_models"] == ["claude-opus-5"]
        assert row["prompt_equals_procedure_plus_batch_line"] and row["prompt_range_matches"]
        assert row["final_message_equals_part_answers"] and row["item_ids_match_range_in_order"]
        assert row["tool_calls"] and all(
            c["name"] == "Read" and c["file_path"].endswith("batch-01.md") for c in row["tool_calls"]
        )
        ranges.append((row["first_number"], row["last_number"]))
        merged.extend(answers)
    assert ranges == expected_ranges
    assert json.loads(members[f"{session}/batch-01.answers.json"]) == merged
    assert [a["item_id"] for a in merged] == batch["item_ids"] and len(merged) == expected_ranges[-1][1]


def test_the_ingest_log_is_an_intact_chain_naming_the_batch(trail: tuple, manifest: dict, members: dict[str, bytes]) -> None:
    _task, session, _directory, ranges, _stem = trail
    entries = [json.loads(line) for line in members[f"ingest/{session}.history.jsonl"].decode("utf-8").splitlines()]
    assert verify_chain(entries, ANNOTATION_ENTRY_FIELDS, session) == []
    assert len(entries) == manifest["ingest"]["history_entries"] == ranges[-1][1]
    assert entries[-1]["entry_sha256"] == manifest["ingest"]["history_head_sha256"]
    for entry in entries:
        assert entry["actor_id"] == "model.claude-opus-5" and entry["action"] == "label"
        assert manifest["batch"]["content_sha256"] in entry["reason"]
        assert manifest["procedure"]["sha256"] in entry["reason"]


def test_the_untracked_transcripts_match_their_recorded_hashes_when_present(trail: tuple, manifest: dict) -> None:
    local = trail[2] / manifest["transcripts"]["file"]
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
