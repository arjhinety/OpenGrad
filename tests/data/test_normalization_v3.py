"""normalization-v3: the pre-classifier artifact built from the canonical-v3 source manifest.

Each test pins one property the Study 002 B-1 unblock depends on: the manifest agrees with the code,
Glaive calls are structured rather than left as text, no behaviour label survives, every version is the
authoritative one, translation cannot move a record's identity, and the build is deterministic.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from opengrad.data import versions
from opengrad.data.normalization_v3 import (
    NormalizationV3Error,
    RowRejected,
    SourceSpec,
    build,
    check_manifest_versions,
    load_source_manifest,
    normalize_row,
    raw_record_hash,
    source_specs,
    verify,
)

GLAIVE_SYSTEM = (
    "SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n"
    '{"name": "get_weather", "description": "Get the weather", "parameters": {"type": "object", '
    '"properties": {"city": {"type": "string"}}, "required": ["city"]}}'
)
GLAIVE_CALL_CHAT = (
    "USER: What is the weather in Paris?\n\n\n"
    'ASSISTANT: <functioncall> {"name": "get_weather", "arguments": \'{"city": "Paris"}\'} '
    "<|endoftext|>\n\n\n"
    'FUNCTION RESPONSE: {"temperature": 20}\n\n\n'
    "ASSISTANT: It is 20 degrees in Paris. <|endoftext|>\n\n\n"
)
GLAIVE_PLAIN_CHAT = (
    "USER: Tell me a joke.\n\n\nASSISTANT: Why did the chicken cross the road? <|endoftext|>\n"
)
GLAIVE_MALFORMED_CHAT = 'USER: Weather?\n\n\nASSISTANT: <functioncall> {"name": "get_weather", "arguments": {"city": }\n'
WHEN2CALL_TOOL = json.dumps(
    {
        "name": "lookup",
        "description": "Look something up",
        "parameters": {
            "type": "dict",
            "properties": {"term": {"type": "str, optional", "description": "t"}},
        },
    }
)


def glaive_spec(raw_path: Path = Path("unused"), sha: str = "0" * 64, rows: int = 0) -> SourceSpec:
    return SourceSpec(
        name="glaive",
        dataset_id="glaive-function-calling-v2",
        upstream_revision="e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac",
        split="train",
        raw_path=raw_path,
        raw_sha256=sha,
        raw_rows=rows,
        adapter_key="glaive_v2",
        adapter_function="adapt_glaive_v2",
        row_label="glaive_function_calling_v2_v2",
        schema_translation=False,
    )


def when2call_spec(
    raw_path: Path = Path("unused"), sha: str = "0" * 64, rows: int = 0
) -> SourceSpec:
    return SourceSpec(
        name="when2call",
        dataset_id="when2call",
        upstream_revision="0582f7749df63a96fdc3070932e83e72396ace53",
        split="train",
        raw_path=raw_path,
        raw_sha256=sha,
        raw_rows=rows,
        adapter_key="when2call",
        adapter_function="adapt_when2call",
        row_label="when2call_v1",
        schema_translation=True,
    )


def when2call_raw(
    user: str = "Look up the term 'otter'.", reply: str = "Which source?"
) -> dict[str, Any]:
    return {
        "tools": [WHEN2CALL_TOOL],
        "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": reply}],
    }


# ── the committed source manifest ───────────────────────────────────────────────────────────────────


def test_committed_source_manifest_agrees_with_the_code() -> None:
    manifest = load_source_manifest()
    check_manifest_versions(manifest)
    specs = {spec.name: spec for spec in source_specs(manifest)}
    assert set(specs) == {"xlam", "glaive", "toolace", "when2call"}
    assert (specs["glaive"].adapter_key, specs["glaive"].adapter_function) == (
        "glaive_v2",
        "adapt_glaive_v2",
    )
    assert (specs["toolace"].adapter_key, specs["toolace"].adapter_function) == (
        "toolace_v2",
        "adapt_toolace_v2",
    )
    assert specs["when2call"].schema_translation and not specs["glaive"].schema_translation
    assert {entry["name"] for entry in manifest["excluded_sources"]} == {"looptool", "button"}


def test_manifest_with_a_stale_adapter_version_is_refused() -> None:
    manifest = load_source_manifest()
    manifest["versions"] = {**manifest["versions"], "adapter_version": "1.0.2"}
    with pytest.raises(versions.ProvenanceVersionMismatch):
        check_manifest_versions(manifest)


def test_manifest_naming_the_wrong_adapter_function_is_refused() -> None:
    manifest = load_source_manifest()
    glaive = next(entry for entry in manifest["sources"] if entry["name"] == "glaive")
    glaive["adapter"] = {**glaive["adapter"], "function": "adapt_glaive"}
    with pytest.raises(NormalizationV3Error, match="adapt_glaive_v2"):
        source_specs(manifest)


# ── rows ────────────────────────────────────────────────────────────────────────────────────────────


def test_glaive_functioncall_becomes_a_structured_call_and_leaves_no_text() -> None:
    item = normalize_row(glaive_spec(), {"system": GLAIVE_SYSTEM, "chat": GLAIVE_CALL_CHAT}, 0)
    calls = [call for message in item["messages"] for call in message.get("tool_calls") or []]
    assert calls == [{"id": "call_0000", "name": "get_weather", "arguments": {"city": "Paris"}}]
    contents = [str(message.get("content")) for message in item["messages"]]
    assert not any("<functioncall>" in text or "<|endoftext|>" in text for text in contents)
    assert item["messages"][-1]["content"] == "It is 20 degrees in Paris."
    structure = item["metadata"]["structure"]
    assert (
        structure["tool_result_before_final"] and not structure["final_assistant_structured_call"]
    )
    assert structure["trajectory_issue_codes"] == []


def test_glaive_malformed_call_is_rejected_not_kept_as_prose() -> None:
    with pytest.raises(RowRejected) as excinfo:
        normalize_row(glaive_spec(), {"system": GLAIVE_SYSTEM, "chat": GLAIVE_MALFORMED_CHAT}, 0)
    assert excinfo.value.code == "ADAPTER_GLAIVE_MALFORMED_CALL"


def test_no_behaviour_label_and_every_version_is_authoritative() -> None:
    rows = [
        normalize_row(glaive_spec(), {"system": GLAIVE_SYSTEM, "chat": GLAIVE_PLAIN_CHAT}, 0),
        normalize_row(when2call_spec(), when2call_raw(), 0),
    ]
    for item in rows:
        metadata = item["metadata"]
        assert "behavior" not in metadata
        assert metadata["adapter_version"] == versions.ADAPTER_VERSION
        assert metadata["supervision"]["adapter_version"] == versions.ADAPTER_VERSION
        assert metadata["schema_normalization_version"] == versions.SCHEMA_NORMALIZATION_VERSION
        assert metadata["normalization_version"] == versions.NORMALIZATION_VERSION
        config = {"adapter_version": versions.ADAPTER_VERSION}
        versions.check_version_agreement(metadata, config)
        versions.check_version_agreement(metadata["supervision"], config)
        # No legacy adapter version anywhere in the row.
        assert '"1.0.0"' not in json.dumps(metadata) and '"1.0.2"' not in json.dumps(metadata)


def test_when2call_translation_does_not_move_the_record_identity() -> None:
    raw = when2call_raw()
    untouched = json.loads(json.dumps(raw))
    item = normalize_row(when2call_spec(), raw, 7)
    assert raw == untouched, "the raw row must not be mutated"
    expected_hash = raw_record_hash(untouched)
    assert item["metadata"]["raw_record_hash"] == expected_hash
    assert item["id"] == "og_" + expected_hash[:16]
    assert item["metadata"]["source"]["raw_row_index"] == 7
    translation = item["metadata"]["schema_translation"]
    assert translation["applied"] and translation["changes"]
    term = item["tools"][0]["parameters"]["properties"]["term"]
    assert term["type"] == "string" and term["nullable"] is True


def test_unrepresentable_when2call_schema_is_quarantined_with_its_reason() -> None:
    tool = json.loads(WHEN2CALL_TOOL)
    tool["parameters"]["properties"]["term"]["type"] = "Tuple[int, str]"
    raw = {**when2call_raw(), "tools": [json.dumps(tool)]}
    with pytest.raises(RowRejected) as excinfo:
        normalize_row(when2call_spec(), raw, 0)
    assert excinfo.value.code.startswith("SRC_SCHEMA_")


# ── the build ──────────────────────────────────────────────────────────────────────────────────────


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def tiny_root(tmp_path: Path) -> Path:
    glaive_rows = [
        {"system": GLAIVE_SYSTEM, "chat": GLAIVE_CALL_CHAT},
        {"system": GLAIVE_SYSTEM, "chat": GLAIVE_PLAIN_CHAT},
        {"system": GLAIVE_SYSTEM, "chat": GLAIVE_PLAIN_CHAT},  # duplicate
        {"system": GLAIVE_SYSTEM, "chat": GLAIVE_MALFORMED_CHAT},  # rejected
    ]
    when2call_rows = [when2call_raw(), when2call_raw("Look up 'badger'.", "Sure.")]
    glaive_sha = _write_parquet(tmp_path / "raw/glaive.parquet", glaive_rows)
    when2call_sha = _write_parquet(tmp_path / "raw/when2call.parquet", when2call_rows)
    manifest = load_source_manifest()
    by_name = {entry["name"]: entry for entry in manifest["sources"]}
    by_name["glaive"]["raw_artifact"] = {
        "path": "raw/glaive.parquet",
        "sha256": glaive_sha,
        "rows": 4,
    }
    by_name["when2call"]["raw_artifact"] = {
        "path": "raw/when2call.parquet",
        "sha256": when2call_sha,
        "rows": 2,
    }
    manifest["sources"] = [by_name["glaive"], by_name["when2call"]]
    (tmp_path / "sources.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return tmp_path


def test_build_is_deterministic_and_verifies(tiny_root: Path) -> None:
    first = build(tiny_root / "a", root=tiny_root, source_manifest=Path("sources.yaml"))
    second = build(tiny_root / "b", root=tiny_root, source_manifest=Path("sources.yaml"), workers=2)
    assert first["fingerprint"] == second["fingerprint"]
    for name in ("glaive", "when2call"):
        a = (tiny_root / "a" / name / "shard-000000.parquet").read_bytes()
        b = (tiny_root / "b" / name / "shard-000000.parquet").read_bytes()
        assert a == b
    assert first["sources"]["glaive"]["counts"] == {
        "source_rows": 4,
        "accepted": 2,
        "rejected": 1,
        "duplicates": 1,
    }
    assert verify(tiny_root / "a", root=tiny_root) == []


def test_verify_detects_a_changed_shard(tiny_root: Path) -> None:
    build(tiny_root / "a", root=tiny_root, source_manifest=Path("sources.yaml"))
    shard = tiny_root / "a" / "glaive" / "shard-000000.parquet"
    shard.write_bytes(shard.read_bytes() + b"\0")
    assert any("bytes changed" in problem for problem in verify(tiny_root / "a", root=tiny_root))


def test_raw_bytes_that_disagree_with_the_manifest_are_refused(tiny_root: Path) -> None:
    (tiny_root / "raw/glaive.parquet").write_bytes(b"not the pinned bytes")
    with pytest.raises(NormalizationV3Error, match="sha256"):
        build(tiny_root / "a", root=tiny_root, source_manifest=Path("sources.yaml"))


def test_build_refuses_to_overwrite(tiny_root: Path) -> None:
    build(tiny_root / "a", root=tiny_root, source_manifest=Path("sources.yaml"))
    with pytest.raises(NormalizationV3Error):
        build(tiny_root / "a", root=tiny_root, source_manifest=Path("sources.yaml"))


# ── ToolACE: the call must not survive as text beside the structured call ────────────────────────────

TOOLACE_SYSTEM = (
    "You are an expert in composing functions.\n"
    '[{"name": "get_weather", "description": "Get the weather", "parameters": {"type": "object", '
    '"properties": {"city": {"type": "string"}}}}]'
)


def toolace_spec() -> SourceSpec:
    return SourceSpec(
        name="toolace",
        dataset_id="toolace",
        upstream_revision="6bda777c88d21e5a204703c1ee45597a8fa4f734",
        split="train",
        raw_path=Path("unused"),
        raw_sha256="0" * 64,
        raw_rows=0,
        adapter_key="toolace_v2",
        adapter_function="adapt_toolace_v2",
        row_label="toolace_v2",
        schema_translation=False,
    )


def toolace_raw(assistant: str) -> dict[str, Any]:
    return {
        "system": TOOLACE_SYSTEM,
        "conversations": [
            {"from": "user", "value": "What is the weather in Paris?"},
            {"from": "assistant", "value": assistant},
        ],
    }


def test_toolace_call_only_turn_keeps_no_call_text() -> None:
    item = normalize_row(toolace_spec(), toolace_raw('[get_weather(city="Paris")]'), 0)
    assistant = item["messages"][-1]
    assert assistant["tool_calls"] == [
        {"id": "call_0000", "name": "get_weather", "arguments": {"city": "Paris"}}
    ]
    assert assistant["content"] is None
    assert item["metadata"]["adapter"] == "toolace_v2"


def test_toolace_keeps_prose_around_the_removed_call() -> None:
    item = normalize_row(
        toolace_spec(),
        toolace_raw('[get_weather(city="Paris")] Let me know if you need another city.'),
        0,
    )
    assistant = item["messages"][-1]
    assert [call["name"] for call in assistant["tool_calls"]] == ["get_weather"]
    assert assistant["content"] == "Let me know if you need another city."


def test_toolace_malformed_call_is_quarantined_not_stripped() -> None:
    with pytest.raises(RowRejected) as excinfo:
        normalize_row(toolace_spec(), toolace_raw("[get_weather(city=)]"), 0)
    assert excinfo.value.code == "INVALID_ARGUMENT_SYNTAX"


def test_toolace_v2_changes_only_the_text_not_the_calls() -> None:
    """The repair must be representation-only: identical calls, identical tools, no duplicated text."""
    from opengrad.data.adapters import adapt_toolace, adapt_toolace_v2

    raw = toolace_raw('[get_weather(city="Paris")]')
    old, new = adapt_toolace(raw, "train"), adapt_toolace_v2(raw, "train")
    assert [m.get("tool_calls") for m in old.messages] == [
        m.get("tool_calls") for m in new.messages
    ]
    assert old.tools == new.tools
    assert old.messages[-1]["content"] == '[get_weather(city="Paris")]'  # the defect, still in v1
    assert new.messages[-1]["content"] is None
