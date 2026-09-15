"""Tests for the canonical-v3 provenance and source-schema-normalization contracts.

Every assertion corresponds to a defect found while building the C1 path, so this file is a regression
net rather than a restatement of the code:

* the adapter-version split-brain (record metadata said 1.0.0 while the manifest said 1.0.2), and
* translation being *stricter* than the canonical validator on three annotation shapes (nested schema
  under ``type``, a JSON-Schema type list, an unparseable type value) -- each of which silently
  rejected rows the canonical layer accepted. Measured before the fix: 68 ToolACE + 1 Glaive rows.
"""

from __future__ import annotations

import json

import pytest

from opengrad.data.source_schema import (
    SRC_HETEROGENEOUS_TUPLE,
    SRC_MULTI_TYPE_UNION,
    SRC_UNREPRESENTABLE_TYPE,
    translate_source_schema,
    translate_source_tools,
    translate_type_annotation,
)
from opengrad.data.versions import (
    ADAPTER_VERSION,
    CANONICAL_SCHEMA_VERSION,
    MANIFEST_VERSION_FIELDS,
    RECORD_VERSION_FIELDS,
    ProvenanceVersionMismatch,
    check_artifact_matches_authoritative_versions,
    check_version_agreement,
    provenance_versions,
    record_versions,
)


# ── provenance versions ─────────────────────────────────────────────────────────────────────────


def test_one_authoritative_source_per_version() -> None:
    versions = provenance_versions()
    assert versions["adapter_version"] == ADAPTER_VERSION
    assert versions["canonical_schema_version"] == CANONICAL_SCHEMA_VERSION
    assert len(set(versions.values())) >= 5, "versions must identify distinct concerns"


def test_record_versions_are_a_subset_of_the_authoritative_map() -> None:
    for field, expected in record_versions().items():
        assert field in RECORD_VERSION_FIELDS
        assert expected == RECORD_VERSION_FIELDS[field]
    assert "adapter_version" in record_versions()


def test_agreement_passes_when_layers_match() -> None:
    manifest = dict(RECORD_VERSION_FIELDS)
    manifest.update(MANIFEST_VERSION_FIELDS)
    check_version_agreement(dict(RECORD_VERSION_FIELDS), manifest)


def test_disagreement_between_record_and_manifest_fails() -> None:
    """The defect this module exists to end: one artifact, two adapter versions."""
    with pytest.raises(ProvenanceVersionMismatch) as excinfo:
        check_version_agreement({"adapter_version": "1.0.0"}, {"adapter_version": "1.0.2"})
    assert "adapter_version" in str(excinfo.value)
    assert "PROV_VERSION_MISMATCH" in str(excinfo.value)


def test_manifest_field_not_matching_the_constant_fails() -> None:
    """A newly built artifact must be traceable to the code that made it."""
    with pytest.raises(ProvenanceVersionMismatch):
        check_artifact_matches_authoritative_versions(
            {"adapter_version": ADAPTER_VERSION, "canonical_schema_version": "tool_use_ir_v0"}
        )


def test_historical_manifest_is_readable() -> None:
    """v1/v2 recorded versions that no longer exist in code; they must not fail a new check.

    The published corpora are not edited to conform, so a manifest whose layers agree at an old
    version is internally consistent and stays readable.
    """
    check_version_agreement({}, {"adapter_version": "1.0.2"})
    check_version_agreement({"adapter_version": "1.0.0"}, {"adapter_version": "1.0.0"})
    with pytest.raises(ProvenanceVersionMismatch):
        check_artifact_matches_authoritative_versions({"adapter_version": "1.0.2"})


# ─ type-annotation translation ────────────────────────────────────────────────────────────────

TRANSLATIONS = [
    ("str", "string", False),
    ("int", "integer", False),
    ("float", "number", False),
    ("bool", "boolean", False),
    ("dict", "object", False),
    ("list", "array", False),
    ("str, optional", "string", True),
    ("int, optional", "integer", True),
    ("bool, optional", "boolean", True),
    ("float, optional", "number", True),
    ("Optional[str]", "string", True),
    ("Union[str, None]", "string", True),
    ("List[int]", "array", False),
    ("List[float]", "array", False),
    ("List[str]", "array", False),
    ("List", "array", False),
    ("Dict", "object", False),
    ("Set[str]", "array", False),
    ("set", "array", False),
    ("Tuple[float, float]", "array", False),
    ("List[List[int]]", "array", False),
    ("object", "object", False),
    ("string", "string", False),
    ("integer", "integer", False),
]


@pytest.mark.parametrize("annotation,expected_type,nullable", TRANSLATIONS)
def test_observed_source_forms_translate(
    annotation: str, expected_type: str, nullable: bool
) -> None:
    result = translate_type_annotation(annotation)
    assert not result.quarantined, f"{annotation} must translate, not quarantine"
    assert result.canonical_type == expected_type
    assert result.nullable is nullable


QUARANTINES = [
    ("Tuple[int, str]", SRC_HETEROGENEOUS_TUPLE),
    ("Union[str, int]", SRC_MULTI_TYPE_UNION),
    ("WeirdThing", SRC_UNREPRESENTABLE_TYPE),
    ("List[WeirdThing]", SRC_UNREPRESENTABLE_TYPE),
]


@pytest.mark.parametrize("annotation,code", QUARANTINES)
def test_unrepresentable_forms_quarantine_with_an_explicit_reason(
    annotation: str, code: str
) -> None:
    result = translate_type_annotation(annotation)
    assert result.quarantined
    assert result.quarantine_reason == code
    assert result.quarantine_detail


def test_canonical_input_passes_through_without_a_translation_record() -> None:
    """Idempotence: an already-canonical schema must produce no records."""
    tool = {
        "name": "lookup",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        },
    }
    translation = translate_source_schema(tool)
    assert not translation.quarantined
    assert translation.records == []
    assert translation.tool == tool


def test_translation_records_what_changed() -> None:
    tool = {
        "name": "t",
        "parameters": {"type": "dict", "properties": {"a": {"type": "str, optional"}}},
    }
    translation = translate_source_schema(tool)
    assert not translation.quarantined
    assert translation.tool["parameters"]["type"] == "object"
    assert translation.tool["parameters"]["properties"]["a"] == {"type": "string", "nullable": True}
    paths = {record["path"] for record in translation.records}
    assert paths == {"parameters", "parameters.properties.a"}


@pytest.mark.parametrize(
    "annotation",
    [
        "object",
        "string",
        "integer",
        "number",
        "boolean",
        "array",
        "null",
        "str",
        "int",
        "float",
        "bool",
        "dict",
        "list",
        "any",
        {"type": "string"},
        {"schema": "nested"},
        5,
        True,
        ["string", "null"],
        ["string", "integer"],
    ],
)
def test_translation_is_never_stricter_than_the_canonical_layer(annotation: object) -> None:
    """The regression this guards.

    The canonical alias pass resolves any non-string ``type`` by dropping the constraint, so a
    translator that quarantines such a row makes normalization *lossy* rather than *recovering*.
    """
    assert not translate_type_annotation(annotation).quarantined


def test_type_list_with_null_is_faithfully_represented() -> None:
    result = translate_type_annotation(["string", "null"])
    assert result.canonical_type == "string"
    assert result.nullable is True
    assert "type_list_nullable" in result.notes


def test_multi_type_list_is_dropped_and_recorded_not_silent() -> None:
    result = translate_type_annotation(["string", "integer"])
    assert not result.quarantined
    assert result.canonical_type is None
    assert any(note.startswith("multi_type_list_dropped:") for note in result.notes)


def test_nested_schema_under_type_is_recovered() -> None:
    """Canonical recovers this shape; translation must not reject it instead."""
    result = translate_type_annotation({"type": "string", "properties": {}})
    assert not result.quarantined
    assert result.canonical_type == "string"
    assert "nested_schema_under_type" in result.notes


def test_heterogeneous_tuple_is_not_silently_reduced_to_array() -> None:
    """A tuple of unlike types cannot be expressed, so it is refused rather than coerced."""
    assert translate_type_annotation("Tuple[int, str]").quarantine_reason == SRC_HETEROGENEOUS_TUPLE


# ── tool-list translation ──────────────────────────────────────────────────────────────────────


def test_tool_list_of_json_strings_is_parsed() -> None:
    """When2Call ships ``tools`` as a list of JSON strings, not a list of objects."""
    tools = [json.dumps({"name": "t", "parameters": {"type": "dict"}})]
    translated, records, reason, detail = translate_source_tools(tools)
    assert reason is None, detail
    assert translated[0]["parameters"]["type"] == "object"
    assert any(record["path"] == "parameters" for record in records)


def test_unparsable_tool_element_quarantines() -> None:
    translated, _records, reason, detail = translate_source_tools(["{not json"])
    assert translated == []
    assert reason == SRC_UNREPRESENTABLE_TYPE
    assert "unparsable JSON" in detail


def test_non_dict_tool_element_quarantines() -> None:
    translated, _records, reason, _detail = translate_source_tools([[1, 2, 3]])
    assert translated == []
    assert reason is not None


def test_absent_tools_is_not_a_failure() -> None:
    translated, records, reason, _detail = translate_source_tools(None)
    assert (translated, records, reason) == ([], [], None)