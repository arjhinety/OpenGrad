from pathlib import Path

import pytest

from opengrad.data.adapters import ADAPTERS, adapt_toolace, adapt_xlam
from opengrad.data.canonical import canonical_dict
from opengrad.data.normalize import iter_jsonl, normalize_records

ROOT = Path(__file__).parents[1] / "fixtures" / "raw"


def test_upstream_shaped_fixtures_are_source_aware():
    expected = {
        "xlam": "query",
        "when2call": "conversation",
        "toolace": "conversations",
        "button": "messages",
        "looptool": "instruction",
        "glaive": "chat",
    }
    _items = list(expected.items())
    assert _items, "nothing to check: an empty collection would pass this test vacuously"
    for name, native_key in _items:
        rows = list(iter_jsonl(ROOT / name / "row.jsonl"))
        row = next((item for item in rows if native_key in item), rows[0])
        assert native_key in row or (name == "button" and "messages" in row)
        result = canonical_dict(ADAPTERS[name](row, "fixture"))
        assert result["schema_version"] == "tool_use_ir_v1"
        assert result["metadata"]["adapter"] != "generic"


def test_xlam_decodes_serialized_tools_and_answers():
    row = next(iter(iter_jsonl(ROOT / "xlam" / "row.jsonl")))
    result = adapt_xlam(row)
    assert result.messages[1]["tool_calls"][0]["arguments"] == {"city": "Manila"}
    assert result.metadata["raw_record_hash"]


def test_toolace_literal_parser_does_not_execute_expressions():
    row = {
        "id": "bad",
        "tools": [{"name": "lookup"}],
        "system": "",
        "conversations": [
            {"from": "human", "value": "x"},
            {"from": "gpt", "value": "[Function lookup(q=__import__('os'))]"},
        ],
    }
    with pytest.raises(ValueError):
        adapt_toolace(row)


def test_button_parses_array_calls_and_embedded_catalogue():
    row = {
        "id": "button-array",
        "messages": [
            {"role": "system", "content": '[{"name":"lookup","parameters":{"type":"object"}}]'},
            {"role": "user", "content": "Find x"},
            {
                "role": "assistant",
                "content": '<call>[{"name":"lookup","arguments":{"q":"x"}}]</call>',
            },
        ],
    }
    result = ADAPTERS["button"](row, "fixture")
    assert result.tools[0]["name"] == "lookup"
    assert result.messages[-1]["tool_calls"][0]["arguments"] == {"q": "x"}


def test_normalization_counts_failures_and_duplicates():
    good = next(iter(iter_jsonl(ROOT / "xlam" / "row.jsonl")))
    rows, counts = normalize_records([good, good, {"id": "missing"}], "xlam")
    assert len(rows) == 1
    assert counts["duplicates"] == 1
    assert counts["parse_failed"] == 1
