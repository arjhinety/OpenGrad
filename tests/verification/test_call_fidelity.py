"""Independent call readers and the conservation/fidelity comparison behind P-DET-COVERAGE-v1 layer A.

The readers must agree with the adapters on real syntax without sharing their code, and every mismatch
kind must be reachable: a check that cannot fail is not evidence.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from opengrad.data.normalization_v3 import normalize_row
from opengrad.verification import call_fidelity as fidelity
from tests.data.test_normalization_v3 import (
    GLAIVE_CALL_CHAT,
    GLAIVE_SYSTEM,
    glaive_spec,
    toolace_raw,
    toolace_spec,
)


def _args(value: dict[str, Any]) -> str:
    return fidelity.canonical_arguments(value)


# ── Glaive ────────────────────────────────────────────────────────────────────────────────────────────


def test_glaive_single_quoted_json_arguments_are_read():
    assert fidelity.glaive_raw_calls(GLAIVE_CALL_CHAT) == [
        ("get_weather", _args({"city": "Paris"}))
    ]


def test_glaive_object_arguments_and_missing_arguments_are_read():
    chat = (
        'ASSISTANT: <functioncall> {"name": "a", "arguments": {"x": true}} <|endoftext|>\n\n\n'
        'FUNCTION RESPONSE: {}\n\n\nASSISTANT: <functioncall> {"name": "b"}\n'
    )
    assert fidelity.glaive_raw_calls(chat) == [("a", _args({"x": True})), ("b", _args({}))]


@pytest.mark.parametrize(
    "chat",
    [
        "ASSISTANT: <functioncall> {\"arguments\": '{}'}",
        'ASSISTANT: <functioncall> {"name": "a", "arguments": \'{"x": }\'}',
        'ASSISTANT: <functioncall> {"name": "a", "arguments": {"x": }}',
    ],
)
def test_glaive_unreadable_calls_raise_instead_of_passing(chat):
    with pytest.raises(fidelity.RawCallError):
        fidelity.glaive_raw_calls(chat)


# ── ToolACE ───────────────────────────────────────────────────────────────────────────────────────────


def test_toolace_names_with_spaces_and_keyword_or_dashed_argument_names_are_read():
    turn = (
        '[Get Weather(city="Paris", from="2024-01-01", page-size=2), news.top(flag=true, x=null)]'
    )
    assert fidelity.toolace_turn_calls(turn) == [
        ("Get Weather", _args({"city": "Paris", "from": "2024-01-01", "page-size": 2})),
        ("news.top", _args({"flag": True, "x": None})),
    ]


def test_toolace_nested_literals_quotes_and_trailing_prose_are_handled():
    turn = '[f(q="a, b(c) = d", ids=[1, 2], opts={"k": [true]})] Then I will summarise.'
    assert fidelity.toolace_turn_calls(turn) == [
        ("f", _args({"q": "a, b(c) = d", "ids": [1, 2], "opts": {"k": [True]}}))
    ]


def test_toolace_empty_call_is_read_with_no_arguments():
    assert fidelity.toolace_turn_calls("[ping()]") == [("ping", _args({}))]


@pytest.mark.parametrize("turn", ['[f("positional")]', "[f(x=some_name)]", "[not a call]"])
def test_toolace_unreadable_turns_raise_instead_of_passing(turn):
    with pytest.raises(fidelity.RawCallError):
        fidelity.toolace_turn_calls(turn)


def test_toolace_only_assistant_bracket_turns_hold_calls():
    conversations = [
        {"from": "user", "value": "[f(x=1)]"},
        {"from": "assistant", "value": "[f(x=1)]"},
        {"from": "assistant", "value": "Plain answer (no call)."},
    ]
    assert fidelity.toolace_raw_calls(conversations) == [("f", _args({"x": 1}))]


# ── xLAM, comparison and status ─────────────────────────────────────────────────────────────────────


def test_xlam_answers_json_is_read():
    answers = '[{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {}}, {"x": 2}]'
    assert fidelity.xlam_raw_calls(answers) == [("a", _args({"x": 1})), ("b", _args({}))]


@pytest.mark.parametrize(
    ("raw", "structured", "outcome"),
    [
        ([("a", "{}")], [("a", "{}")], fidelity.EXACT),
        ([("a", "{}")], [], fidelity.COUNT_MISMATCH),
        ([("a", "{}")], [("b", "{}")], fidelity.NAME_MISMATCH),
        ([("a", '{"x":1}')], [("a", '{"x":"1"}')], fidelity.ARGUMENT_MISMATCH),
    ],
)
def test_every_mismatch_kind_is_reachable(raw, structured, outcome):
    assert fidelity.compare(raw, structured) == outcome


def test_status_is_fail_on_mismatch_incomplete_on_unverified_and_pass_only_when_all_compared():
    clean = {"mismatched_rows": 0, "unverified_rows": 0}
    assert fidelity.fidelity_status({"a": clean}) == "PASS"
    assert (
        fidelity.fidelity_status({"a": clean, "b": {**clean, "unverified_rows": 1}}) == "INCOMPLETE"
    )
    assert fidelity.fidelity_status(
        {"a": {**clean, "mismatched_rows": 1, "unverified_rows": 1}}
    ) == ("FAIL")


# ── rows against real adapter output ────────────────────────────────────────────────────────────────


def _glaive_row(index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = {"system": GLAIVE_SYSTEM, "chat": GLAIVE_CALL_CHAT}
    return normalize_row(glaive_spec(), raw, index), raw


def test_adapter_output_matches_the_independent_reader_and_a_changed_argument_is_caught():
    record, raw = _glaive_row(0)
    tampered = copy.deepcopy(record)
    tampered["id"] = "tampered"
    tampered["metadata"]["source"]["raw_row_index"] = 1
    call_turn = next(message for message in tampered["messages"] if message.get("tool_calls"))
    call_turn["tool_calls"][0]["arguments"] = {"city": "London"}
    cells, mismatch_ids = fidelity.measure_rows("glaive", [record, tampered], {0: raw, 1: raw})
    valid = cells[fidelity.TRAJECTORY_VALID]
    assert (valid["exact"], valid["argument_mismatch"], valid["rows_with_calls"]) == (1, 1, 2)
    assert mismatch_ids == {fidelity.ARGUMENT_MISMATCH: ["tampered"]}


def test_toolace_rows_failing_the_trajectory_gate_are_still_compared():
    raw = toolace_raw('[get_weather(city="Paris")]')
    record = normalize_row(toolace_spec(), raw, 0)
    cells, _ids = fidelity.measure_rows("toolace", [record], {0: raw})
    assert cells[fidelity.TRAJECTORY_ISSUE]["exact"] == 1


def test_rows_without_calls_are_counted_but_not_compared():
    raw = {"system": GLAIVE_SYSTEM, "chat": "USER: Hi\n\n\nASSISTANT: Hello! <|endoftext|>\n"}
    record = normalize_row(glaive_spec(), raw, 0)
    cells, _ids = fidelity.measure_rows("glaive", [record], {0: raw})
    assert cells[fidelity.TRAJECTORY_VALID]["rows"] == 1
    assert cells[fidelity.TRAJECTORY_VALID]["rows_with_calls"] == 0


def test_ledger_accounts_for_calls_in_rows_normalization_did_not_accept():
    ledger = [
        {"disposition": "rejected", "reason": "ADAPTER_GLAIVE_MALFORMED_CALL", "raw_row_index": 0},
        {"disposition": "duplicate", "reason": None, "raw_row_index": 1},
    ]
    raw = {
        0: {"chat": 'ASSISTANT: <functioncall> {"name": "a", "arguments": \'{"x": }\'}'},
        1: {"chat": GLAIVE_CALL_CHAT},
    }
    assert fidelity.measure_ledger("glaive", ledger, raw) == {
        "duplicate:-": {"raw_calls": 1, "rows": 1, "rows_with_calls": 1},
        "rejected:ADAPTER_GLAIVE_MALFORMED_CALL": {"raw_unparseable_independently": 1},
    }


# ── the format-grammar reader (fallback, counted separately) ─────────────────────────────────────────


def test_format_reader_handles_names_with_parentheses_commas_and_json_values():
    turn = '[Stock Price (Real-Time)(symbol="AAPL", live=true), Books, Films(q={"a": null}, n=2)]'
    assert fidelity.toolace_turn_calls_by_format(turn) == [
        ("Stock Price (Real-Time)", _args({"symbol": "AAPL", "live": True})),
        ("Books, Films", _args({"q": {"a": None}, "n": 2})),
    ]


@pytest.mark.parametrize(
    "turn", ["[f(positional)]", "no bracket", "[f(x=not a literal)]", "[f x=1]", "[]"]
)
def test_format_reader_refuses_what_is_not_the_format(turn):
    with pytest.raises(fidelity.RawCallError):
        fidelity.toolace_turn_calls_by_format(turn)


def test_the_fallback_is_used_only_when_the_python_parser_fails_and_is_recorded():
    methods: list[str] = []
    conversations = [
        {"from": "assistant", "value": "[f(x=1)]"},
        {"from": "assistant", "value": "[Price (v2)(x=1)]"},
    ]
    calls = fidelity.toolace_raw_calls(conversations, methods)
    assert calls == [("f", _args({"x": 1})), ("Price (v2)", _args({"x": 1}))]
    assert methods == [fidelity.PRIMARY, fidelity.GRAMMAR_FALLBACK]


def test_rows_read_by_the_fallback_are_counted_in_their_cell():
    raw = toolace_raw('[Get Weather (Now)(city="Paris")]')
    raw["system"] = raw["system"].replace('"get_weather"', '"Get Weather (Now)"')
    record = normalize_row(toolace_spec(), raw, 0)
    cells, _ids = fidelity.measure_rows("toolace", [record], {0: raw})
    cell = cells[fidelity.TRAJECTORY_ISSUE]
    assert (cell["exact"], cell["rows_read_by_format_grammar"]) == (1, 1)
