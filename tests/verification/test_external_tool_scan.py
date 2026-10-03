"""The archive's scan for web-tool calls by the external labelling CLIs (docs/UPSTREAM_ISSUES.md UP-0012)."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
import archive_external_model_labels as archive


def _varint(value: int) -> bytes:
    out = b""
    while value >= 0x80:
        out += bytes([value & 0x7F | 0x80])
        value >>= 7
    return out + bytes([value])


def _ld(number: int, value: bytes) -> bytes:
    """One length-delimited protobuf field."""
    return _varint(number << 3 | 2) + _varint(len(value)) + value


def _call(name: bytes, arguments: dict[str, str]) -> bytes:
    return _ld(20, _ld(7, _ld(1, b"id") + _ld(2, name) + _ld(3, json.dumps(arguments).encode())))


def test_the_tool_name_is_read_from_its_field_path() -> None:
    step = _ld(1, b"search_web appears in free text") + _call(b"view_file", {"path": "input.md"})
    assert archive._field(step, archive.AGY_TOOL_NAME_PATH) == [b"view_file"]
    assert archive._field(b"\xff\xff", (1,)) == []  # bytes that do not parse yield nothing


def test_cline_reports_its_tool_calls_or_is_not_visible() -> None:
    events = [
        {"type": "agent_event", "event": {"type": "iteration_end", "toolCallCount": n}}
        for n in (0, 2)
    ]
    stream = "\n".join(json.dumps(e) for e in events).encode()
    assert archive.cline_tool_calls(stream) == {
        "visible": True,
        "source": "cline --json toolCallCount",
        "tool_calls": 2,
        "any": True,
    }
    assert archive.cline_tool_calls(b'{"type": "run_result"}')["visible"] is False


def test_an_agy_search_is_counted_and_a_missing_conversation_is_not_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(archive, "AGY_CONVERSATIONS", tmp_path)
    run = {"started_at": time.time() - 1, "seconds": 30}
    assert archive.agy_web_calls(run)["visible"] is False  # no conversation: unknown, never "none"
    con = sqlite3.connect(tmp_path / "conversation.db")
    con.execute("create table steps (step_type integer, step_payload blob)")
    con.executemany(
        "insert into steps values (?, ?)",
        [
            (15, _call(b"view_file", {"path": "input.md"})),
            (15, _call(b"search_web", {"query": "q"})),
            (33, b""),
        ],
    )
    con.commit()
    con.close()
    result = archive.agy_web_calls(run)
    assert result["visible"] and result["any"]
    assert result["web_tool_calls"] == {"search_web": 1} and result["web_steps"] == 1
