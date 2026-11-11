"""The upstream issues log is generated from one file per entry, and every entry carries what it must.

`docs/UPSTREAM_ISSUES.md` once had a hand-written table that drifted from its sections (it stopped at UP-0017 while
the sections reached UP-0021). The table and sections are now built by `scripts/reporting/generate_upstream_log.py`
from `docs/upstream/UP-NNNN.md`. These tests make a stale log, or an entry missing the evidence its class or
upstream status needs, a CI failure.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _generator():
    spec = importlib.util.spec_from_file_location(
        "generate_upstream_log", ROOT / "scripts/reporting/generate_upstream_log.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _generator()

BODY = "## UP-0001 — a title (`OURS`)\n\nText."


def _entry(**changes: object) -> dict:
    data = {
        "id": "UP-0001",
        "date": "2026-11-11",
        "where": "OpenGrad tests",
        "class": "OURS",
        "summary": "s",
        "state": "Fixed",
        "upstream": {"status": "not-applicable", "url": None},
    }
    data.update(changes)
    return data


def test_the_log_is_current_and_every_entry_is_valid() -> None:
    committed = (ROOT / gen.LOG).read_text(encoding="utf-8")
    assert committed == gen.render(), "run python scripts/reporting/generate_upstream_log.py"


def test_entries_run_without_gaps_and_each_appears_once_in_the_log() -> None:
    entries = gen.load()
    assert len(entries) >= 30, "nothing to check: the migration carried 30 entries"
    log = (ROOT / gen.LOG).read_text(encoding="utf-8")
    for data, body in entries:
        assert log.count(f"\n| {data['id']} |") == 1
        assert log.count(body) == 1


def test_a_valid_entry_passes() -> None:
    assert gen.validate(_entry(), BODY, "UP-0001.md") == []


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"class": "BUG"}, "class"),
        ({"upstream": {"status": "maybe"}}, "upstream.status"),
        (
            {"upstream": {"status": "filed", "reproduction": "r", "tracker_search": "t"}},
            "upstream.url",
        ),
        ({"upstream": {"status": "candidate", "url": None}}, "upstream.reproduction"),
        ({"class": "UPSTREAM", "upstream": {"status": "not-filed"}}, "'library'"),
        (
            {
                "class": "UPSTREAM",
                "library": "x",
                "version": "1",
                "upstream": {"status": "not-applicable"},
            },
            "not-applicable",
        ),
        (
            {
                "class": "FEATURE GAP",
                "library": "x",
                "version": "1",
                "upstream": {"status": "not-filed"},
            },
            "workaround",
        ),
        ({"summary": ""}, "'summary'"),
    ],
)
def test_an_entry_missing_what_its_class_or_status_needs_fails(
    changes: dict, expected: str
) -> None:
    errors = gen.validate(_entry(**changes), BODY, "UP-0001.md")
    assert any(expected in error for error in errors), errors


def test_the_file_name_and_heading_must_match_the_id() -> None:
    assert gen.validate(_entry(), BODY, "UP-0002.md")
    assert gen.validate(_entry(), "## UP-0002 — wrong", "UP-0001.md")


def test_a_gap_in_the_numbering_fails(tmp_path: Path) -> None:
    (tmp_path / gen.ENTRIES).mkdir(parents=True)
    for number in (1, 3):
        entry_id = f"UP-{number:04d}"
        text = (
            "---\n"
            f"id: {entry_id}\ndate: '2026-11-11'\nwhere: w\nclass: OURS\nsummary: s\nstate: t\n"
            "upstream:\n  status: not-applicable\n"
            f"---\n## {entry_id} — t (`OURS`)\n"
        )
        (tmp_path / gen.ENTRIES / f"{entry_id}.md").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="without gaps"):
        gen.load(tmp_path)


def test_the_heading_must_carry_the_entry_class() -> None:
    errors = gen.validate(_entry(), "## UP-0001 — a title (`UNCLEAR`)\n\nText.", "UP-0001.md")
    assert any("heading" in error for error in errors), errors


def test_a_later_line_for_another_entry_is_rejected() -> None:
    body = BODY + "\n\nLater line for UP-0002, 2026-11-11: belongs elsewhere."
    errors = gen.validate(_entry(), body, "UP-0001.md")
    assert any("UP-0002.md" in error for error in errors), errors
    own = BODY + "\n\nLater line for UP-0001, 2026-11-11: fine."
    assert gen.validate(_entry(), own, "UP-0001.md") == []


def test_a_link_without_a_filed_status_is_rejected() -> None:
    entry = _entry(upstream={"status": "not-filed", "url": "https://example.org/1"})
    assert any("upstream.url is set" in e for e in gen.validate(entry, BODY, "UP-0001.md"))


def test_an_entry_without_an_id_is_reported_not_crashed(tmp_path: Path) -> None:
    (tmp_path / gen.ENTRIES).mkdir(parents=True)
    (tmp_path / gen.ENTRIES / "UP-0001.md").write_text(
        "---\ndate: '2026-11-11'\n---\n## UP-0001 — t (`OURS`)\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing field 'id'"):
        gen.load(tmp_path)
