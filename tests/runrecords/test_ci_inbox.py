"""Failed CI runs land in a draft inbox that has to be dealt with (``opengrad.runrecords.ci_intake``)."""

from __future__ import annotations

import json
from pathlib import Path

from opengrad.runrecords import ci_intake as ci
from opengrad.runrecords.cli import main as runs_main

ROOT = Path(__file__).resolve().parents[2]
SHA = "6f87585fe3a3c2ba2daa97e5c2f66231b4c4ccf9"


def test_every_committed_inbox_entry_is_dealt_with() -> None:
    problems = [
        f"{name}: {error}" for name, entry in ci.entries(ROOT).items() for error in ci.validate_entry(name, entry, ROOT)
    ]
    assert problems == [], "\n".join(problems)


def _run(run_id: int, created: str, conclusion: str = "failure") -> dict:
    return {
        "databaseId": run_id,
        "attempt": 1,
        "workflowName": "quality",
        "event": "push",
        "headBranch": "master",
        "headSha": SHA,
        "createdAt": created,
        "conclusion": conclusion,
    }


JOBS = {
    "jobs": [
        {
            "name": "test",
            "conclusion": "failure",
            "steps": [
                {"name": "Run uv run pytest", "conclusion": "failure", "log": "assert 'held-out text' == ..."},
                {"name": "Run uv run ruff check .", "conclusion": "success"},
            ],
        },
        {"name": "test-windows", "conclusion": "success", "steps": []},
    ]
}


def _fake_gh(runs: list[dict], calls: list[list[str]]):
    def run_gh(args: list[str]) -> str:
        calls.append(args)
        if args[:2] == ["run", "list"]:
            return json.dumps(runs)
        assert args[:2] == ["run", "view"]
        return json.dumps(JOBS)

    return run_gh


def _upstream(root: Path) -> None:
    (root / "docs" / "upstream").mkdir(parents=True)
    (root / "docs" / "upstream" / "UP-0001.md").write_text("x\n", encoding="utf-8")


def test_intake_drafts_each_new_failure_and_needs_attention_until_each_is_dealt_with(tmp_path: Path) -> None:
    _upstream(tmp_path)
    runs = [_run(43437011257, "2026-11-13T13:49:20Z"), _run(43039193997, "2026-11-11T14:30:54Z")]
    calls: list[list[str]] = []
    assert ci.cmd_intake(tmp_path, ci.INTAKE_SINCE, 100, False, _fake_gh(runs, calls)) == 3
    entries = ci.entries(tmp_path)
    assert list(entries) == ["CI-43437011257.json"]  # the older failure is before the inbox's start
    entry = entries["CI-43437011257.json"]
    # Names only: the step's log text never reaches the entry.
    assert entry["failed_jobs"] == [{"name": "test", "failed_steps": ["Run uv run pytest"]}]
    assert "held-out" not in json.dumps(entry)
    assert entry["disposition"] is None and entry["sha"] == SHA
    assert ci.validate_entry("CI-43437011257.json", entry, tmp_path) == [
        "needs a disposition: an upstream log id or a reason"
    ]
    # A second intake writes nothing new and still needs attention.
    calls.clear()
    assert ci.cmd_intake(tmp_path, ci.INTAKE_SINCE, 100, False, _fake_gh(runs, calls)) == 3
    assert [c[:2] for c in calls] == [["run", "list"]]
    assert ci.cmd_dispose(tmp_path, "CI-43437011257", None, "fixed in 875b3e4") == 0
    assert ci.cmd_intake(tmp_path, ci.INTAKE_SINCE, 100, False, _fake_gh(runs, [])) == 0
    assert ci.validate_entry("CI-43437011257.json", ci.entries(tmp_path)["CI-43437011257.json"], tmp_path) == []


def test_check_writes_nothing_and_reports_a_failure_with_no_entry(tmp_path: Path) -> None:
    runs = [_run(43437011257, "2026-11-13T13:49:20Z")]
    assert ci.cmd_intake(tmp_path, ci.INTAKE_SINCE, 100, True, _fake_gh(runs, [])) == 3
    assert ci.entries(tmp_path) == {}
    assert ci.cmd_intake(tmp_path, ci.INTAKE_SINCE, 100, True, _fake_gh([], [])) == 0


def test_a_disposition_must_name_a_real_entry_or_give_a_reason(tmp_path: Path) -> None:
    _upstream(tmp_path)
    assert ci.cmd_intake(tmp_path, ci.INTAKE_SINCE, 100, False, _fake_gh([_run(5, "2026-11-14T00:00:00Z")], [])) == 3
    assert ci.cmd_dispose(tmp_path, "5", "UP-0002", None) == 1  # no such entry
    assert ci.cmd_dispose(tmp_path, "5", "UP-0001", "and a reason") == 2
    assert ci.cmd_dispose(tmp_path, "5", None, "   ") == 2
    assert ci.cmd_dispose(tmp_path, "../5", None, "x") == 2
    assert ci.cmd_dispose(tmp_path, "6", None, "x") == 1  # not in the inbox
    assert ci.cmd_dispose(tmp_path, "5", "UP-0001", None) == 0
    assert ci.entries(tmp_path)["CI-5.json"]["disposition"] == {"upstream": "UP-0001"}


def test_an_entry_with_a_private_detail_or_a_wrong_name_is_refused(tmp_path: Path) -> None:
    entry = ci.draft(_run(7, "2026-11-14T00:00:00Z"), [{"name": "C:\\Users\\someone\\x", "failed_steps": []}])
    entry["disposition"] = {"reason": "x"}
    errors = ci.validate_entry("CI-7.json", entry, tmp_path)
    assert any("absolute path" in e or "home-folder" in e for e in errors), errors
    entry["failed_jobs"] = []
    assert ci.validate_entry("CI-8.json", entry, tmp_path) == ["file name does not match ci_run"]


def test_a_gh_failure_is_reported_not_raised(tmp_path: Path) -> None:
    def broken(args: list[str]) -> str:
        raise RuntimeError("gh run list failed: not logged in")

    assert ci.cmd_intake(tmp_path, ci.INTAKE_SINCE, 100, False, broken) == 1


def test_the_command_line_reaches_the_inbox(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ci, "gh", _fake_gh([], []))
    assert runs_main(["ci-intake", "--check"], tmp_path) == 0
    assert runs_main(["ci-dispose", "CI-9", "--reason", "x"], tmp_path) == 1  # no entry yet
