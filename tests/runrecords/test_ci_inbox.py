"""Failed CI runs land in a draft inbox that has to be dealt with (``opengrad.runrecords.ci_intake``)."""

from __future__ import annotations

import json
from pathlib import Path

from opengrad.runrecords import ci_intake as ci
from opengrad.runrecords.cli import main as runs_main

ROOT = Path(__file__).resolve().parents[2]
SHA = "6f87585fe3a3c2ba2daa97e5c2f66231b4c4ccf9"


def test_every_committed_inbox_entry_is_dealt_with() -> None:
    # A known entry must be found, so moving or renaming the inbox cannot make this pass by checking nothing.
    entries = ci.entries(ROOT)
    assert "CI-43437011257-1.json" in entries, sorted(entries)
    # Every file in the folder is checked, whatever its name or letter case.
    problems = [f"{name}: {error}" for name, entry in entries.items() for error in ci.validate_entry(name, entry, ROOT)]
    assert problems == [], "\n".join(problems)


def _run(
    run_id: int, created: str, conclusion: str = "failure", attempt: int = 1, event: str = "push", branch: str = "master"
) -> dict:
    return {
        "databaseId": run_id,
        "attempt": attempt,
        "workflowName": "quality",
        "event": event,
        "headBranch": branch,
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
        {"name": "slow", "conclusion": "timed_out", "steps": [{"name": "Run rm -rf /tmp/x", "conclusion": "timed_out"}]},
        {"name": "test-windows", "conclusion": "success", "steps": []},
    ]
}


def _fake_gh(runs: list[dict], calls: list[list[str]]):
    """GitHub as ``gh run list --status S`` sees it: only runs whose conclusion is S."""

    def run_gh(args: list[str]) -> str:
        calls.append(args)
        if args[:2] == ["run", "list"]:
            status = args[args.index("--status") + 1]
            assert args[args.index("--event") + 1] == "push"
            branch = args[args.index("--branch") + 1]
            return json.dumps([r for r in runs if r["conclusion"] == status and r["headBranch"] == branch])
        assert args[:2] == ["run", "view"]
        return json.dumps(JOBS)

    return run_gh


def _upstream(root: Path) -> None:
    (root / "docs" / "upstream").mkdir(parents=True)
    (root / "docs" / "upstream" / "UP-0001.md").write_text("x\n", encoding="utf-8")


def test_intake_drafts_each_new_failure_and_needs_attention_until_each_is_dealt_with(tmp_path: Path) -> None:
    _upstream(tmp_path)
    runs = [
        _run(43437011257, "2026-11-13T13:49:20Z"),
        _run(43039193997, "2026-11-11T14:30:54Z"),  # before the inbox's start
        _run(43500000000, "2026-11-14T01:00:00Z", conclusion="timed_out"),
        _run(43500000001, "2026-11-14T02:00:00Z", conclusion="cancelled"),  # superseded on purpose
        _run(43500000002, "2026-11-14T03:00:00Z", branch="feat/x"),  # work in progress on another branch
    ]
    calls: list[list[str]] = []
    assert ci.cmd_intake(tmp_path, ci.INTAKE_SINCE, 100, False, _fake_gh(runs, calls)) == 3
    entries = ci.entries(tmp_path)
    assert sorted(entries) == ["CI-43437011257-1.json", "CI-43500000000-1.json"]
    entry = entries["CI-43437011257-1.json"]
    assert entry is not None
    # Names only: the step's log text never reaches the entry. A timed-out job counts as failed.
    assert entry["failed_jobs"] == [
        {"name": "test", "failed_steps": ["Run uv run pytest"]},
        {"name": "slow", "failed_steps": ["Run rm -rf /tmp/x"]},
    ]
    assert "held-out" not in json.dumps(entry)
    assert entry["disposition"] is None and entry["sha"] == SHA
    assert ci.validate_entry("CI-43437011257-1.json", entry, tmp_path) == [
        "needs a disposition: an upstream log id or a reason"
    ]
    # A second intake writes nothing new and still needs attention.
    calls.clear()
    assert ci.cmd_intake(tmp_path, ci.INTAKE_SINCE, 100, False, _fake_gh(runs, calls)) == 3
    assert {tuple(c[:2]) for c in calls} == {("run", "list")}
    # A step name holding a path from the public workflow file does not block the disposition.
    assert ci.cmd_dispose(tmp_path, "CI-43437011257", 1, None, "fixed in 875b3e4") == 0
    assert ci.cmd_dispose(tmp_path, "43500000000", 1, "UP-0001", None) == 0
    assert ci.cmd_intake(tmp_path, ci.INTAKE_SINCE, 100, False, _fake_gh(runs, [])) == 0
    for name, value in ci.entries(tmp_path).items():
        assert ci.validate_entry(name, value, tmp_path) == []


def test_a_failed_rerun_of_a_run_dealt_with_is_a_new_draft(tmp_path: Path) -> None:
    first = [_run(5, "2026-11-14T00:00:00Z")]
    assert ci.cmd_intake(tmp_path, ci.INTAKE_SINCE, 100, False, _fake_gh(first, [])) == 3
    assert ci.cmd_dispose(tmp_path, "5", 1, None, "flaky runner") == 0
    again = [_run(5, "2026-11-14T00:00:00Z", attempt=2)]
    assert ci.cmd_intake(tmp_path, ci.INTAKE_SINCE, 100, False, _fake_gh(again, [])) == 3
    assert sorted(ci.entries(tmp_path)) == ["CI-5-1.json", "CI-5-2.json"]
    # The name intake prints is accepted as it is; a malformed one is a usage error.
    assert ci.cmd_dispose(tmp_path, "CI-5-x", 1, None, "x") == 2
    assert ci.cmd_dispose(tmp_path, "CI-5-2", 1, None, "flaky again") == 0
    assert ci.cmd_intake(tmp_path, ci.INTAKE_SINCE, 100, False, _fake_gh(again, [])) == 0


def test_another_branch_counts_only_when_named(tmp_path: Path) -> None:
    runs = [_run(9, "2026-11-14T00:00:00Z", branch="feat/x")]
    assert ci.cmd_intake(tmp_path, ci.INTAKE_SINCE, 100, True, _fake_gh(runs, [])) == 0
    assert ci.cmd_intake(tmp_path, ci.INTAKE_SINCE, 100, True, _fake_gh(runs, []), branch="feat/x") == 3


def test_since_is_compared_as_a_time_and_a_bad_one_is_refused(tmp_path: Path) -> None:
    runs = [_run(43437011257, "2026-11-13T13:49:20Z")]
    # 21:00 at +08:00 is 13:00 UTC, before the run: the run counts.
    assert ci.cmd_intake(tmp_path, "2026-11-13T21:00:00+08:00", 100, True, _fake_gh(runs, [])) == 3
    assert ci.cmd_intake(tmp_path, "2026-11-13T22:00:00+08:00", 100, True, _fake_gh(runs, [])) == 0
    assert ci.cmd_intake(tmp_path, "yesterday", 100, True, _fake_gh(runs, [])) == 2
    assert ci.cmd_intake(tmp_path, "2026-11-13T10:00:00", 100, True, _fake_gh(runs, [])) == 2  # no offset


def test_reaching_the_limit_is_an_error_not_a_silent_cut(tmp_path: Path) -> None:
    runs = [_run(10 + i, "2026-11-14T00:00:00Z") for i in range(3)]
    assert ci.cmd_intake(tmp_path, ci.INTAKE_SINCE, 3, True, _fake_gh(runs, [])) == 1
    assert ci.cmd_intake(tmp_path, ci.INTAKE_SINCE, 4, True, _fake_gh(runs, [])) == 3


def test_check_writes_nothing_and_reports_a_failure_with_no_entry(tmp_path: Path) -> None:
    runs = [_run(43437011257, "2026-11-13T13:49:20Z")]
    assert ci.cmd_intake(tmp_path, ci.INTAKE_SINCE, 100, True, _fake_gh(runs, [])) == 3
    assert ci.entries(tmp_path) == {}
    assert ci.cmd_intake(tmp_path, ci.INTAKE_SINCE, 100, True, _fake_gh([], [])) == 0


def test_an_unreadable_inbox_file_is_reported_not_a_crash(tmp_path: Path) -> None:
    folder = tmp_path / ci.INBOX
    folder.mkdir(parents=True)
    (folder / "CI-5-1.json").write_text("{not json", encoding="utf-8")
    (folder / "CI-6-1.JSON").write_text("[]", encoding="utf-8")
    assert ci.cmd_intake(tmp_path, ci.INTAKE_SINCE, 100, True, _fake_gh([], [])) == 3
    assert ci.cmd_dispose(tmp_path, "5", 1, None, "x") == 1
    assert {name: ci.validate_entry(name, e, tmp_path) for name, e in ci.entries(tmp_path).items()} == {
        "CI-5-1.json": ["not a readable JSON object"],
        "CI-6-1.JSON": ["not a readable JSON object"],
    }


def test_a_disposition_must_name_a_real_entry_or_give_a_reason(tmp_path: Path) -> None:
    _upstream(tmp_path)
    assert ci.cmd_intake(tmp_path, ci.INTAKE_SINCE, 100, False, _fake_gh([_run(5, "2026-11-14T00:00:00Z")], [])) == 3
    assert ci.cmd_dispose(tmp_path, "5", 1, "UP-0002", None) == 1  # no such entry
    assert ci.cmd_dispose(tmp_path, "5", 1, "UP-0001", "and a reason") == 2
    assert ci.cmd_dispose(tmp_path, "5", 1, None, "   ") == 2
    assert ci.cmd_dispose(tmp_path, "../5", 1, None, "x") == 2
    assert ci.cmd_dispose(tmp_path, "6", 1, None, "x") == 1  # not in the inbox
    assert ci.cmd_dispose(tmp_path, "5", 1, "UP-0001", None) == 0
    entry = ci.entries(tmp_path)["CI-5-1.json"]
    assert entry is not None and entry["disposition"] == {"upstream": "UP-0001"}
    # A hand-edited disposition with a non-text reason or extra keys is refused.
    for bad in ({"reason": ["x"]}, {"reason": "ok", "note": "text"}):
        assert ci.validate_entry("CI-5-1.json", {**entry, "disposition": bad}, tmp_path) != []
    assert ci.validate_entry("CI-5-1.json", {**entry, "note": "text"}, tmp_path) != []
    assert "event is not push" in " ".join(ci.validate_entry("CI-5-1.json", {**entry, "event": "pull_request"}, tmp_path))


def test_an_entry_with_a_private_detail_or_a_wrong_name_is_refused(tmp_path: Path) -> None:
    entry = ci.draft(_run(7, "2026-11-14T00:00:00Z"), [{"name": "C:\\Users\\someone\\x", "failed_steps": []}])
    entry["disposition"] = {"reason": "x"}
    errors = ci.validate_entry("CI-7-1.json", entry, tmp_path)
    assert any("home-folder" in e for e in errors), errors
    entry["failed_jobs"] = []
    assert ci.validate_entry("CI-8-1.json", entry, tmp_path) == ["file name does not match ci_run and attempt"]
    assert ci.validate_entry("CI-7-1.json", {**entry, "branch": "/opt/x"}, tmp_path) == [
        "the entry contains an absolute path"
    ]


def test_a_gh_failure_is_reported_not_raised(tmp_path: Path) -> None:
    def broken(args: list[str]) -> str:
        raise RuntimeError("gh run list failed: not logged in")

    assert ci.cmd_intake(tmp_path, ci.INTAKE_SINCE, 100, False, broken) == 1


def test_the_command_line_reaches_the_inbox(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ci, "gh", _fake_gh([], []))
    assert runs_main(["ci-intake", "--check"], tmp_path) == 0
    assert runs_main(["ci-dispose", "CI-9", "--attempt", "2", "--reason", "x"], tmp_path) == 1  # no entry yet
