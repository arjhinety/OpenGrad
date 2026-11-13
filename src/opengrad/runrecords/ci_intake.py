"""Failed CI runs land in a draft inbox that has to be dealt with (run-record stage 4).

``python -m opengrad.runrecords ci-intake`` asks GitHub (read-only, through the ``gh`` CLI) for the workflow runs
that failed since ``INTAKE_SINCE`` and writes one draft per run to ``run-records/ci-inbox/CI-<run id>.json``. A
draft has no disposition. ``ci-dispose`` gives it one: the ``docs/upstream/`` entry the failure is logged under,
or a reason it needs none (a fix that landed, a flaky runner). Like a failed local run, a draft is dealt with only
when it has a disposition:

* ``ci-intake`` exits 3 while any draft has none, the same "needs attention" code as ``upload``;
* ``tests/runrecords/test_ci_inbox.py`` fails CI on a committed entry without a valid disposition, so a draft
  cannot be committed until it is dealt with;
* ``ci-intake --check`` writes nothing and exits 3 when a failed run has no disposed entry. CI does not run it
  (whether it should is the owner's decision: it would turn CI red until every earlier failure is dealt with).

An entry keeps only metadata: the run id, attempt, workflow, event, branch, commit, start time, and the names of
the failed jobs and steps. Never log text: a failing test's output can quote the data it compared, so it stays on
GitHub.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

INBOX = Path("run-records") / "ci-inbox"
#: Failures before this were dealt with when they happened (fixed in later commits); the inbox starts here.
INTAKE_SINCE = "2026-11-13T00:00:00Z"
SCHEMA = "opengrad.ci-inbox/1"
RUN_FIELDS = "databaseId,attempt,workflowName,event,headBranch,headSha,createdAt,conclusion"
_RUN_ID = re.compile(r"[1-9][0-9]{0,19}")
_SHA = re.compile(r"[0-9a-f]{40}")

#: Runs ``gh`` with these arguments and returns its standard output (replaced in tests).
Gh = Callable[[list[str]], str]


def gh(args: list[str]) -> str:
    done = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=120, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args[:2])} failed: {done.stderr.strip()[:300]}")
    return done.stdout


def failed_runs(run_gh: Gh, since: str, limit: int) -> list[dict[str, Any]]:
    """Workflow runs whose latest attempt failed, started at or after ``since`` (ISO 8601, UTC)."""
    runs = json.loads(run_gh(["run", "list", "--status", "failure", "--limit", str(limit), "--json", RUN_FIELDS]))
    return [r for r in runs if r.get("conclusion") == "failure" and str(r.get("createdAt", "")) >= since]


def failed_jobs(run_gh: Gh, run_id: int) -> list[dict[str, Any]]:
    """The run's failed jobs, each with the names of its failed steps. Names only, never log text."""
    jobs = json.loads(run_gh(["run", "view", str(run_id), "--json", "jobs"])).get("jobs", [])
    return [
        {
            "name": str(job.get("name")),
            "failed_steps": [str(s.get("name")) for s in job.get("steps", []) if s.get("conclusion") == "failure"],
        }
        for job in jobs
        if job.get("conclusion") == "failure"
    ]


def draft(run: dict[str, Any], jobs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "ci_run": int(run["databaseId"]),
        "attempt": int(run.get("attempt") or 1),
        "workflow": str(run.get("workflowName")),
        "event": str(run.get("event")),
        "branch": str(run.get("headBranch")),
        "sha": str(run.get("headSha")),
        "created_at": str(run.get("createdAt")),
        "failed_jobs": jobs,
        "disposition": None,
    }


def entry_path(root: Path, run_id: int | str) -> Path:
    if not _RUN_ID.fullmatch(str(run_id)):
        raise ValueError(f"not a workflow run id: {run_id!r}")
    return root / INBOX / f"CI-{run_id}.json"


def entries(root: Path) -> dict[str, dict[str, Any]]:
    """File name -> entry, for every entry in the inbox."""
    return {p.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / INBOX).glob("*.json"))}


def disposition_errors(disposition: object, root: Path) -> list[str]:
    from opengrad.runrecords.cli import UPSTREAM_ID  # cli imports this module's callers lazily

    if not isinstance(disposition, dict) or not (disposition.get("upstream") or disposition.get("reason")):
        return ["needs a disposition: an upstream log id or a reason"]
    if disposition.get("upstream") and disposition.get("reason"):
        return ["give an upstream log id or a reason, not both"]
    if disposition.get("upstream"):
        entry = str(disposition["upstream"])
        if not UPSTREAM_ID.fullmatch(entry) or not (root / "docs" / "upstream" / f"{entry}.md").is_file():
            return [f"disposition names {entry!r}, which is not an entry in docs/upstream/"]
    elif not str(disposition["reason"]).strip():
        return ["the reason is empty"]
    return []


def validate_entry(name: str, entry: dict[str, Any], root: Path) -> list[str]:
    """Every problem with one inbox entry; empty when it is dealt with and may be committed."""
    from opengrad.runrecords.cli import ABSOLUTE, private_findings

    errors = []
    if entry.get("schema") != SCHEMA:
        errors.append(f"schema is not {SCHEMA}")
    if not _RUN_ID.fullmatch(str(entry.get("ci_run"))) or name != f"CI-{entry.get('ci_run')}.json":
        errors.append("file name does not match ci_run")
    if not _SHA.fullmatch(str(entry.get("sha"))):
        errors.append("sha is not a full commit id")
    errors += disposition_errors(entry.get("disposition"), root)
    text = json.dumps(entry, ensure_ascii=False)
    errors += [f"the entry contains {finding}" for finding in private_findings(text)]
    if ABSOLUTE.search(text):
        errors.append("the entry contains an absolute path")
    return errors


def _write(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def cmd_intake(root: Path, since: str, limit: int, check: bool, run_gh: Gh | None = None) -> int:
    """Write a draft for each new failed run (none with ``check``); exit 3 while any failed run is not dealt with."""
    run_gh = run_gh or gh
    try:
        runs = failed_runs(run_gh, since, limit)
        new = 0
        for run in runs:
            path = entry_path(root, run["databaseId"])
            if path.is_file() or check:
                continue
            _write(path, draft(run, failed_jobs(run_gh, int(run["databaseId"]))))
            new += 1
    except (RuntimeError, ValueError, KeyError, json.JSONDecodeError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"ci-intake: {exc}")
        return 1
    # Checked from GitHub's list (a failed run with no entry) and from the inbox (an entry with no disposition).
    missing = [str(r["databaseId"]) for r in runs if not entry_path(root, r["databaseId"]).is_file()]
    undealt = [name for name, entry in entries(root).items() if disposition_errors(entry.get("disposition"), root)]
    for run_id in missing:
        print(f"CI-{run_id}: failed, no entry (run without --check to take it in)")
    for name in undealt:
        entry = entries(root)[name]
        steps = "; ".join(f"{j['name']}: {', '.join(j['failed_steps']) or '?'}" for j in entry.get("failed_jobs", []))
        print(f"{name[:-5]}: {entry.get('branch')} {str(entry.get('sha'))[:12]}  {steps}  (needs a disposition)")
    print(f"{new} new draft(s); {len(missing) + len(undealt)} failed run(s) not dealt with")
    return 3 if missing or undealt else 0


def cmd_dispose(root: Path, run_id: str, upstream: str | None, reason: str | None) -> int:
    disposition = {"upstream": upstream} if upstream else {"reason": (reason or "").strip()}
    try:
        path = entry_path(root, run_id.removeprefix("CI-"))
    except ValueError as exc:
        print(exc)
        return 2
    if bool(upstream) == bool(reason) or (reason is not None and not reason.strip()):
        print("give exactly one of --upstream UP-NNNN or --reason TEXT")
        return 2
    if not path.is_file():
        print(f"no inbox entry {path.name}; run ci-intake first")
        return 1
    errors = disposition_errors(disposition, root)
    if errors:
        print("\n".join(errors))
        return 1
    entry = json.loads(path.read_text(encoding="utf-8"))
    entry["disposition"] = disposition
    errors = validate_entry(path.name, entry, root)
    if errors:
        print("\n".join(errors))
        return 1
    _write(path, entry)
    print(f"{path.name[:-5]}: disposition recorded")
    return 0
