"""Failed CI runs land in a draft inbox that has to be dealt with (run-record stage 4).

``python -m opengrad.runrecords ci-intake`` asks GitHub (read-only, through the ``gh`` CLI) for the workflow runs
started by a push to ``BRANCH`` (master; ``--branch`` names another) since ``INTAKE_SINCE`` whose latest attempt failed, timed out or could not start
(``FAILED``), and writes one draft per run attempt to ``run-records/ci-inbox/CI-<run id>-<attempt>.json``. A
failed re-run of a run already dealt with is a new attempt, so it gets its own draft. A draft has no
disposition. ``ci-dispose`` gives it one: the ``docs/upstream/`` entry the failure is logged under, or a reason
it needs none (a fix that landed, a flaky runner). Like a failed local run, a draft is dealt with only when it
has a disposition:

* ``ci-intake`` exits 3 while a failed run attempt has no entry, an entry has no disposition, or an inbox file
  cannot be read; the same "needs attention" code as ``upload``;
* ``tests/runrecords/test_ci_inbox.py`` fails CI on a committed entry without a valid disposition, so a draft
  cannot be committed until it is dealt with;
* ``ci-intake --check`` writes nothing and reports the same. CI runs it on every push (owner's decision,
  2026-11-14), so CI stays red until every earlier failure on master is dealt with: the commit that fixes a
  failure also commits its entry. Failures on other branches are work in progress and are not counted.

What it does not see: a failed attempt re-run to success before the intake runs (GitHub reports the latest
attempt only), and pull-request runs. Pull requests are left out on purpose: a pull request from a fork runs its
own workflow file, so its author chooses the job and step names an entry would copy.

An entry keeps only metadata: the run id, attempt, workflow, event, branch, commit, start time, and the names of
the failed jobs and steps. Never log text: a failing test's output can quote the data it compared, so it stays on
GitHub.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

INBOX = Path("run-records") / "ci-inbox"
#: The branch whose failures must be dealt with: the one everything lands on.
BRANCH = "master"
#: Failures before this were dealt with when they happened (fixed in later commits); the inbox starts here.
INTAKE_SINCE = "2026-11-13T00:00:00Z"
SCHEMA = "opengrad.ci-inbox/1"
#: Run and job conclusions that count as failed. ``cancelled`` is left out: a superseded run is cancelled on purpose.
FAILED = ("failure", "timed_out", "startup_failure")
RUN_FIELDS = "databaseId,attempt,workflowName,event,headBranch,headSha,createdAt,conclusion"
ENTRY_KEYS = frozenset(
    {"schema", "ci_run", "attempt", "workflow", "event", "branch", "sha", "created_at", "failed_jobs", "disposition"}
)
_RUN_ID = re.compile(r"[1-9][0-9]{0,19}")
_NAME = re.compile(r"CI-([1-9][0-9]{0,19})-([1-9][0-9]{0,3})\.json")
_SHA = re.compile(r"[0-9a-f]{40}")

#: Runs ``gh`` with these arguments and returns its standard output (replaced in tests).
Gh = Callable[[list[str]], str]


def gh(args: list[str]) -> str:
    done = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=120, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args[:2])} failed: {done.stderr.strip()[:300]}")
    return done.stdout


def utc(text: str) -> datetime:
    """An ISO 8601 time as an aware UTC datetime; ValueError for anything else (a date alone counts as midnight
    UTC). A time without an offset is refused rather than guessed."""
    value = datetime.fromisoformat(text)
    if len(text) > 10 and value.tzinfo is None:
        raise ValueError(f"{text!r} has no UTC offset; add Z or +HH:MM")
    return (value if value.tzinfo else value.replace(tzinfo=UTC)).astimezone(UTC)


def failed_runs(run_gh: Gh, since: str, limit: int, branch: str = BRANCH) -> list[dict[str, Any]]:
    """Runs triggered by a push to ``branch`` whose latest attempt is in ``FAILED``, started at or after ``since``. Raises when GitHub
    returns ``limit`` runs for a status, since older failures could then be missing."""
    start = utc(since)
    created = f">={start.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    found: dict[int, dict[str, Any]] = {}
    for status in FAILED:
        runs = json.loads(
            run_gh(
                ["run", "list", "--status", status, "--event", "push", "--branch", branch,
                 "--created", created, "--limit", str(limit), "--json", RUN_FIELDS]
            )
        )
        if len(runs) >= limit:
            raise RuntimeError(f"GitHub returned {limit} {status} runs, the limit; raise --limit so none is missed")
        for run in runs:
            if (
                run.get("conclusion") in FAILED
                and utc(str(run["createdAt"])) >= start
                and run.get("event") == "push"
                and run.get("headBranch") == branch
            ):
                found[int(run["databaseId"])] = run
    return [found[k] for k in sorted(found)]


def failed_jobs(run_gh: Gh, run_id: int) -> list[dict[str, Any]]:
    """The run's failed jobs, each with the names of its failed steps. Names only, never log text."""
    jobs = json.loads(run_gh(["run", "view", str(run_id), "--json", "jobs"])).get("jobs", [])
    return [
        {
            "name": str(job.get("name")),
            "failed_steps": [str(s.get("name")) for s in job.get("steps", []) if s.get("conclusion") in FAILED],
        }
        for job in jobs
        if job.get("conclusion") in FAILED
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


def entry_path(root: Path, run_id: int | str, attempt: int | str = 1) -> Path:
    name = f"CI-{run_id}-{attempt}.json"
    if not _NAME.fullmatch(name):
        raise ValueError(f"not a workflow run id and attempt: {run_id!r}, {attempt!r}")
    return root / INBOX / name


def entries(root: Path) -> dict[str, dict[str, Any] | None]:
    """File name -> entry, for every file in the inbox, whatever its name; None for one that is not a JSON object."""
    out: dict[str, dict[str, Any] | None] = {}
    folder = root / INBOX
    for path in sorted(folder.iterdir()) if folder.is_dir() else []:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            value = None
        out[path.name] = value if isinstance(value, dict) else None
    return out


def disposition_errors(disposition: object, root: Path) -> list[str]:
    # cli loads this module lazily, so importing it here makes no cycle.
    from opengrad.runrecords.cli import UPSTREAM_ID

    if not isinstance(disposition, dict) or not (disposition.get("upstream") or disposition.get("reason")):
        return ["needs a disposition: an upstream log id or a reason"]
    if set(disposition) not in ({"upstream"}, {"reason"}):
        return ["a disposition holds exactly one of upstream or reason"]
    value = disposition.get("upstream") or disposition.get("reason")
    if not isinstance(value, str) or not value.strip():
        return ["the disposition must be non-empty text"]
    if "upstream" in disposition and (
        not UPSTREAM_ID.fullmatch(value) or not (root / "docs" / "upstream" / f"{value}.md").is_file()
    ):
        return [f"disposition names {value!r}, which is not an entry in docs/upstream/"]
    return []


def validate_entry(name: str, entry: object, root: Path) -> list[str]:
    """Every problem with one inbox file; empty when it is dealt with and may be committed."""
    from opengrad.runrecords.cli import ABSOLUTE, private_findings

    if not isinstance(entry, dict):
        return ["not a readable JSON object"]
    errors = []
    if set(entry) != ENTRY_KEYS:
        errors.append(f"keys are not {sorted(ENTRY_KEYS)}")
    if entry.get("schema") != SCHEMA:
        errors.append(f"schema is not {SCHEMA}")
    if entry.get("event") != "push":
        errors.append("event is not push: only push-triggered runs are taken in")
    if not all(isinstance(entry.get(k), str) for k in ("workflow", "branch", "created_at")):
        errors.append("workflow, branch and created_at must be text")
    if name != f"CI-{entry.get('ci_run')}-{entry.get('attempt')}.json" or not _NAME.fullmatch(name):
        errors.append("file name does not match ci_run and attempt")
    if not _SHA.fullmatch(str(entry.get("sha"))):
        errors.append("sha is not a full commit id")
    jobs = entry.get("failed_jobs")
    if not isinstance(jobs, list) or not all(
        isinstance(j, dict)
        and set(j) == {"name", "failed_steps"}
        and isinstance(j["name"], str)
        and isinstance(j["failed_steps"], list)
        and all(isinstance(s, str) for s in j["failed_steps"])
        for j in jobs
    ):
        errors.append("failed_jobs is not a list of {name, failed_steps} names")
    errors += disposition_errors(entry.get("disposition"), root)
    text = json.dumps(entry, ensure_ascii=False)
    errors += [f"the entry contains {finding}" for finding in private_findings(text)]
    # Step names are commands from the public workflow file, which may name a scratch path: only the rest must hold
    # no path.
    if ABSOLUTE.search(json.dumps({k: v for k, v in entry.items() if k != "failed_jobs"}, ensure_ascii=False)):
        errors.append("the entry contains an absolute path")
    return errors


def _write(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def _summary(entry: dict[str, Any]) -> str:
    jobs = entry.get("failed_jobs")
    steps = "; ".join(
        f"{j.get('name')}: {', '.join(map(str, j.get('failed_steps') or [])) or '?'}"
        for j in (jobs if isinstance(jobs, list) else [])
        if isinstance(j, dict)
    )
    return f"{entry.get('branch')} {str(entry.get('sha'))[:12]}  {steps}"


def cmd_intake(
    root: Path, since: str, limit: int, check: bool, run_gh: Gh | None = None, branch: str = BRANCH
) -> int:
    """Write a draft for each new failed run attempt (none with ``check``); exit 3 while any is not dealt with."""
    run_gh = run_gh or gh
    try:
        utc(since)
    except ValueError as exc:
        print(f"ci-intake: --since: {exc}")
        return 2
    try:
        runs = failed_runs(run_gh, since, limit, branch)
        new = 0
        for run in runs:
            path = entry_path(root, run["databaseId"], run.get("attempt") or 1)
            if path.is_file() or check:
                continue
            _write(path, draft(run, failed_jobs(run_gh, int(run["databaseId"]))))
            new += 1
    except (RuntimeError, ValueError, KeyError, json.JSONDecodeError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"ci-intake: {exc}")
        return 1
    # Checked from GitHub's list (a failed attempt with no entry) and from the inbox (an entry not dealt with).
    missing = [
        f"CI-{r['databaseId']}-{r.get('attempt') or 1}"
        for r in runs
        if not entry_path(root, r["databaseId"], r.get("attempt") or 1).is_file()
    ]
    for name in missing:
        print(f"{name}: failed, no entry (run without --check to take it in)")
    undealt = 0
    for name, entry in entries(root).items():
        if entry is None:
            print(f"{name}: cannot be read as an inbox entry")
        elif disposition_errors(entry.get("disposition"), root):
            print(f"{name.removesuffix('.json')}: {_summary(entry)}  (needs a disposition)")
        else:
            continue
        undealt += 1
    print(f"{new} new draft(s); {len(missing) + undealt} failed run attempt(s) not dealt with")
    return 3 if missing or undealt else 0


def cmd_dispose(root: Path, run_id: str, attempt: int, upstream: str | None, reason: str | None) -> int:
    if bool(upstream) == bool(reason) or (reason is not None and not reason.strip()):
        print("give exactly one of --upstream UP-NNNN or --reason TEXT")
        return 2
    try:
        # The name intake prints, CI-<id>-<attempt>, is accepted as well as an id with --attempt.
        name = run_id.removeprefix("CI-")
        if "-" in name:
            name, _, given = name.partition("-")
            attempt = int(given) if given.isdigit() else -1
        path = entry_path(root, name, attempt)
    except ValueError as exc:
        print(exc)
        return 2
    if not path.is_file():
        print(f"no inbox entry {path.name}; run ci-intake first")
        return 1
    disposition = {"upstream": upstream} if upstream else {"reason": (reason or "").strip()}
    errors = disposition_errors(disposition, root)
    entry = entries(root).get(path.name)
    if entry is None:
        errors.append(f"{path.name} cannot be read as an inbox entry")
    else:
        entry = {**entry, "disposition": disposition}
        errors = errors or validate_entry(path.name, entry, root)
    if errors or entry is None:
        print("\n".join(errors))
        return 1
    _write(path, entry)
    print(f"{path.name.removesuffix('.json')}: disposition recorded")
    return 0
