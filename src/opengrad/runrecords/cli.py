"""Commands for run records: run, list, pending, dispose, export, upload.

    python -m opengrad.runrecords run --kind script -- python scripts/some_script.py --flag
    python -m opengrad.runrecords list
    python -m opengrad.runrecords pending                       # failed or killed runs with no disposition
    python -m opengrad.runrecords dispose RUN_ID --upstream UP-0031
    python -m opengrad.runrecords dispose RUN_ID --reason "Ctrl-C while testing the launcher"
    python -m opengrad.runrecords export                        # copy finished summaries into run-records/
    python -m opengrad.runrecords upload --repo OWNER/NAME      # owner's word only: durable copy of each log

``export`` refuses a run that did not pass and has no disposition, and ``tests/runrecords/test_committed.py`` fails
CI on any committed summary like that, or one that holds a path, a secret or an email address. ``upload`` is never
run by anything else; it refuses a log holding private details.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from opengrad.hashing import sha256_file
from opengrad.runrecords.record import (
    BEARER,
    COMMITTED_DIR,
    EMAIL,
    GOOGLE_KEY,
    HOME_PATH,
    KINDS,
    SECRET,
    STATUSES,
    URL_CREDENTIALS,
    _write_json,
    local_base,
    log_info,
    read_local,
    run_child,
)

UPSTREAM_ID = re.compile(r"UP-\d{4}")
#: An absolute path in a committed summary would leak the machine's layout. Repository-relative paths and
#: ``<absolute-path>`` placeholders pass. The summary is checked as JSON text, where a backslash is doubled.
ABSOLUTE = re.compile(
    r"(?:file:/+|(?<![\w.])[A-Za-z]:[\\/]|\\\\\\\\[^\\\"]|(?<![\w.\-<])/(?:home|Users|root|mnt|tmp|workspace|opt"
    r"|data|srv|var|private)/)"
)
#: A log or summary matching any of these must not leave the machine. The scanner's SECRET names known token
#: formats only, so emails (labelling transcripts have held one), home paths and other keys are added.
PRIVATE: dict[str, re.Pattern[str]] = {
    "a secret": SECRET,
    "an email address": EMAIL,
    "a home-folder path": HOME_PATH,
    "a Google API key": GOOGLE_KEY,
    "a bearer token": BEARER,
    "credentials in a URL": URL_CREDENTIALS,
}
#: Annotation runs talk to labelling models and have held private details, so their logs stay local unless the
#: owner names the kind explicitly. The email pattern also matches harmless text such as git@github.com: such a
#: log is refused, which is the safe direction.
UPLOAD_KINDS_DEFAULT = ("training", "evaluation", "freeze", "script", "ci")
REQUIRED = (
    "schema_version",
    "run_id",
    "kind",
    "command",
    "started_at",
    "git",
    "status",
    "log",
    "disposition",
)


def private_findings(text: str) -> list[str]:
    return [name for name, pattern in PRIVATE.items() if pattern.search(text)]


def needs_disposition(summary: dict[str, Any]) -> bool:
    return summary.get("status") != "PASS"


def validate_summary(summary: dict[str, Any], root: Path) -> list[str]:
    """Every problem with one committed summary; empty when it may be committed."""
    errors = [f"missing {key!r}" for key in REQUIRED if key not in summary]
    if errors:
        return errors
    if summary["kind"] not in KINDS:
        errors.append(f"kind {summary['kind']!r} is not one of {KINDS}")
    if summary["status"] not in STATUSES:
        errors.append(f"status {summary['status']!r} is not one of {STATUSES}")
    if not (summary.get("log") or {}).get("sha256") and summary["status"] != "INCOMPLETE":
        errors.append("log.sha256 is missing")
    disposition = summary["disposition"]
    if needs_disposition(summary):
        if not isinstance(disposition, dict) or not (
            disposition.get("upstream") or disposition.get("reason")
        ):
            errors.append(
                f"a {summary['status']} run needs a disposition: an upstream log id or a reason"
            )
        elif disposition.get("upstream"):
            entry = str(disposition["upstream"])
            if (
                not UPSTREAM_ID.fullmatch(entry)
                or not (root / "docs" / "upstream" / f"{entry}.md").is_file()
            ):
                errors.append(
                    f"disposition names {entry!r}, which is not an entry in docs/upstream/"
                )
    text = json.dumps(summary, ensure_ascii=False)
    errors += [f"the summary contains {finding}" for finding in private_findings(text)]
    if ABSOLUTE.search(text):
        errors.append("the summary contains an absolute path")
    return errors


def _local_dir(root: Path, run_id: str) -> Path:
    path = local_base(root) / run_id
    if not re.fullmatch(r"[0-9A-Za-z-]+", run_id) or not path.is_dir():
        raise SystemExit(f"no local run record {run_id!r}")
    return path


def _summary(root: Path, run_id: str) -> tuple[Path, dict[str, Any]]:
    """The run's summary. A killed run (start, no summary) gets one now, as INCOMPLETE, so it can be disposed."""
    directory = _local_dir(root, run_id)
    path = directory / "summary.json"
    if path.is_file():
        return path, json.loads(path.read_text(encoding="utf-8"))
    start_path = directory / "start.json"
    if not start_path.is_file():
        raise SystemExit(
            f"{run_id} has neither a start record nor a summary; remove the folder by hand"
        )
    start = json.loads(start_path.read_text(encoding="utf-8"))
    summary = {
        **start,
        "ended_at": None,
        "duration_s": None,
        "exit_code": None,
        "status": "INCOMPLETE",
        "error": "no summary: the run was killed or is still running",
        "log": log_info(directory / "output.log", root, run_id),
        "disposition": None,
    }
    return path, summary


def _say(text: str) -> None:
    """Print what the console can show; a recorded command may hold characters a Windows console cannot."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def cmd_list(root: Path) -> int:
    for record in read_local(root):
        undisposed = needs_disposition(record) and not record.get("disposition")
        note = "  (needs disposition)" if undisposed else ""
        _say(
            f"{record['run_id']}  {record['status']:<11}  {' '.join(record['command'])[:80]}{note}"
        )
    return 0


def cmd_pending(root: Path) -> int:
    pending = [r for r in read_local(root) if needs_disposition(r) and not r.get("disposition")]
    for record in pending:
        _say(f"{record['run_id']}  {record['status']}  {' '.join(record['command'])[:80]}")
    print(f"{len(pending)} run(s) need a disposition")
    return 1 if pending else 0


def cmd_dispose(root: Path, run_id: str, upstream: str | None, reason: str | None) -> int:
    if bool(upstream) == bool(reason) or (reason is not None and not reason.strip()):
        print("give exactly one of --upstream UP-NNNN or --reason TEXT")
        return 2
    path, summary = _summary(root, run_id)
    summary["disposition"] = {"upstream": upstream} if upstream else {"reason": str(reason).strip()}
    errors = [e for e in validate_summary(summary, root) if "disposition" in e]
    if errors:
        print("\n".join(errors))
        return 1
    _write_json(path, summary)
    committed = root / COMMITTED_DIR / f"{run_id}.json"
    if committed.is_file():
        _write_json(committed, summary)
    print(f"{run_id}: disposition recorded")
    return 0


def cmd_export(root: Path) -> int:
    exported = blocked = 0
    for record in read_local(root):
        if record["status"] == "INCOMPLETE" and not record.get("disposition"):
            continue
        errors = validate_summary(record, root)
        if errors:
            blocked += 1
            print(f"not exported {record['run_id']}: " + "; ".join(errors))
            continue
        target = root / COMMITTED_DIR / f"{record['run_id']}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_json(target, record)
        exported += 1
    print(f"exported {exported}, blocked {blocked}")
    return 1 if blocked else 0


def cmd_upload(root: Path, repo: str, kinds: tuple[str, ...]) -> int:
    """A durable copy of each log not yet uploaded, to a private dataset repository the owner names."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    sent = 0
    for record in read_local(root):
        if (
            record["status"] == "INCOMPLETE"
            or record["log"].get("durable")
            or record["kind"] not in kinds
        ):
            continue
        summary_path, summary = _summary(root, record["run_id"])
        log = local_base(root) / record["run_id"] / "output.log"
        data = log.read_bytes()
        found = private_findings(data.decode("utf-8", errors="replace"))
        if found:
            print(
                f"not uploaded {record['run_id']}: the log contains {', '.join(found)}; it stays local"
            )
            continue
        info = api.upload_file(
            path_or_fileobj=data,
            path_in_repo=f"{record['run_id']}/output.log",
            repo_id=repo,
            repo_type="dataset",
            commit_message=f"Run log {record['run_id']}",
        )
        fetched = Path(
            hf_hub_download(
                repo, f"{record['run_id']}/output.log", repo_type="dataset", revision=info.oid
            )
        )
        if sha256_file(fetched) != summary["log"]["sha256"]:
            print(f"FAIL: {record['run_id']} came back different from the local log")
            return 1
        summary["log"]["durable"] = {"repo": repo, "revision": info.oid}
        _write_json(summary_path, summary)
        committed = root / COMMITTED_DIR / f"{record['run_id']}.json"
        if committed.is_file():
            _write_json(committed, summary)
        sent += 1
        print(f"uploaded {record['run_id']} @ {info.oid} (re-fetched, bytes match)")
    print(f"{sent} log(s) uploaded")
    return 0


def main(argv: list[str] | None = None, root: Path | None = None) -> int:
    root = root or Path.cwd()
    parser = argparse.ArgumentParser(
        prog="python -m opengrad.runrecords", description=(__doc__ or "").split("\n")[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run a command as a recorded child process")
    run_p.add_argument("--kind", choices=KINDS, default="script")
    run_p.add_argument("child", nargs=argparse.REMAINDER, help="-- then the command")
    sub.add_parser("list", help="list local run records")
    sub.add_parser(
        "pending", help="runs that did not pass and still need a disposition (exit 1 if any)"
    )
    dis_p = sub.add_parser("dispose", help="record what a failed or killed run was")
    dis_p.add_argument("run_id")
    dis_p.add_argument("--upstream", help="the docs/upstream entry it is logged under, UP-NNNN")
    dis_p.add_argument("--reason", help="why it needs no entry")
    sub.add_parser("export", help="copy finished, valid summaries into run-records/ for committing")
    up_p = sub.add_parser(
        "upload", help="owner's word only: durable copy of each log to a private dataset repo"
    )
    up_p.add_argument(
        "--repo", required=True, help="OWNER/NAME of an existing private dataset repository"
    )
    up_p.add_argument(
        "--kind",
        dest="kinds",
        action="append",
        choices=KINDS,
        help="upload only these kinds (repeatable); default: every kind except annotation",
    )
    args = parser.parse_args(argv)
    if args.command == "run":
        child = args.child[1:] if args.child[:1] == ["--"] else args.child
        if not child:
            parser.error("give the command after --")
        return run_child(args.kind, child, root)
    if args.command == "list":
        return cmd_list(root)
    if args.command == "pending":
        return cmd_pending(root)
    if args.command == "dispose":
        return cmd_dispose(root, args.run_id, args.upstream, args.reason)
    if args.command == "export":
        return cmd_export(root)
    return cmd_upload(root, args.repo, tuple(args.kinds or UPLOAD_KINDS_DEFAULT))


if __name__ == "__main__":
    sys.exit(main())
