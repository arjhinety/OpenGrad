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
run by anything else, and only to a repository the Hub reports as private. Upload is an allowlist
(``opengrad.runrecords.scripts.upload_decision``): only a log whose entry point was reviewed and approved, and
whose run started from the reviewed code, is sent; a run that read held-out data, or whose command cannot be
traced, is never sent. A log holding a credential or a home folder (``private_findings``), and every annotation
log, is sent only as a redacted copy; any other goes byte for byte. The summary records which copy was sent, why,
and the sha256 of both. The committed summary of a run whose log stays local, or of a pinned one, keeps only the
exception type of its error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from opengrad.hashing import sha256_bytes, sha256_file
from opengrad.runrecords.record import (
    BASIC_AUTH,
    BEARER,
    COMMITTED_DIR,
    DASHED_KEY,
    EMAIL,
    EMAIL_ENCODED,
    GOOGLE_KEY,
    HOME_ENCODED,
    HOME_PATH,
    HOME_WITH_NAME,
    KINDS,
    PEM_BLOCK,
    SECRET,
    STATUSES,
    URL_CREDENTIALS,
    _write_json,
    has_secret_field,
    has_secret_flag_value,
    local_base,
    log_info,
    read_local,
    redact_text,
    run_child,
)
from opengrad.runrecords.scripts import PINNED_CONSOLE, command_entry, upload_decision

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
    "a private key": PEM_BLOCK,
    "an API key": DASHED_KEY,
    "basic-auth credentials": BASIC_AUTH,
    "an encoded email address": EMAIL_ENCODED,
}
#: Every kind is considered for upload (owner's decision, 2026-11-11); which logs go is upload_decision's allowlist
#: (owner's decision, 2026-11-13).
UPLOAD_KINDS_DEFAULT = KINDS
#: Annotation runs talk to labelling models and have held private details, so a log of theirs that uploads goes as
#: a redacted copy, whether or not a pattern matches.
ALWAYS_REDACTED_KINDS = ("annotation",)
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


#: The longest line the patterns see, as a safety net. Matching time grows about linearly with a line's length (about
#: 1 s for 64 KB in the worst shape measured), so ordinary long lines (config and JSON dumps) are matched in full. A
#: longer line is not matched at all: it counts as a finding and the redacted copy replaces it whole.
MAX_LINE = 100_000
_LINE_BREAKS = re.compile(r"(\r\n|\n|\r)")


#: Terminal colour and cursor codes: CSI sequences, character-set selections (``ESC ( B`` from ``tput sgr0``) and
#: the two-character cursor codes. Logging libraries put a colour code right before a value, where it would hide a
#: token from patterns that start at a word boundary. Escape strings that carry text (window titles, OSC 8
#: hyperlinks, DCS, SOS, PM, APC) are left in place, so their text is still checked; a token right after a DCS,
#: SOS or APC introducer is glued to the introducer's character (``P``, ``X``, ``_``) and missed, like any token
#: glued to a preceding letter.
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b[()*+][0-~]|\x1b[78=>DEHMc]")


def _bounded(text: str) -> tuple[str, bool]:
    """The text with every line longer than ``MAX_LINE`` replaced, and whether any was."""
    parts = _LINE_BREAKS.split(text)
    cut = False
    for i in range(0, len(parts), 2):
        if len(parts[i]) > MAX_LINE:
            parts[i] = f"<line of {len(parts[i])} characters removed>"
            cut = True
    return "".join(parts), cut


def private_findings(text: str) -> list[str]:
    """What in the text must not leave the machine as is. An ordinary path alone is not on the list (owner's
    choice C, 2026-11-11): a home folder is, in any form, and so is any credential the redactor would hide."""
    found = [name for name, pattern in PRIVATE.items() if pattern.search(text)]
    if (HOME_WITH_NAME.search(text) or HOME_ENCODED.search(text)) and "a home-folder path" not in found:
        found.append("a home-folder path")
    if has_secret_field(text):
        found.append("a credential field")
    if has_secret_flag_value(text):
        found.append("a credential flag value")
    return found


def log_findings(text: str) -> list[str]:
    """``private_findings`` for a log: over-long lines are not matched but are a finding themselves."""
    bounded, cut = _bounded(ANSI.sub("", text))
    return private_findings(bounded) + (["an over-long line"] if cut else [])


def redacted_log(text: str, root: Path) -> str:
    """A log with every private detail replaced: the command redaction (secrets, credential fields, URL
    credentials, bearer tokens, Google keys, absolute paths) applied line by line, then email addresses and home
    paths. Harmless text such as ``git@github.com`` is replaced too, which is the safe direction."""
    # Colour codes go first, so they hide nothing; then a key block, which spans lines, over the whole text.
    text = PEM_BLOCK.sub("<redacted private key>", ANSI.sub("", text))
    text, _ = _bounded(text)
    lines = [redact_text(line, root) for line in text.split("\n")]
    out = EMAIL_ENCODED.sub("<email>", EMAIL.sub("<email>", "\n".join(lines)))
    return HOME_PATH.sub("<absolute-path>/", out)


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
        _write_json(committed, _exportable(summary, root))
    print(f"{run_id}: disposition recorded")
    return 0


def _exportable(record: dict[str, Any], root: Path) -> dict[str, Any]:
    """The summary as committed, by ``export``, ``dispose`` and ``upload`` alike. A run whose log stays local, or
    whose entry point reaches held-out data (``PINNED_CONSOLE``), keeps only its exception type: the message could
    quote held-out data, and redaction removes private details, not test content."""
    decision, _ = upload_decision(record, root)
    error = record.get("error")
    command = [str(t) for t in record.get("command") or []]
    pinned = command_entry(command, root) in PINNED_CONSOLE
    if (decision == "allowed" and not pinned) or not error:
        return record
    kept = {
        "never": "held-out data",
        "unrecognised": "command not recognised",
        "unapproved": "not approved for upload",
    }.get(decision, "held-out data")
    head = str(error).split(":", 1)[0].strip()
    return {**record, "error": f"{head} (message kept local: {kept})"}


def cmd_export(root: Path) -> int:
    exported = blocked = 0
    for record in read_local(root):
        if record["status"] == "INCOMPLETE" and not record.get("disposition"):
            continue
        record = _exportable(record, root)
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


def _report_kept(kept: list[str]) -> bool:
    """List the logs that stayed on this machine only; True when there are any."""
    if kept:
        print(f"INCOMPLETE: {len(kept)} log(s) stay on this machine only, with no durable copy:")
        for run_id in kept:
            print(f"  {run_id}")
    return bool(kept)


def cmd_upload(root: Path, repo: str, kinds: tuple[str, ...]) -> int:
    """A durable copy of each log not yet uploaded, to a private dataset repository the owner names.

    A log with no private detail is sent byte for byte. A log holding one, and every annotation log, is sent as a
    redacted copy (``redacted_log``). It is checked again after redaction and kept local if anything still matches.
    Each upload is downloaded again and compared with the bytes that were sent. Exit codes: 0 every eligible log
    uploaded; 1 a log changed since its run or came back different; 2 the repository was refused; 3 some logs stayed
    local, so they have no durable copy. Logs kept local are listed on exit 0, 3 and 1 (on exit 1, those seen
    before the failure).

    ``upload_decision`` is applied first: a log from a run that read held-out data is never sent and is listed as
    kept by policy (not counted for exit 3); a log whose command cannot be traced to an entry point is never sent
    and is listed (counted for exit 3, since a mapping would give it a backup); so is a log whose entry point is
    not approved for upload yet (an allowlist, owner's decision 2026-11-13). The kind it is uploaded as comes
    from the entry point's policy, so a labelling run launched through ``run --kind script`` is still redacted.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    # A typo in the name, or a repository that is (or later becomes) public, would publish every log: refuse
    # anything the Hub does not report as an existing private dataset repository.
    try:
        visibility = api.repo_info(repo, repo_type="dataset")
    except Exception as exc:  # noqa: BLE001 -- a missing repository or no access: nothing is sent
        print(
            f"BLOCKED: cannot read dataset repository {repo!r} ({type(exc).__name__}); nothing uploaded"
        )
        return 2
    if getattr(visibility, "private", None) is not True:
        print(f"BLOCKED: dataset repository {repo!r} is not private; nothing uploaded")
        return 2
    sent = 0
    kept: list[str] = []
    by_policy: list[str] = []
    unrecognised: list[str] = []
    unapproved: list[str] = []

    def report() -> bool:
        """List every log left on this machine; True when one has no durable copy and should have."""
        if by_policy:
            print(f"{len(by_policy)} log(s) kept local by policy (held-out data), not counted as missing:")
            for run_id in by_policy:
                print(f"  {run_id}")
        if unrecognised:
            print(
                f"{len(unrecognised)} log(s) kept local: the command is not a known entry point, so it cannot be "
                "checked for held-out data (add it in opengrad.runrecords.scripts):"
            )
            for run_id in unrecognised:
                print(f"  {run_id}")
        if unapproved:
            print(
                f"{len(unapproved)} log(s) kept local, not approved for upload: the entry point is not approved "
                "(review its console output, then add it in opengrad.runrecords.scripts), or the run did not start "
                "from reviewed code (push its commit, or git fetch, and upload again; a run with edited code or "
                "configs is not uploaded):"
            )
            for run_id in unapproved:
                print(f"  {run_id}")
        return _report_kept(kept) or bool(unrecognised) or bool(unapproved)

    for record in read_local(root):
        if record["status"] == "INCOMPLETE" or record["log"].get("durable"):
            continue
        decision, kind = upload_decision(record, root)
        if kind not in kinds:  # the policy's kind: a labelling run launched as `run --kind script` is annotation
            continue
        if decision == "never":
            # The run read held-out data (opengrad.runrecords.scripts): its log stays here by policy.
            by_policy.append(record["run_id"])
            continue
        if decision == "unrecognised":
            # Owner's decision, 2026-11-13: a command that cannot be checked for held-out data stays local.
            unrecognised.append(record["run_id"])
            continue
        if decision == "unapproved":
            # Owner's decision, 2026-11-13: only a reviewed, approved entry point uploads.
            unapproved.append(record["run_id"])
            continue
        summary_path, summary = _summary(root, record["run_id"])
        log = local_base(root) / record["run_id"] / "output.log"
        original = log.read_bytes()
        if sha256_bytes(original) != summary["log"]["sha256"]:
            print(f"FAIL: {record['run_id']}: the local log changed since the run; nothing more uploaded")
            report()
            return 1
        try:
            text = original.decode("utf-8")
        except UnicodeDecodeError:
            text = "\x00"
        if "\x00" in text:
            print(f"not uploaded {record['run_id']}: not clean UTF-8 text, so it cannot be checked; it stays local")
            kept.append(record["run_id"])
            continue
        found = log_findings(text)
        # Owner's choice C: a redacted copy only when something private is detected, or for an annotation log.
        # A log whose only absolute paths are ordinary ones is sent byte for byte.
        redact = bool(found) or kind in ALWAYS_REDACTED_KINDS
        data = redacted_log(text, root).encode("utf-8") if redact else original
        left = ", ".join(log_findings(data.decode("utf-8"))) if redact else ""
        if left:
            print(f"not uploaded {record['run_id']}: {left} survived redaction; it stays local")
            kept.append(record["run_id"])
            continue
        name = "output.redacted.log" if redact else "output.log"
        info = api.upload_file(
            path_or_fileobj=data,
            path_in_repo=f"{record['run_id']}/{name}",
            repo_id=repo,
            repo_type="dataset",
            commit_message=f"Run log {record['run_id']}" + (" (redacted copy)" if redact else ""),
        )
        fetched = Path(
            hf_hub_download(
                repo, f"{record['run_id']}/{name}", repo_type="dataset", revision=info.oid
            )
        )
        if sha256_file(fetched) != sha256_bytes(data):
            print(f"FAIL: {record['run_id']} came back different from what was sent")
            report()
            return 1
        summary["log"]["durable"] = {
            "repo": repo,
            "revision": info.oid,
            "path": f"{record['run_id']}/{name}",
            "copy": "redacted" if redact else "original",
            "sha256": sha256_bytes(data),
            "redacted_because": found
            or (["annotation run"] if kind in ALWAYS_REDACTED_KINDS else []),
        }
        _write_json(summary_path, summary)
        committed = root / COMMITTED_DIR / f"{record['run_id']}.json"
        if committed.is_file():
            _write_json(committed, _exportable(summary, root))
        sent += 1
        kind = "redacted copy" if redact else "original"
        print(f"uploaded {record['run_id']} ({kind}) @ {info.oid} (re-fetched, bytes match)")
    print(f"{sent} log(s) uploaded")
    # A log kept local has no backup, so the run does not report success (exit 3); held-out logs are by design.
    return 3 if report() else 0


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
    ci_p = sub.add_parser(
        "ci-intake", help="draft an inbox entry for each failed CI run (exit 3 while any is not dealt with)"
    )
    ci_p.add_argument("--since", default=None, help="ISO time; default: the inbox's start")
    ci_p.add_argument("--limit", type=int, default=100, help="how many failed runs to ask GitHub for")
    ci_p.add_argument("--check", action="store_true", help="write nothing, only report")
    cid_p = sub.add_parser("ci-dispose", help="record what a failed CI run was")
    cid_p.add_argument("ci_run", help="the workflow run id, or CI-<id>")
    cid_p.add_argument("--upstream", help="the docs/upstream entry it is logged under, UP-NNNN")
    cid_p.add_argument("--reason", help="why it needs no entry")
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
        help="upload only these kinds (repeatable); default: every kind",
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
    if args.command in ("ci-intake", "ci-dispose"):
        from opengrad.runrecords import ci_intake

        if args.command == "ci-dispose":
            return ci_intake.cmd_dispose(root, args.ci_run, args.upstream, args.reason)
        return ci_intake.cmd_intake(root, args.since or ci_intake.INTAKE_SINCE, args.limit, args.check)
    return cmd_upload(root, args.repo, tuple(args.kinds or UPLOAD_KINDS_DEFAULT))


if __name__ == "__main__":
    sys.exit(main())
