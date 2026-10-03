"""Archive the audit trail of the three external model annotators of P-DET-COVERAGE-v1 (34 §3).

``scripts/run_external_annotation.py`` ran Gemini 3.8 Flash (High) (``agy``), gpt-5.6-sol (``codex``) and
deepseek-v4.1-flash (``cline``) over both coverage tasks, each in an isolated temporary directory, and kept
every batch, raw answer and run record in the git-ignored ``.annotation/<task>.model-batches/<session>/``.
This script packages that trail, byte for byte, into a new artifact per task:

    reports/pdet-coverage/provenance/external-models/
      <task>.external-models.audit-trail.tar.gz          batches, answers, last messages, run records, ingest logs,
                                                         procedure and runner source
      <task>.external-models.audit-trail.manifest.json   every member's hash, per-batch attempts, CLI versions
      *.sha256                                           sidecars
      local/<task>.external-models.cli-streams.tar.gz    raw CLI stdout and stderr (NOT tracked)

The raw CLI streams stay out of version control: they are unfiltered tool logs (DeepSeek's alone is tens of
megabytes) that may carry account or session details. Their hashes are recorded in the tracked manifest and
in each run record. Before archiving, every file was scanned for the operator's e-mail address and common
credential patterns (none found, 2026-09-17).

For each batch the audit checks: the batch content hash re-derives; every attempt's recorded stdout and
stderr hashes match the files; the attempt that recorded labels has an answers file covering exactly the
batch's items; and each session's ingest log is an intact hash chain with one entry per labelled item.
Batches that were prepared but never run (a run interrupted between preparing and calling the CLI) are kept
and listed as such. It prints counts and hashes only, never item text or answers.

    python scripts/archive_external_model_labels.py --task pdet-coverage-v1
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from archive_pdet_model_batches import canonical_json, listing, sha256, tar_gz, write_new

from opengrad.annotation.provenance import ANNOTATION_ENTRY_FIELDS, verify_chain

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "pdet-coverage" / "provenance" / "external-models"
#: task -> (output directory, procedure, authorizing document). P-DET-COVERAGE-v2 (36) is archived beside its
#: own population.
TASK_SPECS = {
    "pdet-coverage-v1": (OUT, "configs/annotation/pdet-coverage-v1.model-procedure.md", "docs/research/study-002/34-PDET-COVERAGE-MODEL-CONSENSUS-AMENDMENT.md"),
    "pdet-coverage-v1-routing": (
        OUT,
        "configs/annotation/pdet-coverage-v1.model-procedure.md",
        "docs/research/study-002/34-PDET-COVERAGE-MODEL-CONSENSUS-AMENDMENT.md",
    ),
    "pdet-coverage-v2": (
        ROOT / "reports" / "pdet-coverage-v2" / "provenance" / "external-models",
        "configs/annotation/pdet-coverage-v2.model-procedure.md",
        "docs/research/study-002/36-FIRST-REPLY-CONTRACT-AND-PDET-COVERAGE-V2-DRAFT.md",
    ),
    # 41 as amended by 42 (study_002_prereg_v10): two annotators. gpt-5.6-sol's session is archived too: its one
    # prepared batch and three refused attempts are the evidence for 42 section 1.
    "answer-strata-v1": (
        ROOT / "reports" / "study-002" / "answer-strata-v1" / "provenance" / "external-models",
        "configs/annotation/answer-strata-v1.model-procedure.md",
        "docs/research/study-002/42-ANSWER-STRATA-TWO-MODEL-AMENDMENT.md",
    ),
    # 43 (study_002_prereg_v11): the two annotators of 42.
    "punans-v1": (
        ROOT / "reports" / "study-002" / "punans-v1" / "provenance" / "external-models",
        "configs/annotation/punans-v1.model-procedure.md",
        "docs/research/study-002/43-PUNANS-AMENDMENT-DRAFT.md",
    ),
    # 44 (study_002_prereg_v12): the same two annotators, on the trial set and then the main set.
    "punans-v2-trial": (
        ROOT / "reports" / "study-002" / "punans-v2" / "provenance" / "external-models",
        "configs/annotation/punans-v2.model-procedure.md",
        "docs/research/study-002/44-PUNANS-V2-AMENDMENT-DRAFT.md",
    ),
    "punans-v2": (
        ROOT / "reports" / "study-002" / "punans-v2" / "provenance" / "external-models",
        "configs/annotation/punans-v2.model-procedure.rev1.md",
        "docs/research/study-002/44-PUNANS-V2-AMENDMENT-DRAFT.md",
    ),
    # 45 (study_002_prereg_v13): the constructed questions and controls, under 44's revised procedure.
    "punans-v2-constructed": (
        ROOT / "reports" / "study-002" / "punans-v2-constructed" / "provenance" / "external-models",
        "configs/annotation/punans-v2.model-procedure.rev1.md",
        "docs/research/study-002/45-PUNANS-CONSTRUCTED-AMENDMENT-DRAFT.md",
    ),
    # 46 §5 and 47 (study_002_prereg_v15): the flag-set triage, under neutral task ids.
    "first-reply-review-v1-trial": (
        ROOT / "reports" / "study-002" / "flag-triage" / "provenance" / "external-models",
        "configs/annotation/first-reply-review-v1.model-procedure.md",
        "docs/research/study-002/47-PER-SOURCE-FLAG-PRECISION-DRAFT.md",
    ),
    "first-reply-review-v1": (
        ROOT / "reports" / "study-002" / "flag-triage" / "provenance" / "external-models",
        "configs/annotation/first-reply-review-v1.model-procedure.md",
        "docs/research/study-002/47-PER-SOURCE-FLAG-PRECISION-DRAFT.md",
    ),
    "first-reply-review-v1-sample": (
        ROOT / "reports" / "study-002" / "flag-triage" / "provenance" / "external-models",
        "configs/annotation/first-reply-review-v1.model-procedure.md",
        "docs/research/study-002/47-PER-SOURCE-FLAG-PRECISION-DRAFT.md",
    ),
}
#: The status sentence of the archive, where it is not the three-model one of 34.
STATUS = {
    "answer-strata-v1": (
        "Model judgments by two declared non-Claude annotators, each blind and independent; an item's reference "
        "label is the label both give ({authorization}). gpt-5.6-sol, declared by 41, labelled nothing: Codex "
        "refused the model for this account. Not human labels and not human gold."
    ),
    "punans-v1": (
        "Model judgments by two declared non-Claude annotators, each blind and independent; an item's reference "
        "label is the label both give ({authorization}). Not human labels and not human gold."
    ),
    "punans-v2-trial": (
        "Model judgments by two declared non-Claude annotators, each blind and independent, on the trial set, "
        "which never enters P-UNANS ({authorization} section 9). Not human labels and not human gold."
    ),
    "punans-v2": (
        "Model judgments by two declared non-Claude annotators, each blind and independent; an item's reference "
        "label is the label both give ({authorization}). Not human labels and not human gold."
    ),
    "punans-v2-constructed": (
        "Model judgments by two declared non-Claude annotators, each blind and independent, on constructed "
        "questions and answerable controls; an item's reference label is the label both give ({authorization}). "
        "Not human labels and not human gold."
    ),
    "first-reply-review-v1-trial": (
        "Model judgments by two declared non-Claude annotators, each blind and independent, on the flag-set "
        "triage's 100-record trial, which counts toward no floor ({authorization}). Not human labels and not "
        "human gold."
    ),
    "first-reply-review-v1": (
        "Model judgments by two declared non-Claude annotators, each blind and independent, on every record of "
        "the flag set; both labels are kept, and the triage's floors are computed from them ({authorization}). "
        "Not human labels and not human gold."
    ),
    "first-reply-review-v1-sample": (
        "Model judgments by two declared non-Claude annotators, each blind and independent, on the recall "
        "sample of unflagged replies; both labels are kept, and the recall share is computed from them "
        "({authorization}). Not human labels and not human gold."
    ),
}
#: Credential shapes that must never reach an archive (tracked or not).
CREDENTIALS = re.compile(rb"sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AIza[0-9A-Za-z_-]{30,}")
TASKS = tuple(TASK_SPECS)
SESSIONS = {
    "model-gemini": "model.gemini-3.8-flash-high",
    # The tool-gated re-label of INC-0002 (docs/INCIDENT_LOG.md), archived as its own variant.
    "model-gemini-r2": "model.gemini-3.8-flash-high",
    "model-gpt": "model.gpt-5.6-sol",
    "model-deepseek": "model.deepseek-v4.1-flash",
}
RUNNER = "scripts/run_external_annotation.py"
SCHEMA = "opengrad-external-model-label-audit-trail-v1"
STREAM = re.compile(r"batch-\d+\.attempt-\d+\.(stdout|stderr)\.txt")
ATTEMPT = re.compile(r"batch-(\d+)\.attempt-(\d+)\.run\.json")


def history(state: Path, task: str, session: str) -> list[dict[str, Any]]:
    with sqlite3.connect(f"file:{state.as_posix()}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        entries = []
        for row in conn.execute("SELECT * FROM history WHERE task_id = ? AND session_id = ? ORDER BY id", (task, session)):
            entry = dict(row)
            entry["before"] = json.loads(entry.pop("before_json")) if row["before_json"] else None
            entry["after"] = json.loads(entry.pop("after_json")) if row["after_json"] else None
            entries.append(entry)
    return entries


def audit_session(task: str, session: str, members: dict[str, bytes], streams: dict[str, bytes]) -> dict[str, Any]:
    directory = ROOT / ".annotation" / f"{task}.model-batches" / session
    batches: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.iterdir()):
        data = path.read_bytes()
        (streams if STREAM.fullmatch(path.name) else members)[f"{session}/{path.name}"] = data
        if re.fullmatch(r"batch-\d+\.json", path.name):
            batch = json.loads(data)
            content = sha256(canonical_json({"items": batch["items"], "instructions": batch["instructions"]}).encode("utf-8"))
            if content != batch["content_sha256"]:
                raise SystemExit(f"{task}/{session}/{path.name}: batch content no longer matches its hash")
            batches[batch["batch_id"]] = {
                "batch_id": batch["batch_id"],
                "items": len(batch["item_ids"]),
                "item_ids": batch["item_ids"],
                "content_sha256": batch["content_sha256"],
                "procedure_sha256": batch["procedure_sha256"],
                "attempts": [],
            }
    for path in sorted(directory.glob("batch-*.attempt-*.run.json")):
        match = ATTEMPT.fullmatch(path.name)
        if not match:
            continue
        run = json.loads(path.read_bytes())
        stem = path.name.removesuffix(".run.json")
        for stream in ("stdout", "stderr"):
            recorded = run.get(f"{stream}_sha256")
            file = directory / f"{stem}.{stream}.txt"
            if recorded is not None and (not file.is_file() or sha256(file.read_bytes()) != recorded):
                raise SystemExit(f"{task}/{session}/{stem}: {stream} does not match its recorded hash")
        answers_file = directory / f"{stem}.answers.json"
        recorded_labels = bool(run.get("ingested"))
        batch = batches[run["batch_id"]]
        if recorded_labels:
            answers = json.loads(answers_file.read_bytes())
            if sorted(a["item_id"] for a in answers) != sorted(batch["item_ids"]):
                raise SystemExit(f"{task}/{session}/{stem}: recorded answers do not cover exactly the batch items")
        batch["attempts"].append(
            {
                "attempt": run["attempt"],
                "cli_version": run.get("cli_version"),
                "exit_code": run.get("exit_code"),
                "seconds": run.get("seconds"),
                "status": run.get("status"),
                "answers_found": run.get("answers_found"),
                "recorded_labels": run.get("ingested", 0),
                "files_created_in_isolated_dir": run.get("files_created_in_isolated_dir"),
                "prompt_sha256": run.get("prompt_sha256"),
                "stdout_sha256": run.get("stdout_sha256"),
                "stderr_sha256": run.get("stderr_sha256"),
                "answers_file_sha256": sha256(answers_file.read_bytes()) if answers_file.is_file() else None,
                "web_tool_calls": web_tool_calls(session, run, directory / f"{stem}.stdout.txt"),
            }
        )
    entries = history(ROOT / ".annotation" / f"{task}.sqlite3", task, session)
    problems = verify_chain(entries, ANNOTATION_ENTRY_FIELDS, session)
    if problems:
        raise SystemExit(f"{task}/{session}: the ingest log is not an intact chain: {problems[:3]}")
    members[f"ingest/{session}.history.jsonl"] = "".join(
        json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n" for entry in entries
    ).encode("utf-8")
    rows = []
    for batch in sorted(batches.values(), key=lambda b: b["batch_id"]):
        recorded = sum(attempt["recorded_labels"] or 0 for attempt in batch["attempts"])
        rows.append(
            {
                **{k: v for k, v in batch.items() if k != "item_ids"},
                "outcome": (
                    "recorded"
                    if recorded
                    else "prepared, never run (interrupted)"
                    if not batch["attempts"]
                    else "no attempt recorded labels"
                ),
                "labels_recorded": recorded,
            }
        )
    cli_versions = sorted({a["cli_version"] for b in rows for a in b["attempts"] if a["cli_version"]})
    isolated_files = sum(len(a["files_created_in_isolated_dir"] or []) for b in rows for a in b["attempts"])
    return {
        "session_id": session,
        "annotator_id": SESSIONS[session],
        "cli_versions": cli_versions,
        "batches": rows,
        "attempts": sum(len(b["attempts"]) for b in rows),
        "labels_recorded": sum(b["labels_recorded"] for b in rows),
        "files_created_in_isolated_dirs": isolated_files,
        "web_tool_calls": {
            "attempts_scanned": sum(len(b["attempts"]) for b in rows),
            "attempts_not_visible": sum(1 for b in rows for a in b["attempts"] if not a["web_tool_calls"]["visible"]),
            "attempts_with_calls": sum(1 for b in rows for a in b["attempts"] if a["web_tool_calls"].get("any")),
            "batches_with_calls_that_recorded_labels": sorted(
                b["batch_id"] for b in rows if any(a["web_tool_calls"].get("any") and a["recorded_labels"] for a in b["attempts"])
            ),
        },
        "ingest": {
            "history_entries": len(entries),
            "labeled_items": len({entry["item_id"] for entry in entries}),
            "history_head_sha256": entries[-1]["entry_sha256"] if entries else None,
        },
    }


#: Web tools of the two CLIs. agy (Gemini) names its own ``search_web``, ``read_url_content`` and
#: ``browser_subagent``; cline reports only how many tool calls a run made, so any call there counts as unexplained.
AGY_WEB_TOOLS = (b"search_web", b"read_url_content", b"browser_subagent")
#: agy keeps every conversation as a SQLite file of steps (``~/.gemini/antigravity-cli/conversations``). In agy 1.2.x
#: a model step (type 15) names each tool it calls at protobuf field path 20.7.2, and step types 31 and 33 are a URL
#: read and a web search. Read from this machine's stores (2026-10-03), checked against conversations with known
#: searches; not a documented format.
AGY_CONVERSATIONS = Path.home() / ".gemini" / "antigravity-cli" / "conversations"
AGY_MODEL_STEP, AGY_TOOL_NAME_PATH, AGY_WEB_STEPS = 15, (20, 7, 2), (31, 33)


def _varint(data: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = data[i]
        i += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return result, i


def _field(data: bytes, path: tuple[int, ...]) -> list[bytes]:
    """Every length-delimited value at ``path`` in a protobuf message; [] where the bytes do not parse."""
    found: list[bytes] = []
    i = 0
    try:
        while i < len(data):
            key, i = _varint(data, i)
            number, wire = key >> 3, key & 7
            if wire == 0:
                _, i = _varint(data, i)
            elif wire == 1:
                i += 8
            elif wire == 5:
                i += 4
            elif wire == 2:
                size, i = _varint(data, i)
                value = data[i : i + size]
                i += size
                if number == path[0]:
                    found += [value] if len(path) == 1 else _field(value, path[1:])
            else:
                return found
    except IndexError:
        pass
    return found


def agy_web_calls(run: dict[str, Any]) -> dict[str, Any]:
    """Web-tool calls in the agy conversation(s) created while this attempt ran; counts only, never content."""
    # agy creates the attempt's conversation file a few seconds after launch (4-5 s measured); it also opens every
    # older file, so modification times say nothing. The window is the first minute of the attempt.
    start = float(run["started_at"])
    end = start + min(60.0, float(run.get("seconds") or 0))
    files = [f for f in AGY_CONVERSATIONS.glob("*.db") if start <= f.stat().st_ctime <= end] if AGY_CONVERSATIONS.is_dir() else []
    if len(files) != 1:
        return {"visible": False, "why": f"{len(files)} agy conversations were created in the attempt's first minute, not 1"}
    connection = sqlite3.connect(f"file:{files[0]}?mode=ro", uri=True)
    try:
        rows = connection.execute("select step_type, step_payload from steps").fetchall()
    finally:
        connection.close()
    calls: dict[str, int] = {}
    tools: dict[str, int] = {}
    for step_type, payload in rows:
        if step_type == AGY_MODEL_STEP and isinstance(payload, bytes):
            for name in _field(payload, AGY_TOOL_NAME_PATH):
                tool = name.decode("utf-8", "replace")
                tools[tool] = tools.get(tool, 0) + 1
                if name in AGY_WEB_TOOLS:
                    calls[tool] = calls.get(tool, 0) + 1
    web_steps = sum(1 for step_type, _ in rows if step_type in AGY_WEB_STEPS)
    return {
        "visible": True,
        "source": f"agy conversation store, {files[0].name}",
        "steps": len(rows),
        "web_tool_calls": calls,
        "tool_calls": dict(sorted(tools.items())),
        # Executed searches of agy 1.2.x came back as generic result steps, not types 31/33, so 0 here proves nothing.
        "web_steps": web_steps,
        "any": bool(calls or web_steps),
    }


def cline_tool_calls(stdout: bytes) -> dict[str, Any]:
    """Tool calls cline reported in its --json event stream (its iteration-end events carry ``toolCallCount``)."""
    counts: list[int] = []
    for line in stdout.decode("utf-8", "replace").splitlines():
        try:
            event = json.loads(re.sub(r"\x1b\[[0-9;]*m", "", line))
        except ValueError:
            continue
        inner = event.get("event") if isinstance(event, dict) else None
        if isinstance(inner, dict) and "toolCallCount" in inner:
            counts.append(int(inner["toolCallCount"]))
    if not counts:
        return {"visible": False, "why": "no toolCallCount in the cline event stream"}
    return {"visible": True, "source": "cline --json toolCallCount", "tool_calls": sum(counts), "any": sum(counts) > 0}


def web_tool_calls(session: str, run: dict[str, Any], stdout: Path) -> dict[str, Any]:
    if session.startswith("model-gemini"):
        result = agy_web_calls(run)
        gate = run.get("tool_gate")
        if gate is not None and result.get("visible"):
            # A run under the tool gate (scripts/agy_tool_gate.py): a call the hook denied never executed.
            denied = sum(gate["denied"].get(name.decode(), 0) for name in AGY_WEB_TOOLS)
            seen = {
                tool: gate["allowed"].get(tool, 0) + gate["denied"].get(tool, 0)
                for tool in {*gate["allowed"], *gate["denied"]}
            }
            result["tool_gate"] = gate
            # Every call in the store must have passed the gate; a difference means a call bypassed it.
            result["gate_saw_every_call"] = seen == result["tool_calls"]
            result["any"] = bool(
                sum(result["web_tool_calls"].values()) > denied or result["web_steps"] or not result["gate_saw_every_call"]
            )
        return result
    if session.startswith("model-deepseek"):
        return cline_tool_calls(stdout.read_bytes()) if stdout.is_file() else {"visible": False, "why": "no stdout"}
    return {"visible": False, "why": f"no scan for session {session}"}


def scan(*groups: dict[str, bytes]) -> str:
    """Refuse to archive a file holding the operator's git e-mail or a credential shape; return the scan date.

    The P-DET-COVERAGE archives recorded a scan of 2026-09-17 made by hand; this makes it part of the build."""
    email = subprocess.run(
        ["git", "config", "user.email"], cwd=ROOT, capture_output=True, check=False
    ).stdout.strip()
    for group in groups:
        for name, data in group.items():
            if (email and email in data) or CREDENTIALS.search(data):
                raise SystemExit(f"{name}: holds the operator's e-mail address or a credential; not archived")
    return datetime.now(tz=UTC).date().isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--sessions", nargs="+", choices=sorted(SESSIONS), help="default: every session present")
    parser.add_argument("--variant", help="archive under <task>.external-models.<variant>.audit-trail")
    args = parser.parse_args(argv)
    task = args.task
    out, procedure, authorization = TASK_SPECS[task]
    members: dict[str, bytes] = {
        f"procedure/{Path(procedure).name}": (ROOT / procedure).read_bytes(),
        f"runner/{Path(RUNNER).name}": (ROOT / RUNNER).read_bytes(),
    }
    streams: dict[str, bytes] = {}
    sessions = [
        audit_session(task, session, members, streams)
        for session in (args.sessions or SESSIONS)
        if (ROOT / ".annotation" / f"{task}.model-batches" / session).is_dir()
    ]
    scanned = scan(members, streams)
    infix = f".{args.variant}" if args.variant else ""
    stem = f"{task}.external-models{infix}.audit-trail"
    local_name = f"{task}.external-models{infix}.cli-streams.tar.gz"
    archive = tar_gz(members)
    local = tar_gz(streams)
    manifest = {
        "schema": SCHEMA,
        "task_id": task,
        **({"variant": args.variant} if args.variant else {}),
        "authorization": authorization,
        "status": STATUS.get(
            task,
            "Model judgments by three declared non-Claude annotators, each blind and independent. They form the "
            "provisional MODEL_REFERENCE of the three-model consensus ({authorization}); not human labels and not human gold.",
        ).format(authorization=authorization),
        "procedure": {"path": procedure, "sha256": sha256(members[f"procedure/{Path(procedure).name}"])},
        "runner": {"path": RUNNER, "sha256": sha256(members[f"runner/{Path(RUNNER).name}"])},
        "sessions": sessions,
        "archive": {"file": f"{stem}.tar.gz", "sha256": sha256(archive), "bytes": len(archive), "members": listing(members)},
        "cli_streams": {
            "file": f"local/{local_name}",
            "tracked": False,
            "why": (
                "Raw CLI stdout and stderr are unfiltered tool logs that may carry account or session details, and "
                "they are large. They are kept verbatim, out of version control; their hashes are recorded here and "
                "in each run record. A scan for the operator's e-mail address and common credential patterns found "
                f"none ({scanned})."
            ),
            "sha256": sha256(local),
            "bytes": len(local),
            "members": listing(streams),
        },
        "built_by": "scripts/archive_external_model_labels.py",
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    write_new(out / f"{stem}.tar.gz", archive)
    write_new(out / f"{stem}.tar.gz.sha256", f"{sha256(archive)}  {stem}.tar.gz\n".encode())
    write_new(out / f"{stem}.manifest.json", manifest_bytes)
    write_new(out / f"{stem}.manifest.json.sha256", f"{sha256(manifest_bytes)}  {stem}.manifest.json\n".encode())
    write_new(out / "local" / local_name, local)
    print(f"archive  {len(archive)} bytes  sha256 {sha256(archive)}")
    print(f"manifest sha256 {sha256(manifest_bytes)}")
    print(f"streams  {len(local)} bytes  sha256 {sha256(local)}  (not tracked)")
    for s in sessions:
        outcomes: dict[str, int] = {}
        for b in s["batches"]:
            outcomes[b["outcome"]] = outcomes.get(b["outcome"], 0) + 1
        print(
            f"{s['session_id']:15s} cli {s['cli_versions']}  batches {outcomes}  attempts {s['attempts']}  "
            f"labels {s['labels_recorded']}  ingest {s['ingest']['history_entries']}  isolated-dir files {s['files_created_in_isolated_dirs']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
