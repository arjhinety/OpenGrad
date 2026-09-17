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
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from archive_pdet_model_batches import canonical_json, listing, sha256, tar_gz, write_new

from opengrad.annotation.provenance import ANNOTATION_ENTRY_FIELDS, verify_chain

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "pdet-coverage" / "provenance" / "external-models"
TASKS = ("pdet-coverage-v1", "pdet-coverage-v1-routing")
SESSIONS = {
    "model-gemini": "model.gemini-3.8-flash-high",
    "model-gpt": "model.gpt-5.6-sol",
    "model-deepseek": "model.deepseek-v4.1-flash",
}
PROCEDURE = "configs/annotation/pdet-coverage-v1.model-procedure.md"
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
        "ingest": {
            "history_entries": len(entries),
            "labeled_items": len({entry["item_id"] for entry in entries}),
            "history_head_sha256": entries[-1]["entry_sha256"] if entries else None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=TASKS, required=True)
    task = parser.parse_args(argv).task
    members: dict[str, bytes] = {
        f"procedure/{Path(PROCEDURE).name}": (ROOT / PROCEDURE).read_bytes(),
        f"runner/{Path(RUNNER).name}": (ROOT / RUNNER).read_bytes(),
    }
    streams: dict[str, bytes] = {}
    sessions = [audit_session(task, session, members, streams) for session in SESSIONS]
    stem = f"{task}.external-models.audit-trail"
    local_name = f"{task}.external-models.cli-streams.tar.gz"
    archive = tar_gz(members)
    local = tar_gz(streams)
    manifest = {
        "schema": SCHEMA,
        "task_id": task,
        "authorization": "docs/research/study-002/34-PDET-COVERAGE-MODEL-CONSENSUS-AMENDMENT.md",
        "status": (
            "Model judgments by three declared non-Claude annotators, each blind and independent. They form the "
            "provisional MODEL_REFERENCE of study_002_prereg_v5; not human labels and not human gold."
        ),
        "procedure": {"path": PROCEDURE, "sha256": sha256(members[f"procedure/{Path(PROCEDURE).name}"])},
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
                "none (2026-09-17)."
            ),
            "sha256": sha256(local),
            "bytes": len(local),
            "members": listing(streams),
        },
        "built_by": "scripts/archive_external_model_labels.py",
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    write_new(OUT / f"{stem}.tar.gz", archive)
    write_new(OUT / f"{stem}.tar.gz.sha256", f"{sha256(archive)}  {stem}.tar.gz\n".encode())
    write_new(OUT / f"{stem}.manifest.json", manifest_bytes)
    write_new(OUT / f"{stem}.manifest.json.sha256", f"{sha256(manifest_bytes)}  {stem}.manifest.json\n".encode())
    write_new(OUT / "local" / local_name, local)
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
