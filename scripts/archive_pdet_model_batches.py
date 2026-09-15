"""Archive the P-DET `model-a` audit trail out of the git-ignored working directory.

The model labels in session ``model-a`` (``model.claude-opus-5``, amendment 28) were produced in 12 blind
batches. Their trail lived in ``.annotation/pdet-v1.model-batches/model-a/``, which is not tracked. This
script packages it, byte for byte, into a new tracked artifact:

    reports/pdet/provenance/model-a/
      pdet-v1.model-a.audit-trail.tar.gz            batches, raw answers, audits, procedure, ingest log, tools
      pdet-v1.model-a.audit-trail.manifest.json     every member's hash, the batch table, ingest summary
      *.sha256                                      sidecars
      local/pdet-v1.model-a.transcripts.tar.gz      the subagent transcripts, verbatim (NOT tracked)

Nothing is rewritten: every file enters the archive with the bytes it has on disk. The transcripts carry the
session's injected context (including the operator's e-mail address and local paths). They are archived
verbatim and hashed in the tracked manifest, but the archive holding them is kept out of version control.

The archive is deterministic (sorted members, zero timestamps and owners, gzip mtime 0), so re-running
this script on the same inputs gives the same bytes. It refuses to replace a different existing archive.

    python scripts/archive_pdet_model_batches.py [--tools-dir DIR]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import sqlite3
import sys
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BATCHES = ROOT / ".annotation" / "pdet-v1.model-batches" / "model-a"
STATE = ROOT / ".annotation" / "pdet-v1.sqlite3"
PROCEDURE = "configs/annotation/pdet-v1.model-procedure.md"
OUT = ROOT / "reports" / "pdet" / "provenance" / "model-a"
STEM = "pdet-v1.model-a.audit-trail"
TRANSCRIPTS = "pdet-v1.model-a.transcripts.tar.gz"
TOOLS = ("audit_batch.py", "check_prompt.py", "extract_answers.py")
SCHEMA = "opengrad-model-label-audit-trail-v1"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tar_gz(members: dict[str, bytes]) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(members):
            info = tarfile.TarInfo(name)
            info.size = len(members[name])
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(members[name]))
    packed = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=packed, mtime=0, compresslevel=9) as stream:
        stream.write(raw.getvalue())
    return packed.getvalue()


def write_new(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() == data:
            return
        raise SystemExit(f"{path} exists with different contents; an archive is never replaced")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def listing(members: dict[str, bytes]) -> list[dict[str, Any]]:
    return [{"path": name, "sha256": sha256(data), "bytes": len(data)} for name, data in sorted(members.items())]


def canonical_json(value: Any) -> str:
    # Same serialisation as opengrad.annotation.items.canonical_json, which content_sha256 was made with.
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tools-dir", type=Path, help="directory holding the audit scripts that were used")
    args = parser.parse_args(argv)

    batch_files = sorted(path for path in BATCHES.glob("batch-*.json") if re.fullmatch(r"batch-\d+\.json", path.name))
    if not batch_files:
        raise SystemExit(f"no batches under {BATCHES}")
    members: dict[str, bytes] = {}
    transcripts: dict[str, bytes] = {}
    batches: list[dict[str, Any]] = []
    procedure = (ROOT / PROCEDURE).read_bytes()
    members[f"procedure/{Path(PROCEDURE).name}"] = procedure

    for expected, path in enumerate(batch_files, start=1):
        batch_id = path.stem.split("-")[1]
        if batch_id != f"{expected:02d}":
            raise SystemExit(f"batch numbering has a gap: expected {expected:02d}, found {batch_id}")
        files = {suffix: BATCHES / f"batch-{batch_id}{suffix}" for suffix in (".json", ".md", ".answers.json", ".audit.json")}
        for file in files.values():
            members[f"model-a/{file.name}"] = file.read_bytes()
        batch = json.loads(files[".json"].read_bytes())
        audit = json.loads(files[".audit.json"].read_bytes())
        answers_bytes = files[".answers.json"].read_bytes()
        answers = json.loads(answers_bytes)
        content = sha256(canonical_json({"items": batch["items"], "instructions": batch["instructions"]}).encode("utf-8"))
        transcript_path = Path(audit["subagent_transcript"])
        transcript = transcript_path.read_bytes()
        meta_path = transcript_path.with_suffix(".meta.json")
        agent = transcript_path.stem.removeprefix("agent-")
        transcripts[f"transcripts/{transcript_path.name}"] = transcript
        if meta_path.is_file():
            transcripts[f"transcripts/{meta_path.name}"] = meta_path.read_bytes()
        numbers = [item["number"] for item in batch["items"]]
        batches.append(
            {
                "batch_id": batch_id,
                "items": len(batch["item_ids"]),
                "first_number": numbers[0],
                "last_number": numbers[-1],
                "created_at": batch["created_at"],
                "content_sha256": batch["content_sha256"],
                "content_sha256_rederived": content == batch["content_sha256"],
                "procedure_sha256": batch["procedure_sha256"],
                "batch_json_sha256": sha256(members[f"model-a/{files['.json'].name}"]),
                "batch_md_sha256": sha256(members[f"model-a/{files['.md'].name}"]),
                "answers_file_sha256": sha256(answers_bytes),
                "answers_lf_sha256": sha256(answers_bytes.replace(b"\r\n", b"\n")),
                "audit_answers_sha256": audit["answers_sha256"],
                "answers": len(answers),
                "flagged": sum(1 for answer in answers if answer.get("flag")),
                "subagent_id": agent,
                "subagent_models": audit["models"],
                "subagent_tool_calls": len(audit["tool_calls"]),
                "transcript_file": transcript_path.name,
                "transcript_sha256": sha256(transcript),
                "transcript_matches_audit": sha256(transcript) == audit["transcript_sha256"],
            }
        )

    for row in batches:
        if not (row["content_sha256_rederived"] and row["transcript_matches_audit"]):
            raise SystemExit(f"batch {row['batch_id']}: content or transcript no longer matches its record")
        if row["answers_lf_sha256"] != row["audit_answers_sha256"]:
            raise SystemExit(f"batch {row['batch_id']}: answers no longer match the audited answers")

    with sqlite3.connect(f"file:{STATE.as_posix()}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        session = dict(conn.execute("SELECT * FROM sessions WHERE task_id = 'pdet-v1' AND session_id = 'model-a'").fetchone())
        history = []
        for row in conn.execute("SELECT * FROM history WHERE task_id = 'pdet-v1' AND session_id = 'model-a' ORDER BY id"):
            entry = dict(row)
            entry["before"] = json.loads(entry.pop("before_json")) if row["before_json"] else None
            entry["after"] = json.loads(entry.pop("after_json")) if row["after_json"] else None
            history.append(entry)
    members["ingest/model-a.session.json"] = (json.dumps(session, sort_keys=True, indent=2) + "\n").encode("utf-8")
    members["ingest/model-a.history.jsonl"] = "".join(
        json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n" for entry in history
    ).encode("utf-8")
    per_batch: dict[str, int] = {}
    for entry in history:
        match = re.match(r"model batch (\d+);", entry["reason"] or "")
        key = match.group(1) if match else "?"
        per_batch[key] = per_batch.get(key, 0) + 1

    if args.tools_dir:
        for name in TOOLS:
            members[f"tools/{name}"] = (args.tools_dir / name).read_bytes()

    archive = tar_gz(members)
    local = tar_gz(transcripts)
    manifest = {
        "schema": SCHEMA,
        "task_id": "pdet-v1",
        "session_id": "model-a",
        "annotator_id": session["annotator_id"],
        "model": "claude-opus-5",
        "authorization": "docs/research/study-002/28-PDET-MODEL-LABEL-AMENDMENT.md",
        "status": (
            "Provisional model judgments by a declared model annotator. Not human labels, not human gold, not "
            "frozen. A human label in pass-a takes precedence over every label here."
        ),
        "procedure": {"path": PROCEDURE, "sha256": sha256(procedure)},
        "archive": {"file": f"{STEM}.tar.gz", "sha256": sha256(archive), "bytes": len(archive), "members": listing(members)},
        "batches": batches,
        "ingest": {
            "history_entries": len(history),
            "labeled_items": len({entry["item_id"] for entry in history}),
            "entries_per_batch": dict(sorted(per_batch.items())),
            "history_head_sha256": history[-1]["entry_sha256"] if history else None,
            "first_recorded_at": history[0]["recorded_at"] if history else None,
            "last_recorded_at": history[-1]["recorded_at"] if history else None,
        },
        "line_endings": (
            "The batch, answers and audit files were written on Windows through text-mode writes, so they carry "
            "CRLF line endings; they are archived with exactly those bytes. Each audit's answers_sha256 hashed "
            "the model's answer text before that translation, so it equals answers_lf_sha256 (the file with "
            "CRLF turned back into LF), and that text is also the subagent's final message in its transcript. "
            "content_sha256 is computed over the parsed batch content and does not depend on line endings."
        ),
        "transcripts": {
            "file": f"local/{TRANSCRIPTS}",
            "tracked": False,
            "why": (
                "The subagent transcripts are kept verbatim and never edited. They include the session's injected "
                "context -- the operator's e-mail address and local file paths -- so the archive holding them "
                "stays out of version control. Its hash, and each transcript's, are recorded here."
            ),
            "sha256": sha256(local),
            "bytes": len(local),
            "members": listing(transcripts),
        },
        "tools": [f"tools/{name}" for name in TOOLS] if args.tools_dir else [],
        "built_by": "scripts/archive_pdet_model_batches.py",
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    write_new(OUT / f"{STEM}.tar.gz", archive)
    write_new(OUT / f"{STEM}.tar.gz.sha256", f"{sha256(archive)}  {STEM}.tar.gz\n".encode())
    write_new(OUT / f"{STEM}.manifest.json", manifest_bytes)
    write_new(OUT / f"{STEM}.manifest.json.sha256", f"{sha256(manifest_bytes)}  {STEM}.manifest.json\n".encode())
    write_new(OUT / "local" / TRANSCRIPTS, local)
    print(f"archive     {OUT / (STEM + '.tar.gz')}  {len(archive)} bytes  sha256 {sha256(archive)}")
    print(f"manifest    sha256 {sha256(manifest_bytes)}")
    print(f"transcripts {OUT / 'local' / TRANSCRIPTS}  {len(local)} bytes  sha256 {sha256(local)}  (not tracked)")
    print(f"batches     {len(batches)}; answers {sum(row['answers'] for row in batches)}; ingest entries {len(history)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
