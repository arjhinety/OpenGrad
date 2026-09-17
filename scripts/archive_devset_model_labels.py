"""Archive the audit trail of the `model-dev` labels on the classifier development set (33 §3).

Session ``model-dev`` (``model.claude-opus-5``, authorized by doc 33) labelled the 250 items of
``prose-classifier-dev-v1`` in one batch, ``batch-01``. The batch was split across five Claude subagents run
in parallel, one per item range (#1-50, #51-100, ...). Each got the procedure text verbatim, followed by one
line naming the batch file and its range. Their answers were merged in batch order and ingested once.

This script audits each subagent from its transcript and packages the trail, byte for byte, into a new
tracked artifact:

    reports/prose-classifier/dev/provenance/model-dev/
      prose-classifier-dev-v1.model-dev.audit-trail.tar.gz          batch, answers, procedure, ingest log
      prose-classifier-dev-v1.model-dev.audit-trail.manifest.json   hashes, the per-subagent audit, ingest summary
      *.sha256                                                      sidecars
      local/prose-classifier-dev-v1.model-dev.transcripts.tar.gz    subagent transcripts, verbatim (NOT tracked)

For each subagent the audit checks, from the transcript: the prompt is the procedure plus exactly one batch
line; every model turn was claude-opus-5; every tool call was a Read of the batch file; the final message
parses to the saved part answers; the part covers exactly its range, in order. The transcripts carry the
session's injected context (the operator's e-mail address, local paths), so they stay out of version control;
their hashes are recorded. The archive is deterministic and never replaces a different existing one.

    python scripts/archive_devset_model_labels.py
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from archive_pdet_model_batches import (
    canonical_json,
    listing,
    sha256,
    tar_gz,
    write_new,
)

ROOT = Path(__file__).resolve().parents[1]
TASK = "prose-classifier-dev-v1"
SESSION = "model-dev"
BATCHES = ROOT / ".annotation" / f"{TASK}.model-batches" / SESSION
STATE = ROOT / ".annotation" / f"{TASK}.sqlite3"
PROCEDURE = f"configs/annotation/{TASK}.model-procedure.md"
OUT = ROOT / "reports" / "prose-classifier" / "dev" / "provenance" / SESSION
STEM = f"{TASK}.{SESSION}.audit-trail"
TRANSCRIPTS = f"{TASK}.{SESSION}.transcripts.tar.gz"
SUBAGENTS = Path.home() / ".claude/projects/C--Users-arro-DOwnloads-OpenGrad/a4718da9-447f-4538-88d4-459232648e01/subagents"
#: (part, subagent id, first item number, last item number), in batch order.
PARTS = (
    ("1", "a0cb84d2dac24535a", 1, 50),
    ("2", "af27fd02dcfa0ed1d", 51, 100),
    ("3", "a066a49406314937b", 101, 150),
    ("4", "a4ce9c59605a40b5c", 151, 200),
    ("5", "a1cb22f6d78593a44", 201, 250),
)
SCHEMA = "opengrad-model-label-audit-trail-v1"
BATCH_LINE = re.compile(
    r"Batch file: (?P<path>\S+batch-01\.md) -- read the rubric \(lines 1-284\), then label only items "
    r"#(?P<first>\d+) to #(?P<last>\d+) \(lines \d+-\d+ of that file\), returning exactly one object for each "
    r"of those 50 items, in item order\."
)


def procedure_body(text: str) -> str:
    """The part of the procedure file given to each subagent: everything after the ``---`` rule."""
    return text.split("\n---\n", 1)[1].strip()


def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text")


def audit_part(part: str, agent: str, first: int, last: int, body: str, batch: dict[str, Any]) -> tuple[dict[str, Any], bytes, bytes]:
    path = SUBAGENTS / f"agent-{agent}.jsonl"
    transcript = path.read_bytes()
    meta = path.with_suffix(".meta.json").read_bytes()
    events = [json.loads(line) for line in transcript.decode("utf-8").splitlines() if line.strip()]
    prompt = text_of(next(e for e in events if e.get("type") == "user")["message"]["content"]).strip()
    head, _, line = prompt.rpartition("\n")
    match = BATCH_LINE.fullmatch(line.strip())
    models = sorted({e["message"]["model"] for e in events if e.get("type") == "assistant" and e["message"].get("model")})
    calls = [
        {"name": block["name"], **{k: block["input"].get(k) for k in ("file_path", "offset", "limit")}}
        for e in events
        if e.get("type") == "assistant"
        for block in e["message"]["content"]
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]
    final = [text_of(e["message"]["content"]).strip() for e in events if e.get("type") == "assistant"]
    final_text = [t for t in final if t][-1]
    answers_bytes = (BATCHES / f"batch-01.part-{part}.answers.json").read_bytes()
    answers = json.loads(answers_bytes)
    expected = [item["item_id"] for item in batch["items"]][first - 1 : last]
    row = {
        "part": part,
        "first_number": first,
        "last_number": last,
        "subagent_id": agent,
        "subagent_models": models,
        "prompt_equals_procedure_plus_batch_line": head.strip() == body and match is not None,
        "prompt_range_matches": bool(match) and (int(match["first"]), int(match["last"])) == (first, last),
        "tool_calls": calls,
        "only_reads_of_the_batch_file": bool(calls) and all(
            c["name"] == "Read" and str(c["file_path"]).endswith("batch-01.md") for c in calls
        ),
        "final_message_sha256": sha256(final_text.encode("utf-8")),
        "final_message_equals_part_answers": json.loads(final_text[final_text.find("[") : final_text.rfind("]") + 1]) == answers,
        "part_answers_file_sha256": sha256(answers_bytes),
        "answers": len(answers),
        "flagged": sum(1 for a in answers if a.get("flag")),
        "item_ids_match_range_in_order": [a["item_id"] for a in answers] == expected,
        "transcript_file": path.name,
        "transcript_sha256": sha256(transcript),
    }
    return row, transcript, meta


def main() -> int:
    procedure = (ROOT / PROCEDURE).read_bytes()
    body = procedure_body(procedure.decode("utf-8").replace("\r\n", "\n"))
    batch_json = (BATCHES / "batch-01.json").read_bytes()
    batch = json.loads(batch_json)
    content = sha256(canonical_json({"items": batch["items"], "instructions": batch["instructions"]}).encode("utf-8"))
    if content != batch["content_sha256"] or batch["procedure_sha256"] != sha256(procedure):
        raise SystemExit("batch-01: content or procedure hash no longer matches")

    members: dict[str, bytes] = {
        f"procedure/{Path(PROCEDURE).name}": procedure,
        f"{SESSION}/batch-01.json": batch_json,
        f"{SESSION}/batch-01.md": (BATCHES / "batch-01.md").read_bytes(),
    }
    transcripts: dict[str, bytes] = {}
    parts = []
    merged: list[Any] = []
    for part, agent, first, last in PARTS:
        row, transcript, meta = audit_part(part, agent, first, last, body, batch)
        checks = ("prompt_equals_procedure_plus_batch_line", "prompt_range_matches", "only_reads_of_the_batch_file",
                  "final_message_equals_part_answers", "item_ids_match_range_in_order")
        failed = [c for c in checks if not row[c]] + ([] if row["subagent_models"] == ["claude-opus-5"] else ["models"])
        if failed:
            raise SystemExit(f"part {part}: audit failed: {failed}")
        parts.append(row)
        members[f"{SESSION}/batch-01.part-{part}.answers.json"] = (BATCHES / f"batch-01.part-{part}.answers.json").read_bytes()
        merged.extend(json.loads(members[f"{SESSION}/batch-01.part-{part}.answers.json"]))
        transcripts[f"transcripts/agent-{agent}.jsonl"] = transcript
        transcripts[f"transcripts/agent-{agent}.meta.json"] = meta
    merged_bytes = (BATCHES / "batch-01.answers.json").read_bytes()
    if json.loads(merged_bytes) != merged:
        raise SystemExit("batch-01.answers.json is not the five parts merged in batch order")
    members[f"{SESSION}/batch-01.answers.json"] = merged_bytes

    with sqlite3.connect(f"file:{STATE.as_posix()}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        session = dict(conn.execute("SELECT * FROM sessions WHERE task_id = ? AND session_id = ?", (TASK, SESSION)).fetchone())
        history = []
        for row in conn.execute("SELECT * FROM history WHERE task_id = ? AND session_id = ? ORDER BY id", (TASK, SESSION)):
            entry = dict(row)
            entry["before"] = json.loads(entry.pop("before_json")) if row["before_json"] else None
            entry["after"] = json.loads(entry.pop("after_json")) if row["after_json"] else None
            history.append(entry)
    members[f"ingest/{SESSION}.session.json"] = (json.dumps(session, sort_keys=True, indent=2) + "\n").encode("utf-8")
    members[f"ingest/{SESSION}.history.jsonl"] = "".join(
        json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n" for entry in history
    ).encode("utf-8")

    archive = tar_gz(members)
    local = tar_gz(transcripts)
    manifest = {
        "schema": SCHEMA,
        "task_id": TASK,
        "session_id": SESSION,
        "annotator_id": session["annotator_id"],
        "model": "claude-opus-5",
        "authorization": "docs/research/study-002/33-PROSE-DECISION-CLASSIFIER-DEVELOPMENT-PLAN.md",
        "status": (
            "Model judgments by a declared model annotator, used only to develop the prose decision classifier. "
            "Not human labels, not human gold, never evidence of accuracy."
        ),
        "procedure": {"path": PROCEDURE, "sha256": sha256(procedure)},
        "batch": {
            "batch_id": "01",
            "items": len(batch["item_ids"]),
            "created_at": batch["created_at"],
            "content_sha256": batch["content_sha256"],
            "batch_json_sha256": sha256(batch_json),
            "batch_md_sha256": sha256(members[f"{SESSION}/batch-01.md"]),
            "merged_answers_sha256": sha256(merged_bytes),
            "answers": len(merged),
            "flagged": sum(1 for a in merged if a.get("flag")),
        },
        "split": (
            "One batch labelled by five Claude subagents run in parallel, one per 50-item range. Each prompt was "
            "the procedure text followed by one line naming the batch file and its range; each subagent read "
            "the rubric and its own range. The five answer arrays were merged in batch order and ingested once."
        ),
        "parts": parts,
        "archive": {"file": f"{STEM}.tar.gz", "sha256": sha256(archive), "bytes": len(archive), "members": listing(members)},
        "ingest": {
            "history_entries": len(history),
            "labeled_items": len({entry["item_id"] for entry in history}),
            "history_head_sha256": history[-1]["entry_sha256"] if history else None,
            "first_recorded_at": history[0]["recorded_at"] if history else None,
            "last_recorded_at": history[-1]["recorded_at"] if history else None,
        },
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
        "built_by": "scripts/archive_devset_model_labels.py",
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    write_new(OUT / f"{STEM}.tar.gz", archive)
    write_new(OUT / f"{STEM}.tar.gz.sha256", f"{sha256(archive)}  {STEM}.tar.gz\n".encode())
    write_new(OUT / f"{STEM}.manifest.json", manifest_bytes)
    write_new(OUT / f"{STEM}.manifest.json.sha256", f"{sha256(manifest_bytes)}  {STEM}.manifest.json\n".encode())
    write_new(OUT / "local" / TRANSCRIPTS, local)
    print(f"archive     {len(archive)} bytes  sha256 {sha256(archive)}")
    print(f"manifest    sha256 {sha256(manifest_bytes)}")
    print(f"transcripts {len(local)} bytes  sha256 {sha256(local)}  (not tracked)")
    print(f"parts       {len(parts)}; answers {len(merged)}; ingest entries {len(history)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
