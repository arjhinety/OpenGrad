"""Run one declared external model annotator over a task, batch by batch (amendment study_002_prereg_v5).

For each batch this script:

1. prepares the next unlabelled items with :func:`opengrad.annotation.model_batch.prepare_batch`, the audited
   route that renders exactly what the annotation screen shows (no blinded field, no other session's label);
2. builds the model's input from the pinned procedure (its SHA-256 is checked against the task config) followed
   by the rendered batch;
3. runs the external CLI in a **new, empty temporary directory**, outside the repository, with the input on
   standard input and the CLI's tools restricted. The model has no file to open and nothing to find;
4. extracts the final JSON array of answers from the output;
5. records them with :func:`opengrad.annotation.model_batch.ingest_batch`, which validates every answer before
   writing any. An invalid run is retried with a fresh directory, up to ``--attempts`` times.

Everything the CLI printed is kept beside the batch (git-ignored working state), with a run record: the CLI
version, the argv, the exit code and the SHA-256 of the input and outputs. **Nothing here prints item text,
labels or rationales**: Claude, who runs this, builds the classifier these labels will test (34 §3).

    python scripts/run_external_annotation.py pdet-coverage-v1 --annotator model.gpt-5.6-sol --size 20
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from opengrad.annotation.config import load_task_config
from opengrad.annotation.model_batch import (
    BatchError,
    ingest_batch,
    prepare_batch,
    write_batch,
)
from opengrad.annotation.service import Workspace
from opengrad.hashing import sha256_bytes
from opengrad.registry.provenance import portable_path

#: Handed to CLIs that take an instruction argument beside standard input.
STDIN_POINTER = (
    "Your complete instructions and the batch to annotate are provided on standard input. Read all of it and "
    "follow those instructions exactly. Use no tools."
)
#: The input file for CLIs that do not read standard input when started from Python (measured for agy). It
#: is the only file in the isolated directory.
INPUT_FILE = "input.md"
FILE_POINTER = (
    f"Read the file {INPUT_FILE} in your working directory. It contains your complete instructions followed by "
    "the batch to annotate. Follow those instructions exactly. Do not open any other file and run no commands."
)

#: The declared annotators of amendment study_002_prereg_v5 and how each one is run. ``{last}`` is replaced
#: by a path for the CLI's final message, where the CLI can write one.
ANNOTATORS: dict[str, dict[str, Any]] = {
    "model.gemini-3.8-flash-high": {
        "session": "model-gemini",
        "executable": "agy",
        "argv": [
            "--model",
            "Gemini 3.8 Flash (High)",
            "--sandbox",
            "--print-timeout",
            "45m",
            "-p",
            FILE_POINTER,
        ],
        "input": "file",
        "version_argv": ["--version"],
    },
    "model.gpt-5.6-sol": {
        "session": "model-gpt",
        "executable": "codex",
        "argv": [
            "exec",
            "--skip-git-repo-check",
            "-m",
            "gpt-5.6-sol",
            "--sandbox",
            "read-only",
            "--output-last-message",
            "{last}",
            "-",
        ],
        "input": "stdin",
        "version_argv": ["--version"],
    },
    "model.deepseek-v4.1-flash": {
        "session": "model-deepseek",
        "executable": "cline",
        # --json: the text output interleaves terminal colour codes with the answer; the final
        # ``run_result`` event carries the plain final text and the model that ran.
        "argv": [
            "-m",
            "cline-pass/deepseek-v4.1-flash",
            "--auto-approve",
            "false",
            "--json",
            STDIN_POINTER,
        ],
        "input": "stdin",
        "output": "cline_json",
        "version_argv": ["--version"],
    },
}


def extract_answers(text: str, item_ids: list[str]) -> list[dict[str, Any]] | None:
    """The last JSON array in ``text`` whose objects all carry an ``item_id`` from this batch.

    CLIs may print headers, progress or the final message twice around the answer, so the whole output is
    scanned rather than parsed as one document. Returns None when no such array exists.
    """
    wanted = set(item_ids)
    decoder = json.JSONDecoder()
    best: list[dict[str, Any]] | None = None
    index = text.find("[")
    while index != -1:
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            index = text.find("[", index + 1)
            continue
        if (
            isinstance(value, list)
            and value
            and all(
                isinstance(entry, dict) and str(entry.get("item_id")) in wanted for entry in value
            )
        ):
            best = value
            index = text.find("[", end)
        else:
            index = text.find("[", index + 1)
    return best


def cline_final(stdout: str) -> tuple[str, Any]:
    """The final text and reported model of a ``cline --json`` run (its ``run_result`` event)."""
    for line in reversed(stdout.splitlines()):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "run_result":
            return str(event.get("text") or ""), event.get("model")
    return "", None


def procedure_text(config: Any, annotator_id: str) -> bytes:
    declared = config.model_annotator(annotator_id)
    if declared is None:
        raise SystemExit(f"{annotator_id} is not a declared model annotator of {config.task_id}")
    data = config.resolve(declared.procedure).read_bytes()
    if sha256_bytes(data) != declared.procedure_sha256:
        raise SystemExit(f"{declared.procedure} does not match its pinned procedure_sha256")
    return data


def cli_version(executable: str, version_argv: list[str]) -> str:
    try:
        done = subprocess.run(
            [executable, *version_argv], capture_output=True, timeout=120, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {type(exc).__name__}"
    return done.stdout.decode("utf-8", "replace").strip().splitlines()[0] if done.stdout else ""


def run_once(
    spec: dict[str, Any], executable: str, prompt: bytes, directory: Path, stem: str, timeout: int
) -> dict[str, Any]:
    """One isolated CLI run. Outputs are saved under ``directory``; nothing is printed."""
    workdir = Path(tempfile.mkdtemp(prefix="og-annotate-"))
    last = workdir.parent / f"{workdir.name}.last-message.txt"
    argv = [executable, *(arg.replace("{last}", str(last)) for arg in spec["argv"])]
    by_file = spec["input"] == "file"
    if by_file:
        (workdir / INPUT_FILE).write_bytes(prompt)
    started = time.time()
    try:
        done = subprocess.run(
            argv,
            input=None if by_file else prompt,
            stdin=subprocess.DEVNULL if by_file else None,
            capture_output=True,
            cwd=workdir,
            timeout=timeout,
            check=False,
        )
        exit_code, stdout, stderr = done.returncode, done.stdout, done.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code, stdout, stderr = "timeout", exc.stdout or b"", exc.stderr or b""
    ended = time.time()
    created = sorted(
        path.name for path in workdir.iterdir() if not (by_file and path.name == INPUT_FILE)
    )
    last_message = last.read_bytes() if last.is_file() else b""
    shutil.rmtree(workdir, ignore_errors=True)
    last.unlink(missing_ok=True)
    (directory / f"{stem}.stdout.txt").write_bytes(stdout)
    (directory / f"{stem}.stderr.txt").write_bytes(stderr)
    if last_message:
        (directory / f"{stem}.last-message.txt").write_bytes(last_message)
    return {
        # Recorded without the author's absolute paths: the executable and the isolated temp files
        # keep their names only (portable_path; reports/ERRATA.md §22).
        "argv": [portable_path(arg, ROOT) for arg in argv],
        "exit_code": exit_code,
        "started_at": started,
        "seconds": round(ended - started, 1),
        "prompt_sha256": sha256_bytes(prompt),
        "stdout_sha256": sha256_bytes(stdout),
        "stderr_sha256": sha256_bytes(stderr),
        "last_message_sha256": sha256_bytes(last_message) if last_message else None,
        "files_created_in_isolated_dir": created,
        **final_text(spec, stdout, last_message),
    }


def final_text(spec: dict[str, Any], stdout: bytes, last_message: bytes) -> dict[str, Any]:
    """The model's final message, plus the model name the CLI reports when it reports one."""
    text = stdout.decode("utf-8", "replace")
    if spec.get("output") == "cline_json":
        final, model = cline_final(text)
        return {"_text": final, "reported_model": model}
    reported = re.search(r"^model: (.+)$", text, re.MULTILINE)
    return {
        "_text": (last_message or stdout).decode("utf-8", "replace"),
        "reported_model": reported.group(1).strip() if reported else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("task", help="task id, e.g. pdet-coverage-v1")
    parser.add_argument("--annotator", required=True, choices=sorted(ANNOTATORS))
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument(
        "--max-batches", type=int, default=0, help="0 = until every item is labelled"
    )
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=3600, help="seconds per CLI run")
    parser.add_argument(
        "--retry-wait",
        type=int,
        default=90,
        help="seconds to wait before retrying (network drops are common)",
    )
    args = parser.parse_args(argv)

    spec = ANNOTATORS[args.annotator]
    executable = shutil.which(spec["executable"])
    if executable is None:
        raise SystemExit(f"{spec['executable']} is not on PATH")
    config = load_task_config(ROOT / "configs" / "annotation" / f"{args.task}.yaml", root=ROOT)
    procedure = procedure_text(config, args.annotator)
    version = cli_version(executable, spec["version_argv"])
    state_db = config.resolve(config.state_db)
    batch_dir = state_db.parent / f"{config.task_id}.model-batches" / spec["session"]
    batch_dir.mkdir(parents=True, exist_ok=True)

    done_batches = 0
    while not args.max_batches or done_batches < args.max_batches:
        workspace = Workspace.open(config, state_db=state_db)
        try:
            prepared = [
                p for p in batch_dir.glob("batch-*.json") if p.stem[len("batch-") :].isdigit()
            ]
            batch = prepare_batch(
                workspace,
                spec["session"],
                args.annotator,
                size=args.size,
                defer_to=[],
                batch_id=f"{len(prepared) + 1:02d}",
            )
            if not batch["item_ids"]:
                progress = workspace.progress(spec["session"])
                print(
                    f"done      every item is labelled in {spec['session']} ({progress['completed']}/{progress['total']})"
                )
                return 0
            _json_path, md_path = write_batch(batch, config, batch_dir)
        finally:
            workspace.close()

        prompt = procedure + b"\n" + md_path.read_bytes()
        recorded = False
        for attempt in range(1, args.attempts + 1):
            stem = f"batch-{batch['batch_id']}.attempt-{attempt}"
            run = run_once(spec, executable, prompt, batch_dir, stem, args.timeout)
            text = run.pop("_text")
            answers = extract_answers(text, batch["item_ids"])
            record: dict[str, Any] = {
                "task_id": config.task_id,
                "session_id": spec["session"],
                "annotator_id": args.annotator,
                "cli_version": version,
                "batch_id": batch["batch_id"],
                "batch_content_sha256": batch["content_sha256"],
                "procedure_sha256": batch["procedure_sha256"],
                "attempt": attempt,
                **run,
                "answers_found": answers is not None,
            }
            status = "no JSON answer array in the output"
            if answers is not None:
                (batch_dir / f"{stem}.answers.json").write_bytes(
                    (json.dumps(answers, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
                )
                workspace = Workspace.open(config, state_db=state_db)
                try:
                    result = ingest_batch(workspace, batch, answers)
                    record["ingested"] = result["recorded"]
                    status = f"recorded {result['recorded']} labels ({result['flagged']} flagged)"
                    recorded = True
                except BatchError as exc:
                    problems = str(exc).splitlines()
                    record["ingest_problems"] = len(problems) - 1
                    status = f"rejected: {problems[0]}"
                finally:
                    workspace.close()
            record["status"] = status
            (batch_dir / f"{stem}.run.json").write_bytes(
                (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
            )
            print(
                f"batch {batch['batch_id']} ({len(batch['item_ids'])} items) attempt {attempt}: "
                f"exit {run['exit_code']}, {run['seconds']}s, {status}"
            )
            if recorded:
                break
            if attempt < args.attempts:
                time.sleep(args.retry_wait)
        if not recorded:
            print(
                f"stopped   batch {batch['batch_id']} was not recorded after {args.attempts} attempts"
            )
            return 1
        done_batches += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
