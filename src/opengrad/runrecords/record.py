"""Record a run in full: what ran, on what code and machine, what it printed, and how it ended.

Why this exists
---------------
The owner asked (2026-11-11) for every run to be fully logged, so that failures, surprises and workarounds can be
traced, and so that a problem in a library can later go upstream with its evidence (``docs/UPSTREAM_ISSUES.md``).
The experiment store (``runs/``) records configs, metrics and a basic environment, but not the exact command, the
installed library versions, the GPU, the full output or the exit status. This module adds that as a sibling
record. It never writes ``runs/`` or changes ``env_capture.capture()``, ``environment.json`` or ``experiment.json``.

Layout
------
* ``.run-records/<run_id>/`` (git-ignored; ``OPENGRAD_RUN_RECORDS_DIR`` overrides the folder, which the test suite
  uses so no test writes into the repository): ``start.json`` written before the run, ``output.log`` with its
  output, and ``summary.json`` written when it ends. A start with no summary reads as ``INCOMPLETE``.
* ``run-records/<run_id>.json`` (tracked): the summary, exported only when the owner commits it. Writing nothing
  tracked during a run keeps ``git status`` unchanged, so ``env_capture.capture()``'s ``git_dirty`` means what it
  meant before.

What the log holds: ``run_child`` (a command run as a child process) captures everything the child writes to its
standard output and error. ``recorded_call`` (a Python call in this process) captures what is written through
``sys.stdout`` and ``sys.stderr``, including their ``.buffer``; it does not capture output written straight to the
file descriptors by C extensions (CUDA, NCCL) or by child processes the call starts.

Privacy: the command and any error text are redacted (secrets, credential flags and values, URL credentials,
absolute paths); only an allowlist of environment variables is kept; the full log is never committed, only its
sha256. A recorder failure never stops or changes the run: it is reported on standard error instead.
"""

from __future__ import annotations

import codecs
import contextlib
import importlib.metadata
import json
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from opengrad.env_capture import tracked_tree_provenance
from opengrad.hashing import sha256_bytes, sha256_file

RECORD_SCHEMA_VERSION = 1
LOCAL_DIR = Path(".run-records")
LOCAL_DIR_ENV = "OPENGRAD_RUN_RECORDS_DIR"
#: Set by ``run_child`` for its child. A recorded call inside such a child is already captured in full by the
#: parent's record (including C-extension and worker-process output), so it does not start a second record.
PARENT_ENV = "OPENGRAD_RUN_RECORD_PARENT"
COMMITTED_DIR = Path("run-records")
KINDS = ("training", "evaluation", "freeze", "script", "annotation", "ci")
STATUSES = ("PASS", "FAILED", "INTERRUPTED", "INCOMPLETE")

#: Kept byte-identical to ``SECRET`` in ``scripts/repo/check_publication_hygiene.py`` (a test checks it).
SECRET = re.compile(
    r"(?:sk-[A-Za-z0-9]{20,}"
    r"|OPENAI_API_KEY\s*=\s*['\"][A-Za-z0-9_\-]+['\"]"
    r"|ghp_[A-Za-z0-9]{36}"
    r"|gh[ousr]_[A-Za-z0-9]{36}"
    r"|github_pat_[A-Za-z0-9_]{22,}"
    r"|\bhf_[A-Za-z0-9]{30,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----)"
)
#: Kept byte-identical to ``HOME_PATH`` in ``scripts/repo/check_publication_hygiene.py`` (a test checks it).
HOME_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/]+Users(?:\\\\|\\|/)|/Users/[^/\s\"']+/|/home/[^/\s\"']+/)"
)
#: Credentials the scanner's token list does not name.
GOOGLE_KEY = re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")
BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-~+/=]{8,}")
URL_CREDENTIALS = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@")
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
#: Words that mark a name as holding a credential, when one of them is a whole part of the name.
SECRET_WORDS = frozenset(
    {
        "token",
        "secret",
        "pass",
        "password",
        "passwd",
        "key",
        "apikey",
        "auth",
        "credential",
        "credentials",
        "bearer",
    }
)
#: A first name part that makes a flag a true/false switch (``--no-auth``), which takes no value to hide.
SWITCH_PREFIXES = frozenset(
    {"no", "use", "enable", "disable", "skip", "with", "without", "require"}
)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAPS_ENDING = re.compile(r"(?:TOKEN|SECRET|PASSWORD|PASSWD|APIKEY|KEY|AUTH|CREDENTIALS?)$")


def name_parts(name: str) -> list[str]:
    """``X-Api-Key`` -> x, api, key; ``accessToken`` -> access, token; ``db.password`` -> db, password."""
    parts: list[str] = []
    for chunk in re.split(r"[-_.]+", name.strip("-_.")):
        parts += [part.lower() for part in _CAMEL.split(chunk) if part]
    return parts


def is_secret_name(name: str) -> bool:
    """True when a flag or field name says it holds a credential. ``HF_TOKEN``, ``api-key``, ``db.password``,
    ``X-Api-Key``, ``accessToken``, ``secretKey`` and ``MYTOKEN`` do; ``max_tokens``, ``author``, ``monkey``,
    ``passthrough`` and ``tokenizer`` do not."""
    if any(part in SECRET_WORDS for part in name_parts(name)):
        return True
    bare = name.strip("-_.")
    return bare.isupper() and bool(_CAPS_ENDING.search(bare.replace("_", "")))


def is_secret_flag(flag: str) -> bool:
    """A flag whose value is a credential, whatever the value looks like. A switch such as ``--no-auth`` is not."""
    if not flag.startswith("-") or len(flag) < 2:
        return False
    parts = name_parts(flag)
    if parts and parts[0] in SWITCH_PREFIXES:
        return False
    return is_secret_name(flag.lstrip("-"))


#: An exception type such as ``KeyError``: in ``KeyError: 'token'`` what follows is the missing key's name, which an
#: error record needs, not a credential.
EXCEPTION_NAME = re.compile(r"(?:Error|Exception|Warning)$")
#: A NAME=value argument (``env HF_TOKEN=... cmd``); the name is checked with ``is_secret_name``.
ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_.-]*)=.*$", re.DOTALL)
#: A field anywhere in text: JSON, ``key=value``, ``key: value`` (a header), dotted names, an assignment in the
#: middle of a shell string. The value may be double-quoted (with ``\"`` escapes), single-quoted or bare. Only a
#: field whose name passes ``is_secret_name`` has its value hidden.
FIELD = re.compile(
    r"""(["']?)([A-Za-z_][A-Za-z0-9_.-]*)\1(\s*[:=]\s*)("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s,;&"'}\]]+)"""
)


#: An absolute path: drive-letter, UNC, POSIX or file URL. Used for whole arguments and inside text.
ABSOLUTE_START = re.compile(r"^(?:file:/+|[A-Za-z]:[\\/]|\\\\|/)")
ABSOLUTE_INSIDE = re.compile(
    r"(?:file:/+[^\s\"']*|(?<![\w.])[A-Za-z]:[\\/][^\"'\n]*?(?=$|[\"'\n]|\s-)|\\\\[^\s\"']+"
    r"|(?<![\w.\-])/(?:home|Users|root|mnt|tmp|workspace|opt|data|srv|var|private)/[^\s\"']*)"
)
#: Environment variables recorded by name and value. Nothing else from the environment is kept.
ENV_ALLOWLIST = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _warn(message: str) -> None:
    # A warning must never break the run, so a failure to print it is ignored.
    with contextlib.suppress(Exception):
        sys.__stderr__.write(f"[run record] {message}\n")  # type: ignore[union-attr]


def local_base(root: Path) -> Path:
    override = os.environ.get(LOCAL_DIR_ENV)
    return Path(override) if override else root / LOCAL_DIR


def hide_secret_fields(text: str) -> str:
    """Hide the value of every credential-named field. After a field whose name is not a credential, scanning
    resumes at its value, so ``--set=api_key=x`` still reaches ``api_key``."""
    out: list[str] = []
    pos = 0
    while (match := FIELD.search(text, pos)) is not None:
        if is_secret_name(match.group(2)) and not EXCEPTION_NAME.search(match.group(2)):
            value = match.group(4)
            quote = value[0] if value[:1] in {'"', "'"} else ""
            out.append(text[pos : match.start(4)] + f"{quote}<redacted>{quote}")
            pos = match.end(4)
        else:
            out.append(text[pos : match.end(3)])
            pos = match.end(3)
    out.append(text[pos:])
    return "".join(out)


def redact_text(text: str, root: Path) -> str:
    """Free text (an argument, an error message) with credentials and absolute paths removed."""
    text = SECRET.sub("<redacted>", text)
    text = GOOGLE_KEY.sub("<redacted>", text)
    text = BEARER.sub("Bearer <redacted>", text)
    text = URL_CREDENTIALS.sub(r"\1<redacted>@", text)
    text = hide_secret_fields(text)
    resolved = str(root.resolve())
    for prefix in {resolved, resolved.replace("\\", "/")}:
        # Only a full path component: "...\OpenGradX" is not inside "...\OpenGrad".
        text = re.sub(
            re.escape(prefix) + r"(?:[\\/]|$)",
            lambda m: "" if m.group(0)[-1:] in "\\/" else ".",
            text,
        )
    text = ABSOLUTE_INSIDE.sub("<absolute-path>", text)
    return HOME_PATH.sub("<absolute-path>/", text)


def redact_command(argv: Sequence[str], root: Path) -> list[str]:
    out: list[str] = []
    hide_next = False
    resolved = str(root.resolve())
    for value in argv:
        if hide_next:
            out.append("<redacted>")
            hide_next = False
            continue
        assignment = ASSIGNMENT.match(value)
        if assignment and not value.startswith("-") and is_secret_name(assignment.group(1)):
            out.append(f"{assignment.group(1)}=<redacted>")
            continue
        flag, sep, rest = value.partition("=")
        if is_secret_flag(flag):
            if sep:
                out.append(f"{flag}=<redacted>")
            else:
                out.append(value)
                hide_next = True
            continue
        inside = value.startswith((resolved, resolved.replace("\\", "/")))
        if not inside and ABSOLUTE_START.match(value):
            out.append("<absolute-path>")
            continue
        if sep and flag.startswith("-") and ABSOLUTE_START.match(rest):
            out.append(f"{flag}=<absolute-path>")
            continue
        short = re.match(r"^(-[A-Za-z])(.+)$", value)
        if short and ABSOLUTE_START.match(short.group(2)):
            out.append(f"{short.group(1)}<absolute-path>")
            continue
        text = redact_text(value, root)
        out.append(text.replace("\\", "/") if inside else text)
    return out


def _git(root: Path, *args: str) -> bytes | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, check=True, timeout=60
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def git_state(root: Path) -> dict[str, Any]:
    provenance = tracked_tree_provenance(root)
    diff = _git(root, "diff", "HEAD", "--binary") if provenance["dirty"] else b""
    return {
        "sha": provenance["sha"],
        "tracked_dirty": provenance["dirty"],
        "tracked_diff_sha256": sha256_bytes(diff) if diff else None,
    }


def distributions() -> dict[str, str]:
    """Every installed distribution and its version, read from metadata (nothing is imported)."""
    found: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            found[str(name).lower()] = dist.version
    return dict(sorted(found.items()))


def gpu_probe(timeout: float = 10.0) -> dict[str, Any]:
    """What the GPU tools report, without importing torch. ``none`` when neither tool is installed."""
    probes = (
        ("nvidia-smi", ["--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        ("rocm-smi", ["--showproductname", "--showdriverversion"]),
    )
    for tool, args in probes:
        path = shutil.which(tool)
        if not path:
            continue
        try:
            done = subprocess.run(
                [path, *args], capture_output=True, text=True, timeout=timeout, check=False
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return {"probe": tool, "error": type(exc).__name__, "devices": []}
        lines = [line.strip() for line in done.stdout.splitlines() if line.strip()]
        return {"probe": tool, "exit_code": done.returncode, "devices": lines[:16]}
    return {"probe": "none", "devices": []}


def environment_snapshot(root: Path) -> dict[str, Any]:
    lock = root / "uv.lock"
    return {
        "git": git_state(root),
        "uv_lock_sha256": sha256_file(lock) if lock.is_file() else None,
        "python": sys.version.split()[0],
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "distributions": distributions(),
        "gpu": gpu_probe(),
        "env": {name: os.environ[name] for name in ENV_ALLOWLIST if name in os.environ},
    }


def new_run_id(kind: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{kind}-{secrets.token_hex(3)}"


def log_info(path: Path, root: Path, run_id: str) -> dict[str, Any]:
    try:
        shown = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        shown = (LOCAL_DIR / run_id / "output.log").as_posix()
    present = path.exists()
    return {
        "path": shown,
        "bytes": path.stat().st_size if present else 0,
        "sha256": sha256_file(path) if present else None,
        "durable": None,
    }


class RunRecord:
    """One run's files under the local run-record folder."""

    def __init__(
        self, root: Path, kind: str, argv: Sequence[str], run_id: str | None = None
    ) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind {kind!r} is not one of {KINDS}")
        self.root = root
        self.kind = kind
        self.run_id = run_id or new_run_id(kind)
        self.dir = local_base(root) / self.run_id
        self.log_path = self.dir / "output.log"
        self.command = redact_command(argv, root)
        self.started = time.monotonic()
        self.start: dict[str, Any] = {}
        self.finished = False

    def begin(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=False)
        self.start = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "run_id": self.run_id,
            "kind": self.kind,
            "command": self.command,
            "started_at": _now(),
            **environment_snapshot(self.root),
        }
        # Written before the run starts, so a run that is killed still leaves a record (read as INCOMPLETE).
        _write_json(self.dir / "start.json", self.start)
        self.log_path.touch()

    def finish(
        self, exit_code: int | None, status: str, error: str | None = None
    ) -> dict[str, Any] | None:
        """Write the summary. Never raises: a recorder failure must not replace the run's own outcome."""
        if self.finished:
            return None
        self.finished = True
        try:
            summary = {
                **self.start,
                "ended_at": _now(),
                "duration_s": round(time.monotonic() - self.started, 3),
                "exit_code": exit_code,
                "status": status,
                "error": redact_text(error, self.root) if error else None,
                "log": log_info(self.log_path, self.root, self.run_id),
                "disposition": None,
            }
            _write_json(self.dir / "summary.json", summary)
            return summary
        except Exception as exc:  # noqa: BLE001 -- see the docstring
            _warn(f"could not finish {self.run_id}: {type(exc).__name__}: {exc}")
            return None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    tmp.replace(path)


class _BytesTee:
    def __init__(self, stream: Any, log: TextIO) -> None:
        self.stream = stream
        self.log = log

    def write(self, data: bytes) -> int:
        self.log.write(bytes(data).decode("utf-8", errors="replace"))
        self.log.flush()
        return int(self.stream.write(data))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)


class _Tee:
    """Writes to the console and the run's log, including writes through ``.buffer``."""

    def __init__(self, stream: TextIO, log: TextIO) -> None:
        self.stream = stream
        self.log = log

    def write(self, text: str) -> int:
        self.log.write(text)
        self.log.flush()
        return self.stream.write(text)

    def flush(self) -> None:
        self.stream.flush()
        self.log.flush()

    @property
    def buffer(self) -> Any:
        return _BytesTee(self.stream.buffer, self.log)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)


def recorded_call(kind: str, argv: Sequence[str], root: Path, call: Callable[[], int]) -> int:
    """Run ``call()`` in this process as a recorded run and return its exit code.

    An exception or ``SystemExit`` is recorded and then re-raised unchanged. If the record cannot be started, the
    call still runs, unrecorded, with a warning: recording never blocks a run.
    """
    parent = os.environ.get(PARENT_ENV)
    if parent:
        parent_dir = local_base(root) / parent
        if (parent_dir / "start.json").is_file() and not (parent_dir / "summary.json").exists():
            # Launched through `python -m opengrad.runrecords run`, whose open record already holds everything.
            return call()
        # A marker left over in the environment (an exported variable, a shell started by the launcher) must not
        # switch recording off: record anyway.
        _warn(f"{PARENT_ENV}={parent} names no open run record; recording this run itself")
    record = RunRecord(root, kind, argv)
    try:
        record.begin()
        log: TextIO = record.log_path.open(
            "a", encoding="utf-8", errors="backslashreplace", newline="\n"
        )
    except Exception as exc:  # noqa: BLE001 -- see the docstring
        _warn(f"run not recorded ({type(exc).__name__}: {exc})")
        return call()
    out, err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(out, log), _Tee(err, log)
    try:
        code = call()
    except KeyboardInterrupt:
        sys.stdout, sys.stderr = out, err
        log.close()
        record.finish(None, "INTERRUPTED", "KeyboardInterrupt")
        raise
    except SystemExit as exc:
        sys.stdout, sys.stderr = out, err
        log.close()
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        record.finish(code, "PASS" if code == 0 else "FAILED")
        raise
    except BaseException as exc:
        sys.stdout, sys.stderr = out, err
        log.close()
        record.finish(None, "FAILED", f"{type(exc).__name__}: {exc}")
        raise
    sys.stdout, sys.stderr = out, err
    log.close()
    record.finish(code, "PASS" if code == 0 else "FAILED")
    return code


def _console_writer(console: TextIO) -> Callable[[bytes], None]:
    """Decode output incrementally (a character split across chunks is kept) and write what the console can show."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    encoding = getattr(console, "encoding", None) or "utf-8"

    def write(chunk: bytes) -> None:
        text = decoder.decode(chunk)
        if text:
            console.write(
                text.encode(encoding, errors="replace").decode(encoding, errors="replace")
            )
            console.flush()

    return write


def run_child(kind: str, command: Sequence[str], root: Path, console: TextIO | None = None) -> int:
    """Run ``command`` as a child process, record it, and return its exit code.

    The child's output and error are merged and copied to the log and the console as they arrive, so a child that
    is killed still leaves what it printed. Python children are asked to write unbuffered UTF-8, which a Windows pipe
    would otherwise not get. Whatever goes wrong in the recorder, the child is killed and waited on and the record
    is finished.
    """
    console = console if console is not None else sys.stdout
    record = RunRecord(root, kind, command)
    try:
        record.begin()
    except Exception as exc:  # noqa: BLE001 -- recording never blocks a run
        _warn(f"run not recorded ({type(exc).__name__}: {exc}); running it anyway")
        return subprocess.call(list(command), cwd=root)
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        PARENT_ENV: record.run_id,
    }
    try:
        proc = subprocess.Popen(
            list(command),
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=0,
        )
    except OSError as exc:
        record.finish(None, "FAILED", f"{type(exc).__name__}: {exc}")
        return 127
    assert proc.stdout is not None
    stream = proc.stdout.fileno()
    show = _console_writer(console)
    try:
        with record.log_path.open("ab") as log:
            for chunk in iter(lambda: os.read(stream, 65536), b""):
                log.write(chunk)
                log.flush()
                try:
                    show(chunk)
                except Exception as exc:  # noqa: BLE001 -- a console problem must not lose the record
                    _warn(f"console output failed ({type(exc).__name__}); the log keeps everything")
                    show = lambda _chunk: None
        code = proc.wait()
        proc.stdout.close()
    except BaseException as exc:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        proc.stdout.close()
        interrupted = isinstance(exc, KeyboardInterrupt)
        record.finish(
            None, "INTERRUPTED" if interrupted else "FAILED", f"{type(exc).__name__}: {exc}"
        )
        raise
    record.finish(code, "PASS" if code == 0 else "FAILED")
    return code


def read_local(root: Path) -> list[dict[str, Any]]:
    """Every local record, newest first. A start without a summary is reported as INCOMPLETE."""
    records: list[dict[str, Any]] = []
    base = local_base(root)
    if not base.is_dir():
        return records
    for directory in base.iterdir():
        summary, start = directory / "summary.json", directory / "start.json"
        if summary.is_file():
            records.append(json.loads(summary.read_text(encoding="utf-8")))
        elif start.is_file():
            records.append(
                {**json.loads(start.read_text(encoding="utf-8")), "status": "INCOMPLETE"}
            )
    return sorted(records, key=lambda r: (str(r.get("started_at", "")), r["run_id"]), reverse=True)
