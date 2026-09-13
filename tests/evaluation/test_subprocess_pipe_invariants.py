"""No long-running subprocess may be launched with an undrained PIPE.

This encodes a defect that already cost a full evaluation run. `llama-server` was launched with
`stdout=subprocess.PIPE` and nothing ever read that pipe. llama-server logs a few lines per
request; once ~64KB filled the OS pipe buffer the server blocked on write and stopped responding —
no crash, no exit, no error. A 1,277-prompt run sat dead for 35 minutes. With the pipe redirected
to a file the identical workload completed in 213 seconds.

The failure is invisible precisely because a blocked writer looks exactly like a slow one. So it is
caught structurally instead of by observation: `Popen` may not hold a `PIPE` it does not drain.

Acceptable patterns:
  * `stdout=<file object>`   — bounded, inspectable, what the fix uses
  * `stdout=subprocess.DEVNULL`
  * `stdout=subprocess.PIPE` **with** a `.communicate()` in the same function (which drains)
  * `subprocess.run(..., capture_output=True)` — drains internally, not a Popen

`.wait()` after a PIPE is the dangerous one and is treated as a failure: it waits for a process
that may itself be blocked writing into the pipe nobody is reading.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCANNED_DIRS = ("scripts", "release", "src")

# Files allowed to hold a PIPE without a same-function drain, each with a reason. Empty by
# design — an entry here is a deliberate, reviewed exception, not a convenience.
ALLOWLIST: dict[str, str] = {}


def python_files() -> list[Path]:
    out: list[Path] = []
    for name in SCANNED_DIRS:
        base = ROOT / name
        if base.is_dir():
            out.extend(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)
    return out


def _is_pipe(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "PIPE"
        and isinstance(node.value, ast.Name)
        and node.value.id == "subprocess"
    )


def _enclosing_function(tree: ast.AST, target: ast.AST):
    """Smallest function body containing `target`, so the drain search is correctly scoped."""
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start, end = node.lineno, getattr(node, "end_lineno", node.lineno)
            if start <= target.lineno <= end and (best is None or node.lineno > best.lineno):
                best = node
    return best


def _has_drain(scope: ast.AST) -> bool:
    for node in ast.walk(scope):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "communicate"
        ):
            return True
    return False


def find_violations() -> list[tuple[Path, int, str]]:
    violations: list[tuple[Path, int, str]] = []
    for path in python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # not our concern here; other tests cover importability
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr != "Popen":
                continue
            piped = [
                kw.arg for kw in node.keywords
                if kw.arg in ("stdout", "stderr") and _is_pipe(kw.value)
            ]
            if not piped:
                continue
            scope = _enclosing_function(tree, node) or tree
            if _has_drain(scope):
                continue
            rel = path.relative_to(ROOT).as_posix()
            if rel in ALLOWLIST:
                continue
            violations.append((path, node.lineno, ",".join(piped)))
    return violations


def test_no_popen_holds_an_undrained_pipe():
    violations = find_violations()
    rendered = "\n".join(
        f"  {p.relative_to(ROOT).as_posix()}:{line} passes {kinds}=subprocess.PIPE "
        "with no communicate() in the enclosing function"
        for p, line, kinds in violations
    )
    assert not violations, (
        "undrained subprocess PIPE(s) found — this is the llama-server deadlock class:\n"
        + rendered
        + "\nRedirect to a file object or DEVNULL, or drain with communicate()."
    )


def test_llama_server_launcher_logs_to_a_file():
    """Pin the specific fix, so a refactor cannot revert it while still passing the scan."""
    study = ROOT / "scripts/modal/gguf_study.py"
    if not study.is_file():
        pytest.skip("gguf_study.py not present")
    source = study.read_text(encoding="utf-8")
    tree = ast.parse(source)

    server = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "Server"), None
    )
    assert server is not None, "Server class not found"

    popens = [
        n for n in ast.walk(server)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "Popen"
    ]
    assert popens, "Server no longer launches a subprocess; re-check this invariant"
    for call in popens:
        stdout = next((kw.value for kw in call.keywords if kw.arg == "stdout"), None)
        assert stdout is not None, "Server Popen must set stdout explicitly"
        assert not _is_pipe(stdout), (
            "Server must not pipe llama-server's stdout — it deadlocks once the OS buffer fills"
        )
    assert "log_path" in source, "Server should retain file-backed logging"


def test_the_scanner_actually_detects_the_bad_pattern(tmp_path):
    """A scanner that cannot fail proves nothing."""
    bad = tmp_path / "bad.py"
    bad.write_text(
        "import subprocess\n"
        "def launch():\n"
        "    p = subprocess.Popen(['x'], stdout=subprocess.PIPE)\n"
        "    p.wait()\n",
        encoding="utf-8",
    )
    tree = ast.parse(bad.read_text(encoding="utf-8"))
    call = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "Popen"
    )
    assert any(_is_pipe(kw.value) for kw in call.keywords if kw.arg == "stdout")
    assert _has_drain(_enclosing_function(tree, call)) is False

    good = tmp_path / "good.py"
    good.write_text(
        "import subprocess\n"
        "def launch():\n"
        "    with open('/tmp/log', 'w') as fh:\n"
        "        p = subprocess.Popen(['x'], stdout=fh, stderr=subprocess.STDOUT)\n"
        "        p.wait()\n",
        encoding="utf-8",
    )
    tree_good = ast.parse(good.read_text(encoding="utf-8"))
    call_good = next(
        n for n in ast.walk(tree_good)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "Popen"
    )
    assert not any(_is_pipe(kw.value) for kw in call_good.keywords if kw.arg == "stdout")
