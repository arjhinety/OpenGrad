"""The console calls of the entry points in ``PINNED_CONSOLE`` (other than the labelling runner) are pinned.

These entry points reach held-out data only through ``opengrad.data.classifier_input``, which loads the When2Call test
questions as a set to exclude matching training records and never prints them. On 2026-11-13 a read of every
object they print, following helpers into the modules they load, found only counts, ids, hashes, reason codes,
labels, paths and fixed text, so the owner let their logs upload.

This test snapshots, for each of them and every repository file it loads (``opengrad.runrecords.scripts.
import_closure``: imports, relative ones included, and ``.py`` files loaded by path), every console call
(``print``, logging, warnings, traceback, ``os.write``) and every expression that touches ``sys.stdout`` or
``sys.stderr`` (``json.dump(x, sys.stdout)``, an alias). Any change fails until reviewed.

What it does not pin:

* the contents of what a pinned call prints. ``print(json.dumps(summary(...)))`` is pinned as a call; a change to
  what ``summary`` returns is not seen;
* exception text. These entry points are not converted yet and run under ``run_child``, so an uncaught exception's
  traceback lands in the uploaded log (the committed summary keeps only its type);
* console calls the name prefixes below miss (``LOG.info``, ``self.logger.info``, ``rich.print``, a child process
  whose output is not captured). None existed on 2026-11-13.
 Whoever changes such a function in these files reviews the
printed object again (does it now carry record text?). After a review, regenerate the snapshot with
``python tests/runrecords/test_console_pins.py`` and say what was reviewed in the commit message.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from opengrad.runrecords import scripts as rs

ROOT = Path(__file__).parents[2]
PINS = Path(__file__).with_name("console_pins.json")
RUNNER = "scripts/run_external_annotation.py"  # pinned by its own, stricter test
#: Calls that write to the console, by the start of their unparsed function name.
CONSOLE_CALLS = ("print", "os.write", "logging", "logger", "log.", "warnings", "traceback", "pprint", "console.")
#: Any reference to these is pinned wherever it appears.
STREAMS = ("sys.stdout", "sys.stderr", "sys.__stdout__", "sys.__stderr__")


def _tracked() -> set[str]:
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return {p for p in out.split() if p.startswith(("scripts/", "src/"))}


def _console(path: str) -> list[str]:
    found = []
    for node in ast.walk(ast.parse((ROOT / path).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and ast.unparse(node.func).startswith(CONSOLE_CALLS):
            found.append(ast.unparse(node))
        elif isinstance(node, ast.Attribute) and ast.unparse(node).startswith(STREAMS):
            found.append(f"stream: {ast.unparse(node)}")
    return sorted(found)


def snapshot() -> dict[str, dict[str, list[str]]]:
    """Pinned entry point -> {file it loads -> its console calls}, files with none left out."""
    tracked = _tracked()
    out: dict[str, dict[str, list[str]]] = {}
    for entry in sorted(set(rs.PINNED_CONSOLE) - {RUNNER}):
        # The recorder's own files (rs.POLICY_FILES) are left out: an entry point imports them only to record itself,
        # their console handling copies the program's own output, and tests/runrecords covers them.
        loaded = sorted(rs.import_closure(entry, ROOT, tracked) - rs.POLICY_FILES)
        files = {path: _console(path) for path in loaded}
        out[entry] = {path: calls for path, calls in files.items() if calls}
    return out


def test_the_pinned_entry_points_point_here() -> None:
    for entry, test in rs.PINNED_CONSOLE.items():
        if entry != RUNNER:
            assert test == "tests/runrecords/test_console_pins.py", entry


def test_the_console_calls_are_as_reviewed() -> None:
    pinned = json.loads(PINS.read_text(encoding="utf-8"))
    current = snapshot()
    assert sorted(current) == sorted(pinned), "the pinned entry points changed; regenerate after review"
    for entry in current:
        changed = sorted(
            path
            for path in set(current[entry]) | set(pinned[entry])
            if current[entry].get(path) != pinned[entry].get(path)
        )
        assert not changed, f"{entry}: console calls changed in {changed}; review them, then regenerate the pins"


def test_the_snapshot_sees_a_stream_alias_and_a_dump_to_stdout(tmp_path: Path, monkeypatch) -> None:
    module = tmp_path / "x.py"
    module.write_text("import json, sys\nw = sys.stdout.write\njson.dump({}, sys.stderr)\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    assert _console("x.py") == ["stream: sys.stderr", "stream: sys.stdout", "stream: sys.stdout.write"]


if __name__ == "__main__":
    PINS.write_text(json.dumps(snapshot(), indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    sys.stdout.write(f"wrote {PINS.relative_to(ROOT).as_posix()}\n")
