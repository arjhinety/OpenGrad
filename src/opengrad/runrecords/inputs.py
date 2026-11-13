"""Checks on a run's inputs, for run-log upload: held-out markers in its arguments and the configs they name, the
files its entry point loads and whether they match reviewed code. ``opengrad.runrecords.record`` records them when
a run starts; ``opengrad.runrecords.scripts.upload_decision`` reads them. Moved out of ``record.py`` (its size limit).
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: Strings that mark held-out evaluation data: in an entry point's code (``opengrad.runrecords.scripts``), in a
#: run's arguments, or in a ``configs/`` file a run is given. Such a run's log is never uploaded.
HELD_OUT_MARKERS = (
    "frozen_prompts",
    "when2call_test",
    "pconf-v1",
    "answer-records",
    "behavioral-heldout-v2-partition",
    "confirmatory",
    "punans",
)
#: Inputs only (a run's arguments and the configs they name), not the code scan: the frozen held-out set's manifest,
#: partition, quarantine and contamination files all start with it (owner's decision, 2026-11-13, on Fable's and
#: Gemini 4 Argon's advice). Not counted under ``INPUT_EXEMPT_KEYS``.
INPUT_MARKERS = (*HELD_OUT_MARKERS, "behavioral-heldout")
#: Config keys reviewed as documenting or excluding the held-out set, never reading it as data: a value under them
#: is not checked for ``behavioral-heldout`` (the markers in ``HELD_OUT_MARKERS`` still count). A key not listed
#: here counts, so a new key that reads the set keeps its run's log local (fail closed).
INPUT_EXEMPT_KEYS = frozenset({"checkpoint_selection_manifest", "exclude_held_out_namespaces", "contamination_report"})


def argument_values(token: str) -> list[str]:
    """The values an argument can carry: itself, the value of ``--flag=value``, and of an attached ``-fvalue``."""
    value = str(token).replace("\\", "/")
    values = [value]
    if value.startswith("-"):
        if "=" in value:
            values.append(value.split("=", 1)[1])
        if not value.startswith("--") and len(value) > 2:
            values.append(value[2:])
    return values


def names_held_out(token: str) -> bool:
    """True when an argument names a held-out marker (``INPUT_MARKERS``), in any case."""
    folded = str(token).replace("\\", "/").casefold()
    return any(marker in folded for marker in INPUT_MARKERS)


#: A piece of a spaced config value that looks like a path or a file name.
_PATH_LIKE = re.compile(r"[/\\]|\.[A-Za-z0-9]{1,6}$")


def _config_identifiers(path: Path, text: str) -> list[tuple[str, bool]]:
    """What in a config can point at data: its keys and its values with no whitespace (paths, ids, namespaces).

    A description that mentions held-out data in prose ("confirmatory IDs are excluded") does not reach it, just
    as a marker in a code comment does not. In a value with spaces (a command line, a list of paths), the pieces
    that look like paths or file names count. A file that does not parse as YAML, JSON or TOML, or of another
    kind, is read whole: fail closed. Each item comes with whether it sits under a key in ``INPUT_EXEMPT_KEYS``.
    """
    suffix = path.suffix.casefold()
    try:
        if suffix in (".yaml", ".yml"):
            import yaml  # optional extra: without it the whole text is read

            documents = list(yaml.safe_load_all(text))
        elif suffix == ".json":
            documents = [json.loads(text)]
        elif suffix == ".toml":
            import tomllib

            documents = [tomllib.loads(text)]
        else:
            return [(text, False)]
    except Exception:  # noqa: BLE001 -- unparsable: read whole
        return [(text, False)]
    found: list[tuple[str, bool]] = []
    stack: list[tuple[Any, bool]] = [(document, False) for document in documents]
    # A YAML alias can make a node contain itself, or place one node under an exempt key and a reading key both: a
    # node is visited once per exempt state, so its non-exempt visit is never skipped.
    seen: set[tuple[int, bool]] = set()
    while stack:
        item, exempt = stack.pop()
        if isinstance(item, dict | list):
            if (id(item), exempt) in seen:
                continue
            seen.add((id(item), exempt))
        if isinstance(item, dict):
            for key, value in item.items():
                found.append((str(key), exempt))
                stack.append((value, exempt or str(key) in INPUT_EXEMPT_KEYS))
        elif isinstance(item, list):
            stack.extend((entry, exempt) for entry in item)
        elif isinstance(item, str):
            pieces = item.split()
            if len(pieces) == 1:
                found.append((pieces[0], exempt))
            else:
                # A command line or a list of paths in one value: its path-like pieces count, its words do not.
                found.extend((piece, exempt) for piece in pieces if _PATH_LIKE.search(piece))
    return found


def repo_relative(value: str, root: Path) -> str:
    """An argument as a normalised repository-relative path when it is one: an absolute path inside the checkout
    (any drive-letter case) is made relative, a relative one normalised (``./``, ``//``, ``..``)."""
    value = str(value).replace("\\", "/")
    if not value:
        return ""
    if re.match(r"^(?:[A-Za-z]:/|/)", value):
        try:
            resolved = Path(value).resolve()
            base = root.resolve()
            # Compared case-folded: Windows paths ignore case, and a drive letter may be typed either way.
            if os.path.normcase(str(resolved)).startswith(os.path.normcase(str(base)) + os.sep):
                return str(resolved)[len(str(base)) + 1 :].replace("\\", "/")
        except (OSError, ValueError):
            pass
        return value
    return posixpath.normpath(value)


def _resolved_file(value: str, root: Path) -> str | None:
    """The repository path an argument names once resolved (from the checkout's root; links and junctions
    followed), when it is an existing file inside the checkout."""
    path = Path(value)
    candidate = path if path.is_absolute() else root / value
    try:
        if not candidate.is_file():
            return None
        relative = repo_relative(str(candidate.resolve()), root)
    except (OSError, ValueError):
        return None
    return None if re.match(r"^(?:[A-Za-z]:/|/)", relative) else relative


def config_state(token: str, root: Path) -> str:
    """``held_out`` when an argument is a repository ``configs/`` file that names a held-out marker, ``missing`` when
    it names a ``configs/`` file that is not there, else ``""``.

    A run can reach held-out data through the config it is given, which the argument does not show. Config files
    hold settings and paths, not test content, so reading one is safe. One level: a config naming another config
    is not followed. Nothing outside ``configs/`` is read.
    """
    base = (root / "configs").resolve()
    state = ""
    for value in argument_values(token):
        relative = repo_relative(value, root)
        resolved = _resolved_file(value, root)
        if resolved is not None:
            relative = resolved  # a path spelled from another folder, or through a junction
        if not relative.casefold().startswith("configs/"):
            continue
        path = (root / relative).resolve()
        if not path.is_relative_to(base):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            state = "missing"
            continue
        for item, exempt in _config_identifiers(path, text):
            markers = HELD_OUT_MARKERS if exempt else INPUT_MARKERS
            if any(marker in item.casefold() for marker in markers):
                return "held_out"
    return state


def _modified(root: Path) -> set[str] | None:
    """Repository paths with local edits or not tracked (``git status``); None when git cannot say."""
    try:
        done = subprocess.run(
            ["git", "--no-optional-locks", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=root,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    paths: set[str] = set()
    entries = done.stdout.decode("utf-8", "replace").split("\0")
    i = 0
    while i < len(entries):
        entry = entries[i]
        if len(entry) > 3:
            paths.add(entry[3:])
            if set(entry[:2]) & {"R", "C"}:  # a rename or copy (in either column) is followed by its old path
                i += 1
                if i < len(entries) and entries[i]:
                    paths.add(entries[i])
        i += 1
    return paths


def _tracked(root: Path) -> set[str] | None:
    """The paths git tracks in the checkout, less any git is told to ignore edits to (``--skip-worktree``,
    ``--assume-unchanged``: an edit there never shows in ``git status``); None when git cannot say."""
    try:
        done = subprocess.run(
            ["git", "--no-optional-locks", "ls-files", "-v", "-z"],
            cwd=root,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    tracked: set[str] = set()
    for entry in done.stdout.decode("utf-8", "replace").split("\0"):
        if len(entry) < 3:
            continue
        tag, path = entry[0], entry[2:]
        # "S": skip-worktree; a lower-case tag: assume-unchanged. Neither is vouched for.
        if tag != "S" and not tag.islower():
            tracked.add(path)
    return tracked


def blob_sha(data: bytes) -> str:
    """The git blob id of these bytes (what ``git hash-object`` prints), so a pushed version can be found."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _option(command: Sequence[str], name: str) -> str | None:
    tokens = [str(t) for t in command]
    for i, token in enumerate(tokens):
        if token == name and i + 1 < len(tokens):
            return tokens[i + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return None


def consumed_configs(command: Sequence[str], root: Path) -> list[str] | None:
    """The ``configs/`` files a run reads as its configuration: those its arguments name, and for ``opengrad
    evaluate`` the suite file and the benchmark configs it lists (resolved as ``opengrad.benchmarks.runner`` does).
    None when they cannot be determined, for example a benchmark config outside ``configs/`` (fail closed)."""
    from opengrad.runrecords import scripts  # imports this module: loaded at call time

    found: set[str] = set()
    for token in command:
        for value in argument_values(token):
            relative = _resolved_file(value, root)
            if relative is not None and relative.startswith("configs/"):
                found.add(relative)
    if scripts.console_target(command, root) == "opengrad.cli:main" and scripts.subcommand(command, root) == "evaluate":
        import yaml  # the benchmark runner needs it too

        suite = root / "configs" / "benchmark_suites" / f"{_option(command, '--suite') or 'smoke'}.yaml"
        if not suite.is_file():
            return None
        found.add(suite.relative_to(root).as_posix())
        data = yaml.safe_load(suite.read_text(encoding="utf-8"))
        names = data.get("benchmarks") if isinstance(data, dict) else None
        if not isinstance(names, list):
            return None
        base = root / "configs" / "benchmarks"
        for name in map(str, names):
            candidates = (base / f"{name}.yaml", base / f"{name.replace('-', '_')}.yaml")
            hit = next((c for c in candidates if c.is_file()), None)
            if hit is None:
                return None  # the runner would fall back to an arbitrary path: not vouched for
            found.add(hit.relative_to(root).as_posix())
    return sorted(found)


def entry_scan(command: Sequence[str], root: Path) -> tuple[list[str] | None, list[str] | None]:
    """When the run starts: the marker-naming files its entry point loads, scanned as they are on disk, and the files
    it loads (or any ``configs/`` file) with local edits or not tracked. ``(None, None)`` when the command does not
    trace to an entry point.

    The tables in ``opengrad.runrecords.scripts`` describe the reviewed, tracked code. A script not tracked yet may
    reach held-out data they do not list; the scan sees it. An approved entry point uploads only when nothing it
    loads was modified, since an edit can print what the review found it never prints. The configs the run
    consumes (``consumed_configs``) are left out here: ``config_blobs`` records their content, and upload waits
    until those versions are pushed. Anything that fails reads as reaching held-out data, and as modified (fail
    closed).
    """
    try:
        from opengrad.runrecords import scripts  # imports this module: loaded at call time

        entry = scripts.command_entry(command, root)
        if entry is None:
            return None, None
        on_disk = {
            path.relative_to(root).as_posix()
            for folder in ("scripts", "src")
            for path in (root / folder).rglob("*.py")
        }
        loaded = scripts.import_closure(entry, root, on_disk)
        reach = sorted(scripts.marked_files(root, loaded))
        configs = {
            path.relative_to(root).as_posix() for path in (root / "configs").rglob("*") if path.is_file()
        }
        # The run-record package decides what uploads, so its own files are watched too: the scan does not follow
        # imports out of the policy files (they name every marker), so an edit there would otherwise go unseen.
        policy = {
            path.relative_to(root).as_posix()
            for path in (root / "src" / "opengrad" / "runrecords").rglob("*.py")
        } | {p for p in ("src/opengrad/env_capture.py", "src/opengrad/hashing.py") if (root / p).is_file()}
    except Exception:  # noqa: BLE001 -- recording never blocks a run; an unreadable answer fails closed
        return ["<scan failed>"], ["<scan failed>"]
    changed = _modified(root)
    tracked = _tracked(root)
    if changed is None or tracked is None:
        return reach, ["<git status failed>"]
    # The configs the run consumes are vouched for by their content instead (config_blobs: uploaded once those
    # exact versions are on origin/master, UP-0040); any other configs/ change still counts.
    consumed = set(consumed_configs(command, root) or [])
    # Modified or untracked (git status), or not tracked at all, which also catches an ignored file.
    watched = loaded | (configs - consumed) | policy
    # A changed configs/ path counts even when it is gone from disk (a deleted config is a change).
    edited = {path for path in changed if path in watched or (path.startswith("configs/") and path not in consumed)}
    return reach, sorted(edited | (watched - tracked))


def unchecked_arguments(argv: Sequence[str], root: Path) -> bool:
    """True when an argument names an existing file the checks cannot read: not under ``configs/`` (read for
    markers), and not a script under ``scripts/`` or ``src/`` (checked as an entry point). A training config in a
    scratch folder is one. Fail closed: such a run's log is not uploaded."""
    try:
        for token in list(argv)[1:]:
            for value in argument_values(token):
                if not value or value.startswith("-"):
                    continue
                path = Path(value)
                candidates = [path] if path.is_absolute() else [root / value, Path.cwd() / value]
                existing = {c.resolve() for c in candidates if c.is_file()}
                if not existing:
                    continue
                if len(existing) > 1:
                    return True  # the checks read one file, the run may read the other
                relative = repo_relative(str(existing.pop()), root)
                if relative.startswith("configs/"):
                    # Spelled from another folder: config_state reads it from the root, where it is another file.
                    if not path.is_absolute() and (root / value).resolve() != (root / relative).resolve():
                        return True
                    continue
                if relative.startswith(("scripts/", "src/")) and relative.endswith(".py"):
                    continue
                return True
        return False
    except Exception:  # noqa: BLE001 -- recording never blocks a run; an unreadable answer fails closed
        return True


def config_blobs(command: Sequence[str], root: Path) -> dict[str, str] | None:
    """Each consumed config's git blob id as the run read it; None when they cannot be determined."""
    try:
        consumed = consumed_configs(command, root)
        if consumed is None:
            return None
        return {path: blob_sha((root / path).read_bytes()) for path in consumed}
    except Exception:  # noqa: BLE001 -- recording never blocks a run; an unreadable answer fails closed
        return None


def held_out_arguments(argv: Sequence[str], root: Path) -> bool:
    """True when a run's arguments, read before redaction, name held-out data: a marker in any argument (a path
    outside the repository included, which redaction would turn into ``<absolute-path>``), or a ``configs/`` file
    that names one. Checked when the run starts, so a config edited or removed later does not change the answer."""
    try:
        return any(names_held_out(t) or config_state(t, root) == "held_out" for t in argv)
    except Exception:  # noqa: BLE001 -- recording never blocks a run; an unreadable answer fails closed
        return True
