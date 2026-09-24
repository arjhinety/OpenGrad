"""Every tracked file that starts with a shebang is executable in the git index.

Ruff's `EXE001` catches a shebang script committed as mode `100644`, but only on Linux: Windows has no
executable bit, so the owner's machine never sees it and CI stops at `ruff check` before pytest runs
(`plugins/opengrad/skills/opengrad-development/references/known-failures.md`). This test reads the
modes from the index, so it fails on every platform.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _index_entries() -> list[tuple[str, str]]:
    listing = subprocess.run(
        ["git", "ls-files", "-s", "-z"], cwd=ROOT, capture_output=True, check=True
    ).stdout.decode("utf-8")
    entries = []
    for record in listing.split("\0"):
        if not record:
            continue
        meta, path = record.split("\t", 1)
        entries.append((meta.split()[0], path))
    return entries


def _has_shebang(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(2) == b"#!"
    except OSError:
        return False


def test_every_shebang_file_is_executable_in_the_index() -> None:
    entries = _index_entries()
    assert len(entries) > 1000, "the index listing looks empty"
    shebang = [(mode, path) for mode, path in entries if _has_shebang(ROOT / path)]
    assert len(shebang) > 50, "expected dozens of shebang scripts; the scan found almost none"
    not_executable = sorted(path for mode, path in shebang if mode != "100755")
    assert not_executable == [], (
        f"shebang files committed without the executable bit: {not_executable}; "
        "run `git update-index --chmod=+x <path>`"
    )
