"""The directory indexes are current, and the repository map names every tracked top-level directory.

Both exist because the 2026-09-24 review found `reports/`, `scripts/` and `docs/` unindexed and the README's
map listing 8 of 17 top-level directories. A generated view that is not regenerated, and a map that is not
checked, drift back; these tests make the drift a CI failure.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _generator():
    spec = importlib.util.spec_from_file_location(
        "generate_indexes", ROOT / "scripts/reporting/generate_indexes.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_generated_index_is_current() -> None:
    generator = _generator()
    assert set(generator.OUTPUTS) == {"reports/README.md", "scripts/README.md", "docs/README.md"}
    for relative, build in generator.OUTPUTS.items():
        committed = (ROOT / relative).read_text(encoding="utf-8")
        assert committed == build(), f"{relative} is stale: run python scripts/reporting/generate_indexes.py"


def test_every_reports_directory_has_a_study() -> None:
    generator = _generator()
    _items = list(generator.REPORT_DIRECTORIES.values())
    assert _items, "nothing to check: an empty collection would pass this test vacuously"
    for study, _ in _items:
        assert study in {"Study 001", "Study 002"}


def test_the_repository_map_names_every_tracked_top_level_directory() -> None:
    names = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    top = sorted({name.split("/", 1)[0] for name in names if "/" in name})
    layout = (ROOT / "docs/architecture/repository.md").read_text(encoding="utf-8")
    section = layout.split("## Repository layout", 1)[1]
    missing = [directory for directory in top if f"`{directory}/`" not in section]
    assert missing == [], f"add these to docs/architecture/repository.md: {missing}"
    assert len(top) >= 21
