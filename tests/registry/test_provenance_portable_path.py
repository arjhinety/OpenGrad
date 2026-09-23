"""`portable_path`: committed audit trails carry repo-relative paths, never the author's (ERRATA §22)."""

from __future__ import annotations

from pathlib import Path

from opengrad.registry.provenance import portable_path

ROOT = Path(__file__).resolve().parents[2]


def test_a_path_inside_the_repository_becomes_repo_relative() -> None:
    inside = str(ROOT / "reports" / "pdet" / "batch-01.md")
    assert portable_path(inside, ROOT) == "reports/pdet/batch-01.md"


def test_a_windows_path_outside_the_repository_keeps_only_its_name() -> None:
    assert (
        portable_path(r"C:\Users\someone\AppData\Roaming\npm\cline.CMD", ROOT)
        == "<outside-repo>/cline.CMD"
    )
    assert portable_path("C:/Users/someone/AppData/Local/Temp/og-x.last-message.txt", ROOT) == (
        "<outside-repo>/og-x.last-message.txt"
    )
    assert portable_path("/home/someone/.local/bin/codex", ROOT) == "<outside-repo>/codex"


def test_a_value_that_is_not_an_absolute_path_is_unchanged() -> None:
    for value in ("-m", "cline-pass/deepseek-v4.1-flash", "--json", "relative/path.md", ""):
        assert portable_path(value, ROOT) == value
