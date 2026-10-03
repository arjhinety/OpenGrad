"""The INC-0002 re-label comparison's statistics and grouping (reports/incidents/INC-0002-relabel-comparison-plan.md)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
import compare_gemini_relabel as compare


def _ld(number: int, value: bytes) -> bytes:
    def varint(v: int) -> bytes:
        out = b""
        while v >= 0x80:
            out += bytes([v & 0x7F | 0x80])
            v >>= 7
        return out + bytes([v])

    return varint(number << 3 | 2) + varint(len(value)) + value


def _step(name: str, **args: str) -> tuple[int, bytes]:
    call = _ld(2, name.encode()) + _ld(3, json.dumps(args).encode())
    return (15, _ld(20, _ld(7, call)))


def test_newcombe_reproduces_the_published_example() -> None:
    # Newcombe (1998), method 10, example (a): 56/70 against 48/80 gives 0.2 with interval 0.0524 to 0.3339.
    result = compare.newcombe(56, 70, 48, 80)
    assert result is not None
    assert result["difference"] == 0.2
    low, high = result["interval_95"]
    assert round(low, 4) == 0.0524 and round(high, 4) == 0.3339
    assert result["excludes_zero"] is True
    assert compare.newcombe(1, 0, 1, 2) is None


def test_groups_come_from_where_the_tools_pointed() -> None:
    own = _step("view_file", AbsolutePath="X:/u/AppData/Local/Temp/og-annotate-1/input.md")
    assert compare.exposure_group([own]) == ("clean", False)
    home = _step("list_dir", DirectoryPath="X:/u/Documents")
    assert compare.exposure_group([own, home]) == ("D", False)
    repo = _step("grep_search", SearchPath="X:/u/Downloads/opengrad/scripts", Query="x")
    assert compare.exposure_group([own, repo]) == ("B", False)
    answers = _step(
        "view_file",
        AbsolutePath="X:/u/Downloads/opengrad/.annotation/t.model-batches/model-gpt/batch-01.attempt-1.answers.json",
    )
    assert compare.exposure_group([own, repo, answers, _step("search_web", query="q")]) == (
        "A",
        True,
    )
