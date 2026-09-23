"""The publication-hygiene scan runs in CI and fails on anything its allowlist does not name exactly.

Before this test the scanner was never run by CI, exited 1 on HEAD (180 findings), and could not see
`.jsonl` files or tarballs -- where the only live-format token in the repository sits
(`reports/ERRATA.md` §22).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _scanner():
    spec = importlib.util.spec_from_file_location(
        "check_publication_hygiene", ROOT / "scripts/repo/check_publication_hygiene.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_repository_passes_its_hygiene_scan() -> None:
    scanner = _scanner()
    findings = scanner.scan(ROOT)
    problems, accepted = scanner.evaluate(findings, scanner.load_allowlist())
    assert problems == [], "\n".join(problems[:20])
    # The one upstream credential, in the population and in two tarball members.
    assert sum(1 for f in accepted if f["category"] == "api_secret") == 3


def test_the_scanner_catches_tokens_in_text_data_and_tarballs(tmp_path) -> None:
    scanner = _scanner()
    token = "ghp_" + "A1b2C3d4E5" * 3 + "F6g7H8"  # 36 characters after the prefix; not a real token
    assert len(token) == 40
    for text in (f"key = '{token}'", "hf_" + "x" * 34, "AKIA" + "ABCDEFGHIJKLMNOP"):
        assert [f["category"] for f in scanner._lines(text, scanner.PATTERNS, "a.md")] == [
            "api_secret"
        ]
        assert [f["category"] for f in scanner._lines(text, scanner.DATA_PATTERNS, "a.jsonl")] == [
            "api_secret"
        ]


def test_a_new_finding_in_an_allowlisted_file_fails() -> None:
    scanner = _scanner()
    finding = {
        "category": "api_secret",
        "file": "reports/pdet/pdet-v1.population.jsonl",
        "line": 1,
        "text": "",
    }
    entry = {"path": finding["file"], "category": "api_secret", "count": 1, "reason": "r"}
    problems, _ = scanner.evaluate([finding, dict(finding, line=2)], [entry])
    assert problems == [
        "allowlist: api_secret reports/pdet/pdet-v1.population.jsonl: expected exactly 1 finding(s), found 2"
    ]


def test_a_stale_allowlist_entry_fails() -> None:
    scanner = _scanner()
    entry = {"path": "nowhere.md", "category": "private_path", "count": 1, "reason": "r"}
    problems, _ = scanner.evaluate([], [entry])
    assert problems and "found 0" in problems[0]


def test_every_allowlist_entry_carries_a_reason() -> None:
    scanner = _scanner()
    for entry in scanner.load_allowlist():
        assert str(entry["reason"]).strip(), entry
