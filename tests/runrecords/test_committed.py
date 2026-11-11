"""Every committed run summary is valid: a run that did not pass carries a disposition, and no summary holds an
absolute path or anything the secret pattern matches (opengrad.runrecords)."""

from __future__ import annotations

import json
from pathlib import Path

from opengrad.runrecords.cli import validate_summary

ROOT = Path(__file__).resolve().parents[2]


def test_every_committed_run_summary_is_valid() -> None:
    # The folder exists (it holds a README), so a rename cannot make this test pass by finding nothing.
    assert (ROOT / "run-records" / "README.md").is_file()
    problems = []
    for path in sorted((ROOT / "run-records").glob("*.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if path.stem != summary.get("run_id"):
            problems.append(f"{path.name}: file name does not match run_id")
        problems += [f"{path.name}: {error}" for error in validate_summary(summary, ROOT)]
    assert problems == [], "\n".join(problems)
