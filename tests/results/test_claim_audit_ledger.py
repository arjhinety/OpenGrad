"""The Study 001 claim-audit ledger stays complete, and the guardrails only cite findings it holds."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "reports" / "audits" / "study-001-claim-audit"
GUARDRAILS = ROOT / "docs" / "research" / "GUARDRAILS.md"
STATUSES = {"RESOLVED", "PARTIAL", "OPEN"}


def _load(name: str):
    return json.loads((LEDGER / name).read_text(encoding="utf-8"))


def test_every_finding_has_exactly_one_resolution() -> None:
    findings = _load("findings.json")
    resolutions = _load("resolutions.json")
    finding_ids = [f["id"] for f in findings]
    resolution_ids = [r["id"] for r in resolutions]
    assert len(finding_ids) == len(set(finding_ids)) == 93
    assert sorted(resolution_ids) == sorted(finding_ids)
    assert {r["status"] for r in resolutions} <= STATUSES
    assert all(r["where"].strip() for r in resolutions)


def test_severity_counts_match_what_the_documents_quote() -> None:
    findings = _load("findings.json")
    counts = {s: sum(f["severity"] == s for f in findings) for s in ("HIGH", "MED", "LOW")}
    assert counts == {"HIGH": 13, "MED": 42, "LOW": 38}


def test_recheck_repeats_point_at_real_findings() -> None:
    recheck = _load("recheck-2026-09-13.json")
    finding_ids = {f["id"] for f in _load("findings.json")}
    ids = [f["id"] for f in recheck["findings"]]
    assert len(ids) == len(set(ids)) == 7
    for finding in recheck["findings"]:
        assert finding["resolution"]["status"] in STATUSES
        if finding["repeat_of"] is not None:
            assert finding["repeat_of"] in finding_ids


def test_guardrails_cite_only_logged_findings() -> None:
    text = GUARDRAILS.read_text(encoding="utf-8")
    finding_ids = {f["id"] for f in _load("findings.json")}
    recheck_ids = {f["id"] for f in _load("recheck-2026-09-13.json")["findings"]}
    cited = {int(n) for n in re.findall(r"#(\d+)", text)}
    cited_recheck = set(re.findall(r"\bR\d+\b", text))
    assert cited, "GUARDRAILS.md cites no findings"
    assert cited <= finding_ids, f"unknown finding ids: {sorted(cited - finding_ids)}"
    assert cited_recheck <= recheck_ids, f"unknown recheck ids: {sorted(cited_recheck - recheck_ids)}"
