"""registry/source_screening.yaml: decisions follow from verdicts, and every number traces to an artifact.

The screening log records every candidate considered for a purpose, including the rejected ones.
These tests pin that the committed log validates, that each decision rule rejects a candidate that
breaks it, that the generated views are current, and that the numbers the written report quotes
are the ones its artifacts produce (G14).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
from pathlib import Path

import pytest
import yaml

from opengrad.registry.validate import audit
from opengrad.registry.validators import validate_source_screening

ROOT = Path(__file__).parents[2]
SCREENING = "study-002-answer-heldout"
OUT = ROOT / "reports/source-screening" / SCREENING


def _module(relative: str):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _log() -> dict:
    return yaml.safe_load((ROOT / "registry/source_screening.yaml").read_text(encoding="utf-8"))


def test_committed_log_validates_and_every_candidate_is_counted() -> None:
    ids, errors = validate_source_screening(ROOT)
    assert errors == []
    assert len(ids) == len(_log()["screenings"][0]["candidates"]) == 27
    census = audit(ROOT)["screening"]
    assert (census.discovered, census.checked, census.failed) == (27, 27, 0)


def test_the_owner_decision_is_still_pending() -> None:
    # Adopting a source is the study owner's decision (06-SPLIT-SPEC, open item 2 of 20). Changing
    # this status is a research-phase decision and must come with the owner's recorded choice.
    assert _log()["screenings"][0]["owner_decision"] == {"status": "PENDING"}


@pytest.fixture
def scratch(tmp_path: Path) -> Path:
    for name in (
        "registry/source_screening.yaml",
        "registry/source_screening.schema.json",
        "registry/datasets.yaml",
        "docs/research/study-002/06-SPLIT-SPEC.md",
    ):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    shutil.copytree(OUT, tmp_path / OUT.relative_to(ROOT))
    return tmp_path


def _errors_after(root: Path, change) -> list[str]:
    log = copy.deepcopy(_log())
    screening = log["screenings"][0]
    change(screening, {c["id"]: c for c in screening["candidates"]})
    (root / "registry/source_screening.yaml").write_text(yaml.safe_dump(log), encoding="utf-8")
    return validate_source_screening(root)[1]


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (
            lambda s, c: c["bfcl-irrelevance"]["verdicts"]["C3"].update(result="FAIL"),
            "bfcl-irrelevance: SHORTLISTED but fails ['C3']",
        ),
        (
            lambda s, c: c["api-bank"].update(decisive_criteria=["C3"]),
            "api-bank: decisive criterion C3 is not a FAIL verdict",
        ),
        (
            lambda s, c: c["toolqa"]["verdicts"].pop("C7"),
            "toolqa: verdicts",
        ),
        (
            lambda s, c: c["toolqa"].update(evidence=["reports/missing.md"]),
            "toolqa: evidence is neither an https URL nor an existing path",
        ),
        (
            lambda s, c: c["toolqa"].update(evidence=["http://example.org"]),
            "toolqa: evidence is neither an https URL nor an existing path",
        ),
        (
            lambda s, c: s.update(
                owner_decision={"status": "ADOPTED", "adopted": ["bfcl-irrelevance"]}
            ),
            "adopted candidate 'bfcl-irrelevance' is not in registry/datasets.yaml",
        ),
        (
            lambda s, c: s.update(owner_decision={"status": "ADOPTED", "adopted": ["when2call"]}),
            "adopted candidate 'when2call' is not SHORTLISTED",
        ),
        (
            lambda s, c: c["when2tool"].pop("revisit_when"),
            "revisit_when",
        ),
        (
            lambda s, c: c["toolqa"].pop("decisive_criteria"),
            "decisive_criteria",
        ),
    ],
)
def test_each_decision_rule_rejects_a_candidate_that_breaks_it(scratch, change, expected) -> None:
    errors = _errors_after(scratch, change)
    assert any(expected in error for error in errors), errors


def test_generated_views_are_current() -> None:
    views = _module("scripts/reporting/generate_source_views.py")
    for name, text in views.render().items():
        assert (ROOT / name).read_text(encoding="utf-8") == text, f"regenerate {name}"


def test_the_bfcl_estimate_follows_from_the_stored_readings() -> None:
    audit_script = _module("scripts/audit_bfcl_answer_supply.py")
    readings = json.loads((OUT / "answer-sizing-readings.json").read_text(encoding="utf-8"))
    sizing = json.loads((OUT / "bfcl-answer-sizing.json").read_text(encoding="utf-8"))
    total = {"point": 0, "low": 0, "high": 0}
    for key, recorded in readings["readings"].items():
        row = sizing["per_file"][key]
        answer = sum(1 for item in recorded if item["reading"] == "ANSWER")
        borderline = sum(1 for item in recorded if item["reading"] == "BORDERLINE")
        n, population = len(recorded), row["population"]
        assert (n, answer, borderline) == (
            row["sampled"],
            row["readings"]["ANSWER"],
            row["readings"]["BORDERLINE"],
        )
        point = round(answer / n * population)
        low = round(audit_script.wilson(answer, n)[0] * population)
        high = round(audit_script.wilson(answer + borderline, n)[1] * population)
        assert (point, low, high) == (
            row["answer_point_estimate"],
            row["answer_95_low"],
            row["answer_95_high"],
        )
        total = {
            "point": total["point"] + point,
            "low": total["low"] + low,
            "high": total["high"] + high,
        }
    assert total == sizing["total"] == {"point": 389, "low": 274, "high": 600}


def test_the_report_quotes_the_numbers_its_artifacts_produce() -> None:
    report = (OUT / "REPORT.md").read_text(encoding="utf-8")
    sizing = json.loads((OUT / "bfcl-answer-sizing.json").read_text(encoding="utf-8"))["total"]
    overlap = json.loads((OUT / "bfcl-training-overlap.json").read_text(encoding="utf-8"))
    assert (
        f"about {sizing['point']} ANSWER items (95% range {sizing['low']}–{sizing['high']})"
        in report
    )
    assert f"{overlap['exact_matches']} items match exactly" in report
    assert f"{overlap['half_contained']} are at least half-contained" in report
    assert f"together they flag {overlap['flagged']} items" in report
    by_file = overlap["flagged_by_file"]
    assert f"{by_file['irrelevance']} of 240 `irrelevance`" in report
    assert f"{by_file['live_irrelevance']} of 884 `live_irrelevance`" in report
