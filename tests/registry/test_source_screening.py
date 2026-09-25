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
    sizes = {s["id"]: len(s["candidates"]) for s in _log()["screenings"]}
    assert sizes == {"study-002-answer-heldout": 27, "study-002-punans": 17}
    assert len(ids) == sum(sizes.values()) == 44
    census = audit(ROOT)["screening"]
    assert (census.discovered, census.checked, census.failed) == (44, 44, 0)


def test_the_owner_decision_is_recorded_with_its_amendment() -> None:
    # Adopting a source is the study owner's decision (06-SPLIT-SPEC, open item 2 of 20). The owner
    # adopted the recommendation on 2026-09-24 as study_002_prereg_v9 (41), recorded in 03 and ERRATA.
    decision = _log()["screenings"][0]["owner_decision"]
    assert decision["status"] == "ADOPTED"
    assert decision["adopted"] == [
        "bfcl-irrelevance",
        "bfcl-live-irrelevance",
        "natural-questions-dev",
    ]
    assert decision["recorded_in"] == "docs/research/study-002/41-ANSWER-STRATA-AMENDMENT.md"
    for record in ("docs/research/study-002/03-PREREGISTRATION.md", "reports/ERRATA.md"):
        assert "study_002_prereg_v9" in (ROOT / record).read_text(encoding="utf-8"), record


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
            lambda s, c: s.update(owner_decision={"status": "ADOPTED", "adopted": ["toolqa"]}),
            "adopted candidate 'toolqa' is not in registry/datasets.yaml",
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


PUNANS = ROOT / "reports/source-screening/study-002-punans"


def _punans() -> dict:
    return next(s for s in _log()["screenings"] if s["id"] == "study-002-punans")


def test_the_punans_screening_awaits_the_owner_and_pins_every_source_it_read() -> None:
    screening = _punans()
    assert screening["owner_decision"] == {"status": "PENDING"}
    supply = json.loads((PUNANS / "punans-supply.json").read_text(encoding="utf-8"))
    assert supply["overlap_status"] == "COMPLETE"
    candidates = {c["id"]: c for c in screening["candidates"]}
    # Every candidate the audit downloaded carries the revision the audit pinned, and its count.
    for name, source in supply["sources"].items():
        assert candidates[name]["locator"]["revision"] == source["revision"], name
        assert candidates[name]["answer_items"]["estimate_artifact"] == (
            "reports/source-screening/study-002-punans/punans-supply.json"
        )
    kuq = supply["sources"]["kuq"]["by_category"]
    assert (
        candidates["kuq"]["answer_items"]["count"]
        == kuq["future unknown"] + kuq["unsolved problem"]
    )
    for name in ("selfaware", "coconot", "bigbench-known-unknowns"):
        assert (
            candidates[name]["answer_items"]["count"] == supply["sources"][name]["unknowable_items"]
        )
    decisions = {c["decision"] for c in candidates.values()}
    assert decisions == {"SHORTLISTED", "WATCHLIST", "EXCLUDED"}
    assert sorted(i for i, c in candidates.items() if c["decision"] == "SHORTLISTED") == [
        "kuq",
        "selfaware",
    ]


def test_the_punans_report_quotes_the_audit() -> None:
    # G14: every count in the report's supply table is the artifact's.
    supply = json.loads((PUNANS / "punans-supply.json").read_text(encoding="utf-8"))["sources"]
    report = (PUNANS / "REPORT.md").read_text(encoding="utf-8")
    kuq = supply["kuq"]
    past = kuq["names_a_year_up_to_screening_year"]
    rows = {
        "KUQ, future unknown": (kuq["by_category"]["future unknown"], past["future unknown"]),
        "KUQ, unsolved problem": (kuq["by_category"]["unsolved problem"], past["unsolved problem"]),
    }
    for label, (n, dated) in rows.items():
        assert f"| {label} | {n:,} | {dated} |" in report, label
    selfaware = supply["selfaware"]
    assert (
        f"| SelfAware | {selfaware['unknowable_items']:,} | 0 | {selfaware['exact_text_overlap']['training_corpora']} |"
        in report
    )
    coconot = supply["coconot"]
    dated = sum(coconot["names_a_year_up_to_screening_year"].values())
    assert (
        f"| {coconot['unknowable_items']} | {dated} | {coconot['exact_text_overlap']['training_corpora']} |"
        in report
    )
    assert f"all of KUQ's {sum(kuq['by_category'].values()):,} unknowns" in report
    fitting = kuq["by_category"]["future unknown"] + kuq["by_category"]["unsolved problem"]
    assert f"KUQ's two fitting categories ({fitting:,})" in report
    assert f"SelfAware ({selfaware['unknowable_items']:,})" in report
    for name, source in supply.items():
        assert source["revision"] in report, name
        assert source["sha256"][:8] in report, name


def test_the_punans_draft_quotes_the_audit() -> None:
    # G14: the pool sizes in 43 follow from the audit's counts.
    kuq = json.loads((PUNANS / "punans-supply.json").read_text(encoding="utf-8"))["sources"]["kuq"]
    surviving = (
        kuq["by_category"]["false assumption"]
        - kuq["names_a_year_up_to_screening_year"]["false assumption"]
    )
    draft = (ROOT / "docs/research/study-002/43-PUNANS-AMENDMENT-DRAFT.md").read_text(
        encoding="utf-8"
    )
    assert f"the {surviving} surviving candidates" in draft
    assert "Status: DRAFT" in draft
    for record in ("docs/research/study-002/03-PREREGISTRATION.md", "reports/ERRATA.md"):
        # A draft is recorded in neither until the owner adopts it.
        assert "study_002_prereg_v11" not in (ROOT / record).read_text(encoding="utf-8"), record
