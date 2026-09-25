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
    assert sizes == {
        "study-002-answer-heldout": 27,
        "study-002-punans": 17,
        "study-002-punans-v2": 10,
    }
    assert len(ids) == sum(sizes.values()) == 54
    census = audit(ROOT)["screening"]
    assert (census.discovered, census.checked, census.failed) == (54, 54, 0)


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


def test_the_punans_screening_records_the_adoption_and_pins_every_source_it_read() -> None:
    screening = _punans()
    assert screening["owner_decision"] == {
        "status": "ADOPTED",
        "adopted": ["kuq", "selfaware"],
        "recorded_in": "docs/research/study-002/43-PUNANS-AMENDMENT-DRAFT.md",
        "date": "2026-09-25",
    }
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


def test_the_punans_amendment_quotes_the_audit() -> None:
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
    assert "Status: ADOPTED 2026-09-25" in draft
    for record in ("docs/research/study-002/03-PREREGISTRATION.md", "reports/ERRATA.md"):
        # An amendment is recorded in both, in the commit that adopts it.
        assert "study_002_prereg_v11" in (ROOT / record).read_text(encoding="utf-8"), record


PUNANS_V2 = ROOT / "reports/source-screening/study-002-punans-v2"
PUNANS_V2_DRAFT = ROOT / "docs/research/study-002/44-PUNANS-V2-AMENDMENT-DRAFT.md"


def _punans_v2() -> tuple[dict, dict]:
    screening = next(s for s in _log()["screenings"] if s["id"] == "study-002-punans-v2")
    supply = json.loads((PUNANS_V2 / "punans-v2-supply.json").read_text(encoding="utf-8"))
    return screening, supply


def test_the_punans_v2_screening_pins_every_source_it_read_and_records_the_adoption() -> None:
    screening, supply = _punans_v2()
    assert screening["owner_decision"] == {
        "status": "ADOPTED",
        "adopted": ["kuq", "kuqp", "bigbench-known-unknowns"],
        "recorded_in": "docs/research/study-002/44-PUNANS-V2-AMENDMENT-DRAFT.md",
        "date": "2026-09-25",
    }
    assert supply["overlap_status"] == "COMPLETE"
    assert supply["first_attempt_distinct_questions"] == 1063
    candidates = {c["id"]: c for c in screening["candidates"]}
    for name, source in supply["sources"].items():
        assert candidates[name]["locator"]["revision"] == source["revision"], name
        assert candidates[name]["answer_items"]["count"] == source["fresh_after_exclusions"], name
        assert candidates[name]["decision"] == "SHORTLISTED", name
    assert sorted(i for i, c in candidates.items() if c["decision"] == "SHORTLISTED") == sorted(
        supply["sources"]
    )
    # The first attempt measured SelfAware; its exclusion cites that result.
    assert candidates["selfaware"]["decision"] == "EXCLUDED"
    assert (
        "reports/study-002/punans-v1/punans-v1.strata.json" in candidates["selfaware"]["evidence"]
    )


def test_the_punans_v2_report_and_draft_quote_the_audit() -> None:
    # G14: every count the report and 44 quote is the artifact's.
    _, supply = _punans_v2()
    report = (PUNANS_V2 / "REPORT.md").read_text(encoding="utf-8")
    labels = {
        "kuq": "KUQ `unknowns_all.jsonl`, future and unsolved",
        "kuqp": "KUQP future",
        "bigbench-known-unknowns": "BIG-bench Known Unknowns",
    }
    for name, label in labels.items():
        s = supply["sources"][name]
        assert (
            f"| {label} | {s['distinct_questions']:,} | {s['drawn_by_the_first_attempt']} | "
            f"{s['names_a_year_up_to_screening_year']} | {s['exact_text_overlap']['training_corpora']} | "
            f"**{s['fresh_after_exclusions']:,}** |"
        ) in report, name
        assert s["revision"] in report and s["sha256"][:8] in report, name
        assert sum(v for k, v in s["exact_text_overlap"].items() if k != "training_corpora") == 0
    assert f"| **Total** | | | | | **{supply['fresh_total']:,}** |" in report
    kuq = supply["sources"]["kuq"]
    authors, categories = kuq["fresh_by_author"], kuq["fresh_by_category"]
    assert (
        f"GPT {authors['gpt']}, crowdworkers {authors['turk']}, the web {authors['web']}" in report
    )
    assert (
        f"future unknown {categories['future unknown']}, unsolved problem "
        f"{categories['unsolved problem/mistery']}" in report
    )
    beyond = sum(s["fresh_after_exclusions"] for n, s in supply["sources"].items() if n != "kuq")
    kuqp = supply["sources"]["kuqp"]["fresh_after_exclusions"]
    bigbench = supply["sources"]["bigbench-known-unknowns"]["fresh_after_exclusions"]
    assert f"{beyond} fresh questions (KUQP {kuqp}, BIG-bench {bigbench})" in report
    draft = PUNANS_V2_DRAFT.read_text(encoding="utf-8")
    assert (
        f"**{supply['fresh_total']:,}** fresh questions before the §7 screen: KUQ "
        f"{kuq['fresh_after_exclusions']:,}, KUQP {kuqp} and BIG-bench {bigbench}"
    ) in draft
    assert f"It found **{beyond}** fresh questions beyond KUQ" in draft
    for name, source in supply["sources"].items():
        assert source["revision"] in draft and source["sha256"] in draft, name


def test_the_punans_v2_amendment_is_recorded_twice_and_quotes_the_first_attempt() -> None:
    # An adopted amendment is recorded in both 03 and ERRATA, in the commit that adopts it, and quotes the
    # first attempt's result from its artifact.
    draft = PUNANS_V2_DRAFT.read_text(encoding="utf-8")
    assert "Status: ADOPTED 2026-09-25, as drafted, with KUQP." in draft
    for record in ("docs/research/study-002/03-PREREGISTRATION.md", "reports/ERRATA.md"):
        assert "study_002_prereg_v12" in (ROOT / record).read_text(encoding="utf-8"), record
    v1 = json.loads(
        (ROOT / "reports/study-002/punans-v1/punans-v1.strata.json").read_text(encoding="utf-8")
    )
    floor = v1["agreement"]
    assert f"raw agreement **{floor['raw_agreement']:.3f}** over six labels" in draft
    assert f"(Cohen's κ {floor['cohen_kappa']:.3f})" in draft
    assert f"would have held **{v1['strata']['P-UNANS-unknowable']['n']}**" in draft
    kuq = v1["reference_labels_by_source"]["U:kuq"]
    assert f"({kuq['UNKNOWABLE']} of {sum(kuq.values())} jointly labelled" in draft
