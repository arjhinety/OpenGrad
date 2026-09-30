"""The P-UNANS-v2 trial report and main-set stratum (44 §9-§10), on synthetic items and references."""

from __future__ import annotations

import pytest

from opengrad.verification import punans_v2_report as report

G, D = "model.deepseek-v4.1-flash", "model.gemini-3.8-flash-high"
U, N, X = "UNKNOWABLE", "NOT_UNKNOWABLE", "UNKNOWN"


def _item(pid: str, source: str = "kuq", author: str = "gpt") -> dict:
    return {
        "punans_id": pid,
        "source_dataset": source,
        "source_author": author,
        "user_message": "q" * len(pid),
        "tools": [{}],
    }


def _ref(pid: str, a: str, b: str) -> dict:
    agreed = a == b
    return {
        "punans_id": pid,
        "reference_label": a if agreed else None,
        "consensus": "unanimous" if agreed else "NO_CONSENSUS",
        "votes": {G: a, D: b},
    }


def _ten(disagreements: int) -> tuple[list[dict], list[dict]]:
    items = [_item(f"i{k}") for k in range(10)]
    refs = [_ref(f"i{k}", U, N if k < disagreements else U) for k in range(10)]
    return items, refs


def test_the_main_floor_is_raw_agreement_at_or_above_080_with_the_caveat() -> None:
    _, at_floor = _ten(2)
    result = report.agreement(at_floor, floor_applies=True)
    assert (result["raw_agreement"], result["status"]) == (0.8, "PASS")
    assert "easier to reach" in result["caveat"]
    _, below = _ten(3)
    assert report.agreement(below, floor_applies=True)["status"] == "STOP"


def test_the_trial_reports_agreement_but_never_stops() -> None:
    _, below = _ten(5)
    result = report.agreement(below, floor_applies=False)
    assert result["status"] == "TRIAL_NO_FLOOR" and result["at_or_above_floor"] is False
    items, refs = _ten(5)
    document, members = report.build_report(items, refs, "trial")
    assert members == [] and "strata" not in document and "check_4" not in document


def test_the_stratum_is_the_items_both_label_unknowable() -> None:
    items = [_item("a"), _item("b", "kuqp"), _item("c"), _item("d", "bigbench-known-unknowns")]
    refs = [_ref("a", U, U), _ref("b", U, U), _ref("c", N, N), _ref("d", X, X)]
    document, members = report.build_report(items, refs, "main")
    stratum = document["strata"][report.STRATUM]
    assert stratum["n"] == 2 and stratum["by_source"] == {"kuq": 1, "kuqp": 1}
    assert document["check_4"] == {
        "population": report.STRATUM,
        "n": 2,
        "min_n": 385,
        "status": "UNDER_POWERED",
    }
    assert [m["punans_id"] for m in members] == ["a", "b"]
    assert document["reference_labels_by_source"]["kuq"] == {N: 1, U: 1}


def test_below_the_floor_no_membership_is_written() -> None:
    items, refs = _ten(3)
    document, members = report.build_report(items, refs, "main")
    assert document["agreement"]["status"] == "STOP" and members == []
    assert document["strata"][report.STRATUM]["n"] == 7  # counted, not built


def test_agreement_is_reported_per_source_and_per_author() -> None:
    items = [_item("a", "kuq", "gpt"), _item("b", "kuq", "turk"), _item("c", "kuqp", "gpt")]
    refs = [_ref("a", U, U), _ref("b", U, N), _ref("c", N, N)]
    document, _ = report.build_report(items, refs, "main")
    assert document["agreement_by_source"]["kuq"]["same_label"] == 1
    assert document["agreement_by_source"]["kuq"]["items"] == 2
    assert document["agreement_by_author"]["gpt"]["same_label"] == 2


def test_a_reference_that_misses_an_item_or_uses_v1_labels_is_refused() -> None:
    with pytest.raises(report.PUnansV2ReportError):
        report.build_report([_item("a")], [], "main")
    with pytest.raises(report.PUnansV2ReportError):
        report.build_report([_item("a")], [_ref("a", "SUBJECTIVE", "SUBJECTIVE")], "main")


def test_the_committed_trial_report_and_the_readme_agree() -> None:
    # G14: the trial numbers the study README quotes come from the committed report, reference and archive.
    import json
    from pathlib import Path

    root = Path(__file__).parents[2]
    out = root / "reports/study-002/punans-v2"
    document = json.loads((out / "punans-v2-trial.report.json").read_text(encoding="utf-8"))
    assert (
        document["status"] == "TRIAL_REPORT" and document["agreement"]["status"] == "TRIAL_NO_FLOOR"
    )
    assert "strata" not in document
    agreement = document["agreement"]
    refs = [
        json.loads(line)
        for line in (out / "reference/punans-v2-trial.reference.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    split = [r for r in refs if r["votes"][G] != r["votes"][D]]
    pairs = {(r["votes"][D], r["votes"][G]) for r in split}  # (gemini, deepseek)
    assert pairs <= {(N, U), (N, X)}  # every disagreement runs one way
    to_u = sum(1 for r in split if r["votes"][G] == U)
    to_x = sum(1 for r in split if r["votes"][G] == X)
    manifest = json.loads(
        (
            out
            / "provenance/external-models/punans-v2-trial.external-models.audit-trail.manifest.json"
        ).read_text(encoding="utf-8")
    )
    deepseek = next(s for s in manifest["sessions"] if s["session_id"] == "model-deepseek")
    by_version: dict[str, int] = {}
    for batch in deepseek["batches"]:
        for attempt in batch["attempts"]:
            if attempt["recorded_labels"]:
                by_version[attempt["cli_version"]] = (
                    by_version.get(attempt["cli_version"], 0) + batch["items"]
                )
    readme = (root / "docs/research/study-002/README.md").read_text(encoding="utf-8")
    assert (
        f"the models agreed on {agreement['same_label']}: raw {agreement['raw_agreement']:.3f}, "
        f"κ {agreement['cohen_kappa']:.3f}, every disagreement one way"
    ) in readme
    record = (out / "PROCEDURE-REVISION.md").read_text(encoding="utf-8")
    assert f"Every one of its {len(split)} disagreements ran one way" in record
    assert to_u + to_x == len(split)
    flat = " ".join(record.split())
    assert f"{by_version['3.0.65']} from 3.0.65 and {by_version['3.0.66']} from 3.0.66" in flat


def test_the_committed_main_report_and_every_surface_agree() -> None:
    # G14 and G16: the main set's floor and stratum come from the committed report; the surfaces quote them.
    import json
    from collections import Counter
    from pathlib import Path

    root = Path(__file__).parents[2]
    out = root / "reports/study-002/punans-v2"
    document = json.loads((out / "punans-v2.strata.json").read_text(encoding="utf-8"))
    agreement, n = document["agreement"], document["check_4"]["n"]
    assert document["status"] == "MODEL_REFERENCE_PROVISIONAL" and agreement["status"] == "PASS"
    assert agreement["raw_agreement"] >= report.AGREEMENT_FLOOR
    assert document["check_4"]["status"] == "UNDER_POWERED" and n < report.CHECK_4_MIN_N
    members = [
        json.loads(line)
        for line in (out / "punans-v2.strata-members.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(members) == n
    items = {
        json.loads(line)["punans_id"]: json.loads(line)
        for line in (out / "punans-v2.population.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    refs = [
        json.loads(line)
        for line in (out / "reference/punans-v2.reference.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    total: Counter[str] = Counter()
    unknowable: Counter[str] = Counter()
    for ref in refs:
        item = items[ref["punans_id"]]
        key = (
            item["source_dataset"]
            if item["source_dataset"] != "kuq"
            else f"kuq-{item['source_author']}"
        )
        total[key] += 1
        unknowable[key] += ref["reference_label"] == U
    readme = (root / "docs/research/study-002/README.md").read_text(encoding="utf-8")
    assert (
        f"they agreed on {agreement['same_label']} of the 800 main-set questions: raw agreement "
        f"{agreement['raw_agreement']:.4f}, κ {agreement['cohen_kappa']:.3f}"
    ) in readme
    assert f"both models called only {n} of the 800 unknowable" in readme
    assert f"yielded {unknowable['kuq-gpt']} of {total['kuq-gpt']}" in readme
    assert (
        f"yielded {unknowable['kuqp']} of {total['kuqp']} and {unknowable['bigbench-known-unknowns']} of "
        f"{total['bigbench-known-unknowns']}"
    ) in readme
    for surface in ("README.md", "docs/research/STUDIES.md"):
        text = (root / surface).read_text(encoding="utf-8")
        assert f"only {n} questions are agreed unknowable, under the 385" in text, surface


def test_amendment_45_is_recorded_twice_and_quotes_its_artifacts() -> None:
    # An adopted amendment is recorded in both 03 and ERRATA in its adoption commit; its numbers come from
    # artifacts (G14).
    import json
    from pathlib import Path

    root = Path(__file__).parents[2]
    draft = (root / "docs/research/study-002/45-PUNANS-CONSTRUCTED-AMENDMENT-DRAFT.md").read_text(
        encoding="utf-8"
    )
    assert "Status: ADOPTED 2026-10-01, as drafted." in draft
    for record in ("docs/research/study-002/03-PREREGISTRATION.md", "reports/ERRATA.md"):
        assert "study_002_prereg_v13" in (root / record).read_text(encoding="utf-8"), record
    main = json.loads(
        (root / "reports/study-002/punans-v2/punans-v2.strata.json").read_text(encoding="utf-8")
    )
    agreement = main["agreement"]
    assert (
        f"gave the same label on {agreement['same_label']} of 800: raw {agreement['raw_agreement']:.4f}, "
        f"κ {agreement['cohen_kappa']:.3f}"
    ) in draft
    n = main["check_4"]["n"]
    assert f"they agreed on only **{n}** unknowable questions" in draft
    assert f"the union holds about\n  {n} + 480" in draft or f"about {n} + 480" in " ".join(
        draft.split()
    )
    supply = json.loads(
        (
            root / "reports/source-screening/study-002-punans-constructed/wikidata-supply.json"
        ).read_text(encoding="utf-8")
    )["queries"]
    flat = " ".join(draft.split())
    assert (
        f"{supply['competition_series_with_a_winner_named_since_2021']['count']:,} competition series"
        in flat
    )
    assert f"{supply['cities_over_one_million_with_a_population_figure']['count']} cities" in flat
    assert (
        f"Of {supply['conjectures']['count']} items typed as conjectures, only "
        f"{supply['conjectures_recording_who_proved_them']['count']} record who proved them"
    ) in flat
