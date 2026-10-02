"""The Study 002 readiness record (16): fourteen checks, no partial credit, never written twice."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opengrad.verification import study_002_readiness as readiness
from opengrad.verification.accounting import PASS

ROOT = Path(__file__).parents[2]


def test_the_committed_repository_is_blocked_with_six_checks_passing() -> None:
    # The committed state, pinned: a status flip has to be a deliberate change to this test.
    record = readiness.readiness_record(ROOT)
    statuses = {c["number"]: c["status"] for c in record["checks"]}
    assert sorted(statuses) == list(range(1, 15))
    assert [n for n, s in statuses.items() if s == PASS] == [2, 3, 5, 10, 11, 13]
    assert record["status"] == readiness.BLOCKED and record["passed"] == 6
    assert record["preregistration"] == "study_002_prereg_v14"
    for check in record["checks"]:
        assert (check["status"] == PASS) == (not check["gaps"]), check["number"]


def test_check_3_resolves_every_population_or_names_its_declaration() -> None:
    check = readiness.check_3_modes(ROOT)
    over = [p for p, m in check.detail["resolvable_margin_pp"].items() if m > 10]
    assert (
        sorted(over) == sorted(readiness.DESCRIPTIVE_POPULATIONS) == ["ANSWER-natural", "CLARIFY"]
    )


def test_check_11_sees_all_twelve_validators_fail_their_fixtures() -> None:
    check = readiness.check_11_validators(ROOT)
    assert check.status == PASS and len(check.detail["fixtures"]) == 12


def _check(status: str) -> readiness.Check:
    return readiness.Check(0, "x", status)


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ([PASS] * 14, readiness.READY),
        ([PASS] * 13 + [readiness.NOT_RUN], readiness.INCOMPLETE),
        ([PASS] * 12 + [readiness.NOT_RUN, readiness.BLOCKED], readiness.BLOCKED),
        ([PASS] * 13, readiness.INCOMPLETE),
    ],
)
def test_no_partial_credit(statuses, expected) -> None:
    assert readiness.overall([_check(s) for s in statuses]) == expected


def test_a_preflight_record_must_be_ready_compatible_and_name_the_determinism_mode(
    tmp_path: Path,
) -> None:
    path = tmp_path / readiness.PREFLIGHT_RECORD
    path.parent.mkdir(parents=True)
    record = {
        "status": "READY",
        "compatibility": {"result": "COMPATIBLE"},
        "determinism_mode": "NON_DETERMINISTIC_KERNEL",
    }
    path.write_text(json.dumps(record), encoding="utf-8")
    assert readiness.check_1_preflight(tmp_path).status == PASS
    path.write_text(json.dumps(record | {"determinism_mode": None}), encoding="utf-8")
    assert readiness.check_1_preflight(tmp_path).status == readiness.BLOCKED


def test_the_detector_needs_both_precisions_at_the_floor(tmp_path: Path) -> None:
    path = tmp_path / readiness.PDET_OUT_RESULT
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"precision": {"ANSWER": 0.93, "UNSUPPORTED": 0.90}}), encoding="utf-8"
    )
    assert readiness.check_6_detector(tmp_path).status == PASS
    path.write_text(
        json.dumps({"precision": {"ANSWER": 0.93, "UNSUPPORTED": 0.899}}), encoding="utf-8"
    )
    assert readiness.check_6_detector(tmp_path).status == readiness.BLOCKED
    path.write_text(json.dumps({"precision": {"ANSWER": 0.95}}), encoding="utf-8")
    assert readiness.check_6_detector(tmp_path).status == readiness.BLOCKED


def test_an_unreadable_artifact_is_a_gap_not_a_crash(tmp_path: Path) -> None:
    record = readiness.readiness_record(tmp_path)  # an empty root: nothing exists
    assert record["status"] == readiness.BLOCKED
    assert all(c["status"] != PASS or c["number"] in (11, 13) for c in record["checks"])


def test_a_record_is_never_overwritten(tmp_path: Path) -> None:
    out = tmp_path / "record.json"
    out.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        readiness.main(["--root", str(ROOT), "--write", str(out)])
    assert out.read_text(encoding="utf-8") == "{}"


def test_the_surfaces_quote_the_records_passing_checks() -> None:
    # G14: 16's note and the study README name the passing checks the runner computes.
    record = readiness.readiness_record(ROOT)
    passing = [c["number"] for c in record["checks"] if c["status"] == PASS]
    listed = ", ".join(str(n) for n in passing[:-1]) + f" and {passing[-1]}"
    note = (ROOT / "docs/research/study-002/16-GPU-READINESS-GATE.md").read_text(encoding="utf-8")
    assert f"`BLOCKED` with six checks passing ({', '.join(map(str, passing))})" in " ".join(
        note.split()
    )
    readme = (ROOT / "docs/research/study-002/README.md").read_text(encoding="utf-8")
    assert f"is `BLOCKED`: checks {listed} pass" in readme
    assert len(passing) == 6
