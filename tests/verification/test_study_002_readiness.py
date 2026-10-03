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
    assert record["preregistration"] == "study_002_prereg_v15"
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


def test_the_detector_needs_300_agreed_items_3_checkpoints_and_both_precisions(
    tmp_path: Path,
) -> None:
    path = tmp_path / readiness.PDET_OUT_RESULT
    path.parent.mkdir(parents=True)
    good = {
        "agreed_items": 300,
        "sources": ["BASE", "M0", "M1_DPO_CURRENT"],
        "stratified": True,
        "precision": {"ANSWER": 0.93, "UNSUPPORTED": 0.90},
    }
    for result, expected in (
        (good, PASS),
        (good | {"precision": {"ANSWER": 0.93, "UNSUPPORTED": 0.899}}, readiness.BLOCKED),
        (good | {"precision": {"ANSWER": 0.95}}, readiness.BLOCKED),
        (good | {"agreed_items": 10}, readiness.BLOCKED),
        (good | {"sources": ["M0", "M0"]}, readiness.BLOCKED),
        (good | {"stratified": False}, readiness.BLOCKED),
        (good | {"precision": {"ANSWER": float("nan"), "UNSUPPORTED": True}}, readiness.BLOCKED),
    ):
        path.write_text(json.dumps(result), encoding="utf-8")
        assert readiness.check_6_detector(tmp_path).status == expected, result


def test_check_9_needs_a_config_for_every_arm_and_seed(tmp_path: Path) -> None:
    import shutil

    plan = tmp_path / "configs/study_002/seed-plan.yaml"
    plan.parent.mkdir(parents=True)
    shutil.copy(ROOT / "configs/study_002/seed-plan.yaml", plan)
    configs = tmp_path / readiness.ARM_CONFIGS
    configs.mkdir(parents=True)
    (configs / "C0-s0.yaml").write_text(
        "experiment_id: study_002_c0_s0\nstudy_002: {arm: C0}\nreproducibility: {seed: 0}\n",
        encoding="utf-8",
    )
    check = readiness.check_9_seeds(tmp_path)
    assert check.status == readiness.BLOCKED and check.detail["arm_seed_pairs_missing"] == 26
    for arm in readiness.V14_ARMS:
        for seed in (0, 1, 2):
            (configs / f"{arm}-s{seed}.yaml").write_text(
                f"experiment_id: study_002_{arm}_s{seed}\nstudy_002: {{arm: {arm}}}\nreproducibility: {{seed: {seed}}}\n",
                encoding="utf-8",
            )
    assert readiness.check_9_seeds(tmp_path).status == PASS


def test_check_8_stays_blocked_until_the_trainer_gaps_close(tmp_path: Path) -> None:
    plans = tmp_path / readiness.EXPOSURE_PLANS
    plans.mkdir(parents=True)
    (plans / "c0-s0.json").write_text("{}", encoding="utf-8")
    check = readiness.check_8_exposure(tmp_path)
    assert check.status == readiness.BLOCKED
    assert check.gaps == list(readiness.EXPOSURE_TRAINER_GAPS)


def test_an_existing_but_unvalidated_artifact_is_not_run(tmp_path: Path) -> None:
    for path, check in (
        (readiness.FLAG_SET_DISPOSITIONS, readiness.check_7_corpus),
        (readiness.CPU_SMOKE_RECORD, readiness.check_12_smoke),
    ):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text("{}", encoding="utf-8")
        assert check(tmp_path).status == readiness.NOT_RUN


def test_check_2_reads_the_policy_and_parameters_from_code() -> None:
    check = readiness.check_2_gate_version(ROOT)
    assert check.detail["policy"] == "tool_use_promotion_v6"
    prereg = (ROOT / "docs/research/study-002/03-PREREGISTRATION.md").read_text(encoding="utf-8")
    v8 = " ".join(prereg.split("### `study_002_prereg_v8`")[1].split("### ")[0].split())
    assert "more than 2.0× the smaller" in v8 and "2 percentage points" in v8 and "n ≥ 385" in v8
    assert readiness.V8_PARAMETERS == (2.0, 0.02, 385)


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


def test_check_9_flags_a_duplicate_arm_seed_config(tmp_path: Path) -> None:
    import shutil

    plan = tmp_path / "configs/study_002/seed-plan.yaml"
    plan.parent.mkdir(parents=True)
    shutil.copy(ROOT / "configs/study_002/seed-plan.yaml", plan)
    configs = tmp_path / readiness.ARM_CONFIGS
    configs.mkdir(parents=True)
    for arm in readiness.V14_ARMS:
        for seed in (0, 1, 2):
            (configs / f"{arm}-s{seed}.yaml").write_text(
                f"experiment_id: study_002_{arm}_s{seed}\nstudy_002: {{arm: {arm}}}\nreproducibility: {{seed: {seed}}}\n",
                encoding="utf-8",
            )
    (configs / "zz-copy.yaml").write_text(
        "experiment_id: study_002_c0_s0\nstudy_002: {arm: C0}\nreproducibility: {seed: 0}\n",
        encoding="utf-8",
    )
    check = readiness.check_9_seeds(tmp_path)
    assert check.status == readiness.BLOCKED and any("duplicates" in gap for gap in check.gaps)


def test_check_2_finds_study_002_runs_by_the_name_check_9_enforces(tmp_path: Path) -> None:
    (tmp_path / "runs/study_002_c0_s0").mkdir(parents=True)
    (tmp_path / "runs/m0_sft_canonical_v2_final").mkdir(parents=True)
    assert readiness._study_002_runs(tmp_path) == ["study_002_c0_s0"]


def test_check_9_requires_the_study_002_experiment_id(tmp_path: Path) -> None:
    import shutil

    plan = tmp_path / "configs/study_002/seed-plan.yaml"
    plan.parent.mkdir(parents=True)
    shutil.copy(ROOT / "configs/study_002/seed-plan.yaml", plan)
    configs = tmp_path / readiness.ARM_CONFIGS
    configs.mkdir(parents=True)
    for arm in readiness.V14_ARMS:
        for seed in (0, 1, 2):
            (configs / f"{arm}-s{seed}.yaml").write_text(
                f"experiment_id: {arm}_s{seed}\nstudy_002: {{arm: {arm}}}\nreproducibility: {{seed: {seed}}}\n",
                encoding="utf-8",
            )
    check = readiness.check_9_seeds(tmp_path)
    assert check.status == readiness.BLOCKED and any(
        "experiment_id must start" in g for g in check.gaps
    )
