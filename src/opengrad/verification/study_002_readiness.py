"""Study 002's GPU readiness record: the fourteen checks of `16-GPU-READINESS-GATE.md`, under v14.

Each check reads committed artifacts and returns `PASS`, `BLOCKED` (its evidence is missing or does not meet
the pass condition, with the gap named) or `NOT_RUN`. The record is `READY` only when all fourteen pass; any
`BLOCKED` check makes it `BLOCKED`; otherwise a check that did not run makes it `INCOMPLETE` (16, "What blocks
what"). There is no partial credit, and nothing here can turn a missing artifact into a pass.

This is the readiness gate, not the evaluation gate: `study_002_gate_v1` (`study_002_gate.py`) judges a scored
arm; this module decides whether any arm may launch. A record is written once (`--write`), never overwritten;
a correction is a new record (16:7-10).

    python -m opengrad.verification.study_002_readiness            # print the record
    python -m opengrad.verification.study_002_readiness --write PATH
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from opengrad.verification import study_002_gate as gate
from opengrad.verification.accounting import FAIL, PASS
from opengrad.verification.population_validators import (
    v1_mode_coverage,
    v2_metric_denominator,
    v12_resolvable_margin,
)
from opengrad.verification.provenance_validators import self_test as provenance_self_test
from opengrad.verification.resolvability import resolvable_margin

READY = "READY"
BLOCKED = "BLOCKED"
INCOMPLETE = "INCOMPLETE"
NOT_RUN = "NOT_RUN"
RECORD_KIND = "STUDY_002_READINESS_RECORD"

#: 46 §10: the margin a population is used to test, for every comparison between arms.
CLAIM_MARGIN = 0.10
#: Populations that resolve more than CLAIM_MARGIN, each with the amendment that declared its comparisons
#: descriptive before any score.
DESCRIPTIVE_POPULATIONS = {
    "CLARIFY": "study_002_prereg_v8 item D",
    "ANSWER-natural": "41 §10",
}
#: Benchmarks Study 002 runs (09) plus the source of ANSWER-natural (bfcl-v4): each needs a contamination scan.
SCANNED_BENCHMARKS = ("gsm8k", "ifeval", "mmlu-pro", "when2call-eval", "bfcl-v4")
V14_ARMS = {"C0", "R1", "R3", "C2", "X1", "D25", "D50", "S1", "S2"}

# Artifacts later stages write. Each check names the one it needs.
PREFLIGHT_RECORD = "reports/study-002/readiness/gpu-preflight.json"
ARM_CONTAMINATION = "reports/study-002/contamination/answer-strata-vs-arm-corpora.json"
PDET_OUT_RESULT = "reports/study-002/pdet-out/pdet-out.result.json"
FLAG_SET_DISPOSITIONS = "reports/study-002/flag-set/flag-set.dispositions.json"
EXPOSURE_PLANS = "reports/study-002/exposure"
ARM_CONFIGS = "configs/experiments/study_002"
CPU_SMOKE_RECORD = "reports/study-002/smoke/cpu-smoke.json"
STUDY_002_SCORES = "reports/study-002/evaluation"


@dataclass
class Check:
    number: int
    name: str
    status: str
    evidence: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


def _json(root: Path, path: str) -> Any:
    return json.loads((root / path).read_text(encoding="utf-8"))


def _missing(number: int, name: str, artifact: str, why: str) -> Check:
    return Check(number, name, BLOCKED, gaps=[f"{artifact} does not exist: {why}"])


def check_1_preflight(root: Path) -> Check:
    name = "preflight READY"
    if not (root / PREFLIGHT_RECORD).is_file():
        return _missing(
            1,
            name,
            PREFLIGHT_RECORD,
            "the primary GPU is undecided (owner); configs/hardware/gpu_preflight_v1.yaml is Study 001's "
            "inference smoke and does not qualify; the determinism mode (46 §12) is chosen there",
        )
    record = _json(root, PREFLIGHT_RECORD)
    gaps = []
    if record.get("status") != "READY":
        gaps.append(f"preflight status is {record.get('status')!r}")
    if (record.get("compatibility") or {}).get("result") != "COMPATIBLE":
        gaps.append("compatibility.result is not COMPATIBLE")
    if record.get("determinism_mode") not in ("DECLARED_DETERMINISTIC", "NON_DETERMINISTIC_KERNEL"):
        gaps.append("no determinism mode chosen (46 §12)")
    return Check(1, name, BLOCKED if gaps else PASS, [PREFLIGHT_RECORD], gaps)


def _current_prereg(root: Path) -> str | None:
    try:
        header = (root / "docs/research/study-002/03-PREREGISTRATION.md").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"Current version: `(study_002_prereg_v\d+)`", header)
    return match.group(1) if match else None


def check_2_gate_version(root: Path) -> Check:
    gaps = []
    if gate.GATE_VERSION != "study_002_gate_v1" or gate.STUDY_002_GATE_CONTRACT != 3:
        gaps.append(f"gate is {gate.GATE_VERSION} contract {gate.STUDY_002_GATE_CONTRACT}")
    params = gate.ADOPTED_PARAMETERS
    if None in (params.truncation_max_ratio, params.truncation_min_gap, params.p_unans_min_n):
        gaps.append("ADOPTED_PARAMETERS holds an undeclared value")
    prereg = _current_prereg(root)
    if prereg is None:
        gaps.append("03 names no current preregistration version")
    scores = root / STUDY_002_SCORES
    scored = scores.is_dir() and any(scores.iterdir())
    if scored:
        gaps.append(
            f"{STUDY_002_SCORES} holds candidate scores; amendment dates must be checked against them"
        )
    return Check(
        2,
        "gate version frozen",
        BLOCKED if gaps else PASS,
        [
            "src/opengrad/verification/study_002_gate.py",
            "docs/research/study-002/03-PREREGISTRATION.md",
        ],
        gaps,
        {
            "gate": gate.GATE_VERSION,
            "contract": gate.STUDY_002_GATE_CONTRACT,
            "policy": "tool_use_promotion_v6",
            "preregistration": prereg,
            "candidate_scores_exist": scored,
        },
    )


def check_3_modes(root: Path) -> Check:
    path = "reports/study-002/pconf-v1/pconf-v1.partition.json"
    partition = _json(root, path)
    gold, strata = partition["gold_counts"], partition["answer_strata"]
    coverage = v1_mode_coverage(gold, enforce_floor=True)
    gaps = [] if coverage.status == PASS else coverage.all_errors()
    rows = {mode: n for mode, n in gold.items() if mode != "ANSWER"}
    rows |= {"ANSWER": gold["ANSWER"]} | {
        k: v for k, v in strata.items() if k.startswith("ANSWER-")
    }
    margins = {}
    for population, n in rows.items():
        margin = resolvable_margin(int(n))
        margins[population] = round(100 * margin, 2)
        if margin > CLAIM_MARGIN and population not in DESCRIPTIVE_POPULATIONS:
            gaps.append(
                f"{population} resolves {100 * margin:.2f}pp and is not declared descriptive"
            )
    return Check(
        3,
        "all four modes covered",
        BLOCKED if gaps else PASS,
        [path],
        gaps,
        {"resolvable_margin_pp": margins, "descriptive": DESCRIPTIVE_POPULATIONS},
    )


def check_4_strata(root: Path) -> Check:
    manifest = "reports/study-002/answer-strata-v1/answer-strata-v1.manifest.json"
    gaps = []
    if not (root / manifest).is_file():
        gaps.append(f"{manifest} does not exist")
    if not (root / ARM_CONTAMINATION).is_file():
        gaps.append(
            f"{ARM_CONTAMINATION} does not exist: the strata were screened against normalization-v1, "
            "not the arm corpora, which are not built"
        )
    registry = _json(root, "reports/data/benchmark_contamination_registry.json")
    status = {e["benchmark_id"]: e["scan_status"] for e in registry["entries"]}
    unscanned = [b for b in SCANNED_BENCHMARKS if status.get(b) != "SCANNED"]
    if unscanned:
        gaps.append(f"benchmarks not scanned for contamination: {', '.join(unscanned)}")
    return Check(
        4,
        "ANSWER strata materialized and disjoint",
        BLOCKED if gaps else PASS,
        [manifest, "reports/data/benchmark_contamination_registry.json"],
        gaps,
        {"scan_status": {b: status.get(b) for b in SCANNED_BENCHMARKS}},
    )


def check_5_punans(root: Path) -> Check:
    natural_path = "reports/study-002/punans-v2/punans-v2.strata.json"
    constructed_path = "reports/study-002/punans-v2-constructed/punans-v2-constructed.strata.json"
    natural, constructed = _json(root, natural_path), _json(root, constructed_path)
    gaps = []
    if natural["agreement"]["status"] != "PASS":
        gaps.append("P-UNANS-v2's agreement floor did not pass")
    if (
        constructed["agreement"]["status"] != "PASS"
        or constructed["control_rule"]["status"] != "PASS"
    ):
        gaps.append("the constructed set's agreement floor or control rule did not pass")
    if constructed["check_4"]["status"] != "EVALUABLE":
        gaps.append(f"check 4's population is {constructed['check_4']['status']}")
    return Check(
        5,
        "P-UNANS curated with agreement",
        BLOCKED if gaps else PASS,
        [natural_path, constructed_path],
        gaps,
        {
            "raw_agreement": [
                natural["agreement"]["raw_agreement"],
                constructed["agreement"]["raw_agreement"],
            ],
            "check_4_n": constructed["check_4"]["n"],
        },
    )


def check_6_detector(root: Path) -> Check:
    if not (root / PDET_OUT_RESULT).is_file():
        return _missing(
            6,
            "detector measured",
            PDET_OUT_RESULT,
            "classifier v2 is not yet measured on model replies (46 §8); P-DET-OUT is not drawn",
        )
    result = _json(root, PDET_OUT_RESULT)
    gaps = [
        f"{cls} precision {p} is below 0.90"
        for cls, p in (result.get("precision") or {}).items()
        if not isinstance(p, (int, float)) or p < 0.90
    ]
    if set(result.get("precision") or {}) != {"ANSWER", "UNSUPPORTED"}:
        gaps.append("precision must be reported for the answer and the decline class")
    return Check(6, "detector measured", BLOCKED if gaps else PASS, [PDET_OUT_RESULT], gaps)


def check_7_corpus(root: Path) -> Check:
    return _missing(
        7,
        "corpus fingerprint recorded",
        FLAG_SET_DISPOSITIONS,
        "the flag set (46 §5) is not computed or triaged, and no arm corpus is built",
    )


def check_8_exposure(root: Path) -> Check:
    plans = root / EXPOSURE_PLANS
    if not plans.is_dir() or not any(plans.glob("*.json")):
        return _missing(
            8,
            "exposures matched and measurable",
            EXPOSURE_PLANS,
            "the planner exists (training/exposure_planner.py), but no arm corpus is rendered to plan "
            "from, and no run has a batch-composition log",
        )
    return Check(
        8,
        "exposures matched and measurable",
        NOT_RUN,
        gaps=["plans exist; matching them against batch-composition logs is not implemented"],
    )


def check_9_seeds(root: Path) -> Check:
    path = "configs/study_002/seed-plan.yaml"
    plan = yaml.safe_load((root / path).read_text(encoding="utf-8"))
    gaps = []
    if plan.get("seeds") != [0, 1, 2]:
        gaps.append(f"seeds are {plan.get('seeds')}, not 05's 0, 1, 2")
    if {a["id"] for a in plan.get("arms", [])} != V14_ARMS:
        gaps.append("the plan's arms are not v14's")
    if {r["id"] for r in plan.get("repeats", [])} != {"REP-A", "REP-B"}:
        gaps.append("REP-A and REP-B are not both planned")
    if not plan.get("random_sources"):
        gaps.append("no list of random sources (05:35-37)")
    configs = sorted((root / ARM_CONFIGS).glob("*.yaml")) if (root / ARM_CONFIGS).is_dir() else []
    if not configs:
        gaps.append(
            f"no arm config under {ARM_CONFIGS}: seeds are planned but no arm carries them yet"
        )
    for config in configs:
        seed = (
            yaml.safe_load(config.read_text(encoding="utf-8")).get("reproducibility") or {}
        ).get("seed")
        if seed not in plan.get("seeds", []):
            gaps.append(f"{config.relative_to(root).as_posix()} has seed {seed!r}")
    return Check(
        9,
        "seeds fixed",
        BLOCKED if gaps else PASS,
        [path],
        gaps,
        {
            "arm_configs": len(configs),
            "s1_model": next(
                (a.get("model") for a in plan.get("arms", []) if a["id"] == "S1"), None
            ),
        },
    )


def check_10_sentinels(root: Path) -> Check:
    path = "registry/study_002_sentinels.yaml"
    registry = yaml.safe_load((root / path).read_text(encoding="utf-8"))
    declared = {s["id"]: tuple(s["modes"]) for s in registry["sentinels"]}
    gaps = []
    if declared != gate.REQUIRED_SENTINELS:
        gaps.append("the registry and the gate's REQUIRED_SENTINELS disagree")
    for sentinel in registry["sentinels"]:
        if not isinstance(sentinel.get("blocking"), bool) or not sentinel.get("population"):
            gaps.append(f"{sentinel['id']} lacks a blocking status or a population")
    elicit: dict[str, Any] = next((s for s in registry["sentinels"] if s["id"] == "S-ANS-E"), {})
    if not elicit.get("instruction"):
        gaps.append("S-ANS-E has no fixed instruction")
    return Check(
        10,
        "sentinels registered",
        BLOCKED if gaps else PASS,
        [path],
        gaps,
        {"sentinels": len(declared)},
    )


def _population_negatives() -> dict[str, dict[str, Any]]:
    frozen = {"CALL": 453, "UNSUPPORTED": 453, "CLARIFY": 371, "ANSWER": 0}
    three_mode = {
        "ANSWER": {"ANSWER": 0},
        "CALL": {"CALL": 400},
        "CLARIFY": {"CLARIFY": 350},
        "UNSUPPORTED": {"UNSUPPORTED": 440},
    }
    runs: dict[str, tuple[Callable[[], Any], str]] = {
        "V1": (lambda: v1_mode_coverage(frozen), "FAIL_NONVACUOUS"),
        "V2": (
            lambda: v2_metric_denominator(
                {"no_call_accuracy": 0.0, "confusion_matrix": three_mode}
            ),
            "FAIL_VACUOUS_METRIC",
        ),
        "V12": (
            lambda: v12_resolvable_margin([{"id": "R1_vs_C0", "delta": 0.12}]),
            "FAIL_UNRESOLVED_ROW",
        ),
    }
    out = {}
    for validator, (run, code) in runs.items():
        result = run()
        out[validator] = {
            "status": result.status,
            "expected_code": code,
            "matched": result.status == FAIL and any(code in e for e in result.all_errors()),
        }
    return out


def check_11_validators(root: Path) -> Check:
    results = _population_negatives() | provenance_self_test(root)
    expected = {f"V{i}" for i in range(1, 13)}
    gaps = [f"{v} is missing" for v in sorted(expected - set(results))]
    gaps += [
        f"{v} did not fail its fixture with {r['expected_code']}"
        for v, r in results.items()
        if not r["matched"]
    ]
    return Check(
        11,
        "validators exist and have failing tests",
        BLOCKED if gaps else PASS,
        [
            "src/opengrad/verification/population_validators.py",
            "src/opengrad/verification/provenance_validators.py",
        ],
        gaps,
        {"fixtures": {v: r["matched"] for v, r in sorted(results.items())}},
    )


def check_12_smoke(root: Path) -> Check:
    return _missing(
        12,
        "retention paths exercised",
        CPU_SMOKE_RECORD,
        "real SFT refuses CPU (training/sft.py), and the artifact writers of 12 and the evaluator's "
        "P-CONF/P-UNANS loading are not built",
    )


def check_13_gate_can_fail(root: Path) -> Check:
    cases = gate.self_test()
    passed = gate.self_test_passed(cases)
    outcomes = {name: case["overall"] for name, case in sorted(cases.items())}
    return Check(
        13,
        "the gate can fail",
        PASS if passed else BLOCKED,
        ["python -m opengrad.verification.study_002_gate --self-test"],
        [] if passed else ["a self-test case did not return its expected outcome"],
        {
            "cases": len(cases),
            "outcomes": outcomes,
            "failed_cases": sum(1 for o in outcomes.values() if o == FAIL),
        },
    )


def check_14_cost(root: Path) -> Check:
    path = "reports/study-002/cost/cost-ledger.json"
    import jsonschema

    ledger = _json(root, path)
    gaps = []
    try:
        jsonschema.validate(ledger, _json(root, "registry/study_002_cost_ledger.schema.json"))
    except jsonschema.ValidationError as exc:
        gaps.append(f"the ledger does not validate: {exc.message}")
    if ledger.get("available_credit", {}).get("status") != "CONFIRMED":
        gaps.append(
            "available credit is not confirmed by the owner (16:149-152); the envelope is not a balance"
        )
    return Check(14, "cost ceiling and accounting", BLOCKED if gaps else PASS, [path], gaps)


CHECKS: tuple[Callable[[Path], Check], ...] = (
    check_1_preflight,
    check_2_gate_version,
    check_3_modes,
    check_4_strata,
    check_5_punans,
    check_6_detector,
    check_7_corpus,
    check_8_exposure,
    check_9_seeds,
    check_10_sentinels,
    check_11_validators,
    check_12_smoke,
    check_13_gate_can_fail,
    check_14_cost,
)


def overall(checks: list[Check]) -> str:
    statuses = {c.status for c in checks}
    if len(checks) != 14:
        return INCOMPLETE
    if BLOCKED in statuses:
        return BLOCKED
    if NOT_RUN in statuses:
        return INCOMPLETE
    return READY


def _commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def readiness_record(root: Path) -> dict[str, Any]:
    checks = []
    for run in CHECKS:
        try:
            checks.append(run(root))
        except (
            OSError,
            KeyError,
            ValueError,
        ) as exc:  # an unreadable artifact is a gap, never a crash or a pass
            number = int(run.__name__.split("_")[1])
            checks.append(
                Check(number, run.__name__, BLOCKED, gaps=[f"{type(exc).__name__}: {exc}"])
            )
    return {
        "kind": RECORD_KIND,
        "specification": "docs/research/study-002/16-GPU-READINESS-GATE.md",
        "preregistration": _current_prereg(root),
        "commit": _commit(root),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": overall(checks),
        "passed": sum(c.status == PASS for c in checks),
        "checks": [asdict(c) for c in checks],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--write", type=Path, help="write the record here once; never overwrites")
    args = parser.parse_args(argv)
    record = readiness_record(args.root)
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.write is not None:
        if args.write.exists():
            parser.error(f"{args.write} exists; a readiness record is never overwritten (16:7-10)")
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if record["status"] == READY else 1


if __name__ == "__main__":
    raise SystemExit(main())
