"""The healthy fixture and the vacuity self-test of ``study_002_gate_v1``.

16-GPU-READINESS-GATE.md check 13 requires the gate to be shown able to fail. :func:`healthy_bundle` is a
bundle that passes every check; :func:`self_test` mutates one field per case and records whether the gate
fails with the expected code. Split from `study_002_gate.py` to keep it under the module-size limit
(`tests/repo/test_module_size.py`); run it with ``python -m opengrad.verification.study_002_gate --self-test``.
"""

from __future__ import annotations

from typing import Any

from opengrad.promotion.tool_use_policy import CODE_NONVACUOUS
from opengrad.verification.accounting import BLOCKED_INPUT_MISSING, FAIL, PASS
from opengrad.verification.population_validators import (
    CODE_UNDER_POWERED,
    CODE_UNRESOLVED_ROW,
    CODE_VACUOUS_METRIC,
)
from opengrad.verification.provenance_validators import CODE_FINGERPRINT, CODE_UNATTACHED_METRIC
from opengrad.verification.study_002_gate import (
    ADOPTED_PARAMETERS,
    BASELINE_METRICS,
    CANDIDATE_METRICS,
    CODE_ACCOUNTING,
    CODE_INVALID_METRIC,
    CODE_MISSING_METRIC,
    CODE_PROVENANCE_INCOMPLETE,
    CODE_WITHIN_NOISE,
    P_UNANS_PARTITIONS,
    REQUIRED_SENTINELS,
    STUDY_002_PARTITION,
    STUDY_002_PARTITION_SHA256,
    UNDECLARED_PARAMETERS,
    PreregParameters,
    study_002_gate,
)
from opengrad.verification.study_002_gate_runs import CODE_DETERMINISM, CODE_INVALID_COMPARISON

#: The corpus manifest the healthy fixture binds to (LF on every platform, `.gitattributes`).
FIXTURE_CORPUS_MANIFEST = "reports/canonical-v3/canonical-v3.manifest.json"
FIXTURE_CORPUS_FINGERPRINT = "60d3123e1c6cef67f904a81f29669dcff44ecc5aeb0e71178a3b5d4e94bfa75a"


def _metric_row(metric: str, arm: str) -> dict[str, Any]:
    if metric == "refusal_correctness":
        partition_id, fingerprint = "P-UNANS-v2", next(iter(P_UNANS_PARTITIONS.values()))
    else:
        partition_id, fingerprint = "P-CONF-v1", STUDY_002_PARTITION_SHA256
    return {
        "id": f"{arm}:{metric}",
        "metric": metric,
        "metric_version": "07-v1",
        "arm": arm,
        "seed": 0,
        "partition_id": partition_id,
        "partition_fingerprint": fingerprint,
        "n": 1677,
        "denominator": 1677,
        "evaluator_version": "fixture",
        "artifact_sha256": "0" * 64,
    }


# -- the vacuity self-test (16-GPU-READINESS-GATE.md check 13) -------------------------------------


#: C0 seed 0's metrics, which REP-A reproduces exactly under ``DECLARED_DETERMINISTIC`` (14).
_C0_SEED_0_METRICS = {"call_f1": 0.75, "answer_rate": 0.74, "refusal_correctness": 0.80}


def _healthy_runs() -> list[dict[str, Any]]:
    """C0 and R1 on 05's seeds 0, 1, 2, evaluated alike (04), plus REP-A (C0, seed 0, re-run; 05, 14)."""
    runs: list[dict[str, Any]] = [
        {
            "arm": arm,
            "seed": seed,
            "provider": "nvidia",
            "tokenizer_sha256": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
            "chat_template_sha256": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
            "evaluator_version": "fixture",
            "engine": "vllm",
            "engine_version": "0.29.0",
            "max_tokens": {"P-CONF": 512, "gsm8k": 512, "ifeval": 2560, "mmlu_pro": 2048},
            "decoding": {"temperature": 0.0, "top_k": 1, "top_p": 1.0},
            "enable_prefix_caching": False,
            "device_class": "H200",
            "determinism_mode": "DECLARED_DETERMINISTIC",
        }
        for arm in ("C0", "R1")
        for seed in (0, 1, 2)
    ]
    runs[0]["metrics"] = dict(_C0_SEED_0_METRICS)
    runs.append({**runs[0], "repeat": "REP-A", "metrics": dict(_C0_SEED_0_METRICS)})
    return runs


def healthy_bundle() -> dict[str, Any]:
    """A bundle that passes every check once the prereg parameters exist. Tests mutate one field."""
    matrix = {
        "CALL": {"CALL": 430, "CLARIFY": 23},
        "ANSWER": {"ANSWER": 380, "UNSUPPORTED": 20},
        "CLARIFY": {"CLARIFY": 350, "CALL": 21},
        "UNSUPPORTED": {"UNSUPPORTED": 440, "ANSWER": 13},
    }
    candidate = {
        "call_f1": 0.75,
        "call_precision": 0.73,
        "call_recall": 0.77,
        "over_call_rate": 0.15,
        "clarification_accuracy": 0.80,
        "unsupported_accuracy": 0.72,
        "no_call_accuracy": 0.78,
        "must_call_accuracy": 0.77,
        "answer_rate": 0.74,
        "refusal_rate": 0.12,
        "refusal_correctness": 0.80,
        "parse_valid_rate": 0.995,
        "confusion_matrix": matrix,
    }
    return {
        "modes": {"CALL": 453, "ANSWER": 400, "CLARIFY": 371, "UNSUPPORTED": 453},
        "census": {
            "discovered": 1677,
            "checked": 1677,
            "passed": 1677,
            "failed": 0,
            "blocked": 0,
            "skipped": 0,
        },
        "candidate": candidate,
        "baseline": {
            "call_f1": 0.62,
            "answer_rate": 0.98,
            "call_precision": 0.70,
            "call_recall": 0.75,
            "clarification_accuracy": 0.78,
            "unsupported_accuracy": 0.70,
        },
        "p_unans_n": 400,
        "sentinels_ran": {sentinel: list(modes) for sentinel, modes in REQUIRED_SENTINELS.items()},
        "metric_rows": [_metric_row(metric, "R1") for metric in CANDIDATE_METRICS]
        + [_metric_row(metric, "C0") for metric in BASELINE_METRICS],
        "fingerprints": [
            {
                "name": "partition",
                "value": STUDY_002_PARTITION_SHA256,
                "artifact": STUDY_002_PARTITION,
            },
            {
                "name": "corpus",
                "value": FIXTURE_CORPUS_FINGERPRINT,
                "artifact": FIXTURE_CORPUS_MANIFEST,
                "field": "corpus_fingerprint",
            },
        ],
        "arm_runs": _healthy_runs(),
        "truncation": {
            "0-shot": {"C0": 0.02, "R1": 0.03},
            "8-shot": {"C0": 0.01, "R1": 0.01},
            "elicit": {"C0": 0.0, "R1": 0.0},
            "5-shot": {"C0": 0.06, "R1": 0.07},
        },
        "comparisons": [
            {"id": "R1_vs_C0", "population": "P-CONF", "n": 1277, "margin": 0.08, "delta": 0.12}
        ],
    }


def _case(
    bundle: dict[str, Any],
    expected: str,
    code: str | None = None,
    parameters: PreregParameters = ADOPTED_PARAMETERS,
) -> dict[str, Any]:
    report = study_002_gate(bundle, parameters)
    errors = [error for result in report.results for error in result.all_errors()]
    case: dict[str, Any] = {"overall": report.overall, "expected": expected}
    if code is not None:
        case["expects_code"] = code
        case["code_found"] = any(code in error for error in errors)
    return case


def self_test() -> dict[str, Any]:
    """Prove the gate can fail, on the fixtures 16-GPU-READINESS-GATE.md:43-44 names and on the
    failure paths contract 1 missed.

    A gate that has never been observed failing is a gate whose failure path is untested. Every case
    but ``healthy_with_undeclared_parameters`` runs with the adopted parameters, so that a failure is
    attributable to the fixture.
    """
    cases: dict[str, dict[str, Any]] = {}
    cases["healthy_under_adopted_prereg"] = _case(healthy_bundle(), PASS)
    cases["healthy_with_undeclared_parameters"] = _case(
        healthy_bundle(), BLOCKED_INPUT_MISSING, parameters=UNDECLARED_PARAMETERS
    )

    empty_mode = healthy_bundle()
    empty_mode["modes"]["ANSWER"] = 0
    empty_mode["candidate"]["confusion_matrix"]["ANSWER"] = {"ANSWER": 0}
    cases["empty_required_mode"] = _case(empty_mode, FAIL, CODE_NONVACUOUS)

    vacuous = healthy_bundle()
    vacuous["candidate"]["confusion_matrix"]["ANSWER"] = {"ANSWER": 0}
    vacuous["candidate"]["no_call_accuracy"] = 0.0
    cases["vacuous_metric_beside_empty_answer_row"] = _case(vacuous, FAIL, CODE_VACUOUS_METRIC)

    missing_sentinel = healthy_bundle()
    del missing_sentinel["sentinels_ran"]["S-ANS-8"]
    cases["missing_sentinel"] = _case(missing_sentinel, FAIL, CODE_PROVENANCE_INCOMPLETE)

    rejected = healthy_bundle()
    rejected["candidate"]["clarification_accuracy"] = 0.10
    cases["policy_rejects_on_a_floor_the_gate_did_not_read"] = _case(rejected, FAIL, "REJECT")

    under_powered = healthy_bundle()
    under_powered["modes"]["ANSWER"] = 140
    under_powered["candidate"]["confusion_matrix"]["ANSWER"] = {"ANSWER": 130, "UNSUPPORTED": 10}
    under_powered["census"] = {
        **under_powered["census"],
        "discovered": 1417,
        "checked": 1417,
        "passed": 1417,
    }
    cases["under_powered_mode"] = _case(under_powered, FAIL, CODE_UNDER_POWERED)

    idle = healthy_bundle()
    idle["census"] = {
        "discovered": 0,
        "checked": 0,
        "passed": 0,
        "failed": 0,
        "blocked": 0,
        "skipped": 0,
    }
    cases["census_that_scored_nothing"] = _case(idle, FAIL, CODE_ACCOUNTING)

    noise = healthy_bundle()
    noise["comparisons"][0].update(margin=0.01, delta=0.01)
    cases["every_comparison_within_noise"] = _case(noise, FAIL, CODE_WITHIN_NOISE)

    unresolved = healthy_bundle()
    unresolved["comparisons"][0]["margin"] = None
    cases["comparison_without_a_margin"] = _case(unresolved, FAIL, CODE_UNRESOLVED_ROW)

    null_metric = healthy_bundle()
    null_metric["candidate"]["parse_valid_rate"] = None
    cases["null_metric"] = _case(null_metric, FAIL, CODE_MISSING_METRIC)

    cases["empty_bundle"] = _case({}, FAIL)

    # -- contract 3 passed these (reports/ERRATA.md §35) -------------------------------------------
    one_arm = healthy_bundle()
    one_arm["truncation"]["5-shot"] = {"R1": 0.40}
    cases["truncation_stage_with_one_arm"] = _case(one_arm, FAIL, CODE_INVALID_COMPARISON)

    no_stage = healthy_bundle()
    del no_stage["truncation"]["5-shot"]
    cases["truncation_stage_omitted"] = _case(no_stage, FAIL, CODE_INVALID_COMPARISON)

    no_regression_baseline = healthy_bundle()
    del no_regression_baseline["baseline"]["clarification_accuracy"]
    no_regression_baseline["candidate"]["clarification_accuracy"] = 0.55
    cases["baseline_without_a_regression_metric"] = _case(
        no_regression_baseline, FAIL, CODE_MISSING_METRIC
    )

    fractional = healthy_bundle()
    fractional["census"]["failed"] = 0.6
    cases["census_with_a_fractional_count"] = _case(fractional, FAIL, CODE_ACCOUNTING)

    impossible = healthy_bundle()
    impossible["candidate"]["call_f1"] = 3.0
    cases["metric_outside_zero_one"] = _case(impossible, FAIL, CODE_INVALID_METRIC)

    oversized = healthy_bundle()
    oversized["comparisons"][0]["n"] = 10**400
    cases["comparison_n_too_large_for_a_float"] = _case(oversized, FAIL, CODE_UNRESOLVED_ROW)

    # -- contract 4 passed these (reports/ERRATA.md §37) -------------------------------------------
    clarify_only = healthy_bundle()
    clarify_only["comparisons"] = [
        {"id": "R1_vs_C0:CLARIFY", "population": "CLARIFY", "n": 371, "margin": 0.15}
    ]
    cases["comparison_on_a_descriptive_population"] = _case(clarify_only, FAIL, CODE_WITHIN_NOISE)

    renamed = healthy_bundle()
    renamed["comparisons"] = [{"id": "c", "population": "clarify", "n": 1277, "margin": 0.2}]
    cases["comparison_on_a_population_the_gate_does_not_know"] = _case(
        renamed, FAIL, CODE_UNRESOLVED_ROW
    )

    marginless = healthy_bundle()
    marginless["comparisons"].append({"id": "c", "population": "CLARIFY", "n": 371, "margin": None})
    cases["descriptive_row_without_a_margin"] = _case(marginless, FAIL, CODE_UNRESOLVED_ROW)

    corpus_bound = healthy_bundle()
    for row in corpus_bound["metric_rows"]:
        row["partition_fingerprint"] = FIXTURE_CORPUS_FINGERPRINT
    cases["metric_rows_bound_to_the_corpus"] = _case(corpus_bound, FAIL, CODE_FINGERPRINT)

    hatch = healthy_bundle()
    hatch["fingerprints"].append({**hatch["fingerprints"][1], "name": "partition:x"})
    for row in hatch["metric_rows"]:
        row["partition_fingerprint"] = FIXTURE_CORPUS_FINGERPRINT
    cases["metric_rows_bound_to_a_file_named_as_a_partition"] = _case(hatch, FAIL, CODE_FINGERPRINT)

    swapped = healthy_bundle()
    for row in swapped["metric_rows"]:
        row["partition_fingerprint"] = STUDY_002_PARTITION_SHA256
    cases["refusal_correctness_bound_to_p_conf"] = _case(swapped, FAIL, CODE_FINGERPRINT)

    # -- contract 5 passed these (reports/ERRATA.md §38) -------------------------------------------
    other_budget = healthy_bundle()
    other_budget["arm_runs"][3]["max_tokens"] = {
        **other_budget["arm_runs"][3]["max_tokens"],
        "mmlu_pro": 768,
    }
    cases["arm_runs_with_a_different_budget"] = _case(other_budget, FAIL, CODE_INVALID_COMPARISON)

    other_template = healthy_bundle()
    other_template["arm_runs"][5]["chat_template_sha256"] = "1" * 64
    cases["arm_runs_with_a_different_template"] = _case(
        other_template, FAIL, CODE_INVALID_COMPARISON
    )

    extra_arm = healthy_bundle()
    extra_arm["arm_runs"] += [{**run, "arm": "S1"} for run in extra_arm["arm_runs"][:3]]
    cases["arm_runs_with_an_arm_the_truncation_table_lacks"] = _case(
        extra_arm, FAIL, CODE_INVALID_COMPARISON
    )

    mixed_seeds = healthy_bundle()
    for run, seed in zip(mixed_seeds["arm_runs"], (0, 0.0, "0", 0, 0.0, "0")):
        run["seed"] = seed
    cases["one_seed_written_three_ways"] = _case(mixed_seeds, FAIL, CODE_INVALID_COMPARISON)

    placeholder = healthy_bundle()
    for run in placeholder["arm_runs"]:
        run["tokenizer_sha256"] = ""
    cases["arm_runs_with_an_empty_hash"] = _case(placeholder, FAIL, CODE_INVALID_COMPARISON)

    other_seeds = healthy_bundle()
    for run in other_seeds["arm_runs"]:
        run["seed"] += 3
    cases["arms_on_seeds_3_4_5"] = _case(other_seeds, FAIL, CODE_INVALID_COMPARISON)

    rep_a_drift = healthy_bundle()
    rep_a_drift["arm_runs"][-1]["chat_template_sha256"] = "2" * 64
    cases["rep_a_with_a_different_template"] = _case(rep_a_drift, FAIL, CODE_INVALID_COMPARISON)

    # -- contract 6 passed these (reports/ERRATA.md §39) -------------------------------------------
    not_reproduced = healthy_bundle()
    not_reproduced["arm_runs"][-1]["metrics"]["call_f1"] = 0.76
    cases["rep_a_not_reproduced_under_declared_determinism"] = _case(
        not_reproduced, FAIL, CODE_DETERMINISM
    )

    two_modes = healthy_bundle()
    two_modes["arm_runs"][3]["determinism_mode"] = "NON_DETERMINISTIC_KERNEL"
    cases["arms_in_two_determinism_modes"] = _case(two_modes, FAIL, CODE_INVALID_COMPARISON)

    no_rep_a = healthy_bundle()
    del no_rep_a["arm_runs"][-1]
    cases["no_rep_a_record"] = _case(no_rep_a, FAIL, CODE_INVALID_COMPARISON)

    one_seed_short = healthy_bundle()
    del one_seed_short["arm_runs"][5]
    cases["an_arm_with_two_seeds"] = _case(one_seed_short, FAIL, CODE_INVALID_COMPARISON)

    bad_field = healthy_bundle()
    bad_field["fingerprints"][1]["field"] = 7
    cases["fingerprint_field_that_is_not_a_path"] = _case(bad_field, FAIL, CODE_FINGERPRINT)

    inflated = healthy_bundle()
    inflated["comparisons"][0]["n"] = 5000
    cases["comparison_n_not_the_population_n"] = _case(inflated, FAIL, CODE_UNRESOLVED_ROW)

    unbound = healthy_bundle()
    del unbound["comparisons"][0]["population"]
    cases["comparison_naming_no_population"] = _case(unbound, FAIL, CODE_UNRESOLVED_ROW)

    typed = healthy_bundle()
    typed["fingerprints"][0]["value"] = "f" * 64
    cases["fingerprint_not_read_from_its_artifact"] = _case(typed, FAIL, CODE_FINGERPRINT)

    unattached = healthy_bundle()
    unattached["metric_rows"] = [
        row for row in unattached["metric_rows"] if row["metric"] != "refusal_correctness"
    ]
    cases["metric_without_an_attached_row"] = _case(unattached, FAIL, CODE_UNATTACHED_METRIC)

    no_sha = healthy_bundle()
    no_sha["metric_rows"][0]["artifact_sha256"] = "not-a-hash"
    cases["metric_row_without_an_artifact_hash"] = _case(no_sha, FAIL, CODE_UNATTACHED_METRIC)
    return cases


def self_test_passed(cases: dict[str, dict[str, Any]]) -> bool:
    return all(
        case["overall"] == case["expected"] and case.get("code_found", True)
        for case in cases.values()
    )
