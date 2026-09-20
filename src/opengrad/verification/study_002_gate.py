"""``study_002_gate_v1`` — the Study 002 evaluation gate (`docs/research/study-002/11-THRESHOLDS.md`).

This is the gate `16-GPU-READINESS-GATE.md` check 2 requires to exist and check 13 requires to be
shown able to fail. It is a machine-checkable verdict over an **evaluation bundle**, built on the
shared accounting contract (`opengrad.verification.accounting`), and it is deliberately able to say
no: a required input that is absent is ``BLOCKED_INPUT_MISSING`` (never a pass), and a required
population the bundle cannot measure is ``FAIL_NONVACUOUS``.

The fourteen checks, and their failure codes (11-THRESHOLDS.md §``study_002_gate_v1``):

1. ``mode_coverage`` -- every required mode has ``n > 0`` -- ``FAIL_NONVACUOUS``
2. ``census`` -- ``discovered == checked + blocked + skipped`` and ``checked == passed + failed`` -- ``FAIL_ACCOUNTING``
3. ``metric_denominators`` -- no metric reports ``0.0`` from an empty denominator -- ``FAIL_VACUOUS_METRIC``
4. ``safety_regression`` -- ``min_refusal_correctness`` on ``P-UNANS`` -- ``SAFETY_REGRESSION``
5. ``answer_mode`` -- ``min_answer_rate`` / ``max_refusal_rate`` on ``ANSWER``-gold -- ``TOOL_POLICY_REGRESSION``
6. ``no_call_accuracy`` -- ``TOOL_POLICY_REGRESSION``
7. ``call_f1_retention`` -- ``TOOL_POLICY_REGRESSION``
8. ``over_call_rate`` -- ``TOOL_POLICY_REGRESSION``
9. ``macro_recall`` over measured modes only; ``UNMEASURED`` if none -- ``TOOL_POLICY_REGRESSION``
10. ``parse_valid_rate`` -- ``NOT_EVALUABLE``
11. ``sentinels`` -- every required sentinel ran, in every mode -- ``PROVENANCE_INCOMPLETE``
12. ``truncation`` -- truncation reported per stage -- ``INVALID_COMPARISON``
13. ``provenance`` -- complete per 15-PROVENANCE-VALIDATORS.md -- ``PROVENANCE_INCOMPLETE``
14. ``comparison_margins`` -- every comparison prints ``n`` and its margin, or is ``WITHIN_NOISE`` -- ``FAIL_UNRESOLVED_ROW``

Bundle keys (the gate's input contract; a key that is absent blocks its checks rather than passing
them):

* ``modes``: ``{mode: gold_count}`` -- the confirmatory partition's gold counts;
* ``census``: ``{discovered, checked, passed, failed, blocked, skipped}``;
* ``candidate`` / ``baseline``: the promotion-policy metric dicts (see
  :mod:`opengrad.promotion.tool_use_policy`), the candidate carrying a ``confusion_matrix``;
* ``required_sentinels`` / ``sentinels_ran``: sentinel ids;
* ``provenance``: ``{required: [...], present: [...]}``;
* ``truncation``: ``{stage: rate}`` and ``stages``: ``[...]``;
* ``comparisons``: ``[{id, n, margin, delta}]``.

The bundle is produced by the evaluation, not by this module; no artifact of that shape exists yet,
so today the gate is exercised on synthetic bundles by its tests and reports ``BLOCKED_INPUT_MISSING``
on the real repository. That is the point: it converts "what is left" into a machine-checked list.

    python -m opengrad.verification.study_002_gate --self-test
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from opengrad.promotion.tool_use_policy import (
    CODE_NONVACUOUS,
    MACRO_DIMENSIONS,
    REQUIRED_MODES,
    PromotionPolicyV5,
    measurable_dimensions,
)
from opengrad.registry.validate import result_from
from opengrad.verification.accounting import (
    BLOCKED_INPUT_MISSING,
    CONDITIONALLY_REQUIRED,
    FAIL,
    PASS,
    REQUIRED_NONEMPTY,
    ValidationResult,
    VerificationReport,
)

#: The gate's own contract. Bump when the check set, its discovery rules or its non-vacuity
#: requirements change, so a PASS under one contract is never read as a PASS under another.
STUDY_002_GATE_CONTRACT = 1
GATE_VERSION = "study_002_gate_v1"

CODE_VACUOUS_METRIC = "FAIL_VACUOUS_METRIC"
CODE_ACCOUNTING = "FAIL_ACCOUNTING"
CODE_PROVENANCE_INCOMPLETE = "PROVENANCE_INCOMPLETE"
CODE_INVALID_COMPARISON = "INVALID_COMPARISON"
CODE_UNRESOLVED_ROW = "FAIL_UNRESOLVED_ROW"

#: The behavioural checks 4-10, and the v5 dimension each reads.
BEHAVIOUR_CHECKS = (
    ("safety_regression", ("refusal_correctness",)),
    ("answer_mode", ("answer_rate", "refusal_rate", "answer_mode_coverage")),
    ("no_call_accuracy", ("no_call_accuracy",)),
    ("call_f1_retention", ("call_f1_retention",)),
    ("over_call_rate", ("over_call_rate",)),
    ("macro_recall", ("macro_recall",)),
    ("parse_valid_rate", ("parse_valid_rate",)),
)


class Study002Gate:
    """The fourteen checks over one evaluation bundle."""

    def __init__(self, bundle: dict[str, Any]) -> None:
        self.bundle = bundle
        self._verdict: dict[str, Any] | None = None

    # -- inputs ---------------------------------------------------------------------------------

    @property
    def candidate(self) -> dict[str, Any]:
        return dict(self.bundle.get("candidate") or {})

    @property
    def baseline(self) -> dict[str, Any]:
        return dict(self.bundle.get("baseline") or {})

    @property
    def verdict(self) -> dict[str, Any]:
        if self._verdict is None:
            self._verdict = PromotionPolicyV5().evaluate(self.candidate, self.baseline)
        return self._verdict

    def _v5(self, dimension: str) -> dict[str, Any] | None:
        return next((c for c in self.verdict["checks"] if c["dimension"] == dimension), None)

    def _blocked(self, name: str, candidates: list[str], reason: str) -> ValidationResult:
        return result_from(
            name,
            CONDITIONALLY_REQUIRED,
            candidates,
            [],
            blocked_ids=candidates,
            blocked_status=BLOCKED_INPUT_MISSING,
            detail={"reason": reason},
        )

    # -- 1-3: the L1 family ---------------------------------------------------------------------

    def mode_coverage(self) -> ValidationResult:
        modes = dict(self.bundle.get("modes") or {})
        if not modes:
            return self._blocked("mode_coverage", list(REQUIRED_MODES), "no gold-count table in the bundle")
        errors = [
            f"{mode}: {CODE_NONVACUOUS}: gold n={int(modes.get(mode, 0))}"
            for mode in REQUIRED_MODES
            if int(modes.get(mode, 0)) <= 0
        ]
        return result_from("mode_coverage", REQUIRED_NONEMPTY, list(REQUIRED_MODES), errors)

    def census(self) -> ValidationResult:
        census = self.bundle.get("census")
        if not isinstance(census, dict):
            return self._blocked("census", ["census"], "no execution census in the bundle")
        problems = []
        if census.get("discovered") != (
            census.get("checked", 0) + census.get("blocked", 0) + census.get("skipped", 0)
        ):
            problems.append("discovered != checked + blocked + skipped")
        if census.get("checked") != census.get("passed", 0) + census.get("failed", 0):
            problems.append("checked != passed + failed")
        errors = [f"census: {CODE_ACCOUNTING}: {problem}" for problem in problems]
        return result_from("census", REQUIRED_NONEMPTY, ["census"], errors)

    def metric_denominators(self) -> ValidationResult:
        candidate = self.candidate
        if not isinstance(candidate.get("confusion_matrix"), dict):
            return self._blocked(
                "metric_denominators", list(MACRO_DIMENSIONS), "no confusion matrix to read denominators from"
            )
        measurable, unmeasurable = measurable_dimensions(candidate)
        errors = []
        for name in sorted(unmeasurable):
            # A metric computed over an empty class is an absence encoded numerically (the L1 root cause).
            if float(candidate.get(name, 0.0)) == 0.0:
                errors.append(
                    f"{name}: {CODE_VACUOUS_METRIC}: reports 0.0 for a class with an empty denominator"
                )
        return result_from("metric_denominators", REQUIRED_NONEMPTY, list(MACRO_DIMENSIONS), errors,
                           detail={"measurable": len(measurable), "unmeasurable": len(unmeasurable)})

    # -- 4-10: the behavioural family (from the v5 verdict) -------------------------------------

    def behaviour(self) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        for name, dimensions in BEHAVIOUR_CHECKS:
            ids = list(dimensions)
            checks = [self._v5(dimension) for dimension in dimensions]
            if all(check is None for check in checks):
                # The gate cannot measure this family here (e.g. no P-UNANS, no ANSWER population).
                results.append(
                    self._blocked(name, ids, "the bundle does not carry this family's inputs")
                )
                continue
            errors = [
                f"{check['dimension']}: {check.get('code', '')}: observed {check['observed']} "
                f"requires {check['requirement']}"
                for check in checks
                if check is not None and not check["passed"]
            ]
            results.append(result_from(name, CONDITIONALLY_REQUIRED, ids, errors))
        return results

    # -- 11-14: sentinels, truncation, provenance, margins --------------------------------------

    def sentinels(self) -> ValidationResult:
        required = list(self.bundle.get("required_sentinels") or [])
        ran = set(self.bundle.get("sentinels_ran") or [])
        if not required:
            return self._blocked("sentinels", ["sentinel_registration"], "no sentinel registration in the bundle")
        errors = [
            f"{sentinel}: {CODE_PROVENANCE_INCOMPLETE}: did not run"
            for sentinel in required
            if sentinel not in ran
        ]
        return result_from("sentinels", REQUIRED_NONEMPTY, required, errors)

    def truncation(self) -> ValidationResult:
        stages = list(self.bundle.get("stages") or [])
        truncation = self.bundle.get("truncation")
        if not stages or not isinstance(truncation, dict):
            return self._blocked("truncation", stages or ["truncation"], "no per-stage truncation record")
        errors = [
            f"{stage}: {CODE_INVALID_COMPARISON}: no truncation rate reported"
            for stage in stages
            if stage not in truncation
        ]
        return result_from("truncation", REQUIRED_NONEMPTY, stages, errors)

    def provenance(self) -> ValidationResult:
        provenance = self.bundle.get("provenance")
        if not isinstance(provenance, dict) or not provenance.get("required"):
            return self._blocked("provenance", ["provenance"], "no provenance manifest in the bundle")
        required = list(provenance["required"])
        present = set(provenance.get("present") or [])
        errors = [
            f"{item}: {CODE_PROVENANCE_INCOMPLETE}: absent"
            for item in required
            if item not in present
        ]
        return result_from("provenance", REQUIRED_NONEMPTY, required, errors)

    def comparison_margins(self) -> ValidationResult:
        rows = list(self.bundle.get("comparisons") or [])
        if not rows:
            return self._blocked("comparison_margins", ["comparisons"], "no comparison rows in the bundle")
        ids, errors, within_noise = [], [], []
        for row in rows:
            row_id = str(row.get("id", "?"))
            ids.append(row_id)
            if row.get("n") is None or row.get("margin") is None:
                errors.append(f"{row_id}: {CODE_UNRESOLVED_ROW}: row prints no n or no resolvable margin")
            elif row.get("delta") is not None and abs(float(row["delta"])) <= float(row["margin"]):
                within_noise.append(row_id)
        return result_from(
            "comparison_margins", REQUIRED_NONEMPTY, ids, errors, detail={"within_noise": len(within_noise)}
        )

    # -- assembly -------------------------------------------------------------------------------

    def checks(self) -> list[ValidationResult]:
        return [
            self.mode_coverage(),
            self.census(),
            self.metric_denominators(),
            *self.behaviour(),
            self.sentinels(),
            self.truncation(),
            self.provenance(),
            self.comparison_margins(),
        ]

    def report(self) -> VerificationReport:
        report = VerificationReport(contract=STUDY_002_GATE_CONTRACT)
        for result in self.checks():
            report.add(result)
        return report


def study_002_gate(bundle: dict[str, Any]) -> VerificationReport:
    return Study002Gate(bundle).report()


# -- the vacuity self-test (16-GPU-READINESS-GATE.md check 13) -------------------------------------


def healthy_bundle() -> dict[str, Any]:
    """A bundle that passes every check. Tests mutate one field at a time from here."""
    matrix = {
        "CALL": {"CALL": 400, "CLARIFY": 20},
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
        "census": {"discovered": 1277, "checked": 1277, "passed": 1277, "failed": 0, "blocked": 0, "skipped": 0},
        "candidate": candidate,
        "baseline": {"call_f1": 0.62, "answer_rate": 0.98},
        "required_sentinels": ["S-REF", "S-NV"],
        "sentinels_ran": ["S-REF", "S-NV"],
        "provenance": {"required": ["corpus_fingerprint", "partition"], "present": ["corpus_fingerprint", "partition"]},
        "stages": ["0-shot", "8-shot", "elicit"],
        "truncation": {"0-shot": 0.02, "8-shot": 0.01, "elicit": 0.0},
        "comparisons": [{"id": "R1_vs_C0", "n": 1277, "margin": 0.08, "delta": 0.12}],
    }


def self_test() -> dict[str, Any]:
    """Prove the gate can fail, on the three fixtures 16-GPU-READINESS-GATE.md:36-38 names.

    A gate that has never been observed failing is a gate whose failure path is untested.
    """
    cases: dict[str, dict[str, Any]] = {}

    healthy = study_002_gate(healthy_bundle())
    cases["healthy"] = {"overall": healthy.overall, "expected": PASS}

    empty_mode = healthy_bundle()
    empty_mode["modes"]["ANSWER"] = 0
    empty_mode["candidate"]["confusion_matrix"]["ANSWER"] = {"ANSWER": 0}
    result = study_002_gate(empty_mode)
    cases["empty_required_mode"] = {
        "overall": result.overall,
        "expected": FAIL,
        "expects_code": CODE_NONVACUOUS,
    }

    vacuous = healthy_bundle()
    vacuous["candidate"]["confusion_matrix"]["ANSWER"] = {"ANSWER": 0}
    vacuous["candidate"]["no_call_accuracy"] = 0.0
    result = study_002_gate(vacuous)
    cases["vacuous_metric_beside_empty_answer_row"] = {
        "overall": result.overall,
        "expected": FAIL,
        "expects_code": CODE_VACUOUS_METRIC,
    }

    missing_sentinel = healthy_bundle()
    missing_sentinel["sentinels_ran"] = ["S-REF"]
    result = study_002_gate(missing_sentinel)
    cases["missing_sentinel"] = {
        "overall": result.overall,
        "expected": FAIL,
        "expects_code": CODE_PROVENANCE_INCOMPLETE,
    }
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.self_test:
        parser.error("only --self-test is implemented")
    cases = self_test()
    print(json.dumps(cases, indent=2, sort_keys=True))
    observed = {name: case["overall"] for name, case in cases.items()}
    ok = observed.get("healthy") == PASS and all(
        observed[name] == FAIL for name in cases if name != "healthy"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
