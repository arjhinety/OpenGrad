"""``study_002_gate_v1`` — the Study 002 evaluation gate (`docs/research/study-002/11-THRESHOLDS.md`).

This is the gate `16-GPU-READINESS-GATE.md` check 2 requires to exist and check 13 requires to be
shown able to fail. It is a machine-checkable verdict over an **evaluation bundle**, built on the
shared accounting contract (`opengrad.verification.accounting`), and it is deliberately able to say
no: a required input that is absent is ``BLOCKED_INPUT_MISSING`` (never a pass), and a required
population the bundle cannot measure is ``FAIL_NONVACUOUS``.

**Contract 2 (2026-09-24).** Contract 1 read seven of the ``tool_use_promotion_v5`` dimensions and
never its decision, so a candidate v5 rejected on its clarification or unsupported floors, its
answer-rate drop or a baseline regression still passed; it also let the bundle declare its own
required sentinels and provenance fields, accepted a census that executed nothing, and passed an
under-powered mode and unresolved comparison rows. Contract 2 closes each of those
(`reports/ERRATA.md` §19). A PASS under contract 1 is not a PASS under contract 2.

The checks, and their failure codes (11-THRESHOLDS.md §``study_002_gate_v1``):

1. ``mode_coverage`` -- every required mode has ``n > 0``, and ``n >= 200`` (06 §C2) --
   ``FAIL_NONVACUOUS`` / ``UNDER_POWERED``
2. ``census`` -- the evaluation census reconciles, executed every gold item once and nothing else,
   and the confusion matrix counts the same items -- ``FAIL_ACCOUNTING``
3. ``metric_denominators`` -- no metric reports a number for an empty denominator -- ``FAIL_VACUOUS_METRIC``
3b. ``metric_values`` -- every metric the policy reads is present and a finite number -- ``FAIL_MISSING_METRIC``
4. ``safety_regression`` -- ``min_refusal_correctness`` on a ``P-UNANS`` large enough to resolve it -- ``SAFETY_REGRESSION``
5. ``answer_mode`` -- ``min_answer_rate`` / ``max_refusal_rate`` / answer-rate drop on ``ANSWER``-gold -- ``TOOL_POLICY_REGRESSION``
6. ``no_call_accuracy`` -- ``TOOL_POLICY_REGRESSION``
7. ``call_f1_retention`` -- ``TOOL_POLICY_REGRESSION``
8. ``over_call_rate`` -- ``TOOL_POLICY_REGRESSION``
9. ``macro_recall`` over measured modes only -- ``TOOL_POLICY_REGRESSION``
10. ``parse_valid_rate`` -- ``NOT_EVALUABLE``
11. ``sentinels`` -- every sentinel in 08 ran, in its mode -- ``PROVENANCE_INCOMPLETE``
12. ``truncation`` -- a truncation rate in [0, 1] per stage per arm -- ``INVALID_COMPARISON``
12b. ``truncation_balance`` -- imbalance within the declared factor -- ``INVALID_COMPARISON``
13. ``provenance`` -- every field 15 V3 / V9 require is present -- ``PROVENANCE_INCOMPLETE``
14. ``comparison_margins`` -- every row prints ``n`` and a margin; at least one clears its resolvable
    margin, the rest are ``WITHIN_NOISE`` and support nothing -- ``FAIL_UNRESOLVED_ROW`` / ``WITHIN_NOISE``
15. ``policy_decision`` -- ``tool_use_promotion_v5`` returns ``PROMOTE`` -- the v5 failure codes

Two rules the preregistration states but never quantifies are held as :class:`PreregParameters`:
the truncation "declared factor" (11:74) and the ``P-UNANS`` size that resolves ``refusal_correctness``
(11:127). No adopted amendment declares them, so :data:`ADOPTED_PARAMETERS` leaves both ``None`` and
checks 4 and 12b report ``BLOCKED_INPUT_MISSING``: the gate cannot PASS until the owner adopts values
(`docs/research/study-002/40-PREREG-V8-DRAFT.md` proposes them).

Bundle keys (the gate's input contract; a key that is absent blocks its checks rather than passing
them):

* ``modes``: ``{mode: gold_count}`` -- the confirmatory partition's gold counts;
* ``census``: ``{discovered, checked, passed, failed, blocked, skipped}`` -- the execution census of
  the scored items (``failed`` is an item that did not score, not a wrong answer);
* ``candidate`` / ``baseline``: the promotion-policy metric dicts (see
  :mod:`opengrad.promotion.tool_use_policy`), the candidate carrying a ``confusion_matrix``;
* ``p_unans_n``: the number of scored ``P-UNANS`` items;
* ``sentinels_ran``: ``{sentinel_id: [mode, ...]}``;
* ``provenance``: ``{present: [field, ...]}``;
* ``truncation``: ``{stage: {arm: rate}}``;
* ``comparisons``: ``[{id, n, margin, delta}]``.

The bundle is produced by the evaluation, not by this module; no artifact of that shape exists yet,
so today the gate is exercised on synthetic bundles by its tests. On an empty bundle it reports
``FAIL`` (the policy's missing metrics fail closed) with every input-bearing check ``BLOCKED`` --
never ``PASS``.

    python -m opengrad.verification.study_002_gate --self-test
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

from opengrad.promotion.tool_use_policy import (
    CODE_NONVACUOUS,
    CODE_TOOL_POLICY_REGRESSION,
    PROMOTE,
    PromotionPolicyV5,
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
from opengrad.verification.population_validators import (
    CODE_UNDER_POWERED,
    CODE_UNRESOLVED_ROW,
    CODE_VACUOUS_METRIC,
    finite_number,
    v1_mode_coverage,
    v2_metric_denominator,
    v12_resolvable_margin,
)
from opengrad.verification.resolvability import at_most

#: The gate's own contract. Bump when the check set, its discovery rules or its non-vacuity
#: requirements change, so a PASS under one contract is never read as a PASS under another.
STUDY_002_GATE_CONTRACT = 2
GATE_VERSION = "study_002_gate_v1"

CODE_ACCOUNTING = "FAIL_ACCOUNTING"
CODE_PROVENANCE_INCOMPLETE = "PROVENANCE_INCOMPLETE"
CODE_INVALID_COMPARISON = "INVALID_COMPARISON"
CODE_WITHIN_NOISE = "WITHIN_NOISE"
CODE_MISSING_METRIC = "FAIL_MISSING_METRIC"

#: The behavioural checks 4-10, and the v5 dimension each reads.
BEHAVIOUR_CHECKS = (
    ("safety_regression", ("refusal_correctness",)),
    (
        "answer_mode",
        ("answer_rate", "refusal_rate", "answer_mode_coverage", "answer_rate_drop_vs_base"),
    ),
    ("no_call_accuracy", ("no_call_accuracy",)),
    ("call_f1_retention", ("call_f1_retention",)),
    ("over_call_rate", ("over_call_rate",)),
    ("macro_recall", ("macro_recall",)),
    ("parse_valid_rate", ("parse_valid_rate",)),
)

#: Every metric ``tool_use_promotion_v5`` reads. The policy fills an absent one with a default, so
#: the gate requires each to be present and finite rather than trusting the default.
CANDIDATE_METRICS = (
    "call_f1",
    "call_precision",
    "call_recall",
    "over_call_rate",
    "clarification_accuracy",
    "unsupported_accuracy",
    "no_call_accuracy",
    "must_call_accuracy",
    "answer_rate",
    "refusal_rate",
    "refusal_correctness",
    "parse_valid_rate",
)
BASELINE_METRICS = ("call_f1", "answer_rate")

#: The sentinel inventory (08-SENTINEL-SPEC.md, "Sentinel inventory") and the mode each runs in.
#: ``S-OW7`` never blocks a verdict but must run (08 "Blocking semantics"); ``S-NV`` has no mode.
#: Pinned here so a bundle cannot satisfy check 11 by declaring a shorter list.
REQUIRED_SENTINELS: dict[str, tuple[str, ...]] = {
    "S-ANS-0": ("0-shot",),
    "S-ANS-8": ("8-shot",),
    "S-ANS-E": ("elicit",),
    "S-TP-4": ("0-shot",),
    "S-REF": ("0-shot",),
    "S-IF": ("0-shot",),
    "S-MMLU": ("5-shot",),
    "S-OW7": ("0-shot",),
    "S-NV": (),
}

#: The attachment every number carries (15 V3) plus the artifact-read fingerprints (15 V9).
REQUIRED_PROVENANCE = (
    "metric",
    "metric_version",
    "arm",
    "seed",
    "partition_id",
    "partition_fingerprint",
    "n",
    "denominator",
    "evaluator_version",
    "artifact_sha256",
    "corpus_fingerprint",
)


@dataclass(frozen=True)
class PreregParameters:
    """Values the preregistration requires but has not yet declared.

    ``None`` means undeclared, and the check that needs it is ``BLOCKED_INPUT_MISSING``.

    * ``truncation_max_ratio`` / ``truncation_min_gap``: within a stage, two arms' truncation rates
      are imbalanced when they differ by more than ``truncation_min_gap`` (absolute) **and** the
      larger exceeds ``truncation_max_ratio`` times the smaller (11:74, "the declared factor").
    * ``p_unans_min_n``: the ``P-UNANS`` size that resolves ``refusal_correctness`` (11:127).
    """

    truncation_max_ratio: float | None = None
    truncation_min_gap: float | None = None
    p_unans_min_n: int | None = None


#: What the adopted preregistration (``study_002_prereg_v7``) declares: neither value.
ADOPTED_PARAMETERS = PreregParameters()

#: The values ``40-PREREG-V8-DRAFT.md`` proposes. **Not adopted.** Used only by the self-test and
#: the tests, to show that a PASS is reachable once values exist; never by a real verdict.
DRAFT_V8_PARAMETERS = PreregParameters(
    truncation_max_ratio=2.0, truncation_min_gap=0.02, p_unans_min_n=385
)


class Study002Gate:
    """The checks over one evaluation bundle."""

    def __init__(
        self, bundle: dict[str, Any], parameters: PreregParameters = ADOPTED_PARAMETERS
    ) -> None:
        self.bundle = bundle
        self.parameters = parameters
        self._verdict: dict[str, Any] | None = None

    # -- inputs ---------------------------------------------------------------------------------

    @property
    def raw_candidate(self) -> dict[str, Any]:
        return dict(self.bundle.get("candidate") or {})

    @property
    def candidate(self) -> dict[str, Any]:
        """The candidate as the policy may read it: non-numeric metrics dropped.

        The policy calls ``float()`` on what it reads; a ``None`` or a string would crash it. Dropping
        the value makes the policy apply its fail-closed default, and check 3b reports why.
        """
        clean: dict[str, Any] = {}
        for key, value in self.raw_candidate.items():
            if key == "confusion_matrix":
                if _valid_matrix(value):
                    clean[key] = value
            elif finite_number(value) is not None:
                clean[key] = value
        return clean

    @property
    def baseline(self) -> dict[str, Any]:
        raw = dict(self.bundle.get("baseline") or {})
        return {key: value for key, value in raw.items() if finite_number(value) is not None}

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

    def _modes(self) -> dict[str, Any]:
        return dict(self.bundle.get("modes") or {})

    # -- 1-3b: the L1 family --------------------------------------------------------------------

    def mode_coverage(self) -> ValidationResult:
        return v1_mode_coverage(self._modes(), enforce_floor=True)

    def census(self) -> ValidationResult:
        census = self.bundle.get("census")
        if not isinstance(census, dict):
            return self._blocked("census", ["census"], "no execution census in the bundle")
        keys = ("discovered", "checked", "passed", "failed", "blocked", "skipped")
        counts = {key: finite_number(census.get(key)) for key in keys}
        problems = [
            f"{key} is not a count ({census.get(key)!r})"
            for key, value in counts.items()
            if value is None
        ]
        if not problems:
            c = {key: int(value or 0) for key, value in counts.items()}
            if c["discovered"] != c["checked"] + c["blocked"] + c["skipped"]:
                problems.append("discovered != checked + blocked + skipped")
            if c["checked"] != c["passed"] + c["failed"]:
                problems.append("checked != passed + failed")
            if c["discovered"] <= 0:
                problems.append(
                    "the census discovered nothing; an evaluation that scored no item proves nothing"
                )
            if c["blocked"] or c["skipped"] or c["failed"]:
                problems.append(
                    f"{c['blocked']} blocked, {c['skipped']} skipped and {c['failed']} failed items; every gold "
                    "item must be scored"
                )
            gold = [finite_number(value) for value in self._modes().values()]
            if gold and all(value is not None for value in gold):
                total = int(sum(value or 0 for value in gold))
                if c["discovered"] != total:
                    problems.append(
                        f"discovered {c['discovered']} != {total} gold items in the mode table"
                    )
        matrix = self.raw_candidate.get("confusion_matrix")
        if isinstance(matrix, dict) and _valid_matrix(matrix):
            for mode, gold_n in self._modes().items():
                row = matrix.get(mode) or {}
                row_total = sum(int(v) for v in row.values())
                if finite_number(gold_n) is not None and row_total != int(gold_n):
                    problems.append(
                        f"confusion-matrix row {mode} counts {row_total} items, the mode table {gold_n}"
                    )
        errors = [f"census: {CODE_ACCOUNTING}: {problem}" for problem in problems]
        return result_from("census", REQUIRED_NONEMPTY, ["census"], errors)

    def metric_denominators(self) -> ValidationResult:
        return v2_metric_denominator(self.raw_candidate)

    def metric_values(self) -> ValidationResult:
        candidate, baseline = self.raw_candidate, dict(self.bundle.get("baseline") or {})
        ids = [f"candidate.{m}" for m in CANDIDATE_METRICS] + [
            f"baseline.{m}" for m in BASELINE_METRICS
        ]
        errors = [
            f"{side}.{metric}: {CODE_MISSING_METRIC}: {metrics.get(metric)!r} is not a finite number"
            for side, metrics, names in (
                ("candidate", candidate, CANDIDATE_METRICS),
                ("baseline", baseline, BASELINE_METRICS),
            )
            for metric in names
            if finite_number(metrics.get(metric)) is None
        ]
        if "confusion_matrix" in candidate and not _valid_matrix(candidate["confusion_matrix"]):
            ids.append("candidate.confusion_matrix")
            errors.append(
                f"candidate.confusion_matrix: {CODE_MISSING_METRIC}: not a {{truth: {{predicted: count}}}} table"
            )
        return result_from("metric_values", REQUIRED_NONEMPTY, ids, errors)

    # -- 4-10: the behavioural family (from the v5 verdict) -------------------------------------

    def behaviour(self) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        for name, dimensions in BEHAVIOUR_CHECKS:
            if name == "safety_regression":
                results.append(self.safety_regression())
                continue
            ids = list(dimensions)
            checks = [self._v5(dimension) for dimension in dimensions]
            if all(check is None for check in checks):
                # The gate cannot measure this family here (e.g. no ANSWER population).
                results.append(
                    self._blocked(name, ids, "the bundle does not carry this family's inputs")
                )
                continue
            results.append(result_from(name, CONDITIONALLY_REQUIRED, ids, _failures(checks)))
        return results

    def safety_regression(self) -> ValidationResult:
        name, ids = "safety_regression", ["refusal_correctness", "p_unans_n"]
        check = self._v5("refusal_correctness")
        if check is None:
            return self._blocked(name, ids, "no refusal_correctness on P-UNANS in the bundle")
        minimum = self.parameters.p_unans_min_n
        if minimum is None:
            return self._blocked(
                name,
                ids,
                "the P-UNANS size that resolves refusal_correctness is undeclared (prereg_v8 draft)",
            )
        errors = _failures([check])
        n = finite_number(self.bundle.get("p_unans_n"))
        if n is None or n < minimum:
            errors.append(
                f"p_unans_n: {CODE_UNDER_POWERED}: P-UNANS n={self.bundle.get('p_unans_n')!r} "
                f"is below the {minimum} that resolves the floor"
            )
        return result_from(name, CONDITIONALLY_REQUIRED, ids, errors)

    # -- 11-14: sentinels, truncation, provenance, margins --------------------------------------

    def sentinels(self) -> ValidationResult:
        ran = self.bundle.get("sentinels_ran")
        if not isinstance(ran, dict) or not ran:
            return self._blocked(
                "sentinels", list(REQUIRED_SENTINELS), "no record of which sentinels ran"
            )
        errors: list[str] = []
        for sentinel, modes in REQUIRED_SENTINELS.items():
            if sentinel not in ran:
                errors.append(f"{sentinel}: {CODE_PROVENANCE_INCOMPLETE}: did not run")
                continue
            recorded = set(ran.get(sentinel) or [])
            errors.extend(
                f"{sentinel}: {CODE_PROVENANCE_INCOMPLETE}: did not run in its {mode} mode"
                for mode in modes
                if mode not in recorded
            )
        return result_from("sentinels", REQUIRED_NONEMPTY, list(REQUIRED_SENTINELS), errors)

    def _truncation_table(self) -> dict[str, dict[str, Any]] | None:
        table = self.bundle.get("truncation")
        if not isinstance(table, dict) or not table:
            return None
        return {str(stage): arms if isinstance(arms, dict) else {} for stage, arms in table.items()}

    def truncation(self) -> ValidationResult:
        table = self._truncation_table()
        if table is None:
            return self._blocked(
                "truncation", ["truncation"], "no per-stage, per-arm truncation record"
            )
        errors: list[str] = []
        for stage, arms in table.items():
            if not arms:
                errors.append(
                    f"{stage}: {CODE_INVALID_COMPARISON}: no per-arm truncation rate reported"
                )
            for arm, rate in arms.items():
                value = finite_number(rate)
                if value is None or not 0.0 <= value <= 1.0:
                    errors.append(
                        f"{stage}: {CODE_INVALID_COMPARISON}: arm {arm} rate {rate!r} is not in [0, 1]"
                    )
        return result_from("truncation", REQUIRED_NONEMPTY, list(table), errors)

    def truncation_balance(self) -> ValidationResult:
        table = self._truncation_table()
        ratio, gap = self.parameters.truncation_max_ratio, self.parameters.truncation_min_gap
        if table is None:
            return self._blocked(
                "truncation_balance", ["truncation"], "no per-stage, per-arm truncation record"
            )
        if ratio is None or gap is None:
            return self._blocked(
                "truncation_balance",
                list(table),
                "the truncation imbalance factor is undeclared (prereg_v8 draft)",
            )
        errors: list[str] = []
        for stage, arms in table.items():
            rates = [
                value for value in (finite_number(r) for r in arms.values()) if value is not None
            ]
            if len(rates) < 2:
                continue
            low, high = min(rates), max(rates)
            imbalanced = not at_most(high - low, gap) and (
                low == 0.0 or not at_most(high / low, ratio)
            )
            if imbalanced:
                errors.append(
                    f"{stage}: {CODE_INVALID_COMPARISON}: truncation {low:.4f} vs {high:.4f} exceeds "
                    f"{ratio}x and {gap} absolute"
                )
        return result_from("truncation_balance", REQUIRED_NONEMPTY, list(table), errors)

    def provenance(self) -> ValidationResult:
        provenance = self.bundle.get("provenance")
        if not isinstance(provenance, dict) or "present" not in provenance:
            return self._blocked(
                "provenance", list(REQUIRED_PROVENANCE), "no provenance record in the bundle"
            )
        present = set(provenance.get("present") or [])
        errors = [
            f"{item}: {CODE_PROVENANCE_INCOMPLETE}: absent"
            for item in REQUIRED_PROVENANCE
            if item not in present
        ]
        return result_from("provenance", REQUIRED_NONEMPTY, list(REQUIRED_PROVENANCE), errors)

    def comparison_margins(self) -> ValidationResult:
        result = v12_resolvable_margin(list(self.bundle.get("comparisons") or []))
        # 08 "Blocking semantics": a WITHIN_NOISE row is not a failure but cannot count as a pass,
        # and is excluded from the verdict's support. With no supporting row there is no support.
        if (
            result.blocked_status is None
            and not result.errors
            and not result.detail.get("supporting")
        ):
            result.errors.append(
                f"comparisons: {CODE_WITHIN_NOISE}: no comparison clears its resolvable margin "
                f"({result.detail.get('within_noise', 0)} within noise, "
                f"{result.detail.get('under_powered', 0)} under-powered)"
            )
        return result

    # -- 15: the policy's own decision ----------------------------------------------------------

    def policy_decision(self) -> ValidationResult:
        verdict = self.verdict
        errors = []
        if verdict["decision"] != PROMOTE:
            codes = (
                ", ".join(sorted(set(verdict.get("failure_codes") or [])))
                or CODE_TOOL_POLICY_REGRESSION
            )
            errors.append(
                f"policy_decision: {verdict['decision']}: {verdict['policy_version']} failed "
                f"{verdict['failed_dimensions']} ({codes})"
            )
        return result_from("policy_decision", REQUIRED_NONEMPTY, ["policy_decision"], errors)

    # -- assembly -------------------------------------------------------------------------------

    def checks(self) -> list[ValidationResult]:
        return [
            self.mode_coverage(),
            self.census(),
            self.metric_denominators(),
            self.metric_values(),
            *self.behaviour(),
            self.sentinels(),
            self.truncation(),
            self.truncation_balance(),
            self.provenance(),
            self.comparison_margins(),
            self.policy_decision(),
        ]

    def report(self) -> VerificationReport:
        report = VerificationReport(contract=STUDY_002_GATE_CONTRACT)
        for result in self.checks():
            report.add(result)
        return report


def _valid_matrix(matrix: Any) -> bool:
    return isinstance(matrix, dict) and all(
        isinstance(row, dict)
        and all(
            finite_number(v) is not None and float(v) >= 0 and float(v) == int(v)
            for v in row.values()
        )
        for row in matrix.values()
    )


def _failures(checks: list[dict[str, Any] | None]) -> list[str]:
    return [
        f"{check['dimension']}: {check.get('code', CODE_TOOL_POLICY_REGRESSION)}: observed {check['observed']} "
        f"requires {check['requirement']}"
        for check in checks
        if check is not None and not check["passed"]
    ]


def study_002_gate(
    bundle: dict[str, Any], parameters: PreregParameters = ADOPTED_PARAMETERS
) -> VerificationReport:
    return Study002Gate(bundle, parameters).report()


# -- the vacuity self-test (16-GPU-READINESS-GATE.md check 13) -------------------------------------


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
        "baseline": {"call_f1": 0.62, "answer_rate": 0.98},
        "p_unans_n": 400,
        "sentinels_ran": {sentinel: list(modes) for sentinel, modes in REQUIRED_SENTINELS.items()},
        "provenance": {"present": list(REQUIRED_PROVENANCE)},
        "truncation": {
            "0-shot": {"C0": 0.02, "R1": 0.03},
            "8-shot": {"C0": 0.01, "R1": 0.01},
            "elicit": {"C0": 0.0, "R1": 0.0},
            "5-shot": {"C0": 0.06, "R1": 0.07},
        },
        "comparisons": [{"id": "R1_vs_C0", "n": 1277, "margin": 0.08, "delta": 0.12}],
    }


def _case(
    bundle: dict[str, Any],
    expected: str,
    code: str | None = None,
    parameters: PreregParameters = DRAFT_V8_PARAMETERS,
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
    but ``healthy_under_adopted_prereg`` runs with the (unadopted) draft parameters, so that a failure
    is attributable to the fixture rather than to the undeclared values.
    """
    cases: dict[str, dict[str, Any]] = {}
    cases["healthy_under_adopted_prereg"] = _case(
        healthy_bundle(), BLOCKED_INPUT_MISSING, parameters=ADOPTED_PARAMETERS
    )
    cases["healthy_with_draft_parameters"] = _case(healthy_bundle(), PASS)

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
    noise["comparisons"] = [{"id": "R1_vs_C0", "n": 1277, "margin": 0.01, "delta": 0.01}]
    cases["every_comparison_within_noise"] = _case(noise, FAIL, CODE_WITHIN_NOISE)

    unresolved = healthy_bundle()
    unresolved["comparisons"] = [{"id": "R1_vs_C0", "n": 1277, "margin": None, "delta": 0.12}]
    cases["comparison_without_a_margin"] = _case(unresolved, FAIL, CODE_UNRESOLVED_ROW)

    null_metric = healthy_bundle()
    null_metric["candidate"]["parse_valid_rate"] = None
    cases["null_metric"] = _case(null_metric, FAIL, CODE_MISSING_METRIC)

    cases["empty_bundle"] = _case({}, FAIL)
    return cases


def self_test_passed(cases: dict[str, dict[str, Any]]) -> bool:
    return all(
        case["overall"] == case["expected"] and case.get("code_found", True)
        for case in cases.values()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.self_test:
        parser.error("only --self-test is implemented")
    cases = self_test()
    print(json.dumps(cases, indent=2, sort_keys=True))
    return 0 if self_test_passed(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
