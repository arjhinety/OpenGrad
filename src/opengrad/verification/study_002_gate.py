"""``study_002_gate_v1`` — the Study 002 evaluation gate (`docs/research/study-002/11-THRESHOLDS.md`).

This is the gate `16-GPU-READINESS-GATE.md` check 2 requires to exist and check 13 requires to be
shown able to fail. It is a machine-checkable verdict over an **evaluation bundle**, built on the
shared accounting contract (`opengrad.verification.accounting`), and it is deliberately able to say
no: a required input that is absent is ``BLOCKED_INPUT_MISSING`` (never a pass), and a required
population the bundle cannot measure is ``FAIL_NONVACUOUS``.

**Contract 5 (2026-10-13).** Contract 4 checked that provenance fields were *named*, and took each
comparison row's ``n`` on trust. Contract 5 implements, as written, what 15 V3 / V9 and 46 §10 (v14)
state (`reports/ERRATA.md` §37; no amendment): every number the verdict reads carries its V3
attachment, every fingerprint is read from the artifact it names (V9), each comparison row names a
population whose ``n`` is read from an artifact, and a between-arm difference on a population that
resolves more than 10 points (``CLARIFY``, ``ANSWER-natural``) is ``WITHIN_NOISE`` and supports nothing.

**Contract 4 (2026-10-13).** A gate audit found inputs through which a broken candidate still
passed. Contract 4 implements, as written, rules the adopted documents already state
(`reports/ERRATA.md` §35, on the §19 precedent; no amendment): truncation is reported for every
sentinel stage and every arm (40 A, 08 rule 2), so an omitted stage or a one-arm stage no longer
escapes the balance check; the baseline carries every metric the non-regression checks read, so none
is skipped (02 H5: an unmeasured metric falsifies); census counts are whole, non-negative numbers
(check 2); every rate lies in [0, 1] (07); and malformed input fails the check that reads it instead
of raising.

**Contract 3 (2026-09-24, ``study_002_prereg_v8``).** The gate wraps ``tool_use_promotion_v6`` instead
of v5, and runs with the values v8 declares (:data:`ADOPTED_PARAMETERS`). v6 returns
``NOT_EVALUABLE`` on any missing input rather than filling a default, so on an incomplete bundle the
behavioural checks 4-10 are ``BLOCKED`` and check 15 fails; a PASS still requires every input.

**Contract 2 (2026-09-24).** Contract 1 read seven of the ``tool_use_promotion_v5`` dimensions and
never its decision, so a candidate v5 rejected on its clarification or unsupported floors, its
answer-rate drop or a baseline regression still passed; it also let the bundle declare its own
required sentinels and provenance fields, accepted a census that executed nothing, and passed an
under-powered mode and unresolved comparison rows. Contract 2 closes each of those
(`reports/ERRATA.md` §19). A PASS under one contract is not a PASS under another.

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
12. ``truncation`` -- a truncation rate in [0, 1] for every sentinel stage and every arm -- ``INVALID_COMPARISON``
12b. ``truncation_balance`` -- imbalance within the declared factor -- ``INVALID_COMPARISON``
13. ``provenance`` -- every metric the verdict reads has a row carrying 15 V3's attachment, bound to a
    partition fingerprint the bundle read from an artifact (V9), and the ``partition`` and ``corpus``
    fingerprints are read from artifacts -- ``FAIL_UNATTACHED_METRIC`` / ``FAIL_FINGERPRINT`` /
    ``PROVENANCE_INCOMPLETE``
14. ``comparison_margins`` -- every row names a population whose ``n`` is read from an artifact, and
    prints that ``n`` and a margin; a row on a population that resolves more than 10 points is
    ``WITHIN_NOISE`` (46 §10); at least one row clears its resolvable margin, the rest are
    ``WITHIN_NOISE`` and support nothing -- ``FAIL_UNRESOLVED_ROW`` / ``FAIL_FINGERPRINT`` / ``WITHIN_NOISE``
15. ``policy_decision`` -- ``tool_use_promotion_v6`` returns ``PROMOTE`` -- the v6 failure codes

Two rules the preregistration stated but did not quantify until ``study_002_prereg_v8``
(`docs/research/study-002/40-PREREG-V8-DRAFT.md`, adopted 2026-09-24) are held as
:class:`PreregParameters`: the truncation "declared factor" (11:76) and the ``P-UNANS`` size that
resolves ``refusal_correctness`` (11:139). :data:`ADOPTED_PARAMETERS` carries v8's values. A
parameter set that leaves either ``None`` makes checks 4 and 12b ``BLOCKED_INPUT_MISSING``, never PASS.

Bundle keys (the gate's input contract; a key that is absent blocks its checks rather than passing
them):

* ``modes``: ``{mode: gold_count}`` -- the confirmatory partition's gold counts;
* ``census``: ``{discovered, checked, passed, failed, blocked, skipped}`` -- the execution census of
  the scored items (``failed`` is an item that did not score, not a wrong answer);
* ``candidate`` / ``baseline``: the promotion-policy metric dicts (see
  :mod:`opengrad.promotion.tool_use_policy`), the candidate carrying a ``confusion_matrix``;
* ``p_unans_n``: the number of scored ``P-UNANS`` items;
* ``sentinels_ran``: ``{sentinel_id: [mode, ...]}``;
* ``metric_rows``: ``[{metric, metric_version, arm, seed, partition_id, partition_fingerprint, n,
  denominator, evaluator_version, artifact_sha256}]`` -- 15 V3's attachment, at least one row per
  metric the verdict reads;
* ``fingerprints``: ``[{name, value, artifact, field?}]`` -- 15 V9 entries, ``partition`` and ``corpus``
  among them (``field`` is a dotted JSON path; without it the artifact's sha256 is the value);
* ``populations``: ``{name: {n, artifact, field}}`` -- each population's size, read from an artifact;
* ``truncation``: ``{stage: {arm: rate}}``;
* ``comparisons``: ``[{id, population, n, margin, delta}]``.

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
from pathlib import Path
from typing import Any

from opengrad.promotion.tool_use_policy import (
    CODE_NONVACUOUS,
    CODE_TOOL_POLICY_REGRESSION,
    PROMOTE,
    PromotionPolicyV6,
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
from opengrad.verification.provenance_validators import (
    CODE_FINGERPRINT,
    CODE_UNATTACHED_METRIC,
    v3_metric_attachment,
    v9_fingerprint_from_artifact,
)
from opengrad.verification.resolvability import at_most, resolvable_margin

#: The gate's own contract. Bump when the check set, its discovery rules or its non-vacuity
#: requirements change, so a PASS under one contract is never read as a PASS under another.
STUDY_002_GATE_CONTRACT = 5

#: The repository the gate reads fingerprinted artifacts from (15 V9).
REPO_ROOT = Path(__file__).resolve().parents[3]
GATE_VERSION = "study_002_gate_v1"

CODE_ACCOUNTING = "FAIL_ACCOUNTING"
CODE_PROVENANCE_INCOMPLETE = "PROVENANCE_INCOMPLETE"
CODE_INVALID_COMPARISON = "INVALID_COMPARISON"
CODE_WITHIN_NOISE = "WITHIN_NOISE"
CODE_MISSING_METRIC = "FAIL_MISSING_METRIC"
CODE_INVALID_METRIC = "FAIL_INVALID_METRIC"

#: The behavioural checks 4-10, and the policy dimension each reads.
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

#: Every metric the wrapped policy reads. v5 filled an absent one with a default and v6 refuses to
#: evaluate; either way the gate reports each absent or non-finite one by name (check 3b).
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
#: The baseline carries every metric the policy compares it against: ``call_f1`` (retention),
#: ``answer_rate`` (the drop) and the four non-regression metrics. The policy skips a regression check
#: whose baseline value is absent (UP-0014), so contract 4 requires them rather than letting one go
#: unmeasured (02 H5).
BASELINE_METRICS = (
    "call_f1",
    "answer_rate",
    "call_precision",
    "call_recall",
    "clarification_accuracy",
    "unsupported_accuracy",
)

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

#: The truncation stages: "the prompt mode a sentinel runs in" (40-PREREG-V8-DRAFT.md item A), so every
#: mode in the sentinel inventory. Each must report a rate for every arm (08 rule 2); a stage left out,
#: or reported for one arm only, cannot be checked for imbalance.
REQUIRED_STAGES: tuple[str, ...] = tuple(
    sorted({mode for modes in REQUIRED_SENTINELS.values() for mode in modes})
)

#: The fingerprints every bundle reads from an artifact (15 V9: "corpus and partition fingerprints").
REQUIRED_FINGERPRINTS = ("partition", "corpus")

#: 46 §10 (``study_002_prereg_v14``): "The margin a population is used to test is 10 points, for every
#: comparison between arms." A population that resolves more than that supports no comparison.
TESTED_MARGIN = 0.10

#: The populations 46 §10 declares descriptive: they resolve more than 10 points (``CLARIFY`` 10.18pp at
#: n = 371, ``ANSWER-natural`` 11.63pp at n = 284), and "A between-arm difference on those two
#: populations is ``WITHIN_NOISE`` and supports nothing." Named as well as derived from ``n``, so a row
#: cannot escape the rule through its population's size.
DESCRIPTIVE_POPULATIONS = frozenset({"CLARIFY", "ANSWER-natural"})


@dataclass(frozen=True)
class PreregParameters:
    """Values the preregistration requires but has not yet declared.

    ``None`` means undeclared, and the check that needs it is ``BLOCKED_INPUT_MISSING``.

    * ``truncation_max_ratio`` / ``truncation_min_gap``: within a stage, two arms' truncation rates
      are imbalanced when they differ by more than ``truncation_min_gap`` (absolute) **and** the
      larger exceeds ``truncation_max_ratio`` times the smaller (11:76, "the declared factor").
    * ``p_unans_min_n``: the ``P-UNANS`` size that resolves ``refusal_correctness`` (11:139).
    """

    truncation_max_ratio: float | None = None
    truncation_min_gap: float | None = None
    p_unans_min_n: int | None = None


#: What the adopted preregistration declares: ``study_002_prereg_v8`` items A and B (40-PREREG-V8-DRAFT.md,
#: adopted 2026-09-24). Under v7 both were undeclared and the gate blocked checks 4 and 12b.
ADOPTED_PARAMETERS = PreregParameters(
    truncation_max_ratio=2.0, truncation_min_gap=0.02, p_unans_min_n=385
)

#: No value declared -- the state under ``study_002_prereg_v7``. Kept so the self-test still shows an
#: undeclared rule blocking rather than passing.
UNDECLARED_PARAMETERS = PreregParameters()


class Study002Gate:
    """The checks over one evaluation bundle."""

    def __init__(
        self,
        bundle: dict[str, Any],
        parameters: PreregParameters = ADOPTED_PARAMETERS,
        root: Path = REPO_ROOT,
    ) -> None:
        # A bundle that is not a mapping is an empty bundle: every input-bearing check blocks, the
        # policy's missing metrics fail closed, and nothing raises.
        self.bundle = bundle if isinstance(bundle, dict) else {}
        self.parameters = parameters
        self.root = root
        self._verdict: dict[str, Any] | None = None

    # -- inputs ---------------------------------------------------------------------------------

    @property
    def raw_candidate(self) -> dict[str, Any]:
        candidate = self.bundle.get("candidate")
        return dict(candidate) if isinstance(candidate, dict) else {}

    @property
    def raw_baseline(self) -> dict[str, Any]:
        baseline = self.bundle.get("baseline")
        return dict(baseline) if isinstance(baseline, dict) else {}

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
        raw = self.raw_baseline
        return {key: value for key, value in raw.items() if finite_number(value) is not None}

    @property
    def verdict(self) -> dict[str, Any]:
        if self._verdict is None:
            self._verdict = PromotionPolicyV6().evaluate(self.candidate, self.baseline)
        return self._verdict

    def _policy(self, dimension: str) -> dict[str, Any] | None:
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
        modes = self.bundle.get("modes")
        return dict(modes) if isinstance(modes, dict) else {}

    # -- 1-3b: the L1 family --------------------------------------------------------------------

    def mode_coverage(self) -> ValidationResult:
        return v1_mode_coverage(self._modes(), enforce_floor=True)

    def census(self) -> ValidationResult:
        census = self.bundle.get("census")
        if not isinstance(census, dict):
            return self._blocked("census", ["census"], "no execution census in the bundle")
        keys = ("discovered", "checked", "passed", "failed", "blocked", "skipped")
        counts = {key: finite_number(census.get(key)) for key in keys}
        # A count is a whole, non-negative number; int() would silently truncate 0.6 failed items to 0.
        problems = [
            f"{key} is not a count ({census.get(key)!r})"
            for key, value in counts.items()
            if value is None or value < 0 or value != int(value)
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
        # A malformed matrix is reported by check 3b; reading its denominators would raise instead.
        candidate = self.raw_candidate
        if "confusion_matrix" in candidate and not _valid_matrix(candidate["confusion_matrix"]):
            candidate = {k: v for k, v in candidate.items() if k != "confusion_matrix"}
        return v2_metric_denominator(candidate)

    def metric_values(self) -> ValidationResult:
        candidate, baseline = self.raw_candidate, self.raw_baseline
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
        # Every metric the policy reads is a rate: a ratio of counts (07), so it lies in [0, 1]. A value
        # outside that range is not a measurement, whatever threshold it would clear.
        errors += [
            f"{side}.{metric}: {CODE_INVALID_METRIC}: {value} is not a rate in [0, 1]"
            for side, metrics, names in (
                ("candidate", candidate, CANDIDATE_METRICS),
                ("baseline", baseline, BASELINE_METRICS),
            )
            for metric in names
            if (value := finite_number(metrics.get(metric))) is not None and not 0.0 <= value <= 1.0
        ]
        if "confusion_matrix" in candidate and not _valid_matrix(candidate["confusion_matrix"]):
            ids.append("candidate.confusion_matrix")
            errors.append(
                f"candidate.confusion_matrix: {CODE_MISSING_METRIC}: not a {{truth: {{predicted: count}}}} table"
            )
        return result_from("metric_values", REQUIRED_NONEMPTY, ids, errors)

    # -- 4-10: the behavioural family (from the policy verdict) ---------------------------------

    def behaviour(self) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        for name, dimensions in BEHAVIOUR_CHECKS:
            if name == "safety_regression":
                results.append(self.safety_regression())
                continue
            ids = list(dimensions)
            checks = [self._policy(dimension) for dimension in dimensions]
            if all(check is None for check in checks):
                # The gate cannot measure this family here (no ANSWER population, or v6 refused
                # to evaluate an incomplete input and returned only its input.* dimensions).
                results.append(
                    self._blocked(name, ids, "the bundle does not carry this family's inputs")
                )
                continue
            results.append(result_from(name, CONDITIONALLY_REQUIRED, ids, _failures(checks)))
        return results

    def safety_regression(self) -> ValidationResult:
        name, ids = "safety_regression", ["refusal_correctness", "p_unans_n"]
        check = self._policy("refusal_correctness")
        if check is None:
            return self._blocked(name, ids, "no refusal_correctness on P-UNANS in the bundle")
        minimum = self.parameters.p_unans_min_n
        if minimum is None:
            return self._blocked(
                name,
                ids,
                "the P-UNANS size that resolves refusal_correctness is undeclared",
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
            modes_ran = ran.get(sentinel)
            recorded = (
                {m for m in modes_ran if isinstance(m, str)}
                if isinstance(modes_ran, (list, tuple))
                else set()
            )
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
        errors: list[str] = [
            f"{stage}: {CODE_INVALID_COMPARISON}: no truncation rates reported for this sentinel stage"
            for stage in REQUIRED_STAGES
            if stage not in table
        ]
        # Every stage reports the same arms, and at least two, so each stage can be compared (40 A).
        all_arms = {str(arm) for arms in table.values() for arm in arms}
        if len(all_arms) < 2:
            errors.append(
                f"truncation: {CODE_INVALID_COMPARISON}: rates for {len(all_arms)} arm(s); a balance "
                "check needs at least two"
            )
        for stage, arms in table.items():
            if not arms:
                errors.append(
                    f"{stage}: {CODE_INVALID_COMPARISON}: no per-arm truncation rate reported"
                )
            missing = sorted(all_arms - {str(arm) for arm in arms})
            if arms and missing:
                errors.append(
                    f"{stage}: {CODE_INVALID_COMPARISON}: no rate for arm(s) {', '.join(missing)}"
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
                "the truncation imbalance factor is undeclared",
            )
        errors: list[str] = []
        for stage, arms in table.items():
            rates = [
                value for value in (finite_number(r) for r in arms.values()) if value is not None
            ]
            if len(rates) < 2:
                # A one-arm stage cannot be shown balanced; check 12 names the missing arm.
                errors.append(
                    f"{stage}: {CODE_INVALID_COMPARISON}: {len(rates)} usable rate(s); balance not checkable"
                )
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

    def _rows(self, key: str) -> list[dict[str, Any]]:
        """A list input, with every non-mapping entry read as an empty row (which then fails)."""
        rows = self.bundle.get(key)
        if not isinstance(rows, (list, tuple)):
            return []
        return [row if isinstance(row, dict) else {} for row in rows]

    def _read_back(self, entries: list[dict[str, Any]]) -> tuple[set[str], list[str]]:
        """(names, errors) of the V9 entries whose value equals what their artifact holds."""
        if not entries:
            return set(), []
        result = v9_fingerprint_from_artifact(entries, self.root)
        failed = {error.split(":", 1)[0] for error in result.errors}
        names = {
            str(entry.get("name") or f"fingerprint-{index}") for index, entry in enumerate(entries)
        }
        return names - failed, list(result.errors)

    def provenance(self) -> ValidationResult:
        rows = self._rows("metric_rows")
        entries = self._rows("fingerprints")
        if not rows or not entries:
            return self._blocked(
                "provenance",
                ["metric_rows", "fingerprints"],
                "no metric rows or no fingerprints in the bundle",
            )
        errors = list(v3_metric_attachment(rows).errors)
        verified, read_errors = self._read_back(entries)
        errors.extend(read_errors)
        errors.extend(
            f"{name}: {CODE_PROVENANCE_INCOMPLETE}: no {name} fingerprint read from an artifact"
            for name in REQUIRED_FINGERPRINTS
            if name not in verified
        )
        values = {
            str(entry.get("value"))
            for index, entry in enumerate(entries)
            if str(entry.get("name") or f"fingerprint-{index}") in verified
        }
        for index, row in enumerate(rows):
            fingerprint = row.get("partition_fingerprint")
            if fingerprint is not None and str(fingerprint) not in values:
                errors.append(
                    f"{row.get('id') or f'row-{index}'}: {CODE_FINGERPRINT}: its partition "
                    "fingerprint is not one the bundle read from an artifact"
                )
        # The gate's own list: every metric the verdict reads is attached, not only the rows the
        # bundle chose to send.
        sent = {str(row.get("metric")) for row in rows}
        errors.extend(
            f"{metric}: {CODE_UNATTACHED_METRIC}: no metric row"
            for metric in sorted(set(CANDIDATE_METRICS) | set(BASELINE_METRICS))
            if metric not in sent
        )
        return result_from("provenance", REQUIRED_NONEMPTY, ["metric_rows", "fingerprints"], errors)

    def _populations(self) -> tuple[dict[str, int], list[str]]:
        """Each population's ``n``, kept only when its artifact holds that ``n`` (V9)."""
        declared = self.bundle.get("populations")
        if not isinstance(declared, dict):
            return {}, []
        entries, sizes = [], {}
        for name, spec in declared.items():
            spec = spec if isinstance(spec, dict) else {}
            count = finite_number(spec.get("n"))
            if count is None or count <= 0 or count != int(count):
                continue
            sizes[str(name)] = int(count)
            entries.append(
                {
                    "name": str(name),
                    "value": str(int(count)),
                    "artifact": spec.get("artifact"),
                    "field": spec.get("field"),
                }
            )
        verified, errors = self._read_back(entries)
        return {name: n for name, n in sizes.items() if name in verified}, errors

    def comparison_margins(self) -> ValidationResult:
        rows = self._rows("comparisons")
        bound, errors = self._populations()
        tested, descriptive = [], []
        for index, row in enumerate(rows):
            row_id = str(row.get("id") or f"row-{index}")
            population = row.get("population")
            n = finite_number(row.get("n"))
            if not isinstance(population, str) or population not in bound:
                errors.append(
                    f"{row_id}: {CODE_UNRESOLVED_ROW}: names no population whose n is read from "
                    f"an artifact ({population!r})"
                )
            elif n is None or n != bound[population]:
                errors.append(
                    f"{row_id}: {CODE_UNRESOLVED_ROW}: prints n={row.get('n')!r}, but "
                    f"{population} holds {bound[population]}"
                )
            elif (
                population in DESCRIPTIVE_POPULATIONS
                or resolvable_margin(bound[population]) > TESTED_MARGIN
            ):
                descriptive.append(row_id)
            else:
                tested.append(row)
        # A row v12 has not seen yet (a margin it cannot read) still fails there; a descriptive row
        # is counted with the WITHIN_NOISE rows and supports nothing.
        result = v12_resolvable_margin(tested if tested else rows if not descriptive else [])
        if descriptive and not tested:
            result = result_from(
                "comparison_margins",
                REQUIRED_NONEMPTY,
                descriptive,
                [],
                detail={"within_noise": 0, "under_powered": 0, "supporting": 0},
            )
        result.errors[:0] = errors
        result.detail["descriptive"] = len(descriptive)
        result.detail["within_noise"] = result.detail.get("within_noise", 0) + len(descriptive)
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
    bundle: dict[str, Any],
    parameters: PreregParameters = ADOPTED_PARAMETERS,
    root: Path = REPO_ROOT,
) -> VerificationReport:
    return Study002Gate(bundle, parameters, root).report()


#: Committed artifacts the healthy fixture binds to. Both are checked out with LF endings
#: (`.gitattributes`), so the fingerprints below hold on every platform.
PCONF_PARTITION = "reports/study-002/pconf-v1/pconf-v1.partition.json"
PCONF_PARTITION_SHA256 = "caacacb582de85c0daeac037ffea7801897045ec31a2ff5dddc9f5e3bb617c8c"
FIXTURE_CORPUS_MANIFEST = "reports/canonical-v3/canonical-v3.manifest.json"
FIXTURE_CORPUS_FINGERPRINT = "60d3123e1c6cef67f904a81f29669dcff44ecc5aeb0e71178a3b5d4e94bfa75a"


def _metric_row(metric: str, arm: str) -> dict[str, Any]:
    return {
        "id": f"{arm}:{metric}",
        "metric": metric,
        "metric_version": "07-v1",
        "arm": arm,
        "seed": 0,
        "partition_id": "P-CONF-v1",
        "partition_fingerprint": PCONF_PARTITION_SHA256,
        "n": 1677,
        "denominator": 1677,
        "evaluator_version": "fixture",
        "artifact_sha256": "0" * 64,
    }


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
            {"name": "partition", "value": PCONF_PARTITION_SHA256, "artifact": PCONF_PARTITION},
            {
                "name": "corpus",
                "value": FIXTURE_CORPUS_FINGERPRINT,
                "artifact": FIXTURE_CORPUS_MANIFEST,
                "field": "corpus_fingerprint",
            },
        ],
        "populations": {
            "P-CONF": {
                "n": 1277,
                "artifact": PCONF_PARTITION,
                "field": "components.confirmatory_when2call.count",
            },
            "CLARIFY": {"n": 371, "artifact": PCONF_PARTITION, "field": "resolvability.CLARIFY.n"},
            "ANSWER-natural": {
                "n": 284,
                "artifact": PCONF_PARTITION,
                "field": "resolvability.ANSWER-natural.n",
            },
        },
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
