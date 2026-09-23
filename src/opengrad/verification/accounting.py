"""Execution accounting for verification gates.

A gate that returns success without doing any work is indistinguishable from one
that did the work and found nothing wrong. Two checks in this repository failed
exactly that way: the freeze check skipped every canonical corpus because it
looked for a manifest in a field those records do not use, and
``python -m opengrad.registry.validate`` imported its module, ran nothing, and
exited 0.

So a gate here must prove two separate things, and is not allowed to conflate
them:

    A verification gate must prove both that its assertions passed and that the
    expected assertions actually executed.

Every result therefore carries an execution census alongside its errors, and the
census is checked for internal consistency before the status is computed. A
result whose counters do not add up is itself a failure, because counters that
disagree are a symptom of the same class of bug.

Population policy
-----------------

Not every gate has a mandatory input. ``discovered == 0`` means different things
for a corpus the repository certainly contains and for an optional cross-check,
so each gate declares what it is:

``REQUIRED_NONEMPTY``
    The repository contains the inputs by construction. Discovering none is a
    hard failure, because it means discovery is broken.
``CONDITIONALLY_REQUIRED``
    Required only while a stated precondition holds, which the caller passes in.
    If the precondition holds and nothing is discovered, that is a failure.
``OPTIONAL``
    May legitimately have no inputs. Discovering none is recorded as a skip with
    a reason rather than silently counted as success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PASS = "PASS"
FAIL = "FAIL"
BLOCKED_NETWORK = "BLOCKED_NETWORK"
BLOCKED_OPTIONAL_DEPENDENCY = "BLOCKED_OPTIONAL_DEPENDENCY"
# A required input is absent from this checkout (e.g. git-ignored data a clean CI clone does not have).
BLOCKED_INPUT_MISSING = "BLOCKED_INPUT_MISSING"

BLOCKED_STATUSES = (BLOCKED_NETWORK, BLOCKED_OPTIONAL_DEPENDENCY, BLOCKED_INPUT_MISSING)

REQUIRED_NONEMPTY = "REQUIRED_NONEMPTY"
CONDITIONALLY_REQUIRED = "CONDITIONALLY_REQUIRED"
OPTIONAL = "OPTIONAL"

# Raised as an error string prefix so a vacuous pass is greppable and cannot be
# mistaken for a content failure.
NONVACUOUS_PREFIX = "FAIL_NONVACUOUS"
ACCOUNTING_PREFIX = "FAIL_ACCOUNTING"


@dataclass
class Skip:
    """A discovered candidate that was deliberately not evaluated."""

    id: str
    reason: str

    def render(self) -> str:
        return f"{self.id} ({self.reason})"


@dataclass
class ValidationResult:
    """What a gate discovered, what it evaluated, and what it concluded."""

    name: str
    policy: str = OPTIONAL
    discovered: int = 0
    checked: int = 0
    passed: int = 0
    failed: int = 0
    blocked: int = 0
    skipped: list[Skip] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    # Why candidates could not be evaluated. Kept apart from `errors` on purpose:
    # a blocked candidate is not a failed candidate, and a gate that reported FAIL
    # merely because it could not reach the network would be lying in the other
    # direction.
    blocked_reasons: list[str] = field(default_factory=list)
    # Gate-specific breakdown that does not fit the census, e.g. reproduced /
    # corroborated / unresolved for the freeze gate.
    detail: dict[str, Any] = field(default_factory=dict)
    # Set when the gate cannot run at all, e.g. an optional dependency is absent.
    blocked_status: str | None = None
    precondition: str | None = None

    # ── Census consistency ──────────────────────────────────────────────────

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    @property
    def attempted(self) -> int:
        return self.checked + self.blocked

    def accounting_errors(self) -> list[str]:
        """Counters that do not add up are a defect in the gate, not the data."""
        problems: list[str] = []
        if self.discovered != self.attempted + self.skipped_count:
            problems.append(
                f"{ACCOUNTING_PREFIX}: {self.name}: discovered {self.discovered} != "
                f"checked {self.checked} + blocked {self.blocked} + skipped {self.skipped_count}"
            )
        if self.checked != self.passed + self.failed:
            problems.append(
                f"{ACCOUNTING_PREFIX}: {self.name}: checked {self.checked} != "
                f"passed {self.passed} + failed {self.failed}"
            )
        if self.skipped_count and any(not skip.reason.strip() for skip in self.skipped):
            problems.append(f"{ACCOUNTING_PREFIX}: {self.name}: a skip carries no reason")
        return problems

    def vacuity_errors(self) -> list[str]:
        """A required population that discovered nothing proves nothing."""
        required = self.policy == REQUIRED_NONEMPTY or (
            self.policy == CONDITIONALLY_REQUIRED and self.precondition is not None
        )
        if not required:
            return []
        if self.discovered == 0:
            trigger = f" ({self.precondition})" if self.precondition else ""
            return [f"{NONVACUOUS_PREFIX}: {self.name}: expected candidates, discovered 0{trigger}"]
        if self.checked == 0 and self.blocked == 0:
            return [
                (
                    f"{NONVACUOUS_PREFIX}: {self.name}: discovered {self.discovered} "
                    f"candidates and evaluated none"
                )
            ]
        return []

    def all_errors(self) -> list[str]:
        return [*self.accounting_errors(), *self.vacuity_errors(), *self.errors]

    # ── Status ──────────────────────────────────────────────────────────────

    @property
    def status(self) -> str:
        if self.all_errors() or self.failed > 0:
            return FAIL
        if self.blocked_status is not None:
            return self.blocked_status
        if self.blocked > 0:
            return BLOCKED_NETWORK
        return PASS

    def counts(self) -> dict[str, int]:
        return {
            "discovered": self.discovered,
            "checked": self.checked,
            "passed": self.passed,
            "failed": self.failed,
            "blocked": self.blocked,
            "skipped": self.skipped_count,
        }

    def render(self) -> str:
        lines = [f"[{self.status}] {self.name}  {self.counts()}"]
        for key, value in sorted(self.detail.items()):
            lines.append(f"    {key}: {value}")
        lines.extend(f"    skipped: {skip.render()}" for skip in self.skipped)
        lines.extend(f"    blocked: {reason}" for reason in self.blocked_reasons)
        lines.extend(f"    {error}" for error in self.all_errors())
        return "\n".join(lines)


@dataclass
class VerificationReport:
    """An ordered set of gate results and the overall conclusion."""

    contract: int
    results: list[ValidationResult] = field(default_factory=list)

    @property
    def overall(self) -> str:
        statuses = {result.status for result in self.results}
        if FAIL in statuses:
            return FAIL
        for blocked in BLOCKED_STATUSES:
            if blocked in statuses:
                return blocked
        return PASS if self.results else FAIL

    def add(self, result: ValidationResult) -> ValidationResult:
        self.results.append(result)
        return result

    def totals(self) -> dict[str, int]:
        totals = dict.fromkeys(
            ("discovered", "checked", "passed", "failed", "blocked", "skipped"), 0
        )
        for result in self.results:
            for key, value in result.counts().items():
                totals[key] += value
        return totals

    def render(self) -> str:
        width = max((len(result.name) for result in self.results), default=10)
        header = (
            f"{'Gate':<{width}}  {'Disc':>5} {'Chk':>5} {'Pass':>5} "
            f"{'Fail':>5} {'Blk':>5} {'Skip':>5}"
        )
        rows = [header, "-" * len(header)]
        for result in self.results:
            counts = result.counts()
            rows.append(
                f"{result.name:<{width}}  {counts['discovered']:>5} {counts['checked']:>5} "
                f"{counts['passed']:>5} {counts['failed']:>5} {counts['blocked']:>5} "
                f"{counts['skipped']:>5}"
            )
        body = "\n".join(result.render() for result in self.results)
        return (
            f"{body}\n\nEXECUTION COVERAGE (contract v{self.contract})\n{chr(10).join(rows)}"
            f"\n\noverall: {self.overall}"
        )
