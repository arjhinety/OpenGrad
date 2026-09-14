"""Verification gates and their execution accounting.

See ``accounting`` for why a gate has to prove it ran, not only that it passed.
"""

from opengrad.verification.accounting import (
    ACCOUNTING_PREFIX,
    BLOCKED_NETWORK,
    BLOCKED_OPTIONAL_DEPENDENCY,
    BLOCKED_STATUSES,
    CONDITIONALLY_REQUIRED,
    FAIL,
    NONVACUOUS_PREFIX,
    OPTIONAL,
    PASS,
    REQUIRED_NONEMPTY,
    Skip,
    ValidationResult,
    VerificationReport,
)

# The verifier contract version. Bump when the set of checks, their discovery
# rules, or their non-vacuity requirements changes in a way that makes a PASS
# mean something different from before.
#
# v1: publication verification as first implemented. It did not verify that its
#     own checks had executed, so `opengrad.registry.validate` could report PASS
#     without running, and freeze validation could skip every canonical corpus.
# v2: adds executable module validation, correct canonical freeze discovery,
#     non-vacuity enforcement, explicit skipped/blocked accounting, immutable
#     provenance enforcement and reconstructed-event validation.
#
# A PASS reported under v1 does not mean what a PASS under v2 means, and no
# historical record is rewritten to imply otherwise.
VERIFIER_CONTRACT = 2

__all__ = [
    "ACCOUNTING_PREFIX",
    "BLOCKED_NETWORK",
    "BLOCKED_OPTIONAL_DEPENDENCY",
    "BLOCKED_STATUSES",
    "CONDITIONALLY_REQUIRED",
    "FAIL",
    "NONVACUOUS_PREFIX",
    "OPTIONAL",
    "PASS",
    "REQUIRED_NONEMPTY",
    "VERIFIER_CONTRACT",
    "Skip",
    "ValidationResult",
    "VerificationReport",
]
