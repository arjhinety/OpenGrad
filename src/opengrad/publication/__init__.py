"""Publication verification: the gate that runs before a release is declared complete."""

from opengrad.publication.verify import (
    BLOCKED_NETWORK,
    BLOCKED_OPTIONAL_DEPENDENCY,
    FAIL,
    PASS,
    VERIFIER_CONTRACT,
    ValidationResult,
    VerificationReport,
    verify_publication,
)

__all__ = [
    "BLOCKED_NETWORK",
    "BLOCKED_OPTIONAL_DEPENDENCY",
    "FAIL",
    "PASS",
    "VERIFIER_CONTRACT",
    "ValidationResult",
    "VerificationReport",
    "verify_publication",
]
