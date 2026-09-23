"""Regression tests for vacuous success.

Two checks in this repository reported success without doing anything, and both
were found by accident rather than by a test:

1. freeze validation read the artifact path from `processed_dataset_hash.source`
   and `findings.release_manifest`. The canonical corpora name their manifest in
   `source_repository`, so every one of them was skipped and the gate passed.
2. `python -m opengrad.registry.validate` had no `__main__` guard, so it imported
   the module, ran nothing and exited 0 -- indistinguishable from a passing
   validation.

These tests exist so that neither can come back, and so that the third case --
counters that do not add up -- cannot itself be read as a pass.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from opengrad.registry.validate import (
    audit,
    check_freeze,
    validate,
)
from opengrad.verification import (
    ACCOUNTING_PREFIX,
    BLOCKED_NETWORK,
    FAIL,
    NONVACUOUS_PREFIX,
    OPTIONAL,
    PASS,
    REQUIRED_NONEMPTY,
    Skip,
    ValidationResult,
    VerificationReport,
)

REPO_ROOT = Path(__file__).parents[2]
REVISION = "a" * 40
DIGEST = "b" * 64


def _write_registry(root: Path, records: list[dict], models: list[dict] | None = None) -> None:
    (root / "registry").mkdir(parents=True, exist_ok=True)
    (root / "registry/datasets.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "status": "TEST", "datasets": records}),
        encoding="utf-8",
    )
    (root / "registry/models.yaml").write_text(
        yaml.safe_dump({"schema_version": 2, "models": models or []}), encoding="utf-8"
    )


def _record(**overrides) -> dict:
    record = {
        "id": "corpus",
        "display_name": "Corpus",
        "organization": "Org",
        "category": "tool_use",
        "allowed_splits": ["train"],
        "forbidden_splits": [],
        "contamination_status": "UNASSESSED",
        "retrieval_date": "2026-01-01",
        "license": {"value": "Apache-2.0", "verified": True, "source": "https://x/y"},
        "source_revision": {"value": REVISION, "verified": True, "source": "Hugging Face API"},
    }
    record.update(overrides)
    return record


def _manifest_only_reachable_via_source_repository(digest: str = DIGEST, **overrides) -> dict:
    """The schema shape the canonical corpora actually use.

    The manifest is named in `source_repository`, and `processed_dataset_hash`
    carries only a value and a status -- exactly the arrangement the old
    discovery missed.
    """
    return _record(
        processed_dataset_hash={"status": "FULL_DATA_VALIDATED", "value": digest},
        source_repository="release/manifest.json",
        **overrides,
    )


# ── Freeze discovery regression ──────────────────────────────────────────────


def test_a_manifest_reachable_only_via_source_repository_is_discovered(tmp_path: Path):
    """The exact shape that made the freeze gate vacuous must be discovered."""
    (tmp_path / "release").mkdir()
    (tmp_path / "release/manifest.json").write_bytes(b"{}\n")
    _write_registry(tmp_path, [_manifest_only_reachable_via_source_repository()])

    result = check_freeze(tmp_path)
    assert result.discovered == 1, result.render()
    assert result.checked == 1, result.render()
    # It was examined and found wrong (the digest does not match the manifest),
    # which is itself proof the check ran.
    assert result.failed == 1, result.render()
    assert result.status == FAIL


def test_a_discovered_manifest_that_reproduces_its_digest_passes(tmp_path: Path):
    (tmp_path / "release").mkdir()
    manifest = tmp_path / "release/manifest.json"
    manifest.write_bytes(b"{}\n")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    _write_registry(tmp_path, [_manifest_only_reachable_via_source_repository(digest)])
    result = check_freeze(tmp_path)
    assert result.status == PASS, result.render()
    assert result.detail["reproduced_from_committed_artifact"] == 1


def test_a_required_freeze_population_cannot_pass_with_zero_checks(tmp_path: Path):
    """A repository that declares corpora must discover them.

    The requirement binds because canonical corpora are declared, so a discovery
    result of zero means discovery is broken rather than that there was nothing
    to find.
    """
    _write_registry(tmp_path, [_record(id="canonical_fixture")])
    result = check_freeze(tmp_path)
    assert result.discovered == 0
    assert result.status == FAIL
    assert any(NONVACUOUS_PREFIX in error for error in result.all_errors()), result.render()


def test_the_real_repository_discovers_every_canonical_freeze():
    """Every corpus with a validated identity must be found, not skipped."""
    result = check_freeze(REPO_ROOT)
    assert result.discovered >= 4, result.render()
    assert result.discovered == result.checked + result.blocked + result.skipped_count
    assert result.detail["reproduced_from_committed_artifact"] >= 2
    # The remote-anchored corpus is deferred and named, not silently dropped.
    assert result.detail["deferred_to_remote_identity"] == 1
    assert result.detail["unresolved"] == 0
    assert result.failed == 0, result.render()


def test_freeze_validation_is_no_longer_vacuous_on_the_real_repository():
    """The gate must actually examine the corpora it reports PASS/FAIL for."""
    report = audit(REPO_ROOT)
    freeze = report["freeze"]
    assert freeze.checked > 0, freeze.render()
    assert freeze.discovered == freeze.checked + freeze.blocked + freeze.skipped_count


# ── Required-nonempty policy ─────────────────────────────────────────────────


def test_a_required_population_with_zero_discovery_fails():
    result = ValidationResult(name="x", policy=REQUIRED_NONEMPTY, discovered=0)
    assert result.status == FAIL
    assert any(NONVACUOUS_PREFIX in error for error in result.all_errors())


def test_an_optional_population_with_zero_discovery_is_allowed_if_reasoned():
    # The skip is the discovered item: a gate cannot skip something it never saw.
    result = ValidationResult(
        name="x",
        policy=OPTIONAL,
        discovered=1,
        skipped=[Skip(id="<none>", reason="nothing to check")],
    )
    assert result.status == PASS
    assert result.checked == 0


def test_a_skip_cannot_be_recorded_without_discovering_it():
    """discovered=0 with a skip is an accounting contradiction, not a pass."""
    result = ValidationResult(
        name="x", policy=OPTIONAL, discovered=0, skipped=[Skip(id="a", reason="why not")]
    )
    assert result.status == FAIL
    assert any(ACCOUNTING_PREFIX in error for error in result.all_errors())


def test_conditional_requirement_only_binds_when_the_precondition_holds():
    unbound = ValidationResult(name="x", policy="CONDITIONALLY_REQUIRED", discovered=0)
    assert unbound.status == PASS
    bound = ValidationResult(
        name="x",
        policy="CONDITIONALLY_REQUIRED",
        discovered=0,
        precondition="inputs present",
    )
    assert bound.status == FAIL
    assert any(NONVACUOUS_PREFIX in error for error in bound.all_errors())


def test_discovering_candidates_but_evaluating_none_fails():
    result = ValidationResult(
        name="x",
        policy=REQUIRED_NONEMPTY,
        discovered=3,
        checked=0,
        skipped=[Skip(id="a", reason="r"), Skip(id="b", reason="r"), Skip(id="c", reason="r")],
    )
    assert result.status == FAIL
    assert any(NONVACUOUS_PREFIX in error for error in result.all_errors())


# ── Accounting regression ────────────────────────────────────────────────────


def test_inconsistent_counters_cannot_be_a_pass():
    """discovered=3 checked=0 passed=0 failed=0 must not be representable as PASS."""
    result = ValidationResult(
        name="x",
        policy=REQUIRED_NONEMPTY,
        discovered=3,
        checked=0,
        passed=0,
        failed=0,
        blocked=0,
        skipped=[],
    )
    assert result.status == FAIL
    assert any(ACCOUNTING_PREFIX in error for error in result.all_errors())


def test_checked_must_equal_passed_plus_failed():
    result = ValidationResult(
        name="x",
        policy=OPTIONAL,
        discovered=3,
        checked=3,
        passed=1,
        failed=0,
        blocked=0,
    )
    assert any(ACCOUNTING_PREFIX in error for error in result.accounting_errors())


def test_a_skip_without_a_reason_is_an_accounting_error():
    result = ValidationResult(
        name="x",
        policy=OPTIONAL,
        discovered=1,
        checked=0,
        blocked=0,
        skipped=[Skip(id="a", reason="  ")],
    )
    assert any(ACCOUNTING_PREFIX in error for error in result.accounting_errors())


def test_inconsistent_counters_cannot_reach_a_pass_through_a_report():
    bad = ValidationResult(name="x", policy=REQUIRED_NONEMPTY, discovered=2, checked=0)
    report = VerificationReport(contract=2, results=[bad])
    assert report.overall == FAIL


def test_every_real_gate_keeps_its_counters_consistent():
    _items = list(audit(REPO_ROOT).items())
    assert _items, "nothing to check: an empty collection would pass this test vacuously"
    for name, result in _items:
        assert result.accounting_errors() == [], f"{name}: {result.render()}"


def test_a_blocked_population_does_not_become_empty():
    result = ValidationResult(
        name="x",
        policy=REQUIRED_NONEMPTY,
        discovered=8,
        checked=0,
        blocked=8,
        blocked_status=BLOCKED_NETWORK,
    )
    assert result.status == BLOCKED_NETWORK
    assert result.discovered == 8


# ── Module invocation regression ─────────────────────────────────────────────


def _run_module(root: Path) -> subprocess.CompletedProcess[str]:
    """Run the module the way an operator would, as a real process."""
    return subprocess.run(
        [sys.executable, "-m", "opengrad.registry.validate"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env={
            **_env(),
            "PYTHONPATH": str(REPO_ROOT / "src"),
        },
    )


def _env() -> dict:
    import os

    return dict(os.environ)


def _minimal_valid_root(root: Path) -> Path:
    _write_registry(root, [_record()])
    return root


def _minimal_invalid_root(root: Path) -> Path:
    _write_registry(root, [_record(id="dup"), _record(id="dup")])
    return root


def test_module_invocation_agrees_with_the_in_process_validator(tmp_path: Path):
    """The regression is `import succeeds -> exit 0 -> read as validation PASS`.

    So the guard is agreement: whenever the in-process validator finds errors, the
    subprocess must exit non-zero and print them. A synthetic root always has
    errors (it lacks the schemas and contracts the repository carries), which makes
    it the right fixture for the non-zero side; the clean case is exercised against
    the real repository by the test above.
    """
    for name, root in (
        ("invalid", _minimal_invalid_root(tmp_path / "invalid")),
        ("bare", _bare_root(tmp_path / "bare")),
    ):
        errors = validate(root)
        result = _run_module(root)
        assert errors, f"{name}: fixture unexpectedly validates clean"
        assert result.returncode != 0, f"{name}: {result.stdout}"
        assert result.stdout.strip(), f"{name}: the subprocess printed nothing"


def test_module_invocation_without_a_main_guard_would_have_looked_like_a_pass(tmp_path: Path):
    """Importing the module must not be sufficient to produce a success signal."""
    import importlib

    module = importlib.import_module("opengrad.registry.validate")
    root = _minimal_invalid_root(tmp_path)
    # Import alone says nothing about the repository; only calling the validator does.
    assert module.validate(root) != []
    assert bool(module.validate(root)) is (_run_module(root).returncode != 0)


def _bare_root(root: Path) -> Path:
    (root / "registry").mkdir(parents=True, exist_ok=True)
    return root


def test_module_invocation_exits_nonzero_on_invalid_state(tmp_path: Path):
    result = _run_module(_minimal_invalid_root(tmp_path))
    assert result.returncode != 0, result.stdout
    assert "duplicate dataset id" in result.stdout


def test_module_invocation_performs_the_same_validation_as_the_function(tmp_path: Path):
    """An import that exits 0 without validating is the bug being guarded."""
    root = _minimal_invalid_root(tmp_path)
    result = _run_module(root)
    # The in-process validation and the subprocess must agree on the verdict.
    assert bool(validate(root)) is (result.returncode != 0)
    assert bool(validate(root)) is True


def test_module_invocation_on_the_real_repository_is_honest():
    """The module invocation must report the real verdict, not merely exit 0.

    It previously exited 0 without validating at all. Now it runs: for this
    repository that means a clean registry, and the exit code must agree with the
    in-process validation rather than being pinned to either answer.
    """
    result = _run_module(REPO_ROOT)
    errors = validate(REPO_ROOT)
    assert (result.returncode == 0) is (errors == []), result.stdout
    assert ("registry validation: OK" in result.stdout) is (errors == [])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
