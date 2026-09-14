"""The publication gate's status handling.

The property that matters most here is negative: a check that could not run must
never be reported as a check that passed. "We could not verify" and "we verified
and it is fine" are different claims, and collapsing them is how a provenance
defect ships.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opengrad.publication import (
    BLOCKED_NETWORK,
    BLOCKED_OPTIONAL_DEPENDENCY,
    FAIL,
    PASS,
    VERIFIER_CONTRACT,
    ValidationResult,
    VerificationReport,
    verify_publication,
)
from opengrad.verification import OPTIONAL

REPO_ROOT = Path(__file__).parents[2]


def _gate(
    name: str = "gate",
    *,
    status: str = PASS,
    blocked_reasons: list[str] | None = None,
    errors: list[str] | None = None,
    discovered: int = 1,
    checked: int = 1,
    passed: int = 1,
    failed: int = 0,
    blocked: int = 0,
) -> ValidationResult:
    """A result whose counters add up, so only the status logic is under test."""
    return ValidationResult(
        name=name,
        policy=OPTIONAL,
        discovered=discovered,
        checked=checked,
        passed=passed,
        failed=failed,
        blocked=blocked,
        blocked_status=status if status in (BLOCKED_NETWORK, BLOCKED_OPTIONAL_DEPENDENCY) else None,
        errors=errors or [],
        blocked_reasons=blocked_reasons or [],
    )


def _minimal_root(tmp_path: Path, license_block: dict, revision: str | None = None) -> Path:
    import yaml

    record = {
        "id": "fixture",
        "display_name": "Fixture",
        "organization": "Fixture Org",
        "category": "tool_use",
        "allowed_splits": ["train"],
        "forbidden_splits": [],
        "contamination_status": "UNASSESSED",
        "retrieval_date": "2026-01-01",
        "license": license_block,
    }
    if revision is not None:
        record["source_revision"] = {
            "value": revision,
            "verified": True,
            "source": "Hugging Face dataset snapshot API",
        }
        record["hf_id"] = "owner/dataset"
    (tmp_path / "registry").mkdir(parents=True, exist_ok=True)
    (tmp_path / "registry/datasets.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "status": "TEST", "datasets": [record]}),
        encoding="utf-8",
    )
    (tmp_path / "registry/models.yaml").write_text(
        yaml.safe_dump({"schema_version": 2, "models": []}), encoding="utf-8"
    )
    return tmp_path


def _clean_root(tmp_path: Path) -> Path:
    """A root whose only outstanding check is the network one."""
    return _minimal_root(
        tmp_path,
        {
            "value": "Apache-2.0",
            "verified": True,
            "source": "https://huggingface.co/datasets/owner/dataset",
        },
        revision="a" * 40,
    )


# ── Status composition ───────────────────────────────────────────────────────


def test_overall_status_composition():
    assert VerificationReport(contract=VERIFIER_CONTRACT, results=[_gate()]).overall == PASS
    assert (
        VerificationReport(
            contract=VERIFIER_CONTRACT, results=[_gate(), _gate("b", status=FAIL, failed=1)]
        ).overall
        == FAIL
    )
    assert (
        VerificationReport(
            contract=VERIFIER_CONTRACT,
            results=[_gate(), _gate("b", status=BLOCKED_NETWORK, checked=0, passed=0, blocked=1)],
        ).overall
        == BLOCKED_NETWORK
    )
    # A failure always dominates a blocked check.
    assert (
        VerificationReport(
            contract=VERIFIER_CONTRACT,
            results=[
                _gate("a", status=BLOCKED_NETWORK, checked=0, passed=0, blocked=1),
                _gate("b", status=FAIL, failed=1, errors=["boom"]),
            ],
        ).overall
        == FAIL
    )
    assert (
        VerificationReport(
            contract=VERIFIER_CONTRACT,
            results=[
                _gate("a", status=BLOCKED_OPTIONAL_DEPENDENCY, checked=0, passed=0, blocked=1)
            ],
        ).overall
        == BLOCKED_OPTIONAL_DEPENDENCY
    )


def test_nothing_checked_is_not_a_pass():
    assert VerificationReport(contract=VERIFIER_CONTRACT, results=[]).overall != PASS


# ── Expected BLOCKED ─────────────────────────────────────────────────────────


def test_offline_verification_is_blocked_never_passed(tmp_path: Path):
    """On a root with nothing else outstanding, offline is BLOCKED, not PASS."""
    report = verify_publication(_clean_root(tmp_path), network=False)
    assert report.overall == BLOCKED_NETWORK, report.render()
    assert report.overall != PASS


def test_blocked_check_is_not_rendered_as_success():
    report = VerificationReport(
        contract=VERIFIER_CONTRACT,
        results=[
            _gate(
                "remote immutable pins",
                status=BLOCKED_NETWORK,
                blocked_reasons=["unreachable"],
                checked=0,
                passed=0,
                blocked=1,
            )
        ],
    )
    rendered = report.render()
    assert "[BLOCKED_NETWORK]" in rendered
    assert "overall: PASS" not in rendered


def test_offline_mode_reports_the_number_of_unresolved_pins():
    report = verify_publication(REPO_ROOT, network=False)
    remote = next(result for result in report.results if result.name == "remote immutable pins")
    assert remote.status == BLOCKED_NETWORK
    # The registry does pin external sources; the population must not collapse to
    # zero just because the network is unavailable.
    assert remote.blocked == remote.discovered
    assert remote.detail["required"] == remote.discovered
    assert remote.detail["attempted"] == 0


# ── Expected FAIL ────────────────────────────────────────────────────────────


def test_gate_fails_when_a_referenced_artifact_is_missing(tmp_path: Path):
    _minimal_root(
        tmp_path,
        {
            "value": "Apache-2.0",
            "verified": True,
            "source": "reports/does-not-exist.md",
            "source_sha256": "a" * 64,
        },
    )
    report = verify_publication(tmp_path, network=False)
    assert report.overall == FAIL, report.render()


def test_gate_fails_on_a_mutable_branch_pin(tmp_path: Path):
    _minimal_root(
        tmp_path,
        {
            "value": "Apache-2.0",
            "verified": True,
            "source": "https://huggingface.co/owner/model/blob/main/LICENSE",
        },
    )
    report = verify_publication(tmp_path, network=False)
    assert report.overall == FAIL, report.render()


def test_gate_fails_on_an_unknown_legacy_marker(tmp_path: Path):
    _minimal_root(
        tmp_path,
        {"value": "Apache-2.0", "verified": True, "source": "https://example.org/x"},
    )
    releases = tmp_path / "reports/releases"
    releases.mkdir(parents=True)
    (releases / "r.json").write_text(
        json.dumps(
            {"dataset": {"repository": "owner/repo", "provenance_version": "legacy_made_up_v1"}}
        ),
        encoding="utf-8",
    )
    report = verify_publication(tmp_path, network=False)
    assert report.overall == FAIL, report.render()


# ── The real repository ──────────────────────────────────────────────────────

# The repository has no open content failure. Offline it is BLOCKED, because two
# required populations cannot be resolved without the network. It must never be
# reported as PASS in that state.


def test_repository_has_no_offline_failure_and_is_blocked_not_passed():
    report = verify_publication(REPO_ROOT, network=False)
    failures = [
        error for result in report.results if result.status == FAIL for error in result.all_errors()
    ]
    assert failures == [], report.render()
    # Nothing failed, but two populations could not be resolved: not a pass.
    assert report.overall == BLOCKED_NETWORK
    blocked = [r for r in report.results if r.status == BLOCKED_NETWORK]
    assert {r.name for r in blocked} == {"remote freeze identities", "remote immutable pins"}


@pytest.mark.network
def test_network_resolution_verifies_rather_than_merely_unblocking():
    """Connectivity must turn blocked populations into resolved ones, not SKIPs."""
    report = verify_publication(REPO_ROOT)
    by_name = {result.name: result for result in report.results}
    for name in ("remote freeze identities", "remote immutable pins"):
        result = by_name[name]
        assert result.status == PASS, result.all_errors()
        assert result.discovered > 0
        assert result.checked == result.discovered
        assert result.detail["resolved" if name.endswith("pins") else "reproduced"] == (
            result.discovered
        )
    assert report.overall == PASS, report.render()


def test_a_clean_root_passes_every_offline_check(tmp_path: Path):
    """The gates are not simply always-red: a sound root passes them."""
    report = verify_publication(_clean_root(tmp_path), network=False)
    assert [r for r in report.results if r.status == FAIL] == [], report.render()
    assert report.overall == BLOCKED_NETWORK


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_cli_exit_codes(tmp_path: Path, capsys, monkeypatch):
    """Exit 2 means blocked, exit 1 means failed, and the repository is failing."""
    from scripts.verify_publication import main

    monkeypatch.chdir(tmp_path)
    _clean_root(tmp_path)
    assert main(["--offline"]) == 2
    assert "BLOCKED_NETWORK" in capsys.readouterr().out


def test_cli_exits_nonzero_and_distinguishes_failure_from_blockage(tmp_path: Path, capsys):
    """Exit 1 and exit 2 mean different things, and both are non-zero.

    This repository is BLOCKED offline, not failing, so the failure path is
    exercised on a synthetic root whose licence evidence is missing.
    """
    from scripts.verify_publication import main

    assert main(["--offline"]) == 2
    out = capsys.readouterr().out
    assert "overall: BLOCKED_NETWORK" in out

    broken = tmp_path / "broken"
    _minimal_root(
        broken,
        {
            "value": "Apache-2.0",
            "verified": True,
            "source": "reports/does-not-exist.md",
            "source_sha256": "a" * 64,
        },
    )
    assert main(["--root", str(broken), "--offline"]) == 1
    out = capsys.readouterr().out
    assert "overall: FAIL" in out
