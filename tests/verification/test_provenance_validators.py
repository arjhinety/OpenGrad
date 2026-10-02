"""V3-V11 (15-PROVENANCE-VALIDATORS.md): each passes a well-formed input and fails its fixture with its code."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from opengrad.verification import provenance_validators as pv
from opengrad.verification.accounting import BLOCKED_INPUT_MISSING, FAIL, PASS

ROOT = Path(__file__).parents[2]


def _codes(result) -> str:
    return " ".join(result.all_errors())


def test_every_validator_fails_its_15_fixture_with_its_code() -> None:
    results = pv.self_test(ROOT)
    assert set(results) == {"V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11"}
    for validator, outcome in results.items():
        assert outcome["status"] == FAIL and outcome["matched"], validator


@pytest.mark.parametrize(
    "result",
    [
        pv.v3_metric_attachment([]),
        pv.v4_vocabulary_lock([]),
        pv.v5_one_shot([{"partition": "P-DEV"}]),
        pv.v6_row_key([]),
        pv.v7_sentinel_completeness({"S-REF": ("0-shot",)}, {}),
        pv.v8_census_reconciliation({}),
        pv.v9_fingerprint_from_artifact([], ROOT),
        pv.v10_claim_evidence([], ROOT),
        pv.v11_cross_study_comparability([]),
    ],
)
def test_an_absent_input_is_blocked_never_a_pass(result) -> None:
    assert result.status == BLOCKED_INPUT_MISSING


def test_v3_passes_a_full_attachment_and_rejects_a_fake_sha_or_count() -> None:
    assert pv.v3_metric_attachment([pv._ROW]).status == PASS
    assert pv.CODE_UNATTACHED_METRIC in _codes(
        pv.v3_metric_attachment([pv._ROW | {"artifact_sha256": "TO_BE_RECORDED"}])
    )
    assert pv.CODE_UNATTACHED_METRIC in _codes(pv.v3_metric_attachment([pv._ROW | {"n": 0}]))


def test_v4_accepts_canonical_names_and_declared_aliases_only() -> None:
    canonical = {"id": "a", "confusion_matrix": {"ANSWER": {"ANSWER": 3, "CALL": 1}}}
    assert pv.v4_vocabulary_lock([canonical]).status == PASS
    declared = {
        "id": "b",
        "per_mode": {"request_for_info": 0.5},
        "mode_aliases": {"request_for_info": "CLARIFY"},
    }
    assert pv.v4_vocabulary_lock([declared]).status == PASS
    wrong_map = {"id": "c", "per_mode": {"REFUSE": 0.1}, "mode_aliases": {"REFUSE": "ANSWER"}}
    assert pv.CODE_VOCABULARY in _codes(pv.v4_vocabulary_lock([wrong_map]))
    unknown = {"id": "d", "gold_counts": {"DIRECT": 3}}
    assert pv.CODE_VOCABULARY in _codes(pv.v4_vocabulary_lock([unknown]))


def test_v5_allows_other_seeds_protocols_and_named_replacements() -> None:
    base = {"arm": "C0", "seed": 0, "partition": "P-CONF", "protocol": "0-shot"}
    ok = [
        base | {"run_id": "a"},
        base | {"run_id": "b", "seed": 1},
        base | {"run_id": "c", "protocol": "elicit"},
        base | {"run_id": "d", "partition": "P-UNANS"},
        base | {"run_id": "e", "replacement_of": "a"},
    ]
    assert pv.v5_one_shot(ok).status == PASS
    dangling = [base | {"run_id": "a"}, base | {"run_id": "b", "replacement_of": "zzz"}]
    assert pv.CODE_ONE_SHOT in _codes(pv.v5_one_shot(dangling))


def test_v6_and_v7_pass_complete_inputs() -> None:
    row = {
        "arm": "C0",
        "partition": "P-CONF",
        "protocol": "0-shot",
        "device_class": "a100-80gb",
        "provider": "nvidia",
    }
    assert pv.v6_row_key([row]).status == PASS
    required = {"S-ANS-0": ("0-shot",), "S-NV": ()}
    assert (
        pv.v7_sentinel_completeness(required, {"C0": {"S-ANS-0": ["0-shot"], "S-NV": []}}).status
        == PASS
    )
    wrong_mode = {"C0": {"S-ANS-0": ["8-shot"], "S-NV": []}}
    assert pv.CODE_MISSING_SENTINEL in _codes(pv.v7_sentinel_completeness(required, wrong_mode))


def test_v8_passes_a_census_that_adds_up() -> None:
    census = {"discovered": 10, "checked": 8, "passed": 7, "failed": 1, "blocked": 1, "skipped": 1}
    assert pv.v8_census_reconciliation(census).status == PASS
    assert pv.ACCOUNTING_PREFIX in _codes(pv.v8_census_reconciliation(census | {"passed": 8}))


def test_v9_reads_bytes_or_a_json_field(tmp_path: Path) -> None:
    (tmp_path / "corpus.json").write_text('{"identity": {"hash": "abc"}}', encoding="utf-8")
    digest = hashlib.sha256((tmp_path / "corpus.json").read_bytes()).hexdigest()
    good = [
        {"name": "bytes", "value": digest, "artifact": "corpus.json"},
        {"name": "field", "value": "abc", "artifact": "corpus.json", "field": "identity.hash"},
    ]
    assert pv.v9_fingerprint_from_artifact(good, tmp_path).status == PASS
    for bad in (
        {"name": "x", "value": "abd", "artifact": "corpus.json", "field": "identity.hash"},
        {"name": "x", "value": "abc", "artifact": "corpus.json", "field": "identity.missing"},
        {"name": "x", "value": "abc", "artifact": "../outside.json"},
    ):
        assert pv.CODE_FINGERPRINT in _codes(pv.v9_fingerprint_from_artifact([bad], tmp_path)), bad


def test_v10_passes_a_resolving_claim(tmp_path: Path) -> None:
    (tmp_path / "e.json").write_bytes(b"{}")
    claim = {
        "id": "c",
        "number_row": "r",
        "evidence_path": "e.json",
        "evidence_sha256": hashlib.sha256(b"{}").hexdigest(),
    }
    assert pv.v10_claim_evidence([claim], tmp_path).status == PASS
    assert pv.CODE_UNSUPPORTED_CLAIM in _codes(
        pv.v10_claim_evidence([claim | {"evidence_sha256": "0" * 64}], tmp_path)
    )
    assert pv.CODE_UNSUPPORTED_CLAIM in _codes(
        pv.v10_claim_evidence([claim | {"number_row": None}], tmp_path)
    )


def test_v11_accepts_single_study_tables_and_fully_keyed_mixed_ones() -> None:
    single = {"id": "s", "rows": [{"study": "002"}, {"study": "002"}]}
    keyed = {
        "id": "k",
        "rows": [
            {"study": "001", "evaluator_version": "v1", "partition": "conf-1277"},
            {"study": "002", "evaluator_version": "v2", "partition": "P-CONF-v1"},
        ],
    }
    assert pv.v11_cross_study_comparability([single, keyed]).status == PASS
