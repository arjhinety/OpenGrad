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
        pv.v7_sentinel_completeness({"S-REF": ("0-shot",)}, {}, []),
        pv.v8_census_reconciliation({}),
        pv.v9_fingerprint_from_artifact([], ROOT),
        pv.v10_claim_evidence([], ROOT, []),
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
        pv.v7_sentinel_completeness(
            required, {"C0": {"S-ANS-0": ["0-shot"], "S-NV": []}}, ["C0"]
        ).status
        == PASS
    )
    wrong_mode = {"C0": {"S-ANS-0": ["8-shot"], "S-NV": []}}
    assert pv.CODE_MISSING_SENTINEL in _codes(
        pv.v7_sentinel_completeness(required, wrong_mode, ["C0"])
    )


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
    assert pv.v10_claim_evidence([claim], tmp_path, ["r"]).status == PASS
    assert pv.CODE_UNSUPPORTED_CLAIM in _codes(
        pv.v10_claim_evidence([claim | {"evidence_sha256": "0" * 64}], tmp_path, ["r"])
    )
    assert pv.CODE_UNSUPPORTED_CLAIM in _codes(
        pv.v10_claim_evidence([claim | {"number_row": None}], tmp_path, ["r"])
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


def test_v4_fails_a_record_that_names_no_modes() -> None:
    assert pv.CODE_VOCABULARY in _codes(
        pv.v4_vocabulary_lock([{"id": "x", "scores": {"DIRECT": 1}}])
    )


def test_v5_reads_versioned_partitions_and_checks_what_a_replacement_replaces() -> None:
    base = {"arm": "C0", "seed": 0, "protocol": "0-shot"}
    twice = [
        base | {"run_id": "a", "partition": "P-UNANS-v2"},
        base | {"run_id": "b", "partition": "P-UNANS-v2"},
    ]
    assert pv.CODE_ONE_SHOT in _codes(pv.v5_one_shot(twice))
    self_replacing = [base | {"run_id": "a", "partition": "P-CONF-v1", "replacement_of": "a"}]
    assert pv.CODE_ONE_SHOT in _codes(pv.v5_one_shot(self_replacing))
    other_arm = [
        base | {"run_id": "a", "partition": "P-CONF-v1"},
        base | {"run_id": "b", "partition": "P-CONF-v1", "arm": "R1", "replacement_of": "a"},
    ]
    assert pv.CODE_ONE_SHOT in _codes(pv.v5_one_shot(other_arm))
    assert pv.v5_one_shot([base | {"run_id": "a", "partition": "P-DEVELOPMENT"}]).status != PASS


def test_v7_fails_a_required_arm_with_no_artifacts() -> None:
    result = pv.v7_sentinel_completeness({"S-NV": ()}, {"C0": {"S-NV": []}}, ["C0", "R1"])
    assert "R1: " + pv.CODE_MISSING_SENTINEL in _codes(result)


def test_v7_reads_its_modes_from_the_registry() -> None:
    from opengrad.verification.study_002_gate import REQUIRED_SENTINELS

    assert pv.load_sentinel_registry(ROOT) == REQUIRED_SENTINELS


def test_v10_fails_a_claim_whose_number_row_does_not_exist(tmp_path: Path) -> None:
    (tmp_path / "e.json").write_bytes(b"{}")
    claim = {
        "id": "c",
        "number_row": "ghost",
        "evidence_path": "e.json",
        "evidence_sha256": hashlib.sha256(b"{}").hexdigest(),
    }
    assert pv.CODE_UNSUPPORTED_CLAIM in _codes(pv.v10_claim_evidence([claim], tmp_path, ["r"]))


def test_v11_recognises_study_names_and_fails_unknown_ones() -> None:
    spelled = {
        "id": "t",
        "rows": [{"study": "Study 001", "value": 0.5}, {"study": 2, "value": 0.6}],
    }
    assert pv.CODE_INCOMPARABLE in _codes(pv.v11_cross_study_comparability([spelled]))
    unknown = {"id": "u", "rows": [{"study": "S1"}]}
    assert pv.CODE_INCOMPARABLE in _codes(pv.v11_cross_study_comparability([unknown]))


def test_v5_scores_each_p_unans_stratum_separately() -> None:
    base = {"arm": "C0", "seed": 0, "protocol": "0-shot"}
    strata = [
        base | {"run_id": "a", "partition": "P-UNANS-v2"},
        base | {"run_id": "b", "partition": "P-UNANS-v2-constructed"},
    ]
    assert pv.v5_one_shot(strata).status == PASS


def test_v5_refuses_mutual_and_double_replacements() -> None:
    base = {"arm": "C0", "seed": 0, "protocol": "0-shot", "partition": "P-CONF-v1"}
    mutual = [
        base | {"run_id": "a", "replacement_of": "b"},
        base | {"run_id": "b", "replacement_of": "a"},
    ]
    assert pv.CODE_ONE_SHOT in _codes(pv.v5_one_shot(mutual))
    double = [
        base | {"run_id": "a"},
        base | {"run_id": "b", "replacement_of": "a"},
        base | {"run_id": "c", "replacement_of": "a"},
    ]
    assert pv.CODE_ONE_SHOT in _codes(pv.v5_one_shot(double))


def test_v5_refuses_a_ring_of_replacements() -> None:
    base = {"arm": "C0", "seed": 0, "protocol": "0-shot", "partition": "P-CONF-v1"}
    ring = [
        base | {"run_id": "a", "replacement_of": "c"},
        base | {"run_id": "b", "replacement_of": "a"},
        base | {"run_id": "c", "replacement_of": "b"},
    ]
    assert pv.CODE_ONE_SHOT in _codes(pv.v5_one_shot(ring))
    chain = [
        base | {"run_id": "a"},
        base | {"run_id": "b", "replacement_of": "a"},
        base | {"run_id": "c", "replacement_of": "b"},
    ]
    assert pv.v5_one_shot(chain).status == PASS
