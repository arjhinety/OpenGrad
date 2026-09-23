"""The C1 pre-GPU provenance gate (21 phase 6), on synthetic artifacts and the real ones.

Every check gets a fixture that makes it fail, and the failure code is asserted -- a check with no
failing test is itself unverified (docs/research/study-002/15-PROVENANCE-VALIDATORS.md:74-92). The
committed-artifact tests assert the resolved state: the gate passes (the phase-6
``FAIL_CLASSIFIER_VERSION`` finding is fixed) and the counters stay consistent.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from opengrad.data import behaviour_labels, canonical_v3, canonical_v3_balance, versions
from opengrad.data.provenance_gate import (
    FAIL_AUTHORISATION,
    FAIL_CLASSIFIER_VERSION,
    FAIL_RENDERER_IDENTITY,
    FAIL_SELECTION_PLAN,
    gate,
)
from opengrad.data.renderers import Qwen35_2BRenderer
from opengrad.verification.accounting import BLOCKED_INPUT_MISSING, FAIL, PASS

ROOT = Path(__file__).resolve().parents[2]
AUTHORISATION = "docs/research/study-002/38-BALANCING-PERMISSION-AND-C1-AUTHORISATION.md"
TEMPLATE_HASH = "273d8e0e683b885071fb17e08d5f2a5ddfb5309756181681de4f5a1822d80"
SELECTED_SHA = "f8e31f15a6003cf78328ddca1f2965cc1c82764e917d5491d1c9830f254892ca"
SHARD = "shard-000000.parquet"
SOURCE_MANIFEST = Path("configs/releases/toolpolicy_canonical_v3_sources.yaml")


def _classifier() -> dict:
    return {
        "version": behaviour_labels.FROZEN_TAG,
        "tag": behaviour_labels.FROZEN_TAG,
        "source_sha256_lf": behaviour_labels.FROZEN_SOURCE_SHA256_LF,
    }


def _canonical_versions() -> dict:
    return {
        "adapter_version": versions.ADAPTER_VERSION,
        "schema_normalization_version": versions.SCHEMA_NORMALIZATION_VERSION,
        "behavior_taxonomy_version": versions.BEHAVIOR_TAXONOMY_VERSION,
        "canonical_schema_version": versions.CANONICAL_SCHEMA_VERSION,
        "supervision_contract_version": versions.SUPERVISION_CONTRACT_VERSION,
        # the classifier the labels actually came from, so the positive bundle agrees with itself
        "decision_classifier_version": versions.DECISION_CLASSIFIER_V2_VERSION,
    }


def _documents() -> dict[str, dict]:
    identity = {
        "renderer": Qwen35_2BRenderer.renderer_version,
        "model_revision": Qwen35_2BRenderer.model_revision,
        "template_hash": TEMPLATE_HASH,
    }
    return {
        "canonical-v3": {
            "artifact_kind": "CANONICAL_V3_DECISION_BALANCED",
            "authorisation": AUTHORISATION,
            "versions": _canonical_versions(),
            "labels": {"classifier": _classifier(), "input_contract": "prose-decision-input-v2"},
            "gates": {"renderability": {"identity": dict(identity)}},
            "selection_plan": {"selected_ids_sha256": SELECTED_SHA},
            "shards": [{"file": SHARD, "records": 1, "sha256": "0" * 64}],
        },
        "balance": {
            "artifact_kind": "CANONICAL_V3_DECISION_BALANCE",
            "authorisation": AUTHORISATION,
            "labels": {"classifier": _classifier(), "input_contract": "prose-decision-input-v2"},
            "selected_ids_sha256": SELECTED_SHA,
        },
        "labels": {
            "artifact_kind": "BEHAVIOUR_LABELS",
            "authorisation": AUTHORISATION,
            "classifier": _classifier(),
            "input_contract": "prose-decision-input-v2",
        },
        "heldout": {"model_renderer_contract": dict(identity)},
        "snapshot": {
            "renderer": Qwen35_2BRenderer.renderer_version,
            "model_revision": Qwen35_2BRenderer.model_revision,
            "chat_template_hash": TEMPLATE_HASH,
        },
        "source-manifest": {
            "versions": {
                "adapter_version": versions.ADAPTER_VERSION,
                "schema_normalization_version": versions.SCHEMA_NORMALIZATION_VERSION,
                "canonical_schema_version": versions.CANONICAL_SCHEMA_VERSION,
                "supervision_contract_version": versions.SUPERVISION_CONTRACT_VERSION,
            }
        },
    }


def _write(root: Path, path: Path, document: dict) -> None:
    destination = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in {".yaml", ".yml"}:
        destination.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    else:
        destination.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")


def _write_shard(root: Path, versions_block: dict) -> None:
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    from opengrad.data.normalization_v3 import storage_row

    directory = root / canonical_v3.OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    row = {
        "id": "og_1",
        "source": "glaive",
        "tools": [],
        "messages": [],
        "metadata": {
            "adapter_version": versions_block["adapter_version"],
            "schema_normalization_version": versions_block["schema_normalization_version"],
            "supervision": {"adapter_version": versions_block["adapter_version"]},
        },
    }
    pq.write_table(pa.Table.from_pylist([storage_row(row)]), directory / SHARD)


def _root(tmp_path: Path, mutate=None, *, shard: bool = True) -> Path:
    documents = _documents()
    if mutate is not None:
        mutate(documents)
    _write(tmp_path, canonical_v3.REPORT, documents["canonical-v3"])
    _write(tmp_path, canonical_v3_balance.REPORT, documents["balance"])
    _write(tmp_path, behaviour_labels.REPORT, documents["labels"])
    _write(tmp_path, Path("reports/evaluation/behavioral-heldout-v1.manifest.json"), documents["heldout"])
    _write(tmp_path, Path("tests/fixtures/rendered/qwen35_2b_metadata.json"), documents["snapshot"])
    _write(tmp_path, SOURCE_MANIFEST, documents["source-manifest"])
    if shard:
        _write_shard(tmp_path, documents["canonical-v3"]["versions"])
    return tmp_path


def _status(report, name: str) -> str:
    return next(result.status for result in report.results if result.name == name)


def _errors(report, name: str) -> list[str]:
    return next(result.all_errors() for result in report.results if result.name == name)


def test_a_consistent_bundle_passes(tmp_path: Path) -> None:
    report = gate(_root(tmp_path))
    assert report.overall == PASS, [result.all_errors() for result in report.results]


def test_a_wrong_adapter_version_is_rejected(tmp_path: Path) -> None:
    def mutate(documents):
        documents["canonical-v3"]["versions"]["adapter_version"] = "1.0.0"

    report = gate(_root(tmp_path, mutate))
    assert _status(report, "versions_authoritative") == FAIL
    assert any("PROV_VERSION_MISMATCH" in error for error in _errors(report, "versions_authoritative"))


def test_the_classifier_version_mismatch_is_flagged(tmp_path: Path) -> None:
    """The exact defect the gate exists to catch: the versions block names another classifier."""
    def mutate(documents):
        documents["canonical-v3"]["versions"]["decision_classifier_version"] = (
            "prose-decision-classifier-v1"
        )

    report = gate(_root(tmp_path, mutate))
    assert _status(report, "classifier_identity") == FAIL
    assert any(FAIL_CLASSIFIER_VERSION in error for error in _errors(report, "classifier_identity"))


def test_a_non_frozen_classifier_is_rejected(tmp_path: Path) -> None:
    def mutate(documents):
        documents["balance"]["labels"]["classifier"]["source_sha256_lf"] = "b" * 64

    report = gate(_root(tmp_path, mutate))
    assert _status(report, "classifier_identity") == FAIL


def test_a_wrong_renderer_template_is_rejected(tmp_path: Path) -> None:
    def mutate(documents):
        documents["canonical-v3"]["gates"]["renderability"]["identity"]["template_hash"] = "c" * 64

    report = gate(_root(tmp_path, mutate))
    assert _status(report, "renderer_identity") == FAIL
    assert any(FAIL_RENDERER_IDENTITY in error for error in _errors(report, "renderer_identity"))


def test_a_missing_authorisation_is_rejected(tmp_path: Path) -> None:
    def mutate(documents):
        del documents["balance"]["authorisation"]

    report = gate(_root(tmp_path, mutate))
    assert _status(report, "authorisation_recorded") == FAIL
    assert any(FAIL_AUTHORISATION in error for error in _errors(report, "authorisation_recorded"))


def test_a_falsified_selection_plan_is_rejected(tmp_path: Path) -> None:
    def mutate(documents):
        documents["canonical-v3"]["selection_plan"]["selected_ids_sha256"] = "d" * 64

    report = gate(_root(tmp_path, mutate))
    assert _status(report, "selection_plan_agreement") == FAIL
    assert any(FAIL_SELECTION_PLAN in error for error in _errors(report, "selection_plan_agreement"))


def test_absent_local_shards_block_rather_than_pass(tmp_path: Path) -> None:
    """A git-ignored artifact is a blocked check, never a pass (16: a blocked check is not a pass)."""
    report = gate(_root(tmp_path, shard=False))
    result = next(r for r in report.results if r.name == "record_version_agreement")
    assert result.status == BLOCKED_INPUT_MISSING
    assert result.passed == 0
    assert report.overall == BLOCKED_INPUT_MISSING


def test_every_result_keeps_its_counters_consistent(tmp_path: Path) -> None:
    report = gate(_root(tmp_path))
    _items = list(report.results)
    assert _items, "nothing to check: an empty collection would pass this test vacuously"
    for result in _items:
        assert result.accounting_errors() == [], result.render()


# ── the committed artifacts ──────────────────────────────────────────────────────────────────────


def test_the_committed_artifacts_pass_the_provenance_gate() -> None:
    """The phase-6 finding is fixed: canonical-v3's version block names the classifier it used.

    The synthetic negative above proves the check still catches a v1-versus-v2 mismatch; here the
    committed artifacts pass. record_version_agreement needs the git-ignored shards, so on a clean
    checkout it blocks (never fails) and the overall is BLOCKED_INPUT_MISSING rather than PASS.
    """
    report = gate(ROOT)
    assert report.overall != FAIL, [result.all_errors() for result in report.results]
    for name in (
        "versions_authoritative",
        "classifier_identity",
        "renderer_identity",
        "authorisation_recorded",
        "selection_plan_agreement",
    ):
        assert _status(report, name) == PASS, _errors(report, name)
    if _status(report, "record_version_agreement") == PASS:
        assert report.overall == PASS
    else:
        assert _status(report, "record_version_agreement") == BLOCKED_INPUT_MISSING
        assert report.overall == BLOCKED_INPUT_MISSING


def test_the_committed_artifacts_keep_their_counters_consistent() -> None:
    report = gate(ROOT)
    _items = list(report.results)
    assert _items, "nothing to check: an empty collection would pass this test vacuously"
    for result in _items:
        assert result.accounting_errors() == [], result.render()


def test_the_gate_names_the_contract_it_is() -> None:
    report = gate(ROOT)
    assert report.contract == 1
