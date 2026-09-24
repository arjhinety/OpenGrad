"""registry/datasets.yaml schema version 2: the record schema is enforced and every rule bites.

The schema (registry/dataset_record.schema.json) replaced a hand-written list of required field
names on 2026-09-24. These tests pin three things: the committed registry is at version 2 and
conforms; each rule the schema and the reference checks add rejects a record that breaks it; and
the training firewall, which reads this file, allowlists exactly what it did before the migration.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest
import yaml

from opengrad.data import materialize
from opengrad.registry.validators import validate_references, validate_structure

ROOT = Path(__file__).parents[2]


def _registry() -> dict:
    return yaml.safe_load((ROOT / "registry/datasets.yaml").read_text(encoding="utf-8"))


def _by_id(registry: dict) -> dict[str, dict]:
    return {record["id"]: record for record in registry["datasets"]}


def test_committed_registry_is_version_2_and_conforms() -> None:
    registry = _registry()
    assert registry["schema_version"] == 2
    assert registry["record_schema"] == "registry/dataset_record.schema.json"
    # 11 at the v2 migration; +3 evaluation-only sources for the ANSWER strata (study_002_prereg_v9).
    assert len(registry["datasets"]) == 14
    assert validate_structure(ROOT) == []
    assert validate_references(ROOT) == []


def test_every_schema_property_names_its_croissant_counterpart() -> None:
    schema = json.loads((ROOT / "registry/dataset_record.schema.json").read_text(encoding="utf-8"))
    missing = [name for name, spec in schema["properties"].items() if "x-croissant" not in spec]
    assert missing == []


def test_the_training_firewall_allowlist_is_unchanged_by_the_migration() -> None:
    # Captured from the schema-version-1 registry before the migration; a change here changes which
    # sources and splits may enter SFT, which is never a side effect of a metadata change.
    assert {k: sorted(v) for k, v in materialize._training_split_allowlist().items()} == {
        "button": ["train"],
        "canonical_v1": ["train"],
        "canonical_v2": ["train"],
        "canonical_v2_final": ["train"],
        "glaive-function-calling-v2": ["train"],
        "looptool-23k": ["train"],
        "toolace": ["train"],
        "when2call": ["train"],
        "xlam-function-calling-60k": ["train"],
    }


def test_no_duplicate_identity_fields_survive() -> None:
    retired = {"checksum", "exact_revision", "original_sample_count", "retained_sample_count"}
    for record in _registry()["datasets"]:
        assert retired.isdisjoint(record), record["id"]


def test_upstream_distribution_digests_are_pinned_to_the_record_revision() -> None:
    for record in _registry()["datasets"]:
        if record["role"] != "UPSTREAM_SOURCE":
            continue
        assert record["distribution"], record["id"]
        for item in record["distribution"]:
            assert record["source_revision"]["value"] in item["content_url"], record["id"]
    # xLAM's former `checksum` is its one file's Hub LFS digest, not a lost value.
    xlam = _by_id(_registry())["xlam-function-calling-60k"]["distribution"]
    assert [item["sha256"][:8] for item in xlam] == ["4ef5c6f0"]


@pytest.fixture
def scratch(tmp_path: Path) -> Path:
    for name in ("registry/datasets.yaml", "registry/dataset_record.schema.json"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    (tmp_path / "docs/references").mkdir(parents=True)
    shutil.copyfile(ROOT / "docs/references/papers.yaml", tmp_path / "docs/references/papers.yaml")
    return tmp_path


def _write(root: Path, registry: dict) -> None:
    (root / "registry/datasets.yaml").write_text(yaml.safe_dump(registry), encoding="utf-8")


def _mutated(root: Path, record_id: str, change) -> list[str]:
    registry = copy.deepcopy(_registry())
    change(_by_id(registry)[record_id], registry)
    _write(root, registry)
    return validate_structure(root)


@pytest.mark.parametrize(
    ("record_id", "change", "expected"),
    [
        (
            "toolace",
            lambda r, _: r.update(redistribution="PERMITTED_WITH_ATTRIBUTION"),
            "redistribution",
        ),
        ("toolace", lambda r, _: r.pop("role"), "'role' is a required property"),
        ("toolace", lambda r, _: r.update(checksum="0" * 64), "Additional properties"),
        (
            "canonical_v1",
            lambda r, _: r.pop("derived_from"),
            "'derived_from' is a required property",
        ),
        (
            "canonical_v1",
            lambda r, _: r["processed_dataset_hash"].update(value=None),
            "processed_dataset_hash.value",
        ),
        ("apigen-mt-5k", lambda r, _: r.pop("overlap_benchmarks"), "overlap_benchmarks"),
        ("apigen-mt-5k", lambda r, _: r.update(intended_stages=["future_sft"]), "intended_stages"),
        ("canonical_v2", lambda r, _: r.pop("overlap_evidence"), "overlap_evidence"),
        ("button", lambda r, _: r.update(redistribution_basis=None), "redistribution_basis"),
        ("when2call", lambda r, _: r.update(intended_stages=["pretraining"]), "intended_stages"),
        (
            "xlam-function-calling-60k",
            lambda r, _: r["distribution"][0].update(sha256="not-a-digest"),
            "distribution.0.sha256",
        ),
    ],
)
def test_each_schema_rule_rejects_a_record_that_breaks_it(scratch, record_id, change, expected):
    errors = _mutated(scratch, record_id, change)
    assert any(expected in error and record_id in error for error in errors), errors


def test_a_version_2_file_cannot_skip_its_schema(scratch: Path) -> None:
    errors = _mutated(scratch, "toolace", lambda _, registry: registry.pop("record_schema"))
    assert any("requires record_schema" in error for error in errors), errors


@pytest.mark.parametrize(
    ("record_id", "change", "expected"),
    [
        ("toolace", lambda r: r.update(papers=["no-such-paper"]), "unknown id 'no-such-paper'"),
        (
            "canonical_v1",
            lambda r: r.update(overlap_evidence=["reports/data/missing.json"]),
            "evidence path does not exist",
        ),
        (
            "toolace",
            lambda r: r["distribution"][0].update(
                content_url="https://huggingface.co/datasets/Team-ACE/ToolACE/resolve/main/data.json"
            ),
            "does not name the pinned revision",
        ),
    ],
)
def test_each_reference_rule_rejects_a_record_that_breaks_it(scratch, record_id, change, expected):
    registry = copy.deepcopy(_registry())
    change(_by_id(registry)[record_id])
    _write(scratch, registry)
    errors = validate_references(scratch)
    assert any(expected in error for error in errors), errors
