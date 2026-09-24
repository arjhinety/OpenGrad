"""`opengrad croissant`: registry records export as Croissant 1.1, and the mapping is the schema's.

The exporter reads each field's Croissant counterpart from `x-croissant` in
registry/dataset_record.schema.json. These tests pin that the exported keys agree with those
annotations, that the pinned files and licences carry through, and which records are incomplete
and why: an upstream publication date or a file digest is never invented to fill a gap. The output
was also checked once with MLCommons' validator (mlcroissant 1.1.0): 0 errors on all 11 records.
Its only warnings are a commit SHA as `version` (not semver) and no `citeAs` for the two records
without a paper.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import yaml

from opengrad.registry import croissant

ROOT = Path(__file__).parents[2]


def _exported() -> dict[str, tuple[dict, list[str]]]:
    return croissant.export_all(ROOT)


def test_every_record_exports_and_only_derived_corpora_lack_files() -> None:
    missing = {key: gaps for key, (_, gaps) in _exported().items() if gaps}
    # Derived corpora are identified by processed_dataset_hash, and their Hub files live at a Hub
    # revision the record does not hold yet; the M1 pairs are a repository file with no public URL.
    assert missing == {
        "canonical_v1": ["distribution"],
        "canonical_v2": ["distribution"],
        "canonical_v2_final": ["distribution"],
        "m1_calibration_preference_pairs_v1": ["url", "distribution"],
    }


def test_exported_keys_follow_the_schema_annotations() -> None:
    mapping = croissant.croissant_mapping(croissant.load_schema(ROOT))
    document, _ = _exported()["xlam-function-calling-60k"]
    for field in croissant.SCALAR_FIELDS:
        assert croissant._key(mapping[field]) in document, field
    # The transformed fields keep the property their annotation names.
    assert mapping["license"] == "sc:license" and "license" in document
    assert mapping["distribution"] == "sc:distribution" and "distribution" in document
    assert mapping["source_repository"] == "sc:url" and "url" in document
    assert mapping["organization"] == "sc:creator" and "creator" in document
    assert mapping["papers"] == "cr:citeAs" and "citeAs" in document
    assert mapping["derived_from"] == "prov:wasDerivedFrom"
    assert "prov:wasDerivedFrom" in _exported()["canonical_v2_final"][0]


def test_pinned_files_carry_through_with_their_digests() -> None:
    registry = yaml.safe_load((ROOT / "registry/datasets.yaml").read_text(encoding="utf-8"))
    when2call = next(r for r in registry["datasets"] if r["id"] == "when2call")
    document, _ = _exported()["when2call"]
    assert [(f["contentUrl"], f["sha256"]) for f in document["distribution"]] == [
        (item["content_url"], item["sha256"]) for item in when2call["distribution"]
    ]
    assert document["version"] == when2call["source_revision"]["value"]
    assert document["datePublished"] == "2025-04-26"


def test_composite_licences_resolve_to_the_upstream_licences() -> None:
    exported = _exported()
    four_sources = {
        "https://spdx.org/licenses/Apache-2.0.html",
        "https://spdx.org/licenses/CC-BY-4.0.html",
    }
    assert set(exported["canonical_v2_final"][0]["license"]) == four_sources
    # The M1 pairs cite canonical_v2_final, itself a composite, and resolve through it.
    assert set(exported["m1_calibration_preference_pairs_v1"][0]["license"]) == four_sources


def test_the_context_is_the_standard_croissant_context() -> None:
    document, _ = _exported()["toolace"]
    context = document["@context"]
    assert context["@vocab"] == "https://schema.org/"
    assert context["cr"] == "http://mlcommons.org/croissant/"
    assert context["conformsTo"] == "dct:conformsTo"
    assert document["conformsTo"] == [croissant.CROISSANT]


def test_responsible_use_fields_export_as_croissant_rai() -> None:
    registry = yaml.safe_load((ROOT / "registry/datasets.yaml").read_text(encoding="utf-8"))
    records = registry["datasets"]
    record = copy.deepcopy(next(r for r in records if r["id"] == "when2call"))
    record["responsible_use"] = {
        "data_collection": "Generated from xlam-function-calling-60k by an LLM pipeline.",
        "data_limitations": ["Direct answers are always labelled wrong."],
        "evidence": ["https://arxiv.org/abs/2504.18851"],
    }
    document, _ = croissant.export(
        record, records, croissant.load_schema(ROOT), croissant._papers(ROOT)
    )
    assert document["rai:dataCollection"].startswith("Generated from")
    assert document["rai:dataLimitations"] == ["Direct answers are always labelled wrong."]
    assert croissant.CROISSANT_RAI in document["conformsTo"]
    assert "evidence" not in json.dumps(document)


def test_the_command_prints_one_record_and_rejects_an_unknown_id(monkeypatch, capsys) -> None:
    from opengrad import cli

    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "argv", ["opengrad", "croissant", "toolace"])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["identifier"] == "toolace"
    monkeypatch.setattr(sys, "argv", ["opengrad", "croissant", "no-such-id"])
    assert cli.main() == 2
