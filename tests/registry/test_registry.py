from pathlib import Path

import yaml

from opengrad.registry.validate import validate


def test_registries_validate():
    """The registry validates clean.

    It previously reported one open finding -- `canonical_v2`'s derived identity
    was not reproducible from committed bytes and two records appeared to
    disagree about it. Both identities were subsequently recovered from the
    published artifact and shown to be two editions of the same manifest
    (reports/ERRATA.md section 11), so the finding is closed. This assertion
    fails if any finding returns.
    """
    from pathlib import Path

    assert validate(Path(__file__).parents[2]) == []


def test_dataset_metadata_is_field_level_and_clean_exclusion_is_preserved():
    root = Path(__file__).parents[2]
    records = yaml.safe_load((root / "registry/datasets.yaml").read_text())["datasets"]
    by_id = {record["id"]: record for record in records}
    assert by_id["xlam-function-calling-60k"]["sample_count"]["published"] == 60000
    assert by_id["xlam-function-calling-60k"]["license"]["verified"] is True
    assert by_id["xlam-function-calling-60k"]["processed_dataset_hash"]["status"] == (
        "FULL_DATA_VALIDATED"
    )
    assert by_id["xlam-function-calling-60k"]["retained_after_filtering"]["value"] == 59370
    assert by_id["apigen-mt-5k"]["contamination_status"] == "EXCLUDED_FROM_CLEAN_DEFAULT"
    assert "clean_default" in by_id["apigen-mt-5k"]["forbidden_splits"]
