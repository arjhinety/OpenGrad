from pathlib import Path

from opengrad.data.manifest import DatasetManifest, compute_dataset_fingerprint
from opengrad.data.validator import validate_records


def test_compute_dataset_fingerprint() -> None:
    h1 = ["hash_b", "hash_a", "hash_c"]
    h2 = ["hash_c", "hash_a", "hash_b"]
    # Order-independent fingerprint
    assert compute_dataset_fingerprint(h1) == compute_dataset_fingerprint(h2)


def test_dataset_manifest_roundtrip(tmp_path: Path) -> None:
    manifest = DatasetManifest(
        dataset_name="test_dataset",
        source="test_source",
        upstream_revision="rev123",
        adapter="canonical",
        schema_version="1.0",
        split="train",
        raw_record_count=100,
        valid_record_count=98,
        rejected_record_count=2,
        deduplicated_count=0,
        final_count=98,
        checksum="sha256_checksum",
        deterministic_fingerprint="fp_123",
        tokenizer="Qwen/Qwen3.5-2B",
    )
    dest = tmp_path / "manifest.json"
    manifest.write(dest)
    assert dest.exists()
    loaded = DatasetManifest.from_dict(manifest.to_dict())
    assert loaded.final_count == 98


def test_validator_detects_invalid_dpo_pair() -> None:
    dpo_records = [
        {"id": "d1", "prompt": "Hello", "chosen": "Hi there!", "rejected": "Hi there!"},  # chosen == rejected
        {"id": "d2", "prompt": "2+2?", "chosen": "4", "rejected": "5"},  # valid
    ]
    report = validate_records(dpo_records, dataset_name="dpo_test", mode="dpo")
    assert report.status == "FAIL"
    assert report.invalid_records == 1
    assert "CHOSEN_EQUALS_REJECTED" in report.reason_counts


def test_validator_passes_valid_sft_record() -> None:
    records = [
        {
            "id": "r1",
            "source": {"dataset_id": "test"},
            "tools": [{"name": "lookup", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}],
            "messages": [
                {"role": "user", "content": "Search for x"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "c1", "name": "lookup", "arguments": {"q": "x"}}],
                },
                {"role": "tool", "content": "result", "tool_call_id": "c1"},
                {"role": "assistant", "content": "Done."},
            ],
            "metadata": {"split": "train"},
        }
    ]
    report = validate_records(records, dataset_name="sft_test", mode="sft")
    assert report.status == "PASS"
    assert report.valid_records == 1
