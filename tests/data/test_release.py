from pathlib import Path

from opengrad.data.release import _load_config, _parquet_row, validate_release


def test_release_row_preserves_source_lineage():
    row = {
        "id": "og-1",
        "schema_version": "tool_use_ir_v1",
        "canonical_hash": "hash",
        "tools": [],
        "messages": [],
        "metadata": {
            "split": "train",
            "adapter": "fixture",
            "adapter_version": "1.0.0",
            "source": {"upstream_id": "source-1"},
            "behavior": {"decision": "ANSWER", "confidence": "known", "capabilities": []},
        },
    }
    result = _parquet_row(
        row,
        {
            "id": "fixture",
            "source_revision": "rev",
            "upstream_license": "CC-BY-4.0",
            "redistribution_status": "PERMITTED_WITH_ATTRIBUTION",
            "upstream_access_mode": "gated",
        },
    )
    assert result["opengrad_id"] == "og-1"
    assert result["source_record_id"] == "source-1"
    assert result["source_revision"] == "rev"
    assert result["source_license"] == "CC-BY-4.0"
    assert result["redistribution_status"] == "PERMITTED_WITH_ATTRIBUTION"
    assert result["upstream_access_mode"] == "gated"
    assert result["downstream_access_requirement"] == "public_allowed"
    assert result["behavior_decision"] == "ANSWER"


def test_xlam_upstream_gate_does_not_block_downstream_release():
    config = _load_config(Path("configs/releases/toolpolicy_canonical_v1.yaml"))
    xlam = next(
        item for item in config["included_sources"] if item["id"] == "xlam-function-calling-60k"
    )
    assert xlam["upstream_access_mode"] == "gated"
    assert xlam["redistribution_status"] == "PERMITTED_WITH_ATTRIBUTION"
    assert xlam["downstream_access_requirement"] == "public_allowed"
    assert config["allow_gated_sources"] is False


def test_release_validator_fails_closed_without_manifest(tmp_path: Path):
    assert validate_release(tmp_path) == ["release-manifest.json missing"]


def test_rebuilding_an_identical_payload_preserves_the_recorded_manifest(monkeypatch, tmp_path):
    """A release's identity is its payload.

    Rebuilding the same shards at a later commit used to rewrite `opengrad_git_commit` and
    therefore the manifest hash, which silently orphaned every recorded dataset hash and every
    experiment pinned to that release while the data itself was unchanged.
    """
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from opengrad.data import release as release_module

    output = tmp_path / "release"
    output.mkdir()
    table = pa.table({"opengrad_id": ["a", "b"]})
    pq.write_table(table, output / "shard-000000.parquet")

    config_path = tmp_path / "release.yaml"
    config_path.write_text(
        "release_name: fixture\n"
        "release_version: v0\n"
        "release_class: PRE_TRAINING_CANONICAL_RELEASE\n"
        "hub_repository: local/fixture\n"
        "canonical_schema_version: tool_use_ir_v1\n"
        "behavior_taxonomy_version: tax\n"
        "card_directory: release/huggingface/fixture\n"
        "included_sources: []\n",
        encoding="utf-8",
    )
    (tmp_path / "release" / "huggingface" / "fixture").mkdir(parents=True)
    (tmp_path / "release" / "huggingface" / "fixture" / "README.template.md").write_text(
        "card\n", encoding="utf-8"
    )
    (tmp_path / "release" / "huggingface" / "fixture" / "source-licenses.md").write_text(
        "licenses\n", encoding="utf-8"
    )
    (tmp_path / "release" / "huggingface" / "fixture" / "CITATIONS.bib").write_text(
        "cites\n", encoding="utf-8"
    )

    monkeypatch.setattr(release_module, "_git_commit", lambda root: "commit-one")
    release_module.build_release(tmp_path, config_path, output)
    first = (output / "release-manifest.json").read_bytes()
    assert json.loads(first)["opengrad_git_commit"] == "commit-one"

    # Same payload, later commit: the recorded identity must not move.
    monkeypatch.setattr(release_module, "_git_commit", lambda root: "commit-two")
    release_module.build_release(tmp_path, config_path, output)
    second = (output / "release-manifest.json").read_bytes()
    assert second == first
    assert json.loads(second)["opengrad_git_commit"] == "commit-one"

    # A changed payload is a new release, so the manifest is rewritten.
    pq.write_table(pa.table({"opengrad_id": ["a", "b", "c"]}), output / "shard-000000.parquet")
    release_module.build_release(tmp_path, config_path, output)
    third = json.loads((output / "release-manifest.json").read_bytes())
    assert third["opengrad_git_commit"] == "commit-two"
