"""Candidate evaluation tests.

These are pure-logic tests: no GPU, no engine, no held-out data. They pin the contract that
makes a candidate measurement comparable to B0 — the model may change and nothing else — since
a candidate that quietly altered the engine, template, generation settings, or manifest would
produce a delta that looks like a model improvement.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from opengrad.evaluation.candidate import (
    BASELINE_CONFIG,
    CANDIDATE_STATUS,
    compare_routing,
    validate_candidate_config,
    write_candidate_config,
)

ROOT = Path(__file__).resolve().parents[2]


def _checkpoint(tmp_path: Path, name: str = "checkpoint-400") -> Path:
    path = tmp_path / name
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    return path


def _candidate(tmp_path: Path, **overrides) -> dict:
    baseline = yaml.safe_load((ROOT / BASELINE_CONFIG).read_text(encoding="utf-8"))
    config = dict(baseline)
    config["status"] = CANDIDATE_STATUS
    config["model_id"] = str(_checkpoint(tmp_path))
    config["model_revision"] = None
    config["provenance"] = dict(baseline["provenance"])
    config["provenance"].update(
        {
            "parent_experiment_id": "exp",
            "checkpoint_id": "checkpoint-400",
            "checkpoint_step": 400,
            "base_model_id": baseline["model_id"],
            "base_model_revision": baseline["model_revision"],
        }
    )
    config.update(overrides)
    return config


def test_a_faithful_candidate_is_accepted(tmp_path):
    validate_candidate_config(_candidate(tmp_path), ROOT)


def test_the_frozen_baseline_config_is_not_itself_a_candidate(tmp_path):
    """The baseline cannot be smuggled in as its own candidate; status is the discriminator."""
    baseline = yaml.safe_load((ROOT / BASELINE_CONFIG).read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="status must be"):
        validate_candidate_config(baseline, ROOT)


@pytest.mark.parametrize(
    "field,value",
    [
        (("renderer",), "some_other_renderer"),
        (("template_hash",), "0" * 64),
        (("seed",), 1),
        (("generation", "temperature"), 0.7),
        (("generation", "max_new_tokens"), 128),
        (("runtime", "backend"), "transformers"),
        (("runtime", "max_model_len"), 2048),
        (("evaluations", "behavioral_manifest"), "some/other/manifest.json"),
    ],
)
def test_changing_anything_that_defines_the_measurement_is_refused(tmp_path, field, value):
    config = _candidate(tmp_path)
    target = config
    for key in field[:-1]:
        target = target[key]
    target[field[-1]] = value
    with pytest.raises(ValueError, match="measurement contract"):
        validate_candidate_config(config, ROOT)


def test_tokenizer_must_stay_the_pinned_one(tmp_path):
    with pytest.raises(ValueError, match="pinned tokenizer revision"):
        validate_candidate_config(_candidate(tmp_path, tokenizer_revision="deadbeef"), ROOT)


def test_model_id_must_be_a_real_checkpoint_directory(tmp_path):
    with pytest.raises(ValueError, match="existing checkpoint directory"):
        validate_candidate_config(_candidate(tmp_path, model_id=str(tmp_path / "nope")), ROOT)


def test_incomplete_provenance_is_refused(tmp_path):
    config = _candidate(tmp_path)
    del config["provenance"]["checkpoint_step"]
    with pytest.raises(ValueError, match="checkpoint_step"):
        validate_candidate_config(config, ROOT)


def test_missing_required_field_is_refused(tmp_path):
    config = _candidate(tmp_path)
    del config["outputs"]
    with pytest.raises(ValueError, match="missing required field"):
        validate_candidate_config(config, ROOT)


def test_generated_candidate_config_is_accepted_and_namespaced(tmp_path):
    """The generator must produce something the validator accepts, without hand-editing."""
    checkpoint = _checkpoint(tmp_path)
    out = tmp_path / "candidate.yaml"
    write_candidate_config(
        ROOT,
        checkpoint=checkpoint,
        checkpoint_id="checkpoint-400",
        checkpoint_step=400,
        parent_experiment_id="qwen35_2b_m0_sft_full_v3",
        out_path=out,
    )
    config = yaml.safe_load(out.read_text(encoding="utf-8"))
    validate_candidate_config(config, ROOT)
    assert config["model_id"] == str(checkpoint)
    for value in config["outputs"].values():
        assert "qwen35_2b_m0_sft_full_v3" in str(value), value
        assert not str(value).startswith("reports/"), (
            "a candidate must never write B0 evidence paths"
        )


def test_comparison_reports_direction_and_verdict():
    baseline = {
        "call_f1": 0.6191,
        "call_precision": 0.4542,
        "over_call_rate": 0.643,
        "under_call_rate": 0.014,
        "parse_valid_rate": 0.99899,
    }
    candidate = {
        "call_f1": 0.7000,
        "call_precision": 0.6000,
        "over_call_rate": 0.500,
        "under_call_rate": 0.030,
        "parse_valid_rate": 0.99899,
    }
    comparison = compare_routing(baseline, candidate, "tool_calling/qwen35_2b/baseline")
    assert comparison["improved"] == ["call_f1", "call_precision", "over_call_rate"]
    # An error rate that rose is a regression even though the number went up.
    assert comparison["regressed"] == ["under_call_rate"]
    assert comparison["metrics"]["over_call_rate"]["direction"] == "lower_is_better"
    assert comparison["metrics"]["call_f1"]["delta"] == pytest.approx(0.0809, abs=1e-4)


def test_comparison_ignores_metrics_absent_on_one_side():
    """A future evaluator adding a metric must not read as a regression here."""
    comparison = compare_routing({"call_f1": 0.5, "brand_new": 0.9}, {"call_f1": 0.6}, "b")
    assert set(comparison["metrics"]) == {"call_f1"}
