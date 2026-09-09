import pytest

from opengrad.data.behavior import capability_ids, load_taxonomy, validate_behavior
from opengrad.data.canonical import ToolConversation
from opengrad.data.mixture import load_mixture, validate_mixture
from opengrad.data.residuals import residual_to_weights, validate_residual_profile
from opengrad.data.selection import validate_counterfactual_groups, validate_sft_selection


def test_taxonomy_has_four_decisions_and_descriptions():
    taxonomy = load_taxonomy()
    assert set(taxonomy["decisions"]) == {"CALL", "ANSWER", "CLARIFY", "UNSUPPORTED"}
    assert {"must_call", "direct_answer_retention"} <= capability_ids()
    validate_behavior("CLARIFY", ["clarify_before_call"], "derived")


def test_unknown_behavior_rejected():
    with pytest.raises(ValueError, match="unknown behavior capability"):
        validate_behavior("CALL", ["invented_capability"])


def test_canonical_preserves_behavior_and_catalogue_metadata():
    conversation = ToolConversation(
        "fixture-call",
        "synthetic",
        [{"name": "lookup"}],
        [{"role": "user", "content": "lookup x"}],
        {
            "split": "train",
            "behavior": {"decision": "CALL", "capabilities": ["must_call"]},
            "tool_context": {"tool_count": 1, "distractor_count": 0},
        },
    )
    conversation.validate()


def test_mixture_configs_expose_m0_m1_and_schema_ready_m2():
    assert (
        load_mixture("configs/data/tool_calling/source_baseline_v1.yaml")["mixture_class"]
        == "source_oriented"
    )
    assert (
        load_mixture("configs/data/tool_calling/balanced_policy_v1.yaml")["mixture_class"]
        == "behavior_balanced"
    )
    assert (
        load_mixture("configs/data/tool_calling/residual_policy_v1.yaml")["status"]
        == "SCHEMA_READY"
    )


def test_residual_mapping_is_deterministic_and_requires_evidence():
    profile = {
        "model": "fixture",
        "baseline_experiment": "b0",
        "sample_count": 100,
        "revision": "r1",
        "residuals": {"UNDER_CALL": 0.8, "OVER_CALL": 0.2},
    }
    result = residual_to_weights(profile)
    assert result["baseline_experiment"] == "b0"
    assert result["target_weights"]["must_call"] > result["target_weights"]["must_not_call"]
    with pytest.raises(ValueError):
        validate_residual_profile(
            {"model": "fixture", "baseline_experiment": "b0", "sample_count": 0, "residuals": {}}
        )


def test_counterfactual_and_sft_boundaries_fail_loudly():
    with pytest.raises(ValueError, match="at least two"):
        validate_counterfactual_groups([{"metadata": {"counterfactual": {"group_id": "one"}}}])
    with pytest.raises(ValueError, match="forbidden"):
        validate_sft_selection(["when2call:mcq"])
    with pytest.raises(ValueError, match="contaminated"):
        validate_sft_selection(["apigen-mt-5k"], {"apigen-mt-5k"})


def test_malformed_mixture_weights_fail_closed():
    with pytest.raises(TypeError, match="behavior_weights must be a mapping"):
        validate_mixture(
            {
                "mixture_class": "behavior_balanced",
                "status": "HYPOTHESIS_ONLY",
                "behavior_weights": ["must_call"],
            }
        )
