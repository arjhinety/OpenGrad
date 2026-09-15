"""Classifier input contract prose-decision-input-v1: eligibility and the typed input object.

No classification logic exists or is tested here. These tests pin what the future classifier may read,
which records it may never receive, and that the serialization is deterministic and source-blind.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from opengrad.data import versions
from opengrad.data.classifier_input import (
    CONTRACT_VERSION,
    EVALUATION_ONLY_OR_HELDOUT,
    EXCLUSION_ORDER,
    FEATURE_FIELDS,
    MALFORMED_OR_UNRENDERABLE,
    MULTI_TURN_UNSUPPORTED,
    POST_TOOL_RESULT_RESPONSE,
    STRUCTURAL_CALL,
    ClassifierFeatures,
    ClassifierInput,
    ClassifierProvenance,
    ContractViolation,
    HeldoutIndex,
    IneligibleRecord,
    build_classifier_input,
    eligibility,
    normalize_prompt,
)
from opengrad.data.normalization_v3 import normalize_row
from tests.data.test_normalization_v3 import (
    GLAIVE_CALL_CHAT,
    GLAIVE_SYSTEM,
    glaive_spec,
    when2call_raw,
    when2call_spec,
)

ROOT = Path(__file__).resolve().parents[2]
NO_HELDOUT = HeldoutIndex()


def glaive_record(chat: str) -> dict[str, Any]:
    return normalize_row(glaive_spec(), {"system": GLAIVE_SYSTEM, "chat": chat}, 0)


def plain_record() -> dict[str, Any]:
    return normalize_row(when2call_spec(), when2call_raw(), 0)


# ── eligibility ────────────────────────────────────────────────────────────────────────────────────


def test_tool_free_single_exchange_is_eligible() -> None:
    result = eligibility(plain_record(), NO_HELDOUT)
    assert result.eligible and result.reasons == ()


def test_structured_call_goes_to_structural_routing() -> None:
    single_call = glaive_record(
        "USER: Weather in Paris?\n\n\nASSISTANT: <functioncall> "
        '{"name": "get_weather", "arguments": \'{"city": "Paris"}\'}\n'
    )
    result = eligibility(single_call, NO_HELDOUT)
    assert STRUCTURAL_CALL in result.reasons
    assert POST_TOOL_RESULT_RESPONSE not in result.reasons


def test_prose_after_a_tool_result_is_excluded_but_not_called_multi_turn() -> None:
    result = eligibility(glaive_record(GLAIVE_CALL_CHAT), NO_HELDOUT)
    assert result.reasons == (STRUCTURAL_CALL, POST_TOOL_RESULT_RESPONSE)
    with pytest.raises(IneligibleRecord):
        build_classifier_input(glaive_record(GLAIVE_CALL_CHAT), NO_HELDOUT)


def test_multi_turn_is_excluded_and_never_flattened_to_its_final_message() -> None:
    record = glaive_record(
        "USER: Hi\n\n\nASSISTANT: Hello! How can I help?\n\n\n"
        "USER: Tell me a joke.\n\n\nASSISTANT: Why did the chicken cross the road?\n"
    )
    result = eligibility(record, NO_HELDOUT)
    assert result.reasons == (MULTI_TURN_UNSUPPORTED,)
    assert "shape:U-A-U-A" in result.details
    with pytest.raises(IneligibleRecord) as excinfo:
        build_classifier_input(record, NO_HELDOUT)
    assert excinfo.value.result.primary_reason == MULTI_TURN_UNSUPPORTED


def test_heldout_prompt_is_excluded() -> None:
    record = plain_record()
    heldout = HeldoutIndex(prompts=frozenset({normalize_prompt("  LOOK up the term 'otter'. ")}))
    result = eligibility(record, heldout)
    assert result.reasons == (EVALUATION_ONLY_OR_HELDOUT,)
    assert "heldout_prompt" in result.details


def test_evaluation_only_record_is_excluded() -> None:
    record = plain_record()
    record["metadata"]["eligibility"] = "evaluation_only"
    assert eligibility(record, NO_HELDOUT).primary_reason == EVALUATION_ONLY_OR_HELDOUT


def test_empty_final_response_is_malformed() -> None:
    record = plain_record()
    record["messages"][-1]["content"] = "   "
    result = eligibility(record, NO_HELDOUT)
    assert result.primary_reason == MALFORMED_OR_UNRENDERABLE
    assert "final_assistant_empty" in result.details


def test_reasons_follow_the_declared_precedence() -> None:
    record = glaive_record(GLAIVE_CALL_CHAT)
    record["metadata"]["eligibility"] = "evaluation_only"
    reasons = eligibility(record, NO_HELDOUT).reasons
    assert list(reasons) == [reason for reason in EXCLUSION_ORDER if reason in reasons]
    assert reasons[0] == EVALUATION_ONLY_OR_HELDOUT


@pytest.mark.parametrize(
    "mutate",
    [
        lambda metadata: metadata.pop("normalization_version"),
        lambda metadata: metadata.update(adapter_version="1.0.0"),
        lambda metadata: metadata.update(
            behavior={"decision": "ANSWER", "capabilities": [], "confidence": "derived"}
        ),
    ],
    ids=["not-v3", "legacy-adapter-version", "carries-behaviour-label"],
)
def test_a_record_outside_normalization_v3_is_refused_outright(mutate: Any) -> None:
    record = plain_record()
    mutate(record["metadata"])
    with pytest.raises(ContractViolation):
        eligibility(record, NO_HELDOUT)


# ── the input object ───────────────────────────────────────────────────────────────────────────────


def test_features_are_exactly_the_contract_fields() -> None:
    payload = build_classifier_input(plain_record(), NO_HELDOUT).features_payload()
    assert set(payload) == {"contract_version", *FEATURE_FIELDS}
    assert (
        payload["contract_version"]
        == CONTRACT_VERSION
        == versions.CLASSIFIER_INPUT_CONTRACT_VERSION
    )
    assert payload["user_message"] == "Look up the term 'otter'."
    assert payload["assistant_response"] == "Which source?"
    assert payload["structured_call_present"] is False
    assert [tool["name"] for tool in payload["tools"]] == ["lookup"]


def test_source_identity_is_provenance_not_a_feature() -> None:
    record = plain_record()
    built = build_classifier_input(record, NO_HELDOUT)
    features = built.features_json()
    metadata = record["metadata"]
    for identity in (
        metadata["source"]["dataset_id"],
        metadata["source"]["source_name"],
        metadata["raw_record_hash"],
        record["id"],
        record["canonical_hash"],
        metadata["adapter"],
    ):
        assert identity not in features
    assert built.provenance.source_name == "when2call"
    assert built.provenance.record_key == f"when2call:{metadata['raw_record_hash']}"

    relabelled = copy.deepcopy(record)
    relabelled["metadata"]["source"].update(dataset_id="other", source_name="other")
    relabelled["metadata"]["raw_record_hash"] = "f" * 64
    other = build_classifier_input(relabelled, NO_HELDOUT)
    assert other.features_sha256() == built.features_sha256()
    assert other.provenance_json() != built.provenance_json()


def test_serialization_is_deterministic_and_key_order_free() -> None:
    record = plain_record()
    shuffled = copy.deepcopy(record)
    shuffled["tools"] = [dict(reversed(list(tool.items()))) for tool in shuffled["tools"]]
    a = build_classifier_input(record, NO_HELDOUT)
    b = build_classifier_input(shuffled, NO_HELDOUT)
    assert a.features_json() == b.features_json()
    assert a.features_json() == build_classifier_input(record, NO_HELDOUT).features_json()


def test_features_hash_is_pinned() -> None:
    """A literal: any change to field names, order rules or encoding must move this value."""
    features = ClassifierFeatures(
        user_message="Is it raining in Oslo?",
        assistant_response="I can't check live weather.",
        tools=({"name": "get_weather", "parameters": {"type": "object", "properties": {}}},),
        structured_call_present=False,
    )
    provenance = ClassifierProvenance(*(["x"] * 10))
    item = ClassifierInput(features, provenance)
    assert item.features_json() == (
        '{"assistant_response":"I can\'t check live weather.","contract_version":'
        '"prose-decision-input-v1","structured_call_present":false,"tools":[{"name":"get_weather",'
        '"parameters":{"properties":{},"type":"object"}}],"user_message":"Is it raining in Oslo?"}'
    )
    assert item.features_sha256() == (
        "0e07ddc2ee718981327e5d25032c5ed52f4f6683013b32fffa4aa069b960463f"
    )


# ── the real held-out index ────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not (ROOT / ".cache/normalization/raw/when2call_test_mcq.parquet").is_file(),
    reason="raw When2Call test files are a local cache",
)
def test_heldout_index_loads_the_pinned_inputs() -> None:
    index = HeldoutIndex.load(ROOT)
    # 3,650 partition ids (P-DEV 2,373 + P-CONF 1,277) plus the 2 quarantined ids outside it.
    assert len(index.record_ids) == 3652
    # 3,652 MCQ rows normalize to 2,295 distinct prompts; all 286 LLM-judge prompts are among them.
    assert len(index.prompts) == 2295
    assert index.inputs[0][0].endswith("when2call_test_mcq.parquet")


@pytest.mark.parametrize(
    "path",
    [
        "src/opengrad/data/classifier_input.py",
        "src/opengrad/data/normalization_v3.py",
        "scripts/audit_normalization_v3_structure.py",
    ],
)
def test_structural_code_uses_no_decision_cues(path: str) -> None:
    """Structure only: none of these may reach for refusal, question or direct-answer cues."""
    source = (ROOT / path).read_text(encoding="utf-8")
    for cue in ("detect_refusal", "QUESTION_CUES", "HEDGE_CUES", "capability import", "stratum("):
        assert cue not in source, f"{path} references {cue}"
