"""Classifier input contract prose-decision-input-v2: the first reply (docs/research/study-002/36 §2).

v2 admits the first assistant reply of any record, whatever follows it. These tests pin that the unit is exactly
that reply, that nothing after it is ever read, that continuation is provenance and never a feature, and that v1 is
unchanged.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from opengrad.data import versions
from opengrad.data.classifier_input import (
    CONTRACT_V2_VERSION,
    EVALUATION_ONLY_OR_HELDOUT,
    FEATURE_FIELDS,
    FIRST_REPLY_EMPTY,
    FIRST_REPLY_EXCLUSION_ORDER,
    FIRST_REPLY_IS_STRUCTURAL_CALL,
    FIRST_TURN_NOT_ONE_USER_THEN_ASSISTANT,
    MALFORMED_OR_UNRENDERABLE,
    MULTI_TURN_UNSUPPORTED,
    UNIT_HAS_CONTINUATION,
    UNIT_SINGLE_EXCHANGE,
    ClassifierFeatures,
    ClassifierInput,
    ContractViolation,
    FirstReplyProvenance,
    HeldoutIndex,
    IneligibleRecord,
    build_classifier_input,
    build_first_reply_input,
    eligibility,
    first_reply_eligibility,
    normalize_prompt,
)
from tests.data.test_classifier_input import glaive_record, plain_record
from tests.data.test_normalization_v3 import GLAIVE_CALL_CHAT

NO_HELDOUT = HeldoutIndex()
CONTINUED_WITH_CALL = (
    "USER: Can you help me with the weather?\n\n\n"
    "ASSISTANT: Of course. Which city? <|endoftext|>\n\n\n"
    "USER: Paris\n\n\n"
    'ASSISTANT: <functioncall> {"name": "get_weather", "arguments": \'{"city": "Paris"}\'} <|endoftext|>\n\n\n'
    'FUNCTION RESPONSE: {"temperature": 20}\n\n\n'
    "ASSISTANT: It is 20 degrees in Paris. <|endoftext|>\n\n\n"
)
CONTINUED_IN_PROSE = (
    "USER: Hi\n\n\nASSISTANT: Hello! How can I help? <|endoftext|>\n\n\n"
    "USER: Tell me a joke.\n\n\nASSISTANT: Why did the chicken cross the road? <|endoftext|>\n"
)


# ── the unit ───────────────────────────────────────────────────────────────────────────────────────


def test_a_v1_single_exchange_is_eligible_with_the_same_features_and_a_different_hash() -> None:
    record = plain_record()
    v1 = build_classifier_input(record, NO_HELDOUT)
    v2 = build_first_reply_input(record, NO_HELDOUT)
    assert v2.features == v1.features
    assert v2.contract_version == CONTRACT_V2_VERSION == versions.CLASSIFIER_INPUT_CONTRACT_V2_VERSION
    assert v2.features_sha256() != v1.features_sha256()
    assert isinstance(v2.provenance, FirstReplyProvenance)
    assert v2.provenance.unit_kind == UNIT_SINGLE_EXCHANGE


@pytest.mark.parametrize("chat", [CONTINUED_WITH_CALL, CONTINUED_IN_PROSE], ids=["then-a-tool-call", "then-prose"])
def test_the_first_reply_of_a_continuing_conversation_is_eligible_under_v2_only(chat: str) -> None:
    record = glaive_record(chat)
    assert not eligibility(record, NO_HELDOUT).eligible
    with pytest.raises(IneligibleRecord):
        build_classifier_input(record, NO_HELDOUT)
    result = first_reply_eligibility(record, NO_HELDOUT)
    assert result.eligible and result.reasons == ()
    built = build_first_reply_input(record, NO_HELDOUT)
    assert built.provenance.unit_kind == UNIT_HAS_CONTINUATION
    assert built.features.structured_call_present is False
    assert set(built.features_payload()) == {"contract_version", *FEATURE_FIELDS}


def test_nothing_after_the_first_reply_is_read() -> None:
    record = glaive_record(CONTINUED_IN_PROSE)
    built = build_first_reply_input(record, NO_HELDOUT)
    assert built.features.user_message.strip() == "Hi"
    assert built.features.assistant_response.strip() == "Hello! How can I help?"
    for later in ("joke", "chicken"):
        assert later not in built.features_json()
    changed = copy.deepcopy(record)
    body = [m for m in changed["messages"] if m["role"] != "system"]
    body[2]["content"] = "Something else entirely."
    body[3]["content"] = "A different ending."
    assert build_first_reply_input(changed, NO_HELDOUT).features_sha256() == built.features_sha256()


def test_continuation_is_provenance_and_never_changes_the_features() -> None:
    continued = glaive_record(CONTINUED_IN_PROSE)
    single = copy.deepcopy(continued)
    first_user = next(i for i, m in enumerate(single["messages"]) if m["role"] == "user")
    single["messages"] = single["messages"][: first_user + 2]
    a = build_first_reply_input(continued, NO_HELDOUT)
    b = build_first_reply_input(single, NO_HELDOUT)
    assert a.features_sha256() == b.features_sha256()
    assert (a.provenance.unit_kind, b.provenance.unit_kind) == (UNIT_HAS_CONTINUATION, UNIT_SINGLE_EXCHANGE)
    assert "unit_kind" not in a.features_json()
    assert a.provenance_json() != b.provenance_json()


# ── eligibility ────────────────────────────────────────────────────────────────────────────────────


def test_a_first_reply_that_is_a_structured_call_goes_to_structural_routing() -> None:
    result = first_reply_eligibility(glaive_record(GLAIVE_CALL_CHAT), NO_HELDOUT)
    assert result.reasons == (FIRST_REPLY_IS_STRUCTURAL_CALL,)


def test_an_empty_first_reply_is_excluded() -> None:
    record = glaive_record(CONTINUED_IN_PROSE)
    next(m for m in record["messages"] if m["role"] == "assistant")["content"] = "   "
    assert FIRST_REPLY_EMPTY in first_reply_eligibility(record, NO_HELDOUT).reasons


def test_a_record_that_does_not_open_with_one_user_turn_then_an_assistant_turn_is_excluded() -> None:
    record = glaive_record(CONTINUED_IN_PROSE)
    body_start = next(i for i, m in enumerate(record["messages"]) if m["role"] == "user")
    record["messages"].insert(body_start, dict(record["messages"][body_start]))
    result = first_reply_eligibility(record, NO_HELDOUT)
    assert FIRST_TURN_NOT_ONE_USER_THEN_ASSISTANT in result.reasons
    assert "head:U-U" in result.details


def test_a_held_out_prompt_in_any_later_turn_still_excludes_the_record() -> None:
    record = glaive_record(CONTINUED_IN_PROSE)
    heldout = HeldoutIndex(prompts=frozenset({normalize_prompt("tell me a JOKE.")}))
    result = first_reply_eligibility(record, heldout)
    assert result.reasons == (EVALUATION_ONLY_OR_HELDOUT,)
    assert "heldout_prompt" in result.details


def test_final_message_checks_do_not_apply_but_other_malformations_do() -> None:
    ends_on_user = glaive_record(CONTINUED_IN_PROSE)
    ends_on_user["messages"] = ends_on_user["messages"][:-1]
    assert "final_role:user" in eligibility(ends_on_user, NO_HELDOUT).details
    assert first_reply_eligibility(ends_on_user, NO_HELDOUT).eligible

    not_valid = glaive_record(CONTINUED_IN_PROSE)
    not_valid["metadata"]["parse_status"] = "PARSE_FAILED"
    result = first_reply_eligibility(not_valid, NO_HELDOUT)
    assert result.primary_reason == MALFORMED_OR_UNRENDERABLE


def test_reasons_follow_the_declared_precedence() -> None:
    record = glaive_record(GLAIVE_CALL_CHAT)
    record["metadata"]["eligibility"] = "evaluation_only"
    reasons = first_reply_eligibility(record, NO_HELDOUT).reasons
    assert list(reasons) == [reason for reason in FIRST_REPLY_EXCLUSION_ORDER if reason in reasons]
    assert reasons == (EVALUATION_ONLY_OR_HELDOUT, FIRST_REPLY_IS_STRUCTURAL_CALL)


def test_a_record_outside_normalization_v3_is_refused_outright() -> None:
    record = glaive_record(CONTINUED_IN_PROSE)
    record["metadata"]["adapter_version"] = "1.0.0"
    with pytest.raises(ContractViolation):
        first_reply_eligibility(record, NO_HELDOUT)


def test_v1_is_unchanged_for_multi_turn_records() -> None:
    result = eligibility(glaive_record(CONTINUED_IN_PROSE), NO_HELDOUT)
    assert result.reasons == (MULTI_TURN_UNSUPPORTED,)


def test_source_identity_is_provenance_not_a_feature() -> None:
    record = glaive_record(CONTINUED_WITH_CALL)
    built = build_first_reply_input(record, NO_HELDOUT)
    relabelled = copy.deepcopy(record)
    relabelled["metadata"]["source"].update(dataset_id="other", source_name="other")
    relabelled["metadata"]["raw_record_hash"] = "f" * 64
    other = build_first_reply_input(relabelled, NO_HELDOUT)
    assert other.features_sha256() == built.features_sha256()
    assert other.provenance_json() != built.provenance_json()
    for identity in (record["metadata"]["source"]["source_name"], record["metadata"]["raw_record_hash"], record["id"]):
        assert identity not in built.features_json()


def test_features_hash_is_pinned() -> None:
    """A literal: v2 serializes as v1 does, under its own contract version."""
    features = ClassifierFeatures(
        user_message="Is it raining in Oslo?",
        assistant_response="I can't check live weather.",
        tools=({"name": "get_weather", "parameters": {"type": "object", "properties": {}}},),
        structured_call_present=False,
    )
    provenance: Any = FirstReplyProvenance(*(["x"] * 10))
    item = ClassifierInput(features, provenance, contract_version=CONTRACT_V2_VERSION)
    assert item.features_json() == (
        '{"assistant_response":"I can\'t check live weather.","contract_version":'
        '"prose-decision-input-v2","structured_call_present":false,"tools":[{"name":"get_weather",'
        '"parameters":{"properties":{},"type":"object"}}],"user_message":"Is it raining in Oslo?"}'
    )
    assert item.features_sha256() == "dce2b8f061b635abb9372e0d567d92513ca18c64096dc4bf05d5b81ad7650ab5"
