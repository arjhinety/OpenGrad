"""Regression tests for heterogeneous supervision contracts.

The contract exists so OpenGrad can train on datasets that supervise different portions of the
tool-use process without weakening validation. These tests pin both halves of that: the
relaxation is exactly as wide as one declared target, and every other rule still fails closed.

Numbering follows the requirement list in the task that introduced this contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opengrad.data.adapters import adapt_xlam
from opengrad.data.canonical import CanonicalSFTExample, ToolConversation
from opengrad.data.schema import effective_schema
from opengrad.data.semantic import terminal_call_ids, validate_training_trajectory
from opengrad.data.supervision import (
    CONTRACT_VERSION,
    CONTRACTS,
    SUPERVISION_METADATA_KEY,
    SupervisionAssignment,
    SupervisionKind,
    contract_for_kind,
    declared_kinds,
    resolve_contract,
    supervision_block,
    validate_supervision_block,
)

CALL_PREDICTION = SupervisionKind.CALL_PREDICTION
COMPLETE_TRAJECTORY = SupervisionKind.COMPLETE_TRAJECTORY


# --- fixtures -----------------------------------------------------------------------


def _tool(name: str = "lookup") -> dict:
    return {
        "name": name,
        "description": "look something up",
        "parameters": {
            "type": "object",
            "properties": {"q": {"type": "string"}, "limit": {"type": "integer"}},
        },
    }


def _conversation(
    messages: list[dict],
    *,
    kind: SupervisionKind | None,
    tools: list[dict] | None = None,
    assignment: SupervisionAssignment = SupervisionAssignment.SOURCE_ADAPTER,
) -> ToolConversation:
    metadata: dict = {"split": "train"}
    if kind is not None:
        metadata[SUPERVISION_METADATA_KEY] = supervision_block(
            kind, assignment=assignment, adapter="test_adapter", adapter_version="1.0.0"
        )
    return ToolConversation(
        id="t1",
        source="test",
        tools=tools if tools is not None else [_tool()],
        messages=messages,
        metadata=metadata,
    )


def _user(content: str = "find a thing") -> dict:
    return {"role": "user", "content": content}


def _call(call_id: str = "call_0", name: str = "lookup", args: dict | None = None) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call_id, "name": name, "arguments": args if args is not None else {"q": "x"}}
        ],
    }


def _result(call_id: str = "call_0", content: str = "42") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "name": "lookup", "content": content}


def _answer(content: str = "the answer is 42") -> dict:
    return {"role": "assistant", "content": content}


def _codes(example: ToolConversation, kind: SupervisionKind) -> list[str]:
    return [issue.code for issue in validate_training_trajectory(example, contract_for_kind(kind))]


# --- 1. valid terminal CALL_PREDICTION tool call ------------------------------------


def test_1_valid_terminal_call_prediction_is_accepted() -> None:
    example = _conversation([_user(), _call()], kind=CALL_PREDICTION)
    assert _codes(example, CALL_PREDICTION) == []
    CanonicalSFTExample(example).validate()


def test_1_call_prediction_accepts_resolved_intermediate_calls() -> None:
    """A resolved earlier call is context; only the final call is the target."""
    example = _conversation(
        [_user(), _call("call_0"), _result("call_0"), _call("call_1", args={"q": "y"})],
        kind=CALL_PREDICTION,
    )
    assert _codes(example, CALL_PREDICTION) == []


# --- 2. invalid tool name in CALL_PREDICTION ---------------------------------------


def test_2_invalid_tool_name_is_rejected_under_call_prediction() -> None:
    example = _conversation([_user(), _call(name="not_declared")], kind=CALL_PREDICTION)
    assert "UNDECLARED_TOOL" in _codes(example, CALL_PREDICTION)
    with pytest.raises(ValueError, match="SEM_UNDECLARED_TOOL"):
        CanonicalSFTExample(example).validate()


def test_2_invalid_tool_name_is_rejected_under_complete_trajectory() -> None:
    example = _conversation([_user(), _call(name="nope"), _answer()], kind=COMPLETE_TRAJECTORY)
    assert "UNDECLARED_TOOL" in _codes(example, COMPLETE_TRAJECTORY)


# --- 3. invalid arguments in CALL_PREDICTION ----------------------------------------


def test_3_invalid_arguments_are_rejected_under_call_prediction() -> None:
    """The relaxation covers the missing result only, never the arguments."""
    example = _conversation([_user(), _call(args={"q": 5})], kind=CALL_PREDICTION)
    assert "ARG_TYPE" in _codes(example, CALL_PREDICTION)
    with pytest.raises(ValueError, match="SEM_ARGUMENT_INVALID"):
        CanonicalSFTExample(example).validate()


def test_3_malformed_argument_json_is_rejected_under_call_prediction() -> None:
    call = _call()
    call["tool_calls"][0]["arguments"] = "{not json"
    example = _conversation([_user(), call], kind=CALL_PREDICTION)
    assert "MALFORMED_TOOL_ARGUMENTS" in _codes(example, CALL_PREDICTION)


# --- 4. no fabricated tool result ---------------------------------------------------


def test_4_adapter_does_not_fabricate_a_tool_result() -> None:
    record = {
        "id": "x1",
        "query": "what is the weather",
        "tools": [{"name": "weather", "parameters": {"city": {"type": "str"}}}],
        "answers": [{"name": "weather", "arguments": {"city": "Paris"}}],
    }
    conversation = adapt_xlam(record)
    assert [message["role"] for message in conversation.messages] == ["user", "assistant"]
    assert not any(message["role"] == "tool" for message in conversation.messages)
    assert conversation.messages[-1]["tool_calls"][0]["name"] == "weather"


def test_4_no_placeholder_result_is_injected_for_any_synthetic_reason() -> None:
    """A null/empty/'success' observation would change the learned task."""
    record = {
        "id": "x2",
        "query": "q",
        "tools": [{"name": "t", "parameters": {"a": {"type": "str"}}}],
        "answers": [{"name": "t", "arguments": {"a": "1"}}],
    }
    payload = json.dumps(conversation_to_dict(adapt_xlam(record)))
    for placeholder in ('"tool_result": null', '"content": "success"', '"tool_result": ""'):
        assert placeholder not in payload


def conversation_to_dict(conversation: ToolConversation) -> dict:
    return {
        "tools": conversation.tools,
        "messages": conversation.messages,
        "metadata": conversation.metadata,
    }


# --- 5. complete trajectory with matched tool result --------------------------------


def test_5_complete_trajectory_with_result_is_accepted() -> None:
    example = _conversation([_user(), _call(), _result(), _answer()], kind=COMPLETE_TRAJECTORY)
    assert _codes(example, COMPLETE_TRAJECTORY) == []
    CanonicalSFTExample(example).validate()


# --- 6. unresolved intermediate call -> reject --------------------------------------


def test_6_unresolved_intermediate_call_is_rejected() -> None:
    """An earlier call with no result stays invalid: the exemption is only for the target."""
    example = _conversation([_user(), _call("call_0"), _call("call_1")], kind=CALL_PREDICTION)
    codes = _codes(example, CALL_PREDICTION)
    assert "MISSING_TOOL_RESULT" in codes


def test_6b_unresolved_call_is_rejected_under_complete_trajectory() -> None:
    example = _conversation([_user(), _call(), _answer()], kind=COMPLETE_TRAJECTORY)
    assert "MISSING_TOOL_RESULT" in _codes(example, COMPLETE_TRAJECTORY)


# --- 7. multiple calls/results in a complete trajectory -----------------------------


def test_7_multiple_call_result_pairs_in_order_are_accepted() -> None:
    example = _conversation(
        [
            _user(),
            _call("call_0"),
            _result("call_0"),
            _call("call_1", args={"q": "y"}),
            _result("call_1", "43"),
            _answer("done"),
        ],
        kind=COMPLETE_TRAJECTORY,
    )
    assert _codes(example, COMPLETE_TRAJECTORY) == []


def test_7_fifo_order_violation_is_rejected() -> None:
    example = _conversation(
        [
            _user(),
            _call("call_0"),
            _call("call_1", args={"q": "y"}),
            _result("call_1"),
            _result("call_0"),
            _answer(),
        ],
        kind=COMPLETE_TRAJECTORY,
    )
    assert "INVALID_MESSAGE_SEQUENCE" in _codes(example, COMPLETE_TRAJECTORY)


# --- 8. incorrect call/result identity -> reject ------------------------------------


def test_8_orphan_tool_result_is_rejected() -> None:
    example = _conversation([_user(), _result("call_missing")], kind=COMPLETE_TRAJECTORY)
    assert "ORPHAN_TOOL_RESULT" in _codes(example, COMPLETE_TRAJECTORY)


def test_8_orphan_tool_result_is_rejected_under_call_prediction_too() -> None:
    """The relaxation is about a missing result, never about an invented one."""
    example = _conversation([_user(), _result("call_ghost"), _call()], kind=CALL_PREDICTION)
    assert "ORPHAN_TOOL_RESULT" in _codes(example, CALL_PREDICTION)


def test_8_result_name_mismatch_is_rejected() -> None:
    example = _conversation(
        [
            _user(),
            _call(),
            {"role": "tool", "tool_call_id": "call_0", "name": "other", "content": "x"},
        ],
        kind=COMPLETE_TRAJECTORY,
    )
    assert "TOOL_RESULT_NAME_MISMATCH" in _codes(example, COMPLETE_TRAJECTORY)


def test_8_duplicate_call_ids_are_rejected() -> None:
    example = _conversation([_user(), _call("call_0"), _call("call_0")], kind=CALL_PREDICTION)
    assert "DUPLICATE_TOOL_CALL_ID" in _codes(example, CALL_PREDICTION)


# --- the anti-pattern guard ---------------------------------------------------------


def test_orphaned_call_in_a_complete_trajectory_still_fails_exactly_as_before() -> None:
    """The architectural change must not turn arbitrary orphaned calls into valid data.

    A record that declares COMPLETE_TRAJECTORY, and a record that declares nothing at all (the
    legacy reading), must both reject an unresolved call.
    """
    declared = _conversation([_user(), _call()], kind=COMPLETE_TRAJECTORY)
    legacy = _conversation([_user(), _call()], kind=None)
    assert _codes(declared, COMPLETE_TRAJECTORY) == ["MISSING_TOOL_RESULT"]
    assert _codes(legacy, COMPLETE_TRAJECTORY) == ["MISSING_TOOL_RESULT"]
    # And the training boundary raises, rather than silently accepting either.
    for example in (declared, legacy):
        with pytest.raises(ValueError, match="SEM_UNRESOLVED_CALL"):
            CanonicalSFTExample(example).validate()


def test_call_prediction_without_a_terminal_call_is_rejected() -> None:
    """A prose-only record cannot claim the call-prediction contract."""
    example = _conversation([_user(), _answer()], kind=CALL_PREDICTION)
    assert "SUPERVISION_TARGET_MISSING" in _codes(example, CALL_PREDICTION)


def test_call_prediction_target_must_be_the_final_assistant_turn() -> None:
    """A trailing prose turn must not hide the fact that the call is not terminal."""
    example = _conversation([_user(), _call(), _answer()], kind=CALL_PREDICTION)
    assert "SUPERVISION_TARGET_MISSING" in _codes(example, CALL_PREDICTION)


def test_terminal_call_ids_reads_only_the_final_assistant_turn() -> None:
    assert terminal_call_ids([_user(), _call(), _answer()]) == set()
    assert terminal_call_ids([_user(), _call()]) == {"call_0"}


# --- 9. source-independent classification -------------------------------------------


def test_9_classification_does_not_depend_on_the_source_name() -> None:
    """A CALL_PREDICTION record from any source validates; the same shape as COMPLETE does not."""
    messages = [_user(), _call()]
    for source in ("xlam-function-calling-60k", "some-other-corpus", "synthetic"):
        example = ToolConversation(
            id="r",
            source=source,
            tools=[_tool()],
            messages=messages,
            metadata={
                "split": "train",
                SUPERVISION_METADATA_KEY: supervision_block(
                    CALL_PREDICTION,
                    assignment=SupervisionAssignment.SOURCE_ADAPTER,
                    adapter="a",
                    adapter_version="1",
                ),
            },
        )
        assert _codes(example, CALL_PREDICTION) == [], source
        assert "MISSING_TOOL_RESULT" in _codes(example, COMPLETE_TRAJECTORY), source


def test_9_no_source_name_conditional_exists_in_the_validator() -> None:
    source = Path("src/opengrad/data/semantic.py").read_text(encoding="utf-8")
    lowered = source.lower()
    for marker in ("xlam", "glaive", "toolace", "looptool", "when2call", "button"):
        assert marker not in lowered, f"validator branches on source name: {marker}"


# --- 10. unknown supervision kind -> reject/quarantine ------------------------------


def test_10_unknown_supervision_kind_is_rejected_not_defaulted() -> None:
    with pytest.raises(ValueError, match="unknown supervision kind"):
        contract_for_kind("SOME_FUTURE_KIND")
    example = _conversation([_user(), _call()], kind=None)
    example.metadata[SUPERVISION_METADATA_KEY] = {"kind": "SOME_FUTURE_KIND"}
    with pytest.raises(ValueError, match="unknown supervision kind"):
        CanonicalSFTExample(example).validate()


def test_10_missing_kind_inside_a_present_block_is_rejected() -> None:
    with pytest.raises(ValueError, match="kind is required"):
        validate_supervision_block({SUPERVISION_METADATA_KEY: {"assignment": "source_adapter"}})


def test_10_unknown_assignment_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown supervision assignment"):
        validate_supervision_block(
            {SUPERVISION_METADATA_KEY: {"kind": "CALL_PREDICTION", "assignment": "vibes"}}
        )


def test_10_absent_supervision_reads_as_the_legacy_default_only() -> None:
    """Absence is the stricter contract, reported as inherited rather than declared."""
    contract, assignment = resolve_contract({})
    assert contract.kind is COMPLETE_TRAJECTORY
    assert assignment == SupervisionAssignment.LEGACY_DEFAULT.value


def test_10_mismatched_validation_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="validation_policy"):
        validate_supervision_block(
            {
                SUPERVISION_METADATA_KEY: {
                    "kind": "CALL_PREDICTION",
                    "validation_policy": "complete_target_v1",
                }
            }
        )


# --- 11 & 12. deterministic rendering and loss masking ------------------------------


def _renderer_or_skip():  # type: ignore[no-untyped-def]
    from opengrad.data.renderers import Qwen35_2BRenderer

    renderer = Qwen35_2BRenderer()
    try:
        renderer._load()
    except Exception:  # noqa: BLE001 - tokenizer download is an environment property
        pytest.skip("tokenizer unavailable")
    return renderer


def test_11_rendering_is_deterministic_per_contract() -> None:
    renderer = _renderer_or_skip()
    example = _conversation([_user(), _call()], kind=CALL_PREDICTION)
    first = renderer.render_sft(example).text
    second = renderer.render_sft(example).text
    assert first == second
    assert "<|im_start|>tool" not in first


def test_12_loss_mask_covers_the_call_prediction_target() -> None:
    from opengrad.training.sft_data import build_sample

    renderer = _renderer_or_skip()
    example = _conversation([_user(), _call()], kind=CALL_PREDICTION)
    sample = build_sample(renderer, example, max_seq_length=2048)
    assert sample.status == "OK"
    assert sample.supervision_kind == "CALL_PREDICTION"
    assert sample.supervised, "the call-prediction target must carry loss"
    assert sum(sample.loss_mask()) == len(sample.supervised)
    # The supervised span is the assistant call turn, and it is the last span.
    start, end = sample.detail["target_span"]
    assert start < end
    assert all(start <= index < end for index in sample.supervised)


def test_12_loss_mask_is_deterministic() -> None:
    from opengrad.training.sft_data import build_sample

    renderer = _renderer_or_skip()
    example = _conversation([_user(), _call()], kind=CALL_PREDICTION)
    first = build_sample(renderer, example, max_seq_length=2048)
    second = build_sample(renderer, example, max_seq_length=2048)
    assert first.loss_mask() == second.loss_mask()
    assert first.supervised == second.supervised


def test_12_complete_trajectory_masks_every_assistant_turn_not_the_results() -> None:
    from opengrad.training.sft_data import build_sample

    renderer = _renderer_or_skip()
    example = _conversation(
        [_user(), _call(), _result("call_0", "42"), _answer("done")], kind=COMPLETE_TRAJECTORY
    )
    sample = build_sample(renderer, example, max_seq_length=2048)
    assert sample.status == "OK"
    assert sample.supervision_kind == "COMPLETE_TRAJECTORY"
    assert len(sample.detail["spans"]) == 2, "both assistant turns are supervised"


# --- 13. serialization round-trip ---------------------------------------------------


def test_13_supervision_metadata_survives_serialization() -> None:
    from opengrad.data.canonical import canonical_dict
    from opengrad.data.materialize import _storage_row
    from opengrad.data.real_analysis import _restore_row

    example = _conversation([_user(), _call()], kind=CALL_PREDICTION)
    stored = _storage_row(canonical_dict(example))
    restored = _restore_row(stored)
    block = restored["metadata"][SUPERVISION_METADATA_KEY]
    assert block["kind"] == "CALL_PREDICTION"
    assert block["assignment"] == SupervisionAssignment.SOURCE_ADAPTER.value
    assert block["contract_version"] == CONTRACT_VERSION
    assert block["adapter"] == "test_adapter"

    rebuilt = ToolConversation(
        restored["id"],
        restored["source"],
        restored["tools"],
        restored["messages"],
        restored["metadata"],
    )
    contract, assignment = resolve_contract(rebuilt.metadata)
    assert contract.kind is CALL_PREDICTION
    assert assignment == SupervisionAssignment.SOURCE_ADAPTER.value


def test_13_supervision_block_is_json_serializable() -> None:
    block = supervision_block(
        CALL_PREDICTION,
        assignment=SupervisionAssignment.UPSTREAM_DECLARED,
        adapter="a",
        adapter_version="1",
        note="n",
    )
    assert json.loads(json.dumps(block)) == block


# --- 17. already-valid datasets unchanged -------------------------------------------


def test_17_a_legacy_complete_trajectory_is_unaffected() -> None:
    """A record with no supervision block validates exactly as it did before."""
    example = _conversation([_user(), _call(), _result(), _answer()], kind=None)
    assert _codes(example, COMPLETE_TRAJECTORY) == []
    CanonicalSFTExample(example).validate()


def test_17_legacy_orphan_still_fails_with_the_same_code() -> None:
    example = _conversation([_user(), _call()], kind=None)
    with pytest.raises(ValueError, match="SEM_UNRESOLVED_CALL"):
        CanonicalSFTExample(example).validate()


# --- contract registry --------------------------------------------------------------


def test_every_declared_kind_has_a_contract() -> None:
    for kind in SupervisionKind:
        assert kind in CONTRACTS, kind
        assert contract_for_kind(kind).validation_policy


def test_contracts_differ_only_in_the_terminal_result_requirement() -> None:
    """The whole point: one rule differs, and it must be the declared one."""
    complete = contract_for_kind(COMPLETE_TRAJECTORY)
    call = contract_for_kind(CALL_PREDICTION)
    assert complete.tool_result_required_after_terminal_call is True
    assert call.tool_result_required_after_terminal_call is False
    assert complete.allow_intermediate_calls == call.allow_intermediate_calls
    assert complete.require_intermediate_results == call.require_intermediate_results
    assert complete.supervised_message_roles == call.supervised_message_roles


def test_contract_serializes_all_required_fields() -> None:
    payload = contract_for_kind(CALL_PREDICTION).as_dict()
    for field_name in (
        "kind",
        "terminal_target_type",
        "tool_result_required_after_terminal_call",
        "allow_intermediate_calls",
        "require_intermediate_results",
        "supervised_message_roles",
        "validation_policy",
    ):
        assert field_name in payload


def test_declared_kinds_are_stable() -> None:
    assert declared_kinds() == {"COMPLETE_TRAJECTORY", "CALL_PREDICTION"}


def test_xlam_adapter_declares_call_prediction_with_upstream_evidence() -> None:
    """The motivating case: classified from upstream semantics, not from a source-name rule."""
    record = {
        "id": "x3",
        "query": "q",
        "tools": [{"name": "t", "parameters": {"a": {"type": "str"}}}],
        "answers": [{"name": "t", "arguments": {"a": "1"}}],
    }
    conversation = adapt_xlam(record)
    contract, assignment = resolve_contract(conversation.metadata)
    assert contract.kind is CALL_PREDICTION
    assert assignment == SupervisionAssignment.UPSTREAM_DECLARED.value
    block = conversation.metadata[SUPERVISION_METADATA_KEY]
    assert block["adapter"] == "xlam_function_calling_60k_v2"
    assert "tool-result turn" in block["note"]


def test_tool_schemas_still_validate_after_the_contract_change() -> None:
    """Sanity: the schema layer is untouched by this work."""
    assert effective_schema(_tool())["type"] == "object"


# --- 16. optional filtering / ablation by supervision kind ---------------------------


def test_16_supervision_include_selects_kinds() -> None:
    from opengrad.experiments.schema import validate_supervision_selection

    assert validate_supervision_selection(None) == {}
    assert validate_supervision_selection({"include": ["CALL_PREDICTION"]}) == {
        "include": ["CALL_PREDICTION"]
    }
    assert validate_supervision_selection(
        {"include": ["CALL_PREDICTION", "COMPLETE_TRAJECTORY"]}
    ) == {"include": ["CALL_PREDICTION", "COMPLETE_TRAJECTORY"]}


@pytest.mark.parametrize(
    "block",
    [
        {},
        {"include": None},
        {"include": ["NOT_A_KIND"]},
        {"include": []},
        {"include": ["CALL_PREDICTION", "CALL_PREDICTION"]},
        {"unexpected_key": 1},
        "not-a-mapping",
    ],
)
def test_16_invalid_supervision_selection_is_rejected(block: object) -> None:
    from opengrad.experiments.schema import validate_supervision_selection

    with pytest.raises((ValueError, TypeError)):
        validate_supervision_selection(block)


def test_16_sampling_weights_are_rejected_rather_than_silently_ignored() -> None:
    """A no-op weight would change an experiment's meaning without changing its output."""
    from opengrad.experiments.schema import validate_supervision_selection

    with pytest.raises(ValueError, match="not implemented"):
        validate_supervision_selection({"sampling_weights": {"CALL_PREDICTION": 1.0}})
    with pytest.raises(ValueError, match="not implemented"):
        validate_supervision_selection({"sampling_weights": {}})
    with pytest.raises(ValueError, match="unknown kinds"):
        validate_supervision_selection({"sampling_weights": {"NOPE": 1.0}})


def test_16_supervision_selection_survives_the_experiment_config_round_trip() -> None:
    from opengrad.experiments.schema import ExperimentConfig

    config = ExperimentConfig.from_file("configs/experiments/qwen35_2b_m0_sft_v2corpus.yaml")
    assert config.supervision == {}
    payload = config.to_dict()
    payload["supervision"] = {"include": ["CALL_PREDICTION"]}
    parsed = ExperimentConfig.from_dict(payload)
    assert parsed.supervision == {"include": ["CALL_PREDICTION"]}
    assert parsed.to_dict()["supervision"] == {"include": ["CALL_PREDICTION"]}


def test_16_filtering_is_part_of_the_cache_identity() -> None:
    """Without this, a filtered run could reuse an unfiltered sample set."""
    from opengrad.training.preprocess import CacheIdentity

    base = {
        "model_id": "m",
        "model_revision": "r",
        "tokenizer_revision": "r",
        "renderer": "qwen3_5_2b_v1",
        "template_hash": "h",
        "max_seq_length": 2048,
        "corpus_manifest_sha256": "c",
    }
    unfiltered = CacheIdentity(**base)
    filtered = CacheIdentity(**base, supervision_include=("CALL_PREDICTION",))
    assert unfiltered.to_dict() != filtered.to_dict()
    assert unfiltered.to_dict()["supervision_include"] == []
    assert filtered.to_dict()["supervision_include"] == ["CALL_PREDICTION"]
    assert CacheIdentity.from_dict(filtered.to_dict()).supervision_include == ("CALL_PREDICTION",)


# --- 15. per-kind readiness statistics ----------------------------------------------


def test_15_readiness_reports_composition_by_kind(tmp_path: Path) -> None:
    from opengrad.readiness import _supervision_composition_state

    report = tmp_path / "yield.json"
    report.write_text(
        json.dumps(
            {
                "sources": [
                    {"source": "xlam", "supervision_kinds_trainable": {"CALL_PREDICTION": 40000}},
                    {
                        "source": "glaive",
                        "supervision_kinds_trainable": {"COMPLETE_TRAJECTORY": 97000},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    ok, detail, code = _supervision_composition_state(
        tmp_path, {"datasets": {"yield_report": "yield.json"}}
    )
    assert ok is True and code is None
    assert "CALL_PREDICTION" in detail and "COMPLETE_TRAJECTORY" in detail


def test_15_readiness_blocks_a_selection_the_corpus_cannot_satisfy(tmp_path: Path) -> None:
    from opengrad.readiness import _supervision_composition_state

    (tmp_path / "yield.json").write_text(
        json.dumps({"sources": [{"supervision_kinds_trainable": {"CALL_PREDICTION": 5}}]}),
        encoding="utf-8",
    )
    ok, _detail, code = _supervision_composition_state(
        tmp_path,
        {
            "datasets": {"yield_report": "yield.json"},
            "supervision": {"include": ["COMPLETE_TRAJECTORY"]},
        },
    )
    assert ok is False
    assert code == "SUPERVISION_SELECTION_MISMATCH"


def test_15_readiness_blocks_unclassified_trainable_records(tmp_path: Path) -> None:
    from opengrad.readiness import _supervision_composition_state

    (tmp_path / "yield.json").write_text(
        json.dumps({"sources": [{"supervision_kinds_trainable": {"UNCLASSIFIED": 7}}]}),
        encoding="utf-8",
    )
    ok, _detail, code = _supervision_composition_state(
        tmp_path, {"datasets": {"yield_report": "yield.json"}}
    )
    assert ok is False
    assert code == "SUPERVISION_UNCLASSIFIED"


def test_15_readiness_treats_zero_count_as_absent(tmp_path: Path) -> None:
    from opengrad.readiness import _supervision_composition_state

    (tmp_path / "yield.json").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source": "filtered",
                        "supervision_kinds_trainable": {"CALL_PREDICTION": 0},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    ok, _detail, code = _supervision_composition_state(
        tmp_path,
        {
            "datasets": {"yield_report": "yield.json"},
            "supervision": {"include": ["CALL_PREDICTION"]},
        },
    )
    assert ok is False
    assert code == "SUPERVISION_SELECTION_MISMATCH"


def test_15_readiness_fails_when_active_report_has_no_composition(tmp_path: Path) -> None:
    from opengrad.readiness import _supervision_composition_state

    (tmp_path / "yield.json").write_text(json.dumps({"sources": []}), encoding="utf-8")
    ok, _detail, code = _supervision_composition_state(
        tmp_path, {"datasets": {"yield_report": "yield.json"}}
    )
    assert ok is False
    assert code == "SUPERVISION_COMPOSITION_MISSING"


def test_15_minus_xlam_configs_select_only_the_contract_present_after_filtering() -> None:
    import yaml

    from opengrad.experiments.schema import ExperimentConfig
    from opengrad.readiness import _supervision_composition_state

    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "configs/experiments/m0_v2_final_minus_xlam_fixed_compute.yaml",
        root / "configs/experiments/m0_v2_final_minus_xlam_matched_exposure.yaml",
    ]
    for path in paths:
        config = ExperimentConfig.from_file(path)
        assert config.datasets["exclude_sources"] == ["xlam-function-calling-60k"]
        assert config.supervision == {"include": ["COMPLETE_TRAJECTORY"]}
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        ok, detail, code = _supervision_composition_state(root, raw)
        assert ok is True and code is None
        assert "'COMPLETE_TRAJECTORY': 105876" in detail
        assert "CALL_PREDICTION" not in detail


def test_15_minus_xlam_call_prediction_selection_is_a_hard_mismatch() -> None:
    import yaml

    from opengrad.readiness import _supervision_composition_state

    root = Path(__file__).resolve().parents[2]
    path = root / "configs/experiments/m0_v2_final_minus_xlam_fixed_compute.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["supervision"] = {"include": ["CALL_PREDICTION"]}
    ok, detail, code = _supervision_composition_state(root, raw)
    assert ok is False
    assert code == "SUPERVISION_SELECTION_MISMATCH"
    assert "CALL_PREDICTION" in detail


def test_16_supervision_composition_designs_are_separate_and_explicit() -> None:
    from opengrad.experiments.schema import ExperimentConfig

    root = Path(__file__).resolve().parents[2]
    expected = {
        "m0_v2_final_supervision_call_prediction_only.yaml": "CALL_PREDICTION",
        "m0_v2_final_supervision_complete_trajectory_only.yaml": "COMPLETE_TRAJECTORY",
    }
    for filename, kind in expected.items():
        config = ExperimentConfig.from_file(root / "configs/experiments" / filename)
        assert config.supervision == {"include": [kind]}
        assert config.datasets["exclude_sources"] == []
        assert "PLANNED, NOT LAUNCHED" in config.hypothesis
        assert "source-confounded" in config.hypothesis


def test_15_composition_gate_is_dormant_without_a_yield_report(tmp_path: Path) -> None:
    from opengrad.readiness import _supervision_composition_state

    ok, detail, code = _supervision_composition_state(tmp_path, {"datasets": {}})
    assert ok is True and code is None and "not measured" in detail
