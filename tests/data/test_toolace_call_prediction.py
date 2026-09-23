"""ToolACE call-final records read as next-call targets (``adapt_toolace_v3``, adapter version 2.2.0).

The semantic contract under test:

* a record of the validated shape declares ``CALL_PREDICTION`` and passes the trajectory gate;
* it says so as OpenGrad's inference (``source_adapter`` with a note), never as ``upstream_declared``;
* every other shape keeps ``COMPLETE_TRAJECTORY`` and is quarantined exactly as before;
* the messages are the v2 adapter's, and the adapters and release definitions Canonical-v2 was built
  from are unchanged.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from opengrad.data import adapters
from opengrad.data.adapters import (
    ADAPTERS,
    TOOLACE_CALL_PREDICTION_EVIDENCE,
    adapt_toolace,
    adapt_toolace_v2,
    adapt_toolace_v3,
)
from opengrad.data.canonical import canonical_dict
from opengrad.data.normalization_v3 import load_source_manifest, source_specs
from opengrad.data.semantic import validate_training_trajectory

ROOT = Path(__file__).resolve().parents[2]

SYSTEM = (
    "You are an expert in composing functions.\n"
    '[{"name": "get_weather", "description": "Get the weather", "parameters": {"type": "object", '
    '"properties": {"city": {"type": "string"}}}}, '
    '{"name": "get_time", "description": "Get the time", "parameters": {"type": "object", '
    '"properties": {"city": {"type": "string"}}}}]'
)
CALL = '[get_weather(city="Paris")]'
TWO_CALLS = '[get_weather(city="Paris"), get_time(city="Paris")]'
RESULT = '{"temperature": 21}'


def record(*turns: tuple[str, str]) -> dict[str, Any]:
    return {
        "system": SYSTEM,
        "conversations": [{"from": role, "value": value} for role, value in turns],
    }


def kind_and_issues(raw: dict[str, Any]) -> tuple[str, list[str]]:
    conversation = adapt_toolace_v3(raw, "train")
    issues = sorted({issue.code for issue in validate_training_trajectory(conversation)})
    return conversation.metadata["supervision"]["kind"], issues


# ── 1. the validated shapes ──────────────────────────────────────────────────────────────────────────


def test_single_user_then_call_is_a_call_prediction_target_that_passes_the_gate() -> None:
    raw = record(("user", "Weather in Paris?"), ("assistant", CALL))
    conversation = adapt_toolace_v3(raw, "train")
    block = conversation.metadata["supervision"]
    assert block["kind"] == "CALL_PREDICTION"
    assert block["validation_policy"] == "call_prediction_v1"
    assert validate_training_trajectory(conversation) == []
    assert conversation.messages[-1]["content"] is None
    assert [call["name"] for call in conversation.messages[-1]["tool_calls"]] == ["get_weather"]


def test_parallel_calls_in_the_final_turn_are_one_target() -> None:
    assert kind_and_issues(record(("user", "Weather and time?"), ("assistant", TWO_CALLS))) == (
        "CALL_PREDICTION",
        [],
    )


def test_multi_turn_with_every_earlier_call_answered_is_a_call_prediction_target() -> None:
    raw = record(
        ("user", "Weather in Paris?"),
        ("assistant", CALL),
        ("tool", RESULT),
        ("assistant", "It is 21 degrees."),
        ("user", "And the time?"),
        ("assistant", '[get_time(city="Paris")]'),
    )
    assert kind_and_issues(raw) == ("CALL_PREDICTION", [])


# ── 2. the inference is never presented as an upstream label ────────────────────────────────────────


def test_the_declaration_is_recorded_as_opengrads_inference_not_upstreams() -> None:
    block = adapt_toolace_v3(
        record(("user", "Weather in Paris?"), ("assistant", CALL)), "train"
    ).metadata["supervision"]
    assert block["assignment"] == "source_adapter"
    assert block["assignment"] != "upstream_declared"
    assert "not an upstream ToolACE annotation" in block["note"]
    assert TOOLACE_CALL_PREDICTION_EVIDENCE in block["note"]
    assert block["adapter"] == "toolace_v3"


def test_the_source_manifest_states_the_same_basis_as_the_code() -> None:
    manifest = load_source_manifest()
    specs = {spec.name: spec for spec in source_specs(manifest)}
    assert (specs["toolace"].adapter_key, specs["toolace"].row_label) == (
        "toolace_v3",
        "toolace_v3",
    )
    entry = next(item for item in manifest["sources"] if item["name"] == "toolace")
    supervision = entry["supervision"]
    assert supervision["assignment"] == "source_adapter"
    assert supervision["upstream_declared"] is False
    assert TOOLACE_CALL_PREDICTION_EVIDENCE in " ".join(supervision["basis"].split())
    assert (ROOT / TOOLACE_CALL_PREDICTION_EVIDENCE).is_file()


# ── 3. shapes outside the validated pattern stay quarantined ────────────────────────────────────────


def test_an_earlier_call_without_its_result_stays_a_quarantined_trajectory() -> None:
    raw = record(
        ("user", "Weather in Paris?"),
        ("assistant", CALL),
        ("assistant", "Checking."),
        ("user", "And the time?"),
        ("assistant", '[get_time(city="Paris")]'),
    )
    kind, issues = kind_and_issues(raw)
    assert kind == "COMPLETE_TRAJECTORY"
    assert "MISSING_TOOL_RESULT" in issues


def test_an_earlier_parallel_call_with_fewer_results_than_calls_stays_quarantined() -> None:
    raw = record(
        ("user", "Weather and time?"),
        ("assistant", TWO_CALLS),
        ("tool", RESULT),
        ("assistant", "Done."),
        ("user", "Once more?"),
        ("assistant", CALL),
    )
    kind, issues = kind_and_issues(raw)
    assert kind == "COMPLETE_TRAJECTORY"
    assert issues  # quarantined; the v2 build quarantines it for the same reasons


def test_a_final_call_after_a_tool_result_is_not_read_as_a_target() -> None:
    """A mid-execution call is where a lost result is most plausible, so the rule does not admit it."""
    raw = record(
        ("user", "Weather, then time?"),
        ("assistant", CALL),
        ("tool", RESULT),
        ("assistant", '[get_time(city="Paris")]'),
    )
    kind, issues = kind_and_issues(raw)
    assert kind == "COMPLETE_TRAJECTORY"
    assert issues == ["MISSING_TOOL_RESULT"]


def test_a_final_turn_with_prose_beside_the_call_is_not_read_as_a_target() -> None:
    raw = record(("user", "Weather in Paris?"), ("assistant", f"{CALL} I will check."))
    kind, issues = kind_and_issues(raw)
    assert kind == "COMPLETE_TRAJECTORY"
    assert issues == ["MISSING_TOOL_RESULT"]


def test_a_prose_final_record_keeps_the_trajectory_contract() -> None:
    assert kind_and_issues(record(("user", "Hello"), ("assistant", "Hi, how can I help?"))) == (
        "COMPLETE_TRAJECTORY",
        [],
    )
    answered = record(
        ("user", "Weather in Paris?"), ("assistant", CALL), ("tool", RESULT), ("assistant", "21.")
    )
    assert kind_and_issues(answered) == ("COMPLETE_TRAJECTORY", [])


def test_a_malformed_call_is_rejected_not_reinterpreted() -> None:
    with pytest.raises(ValueError):
        adapt_toolace_v3(
            record(("user", "Weather?"), ("assistant", "[get_weather(city=)]")), "train"
        )


def test_a_target_calling_an_undeclared_tool_is_rejected_not_reinterpreted() -> None:
    raw = record(("user", "Stock price?"), ("assistant", '[get_stock(symbol="X")]'))
    with pytest.raises(ValueError, match="unknown tool: get_stock"):
        adapt_toolace_v3(raw, "train")


# ── 4. determinism and the unchanged parts ──────────────────────────────────────────────────────────


def test_classification_is_deterministic_across_runs() -> None:
    raw = record(("user", "Weather in Paris?"), ("assistant", CALL))
    dumps = {
        json.dumps(
            canonical_dict(adapt_toolace_v3(json.loads(json.dumps(raw)), "train")), sort_keys=True
        )
        for _ in range(5)
    }
    assert len(dumps) == 1


@pytest.mark.parametrize(
    "turns",
    [
        (("user", "Weather in Paris?"), ("assistant", CALL)),
        (
            ("user", "Weather?"),
            ("assistant", CALL),
            ("tool", RESULT),
            ("assistant", '[get_time(city="P")]'),
        ),
        (("user", "Hello"), ("assistant", "Hi.")),
    ],
)
def test_v3_changes_only_the_declaration_never_the_messages_tools_or_identity(
    turns: tuple[tuple[str, str], ...],
) -> None:
    raw = record(*turns)
    old, new = adapt_toolace_v2(raw, "train"), adapt_toolace_v3(raw, "train")
    assert new.messages == old.messages
    assert new.tools == old.tools
    assert new.id == old.id
    assert new.metadata["raw_record_hash"] == old.metadata["raw_record_hash"]


def test_the_adapters_canonical_v2_was_built_from_still_declare_a_trajectory() -> None:
    raw = record(("user", "Weather in Paris?"), ("assistant", CALL))
    for adapter in (adapt_toolace, adapt_toolace_v2):
        conversation = adapter(raw, "train")
        assert conversation.metadata["supervision"]["kind"] == "COMPLETE_TRAJECTORY"
        assert "MISSING_TOOL_RESULT" in {
            issue.code for issue in validate_training_trajectory(conversation)
        }
    assert ADAPTERS["toolace"] is adapt_toolace
    assert ADAPTERS["toolace_v2"] is adapt_toolace_v2


#: sha256 of the LF source text at commit 009432b, before `adapt_toolace_v3` existed.
FROZEN_SOURCES = {
    "adapt_toolace": "c41358977717612637e951a677718b6d02785b4702e328070e7bf2c2d84123f1",
    "adapt_toolace_v2": "b7f8b85783c2b3c50a97c2ae8842babd8c3be39f8cd12f958810989ff23c90f6",
}
FROZEN_RELEASES = {
    "configs/releases/toolpolicy_canonical_v2.yaml": "ae2b1e63d5a8c0bb2185e1e4ea4caeb14a2fce80eef57bf83483043c4078cd31",
    "configs/releases/toolpolicy_canonical_v2_final.yaml": "b7692adde93ba56a0e6101955c53a99112214b45f68b74bf674122acafd523ea",
}


def test_the_older_toolace_adapters_are_byte_for_byte_unchanged() -> None:
    _items = list(FROZEN_SOURCES.items())
    assert _items, "nothing to check: an empty collection would pass this test vacuously"
    for name, digest in _items:
        source = inspect.getsource(getattr(adapters, name)).replace("\r\n", "\n")
        assert hashlib.sha256(source.encode("utf-8")).hexdigest() == digest, name


def test_the_canonical_v2_release_definitions_are_unchanged() -> None:
    _items = list(FROZEN_RELEASES.items())
    assert _items, "nothing to check: an empty collection would pass this test vacuously"
    for path, digest in _items:
        data = (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest() == digest, path
        assert "toolace_v3" not in data.decode("utf-8")


# ── 5. the evidence the rule rests on, regenerated from the pinned raw rows ─────────────────────────

RAW = ROOT / ".cache/normalization/raw/toolace_default_train.parquet"


@pytest.mark.skipif(not RAW.is_file(), reason="pinned raw ToolACE parquet is not cached")
def test_committed_evidence_regenerates_and_supports_the_rule() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "audit_toolace_call_final_shape", ROOT / "scripts/audit_toolace_call_final_shape.py"
    )
    assert spec and spec.loader
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    from opengrad.data.normalization_v3 import iter_raw

    committed = json.loads((ROOT / TOOLACE_CALL_PREDICTION_EVIDENCE).read_text(encoding="utf-8"))
    measured = audit.measure(list(iter_raw(RAW)))
    assert measured == committed["result"]
    endings = measured["row_endings"]
    assert set(endings) == {
        "assistant_call",
        "assistant_prose",
    }  # no row ends on a result or a user turn
    assert measured["rows_that_are_a_proper_prefix_of_another_row"] == 0
    assert measured["call_final_rows"]["with_an_earlier_call_lacking_a_result"] == 0
    yaml.safe_dump(committed)  # plain data only
