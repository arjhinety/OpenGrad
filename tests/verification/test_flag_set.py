"""The Study 002 flag set (46 §5): the committed artifact and the first-reply rules. Counts and hashes only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from opengrad.data.decision_classifier_v2 import UNSUPPORTED
from opengrad.hashing import sha256_bytes
from opengrad.verification import flag_set as fs

ROOT = Path(__file__).parents[2]


def _manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(
        (ROOT / fs.OUTPUT_DIR / fs.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    return manifest


def test_the_committed_flag_set_is_internally_consistent() -> None:
    manifest = _manifest()
    flags = manifest["flag_set"]
    assert sum(flags["by_source"].values()) == flags["n"] == sum(flags["by_unit_kind"].values())
    assert (
        manifest["answer_labelled_first_replies"]["eligible_by_predicted_label"]["UNSUPPORTED"]
        == flags["n"]
    )
    members = (ROOT / fs.OUTPUT_DIR / fs.MEMBERS_NAME).read_bytes()
    assert sha256_bytes(members) == manifest["members"]["sha256"]
    assert len(members.splitlines()) == manifest["members"]["rows"] == flags["n"]
    assert manifest["corpus"]["record_count"] == manifest["records_read"]


def test_the_recomputed_prior_is_study_001s_audit() -> None:
    # The v1 regex rerun here must reproduce the 18,114 of 03's prior, source by source; otherwise the corpus
    # or the first-reply reading differs from the audit the study was designed around.
    audit = json.loads(
        (
            ROOT
            / "results/benchmarks/h200/capability_v1/sft_refusal_supervision_audit_canonical_v2.json"
        ).read_text(encoding="utf-8")
    )
    prior = _manifest()["prior_heuristic_regex_v1"]
    assert prior["n"] == audit["totals"]["refusal_targets"]
    expected = {
        k: v["refusal_targets"] for k, v in audit["per_source"].items() if v["refusal_targets"]
    }
    assert prior["by_source"] == expected
    assert prior["also_in_flag_set"] <= prior["n"]


def test_the_builder_code_is_the_code_that_built_it() -> None:
    for path, digest in _manifest()["code_sha256_lf"].items():
        assert fs._lf_sha256(ROOT, path) == digest, path


def test_verify_passes_or_reports_its_input_missing() -> None:
    # Hashes only here; `--verify` also rebuilds from the release when it is in the local cache.
    result = fs.verify(ROOT, rebuild=False)
    assert result["errors"] == [] and result["status"] == "PASS"


def _record(
    messages: list[dict[str, Any]], decision: str = "ANSWER", tools: list[Any] | None = None
) -> dict[str, Any]:
    return {
        "messages": json.dumps(messages),
        "tools": json.dumps(tools or []),
        "behavior_decision": decision,
    }


def test_the_first_reply_follows_contract_v2s_shape() -> None:
    system = {"role": "system", "content": "s"}
    user, reply = {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}
    assert fs.first_reply([system, user, reply])[2:] == (None, fs.UNIT_SINGLE_EXCHANGE)
    later = [user, reply, {"role": "user", "content": "q2"}, {"role": "assistant", "content": "b"}]
    assert fs.first_reply(later)[2:] == (None, fs.UNIT_HAS_CONTINUATION)
    assert fs.first_reply([reply])[2] == fs.NOT_USER_THEN_ASSISTANT
    call = {"role": "assistant", "content": "", "tool_calls": [{"name": "x", "arguments": {}}]}
    assert fs.first_reply([user, call])[2] == fs.STRUCTURAL_CALL
    assert fs.first_reply([user, {"role": "assistant", "content": "  "}])[2] == fs.EMPTY_REPLY


def test_a_plain_decline_is_flagged_and_an_answer_is_not() -> None:
    tools = [{"name": "get_exchange_rate", "description": "d", "parameters": {"type": "object"}}]
    user = {"role": "user", "content": "Can you book a flight for me?"}
    decline = {
        "role": "assistant",
        "content": "I'm sorry, but I can't assist with that. My capabilities are limited to the functions provided.",
    }
    answer = {"role": "assistant", "content": "The capital of France is Paris."}
    assert fs.evaluate(_record([user, decline], tools=tools))["label"] == UNSUPPORTED
    assert fs.evaluate(_record([user, answer], tools=tools))["label"] != UNSUPPORTED


def test_the_detector_fingerprint_covers_everything_the_detector_reads() -> None:
    # The manifest pins detect_refusal, not all of capability.py. If the detector starts reading another
    # module-level name, the fingerprint would miss it: extend DETECTOR_GLOBALS and detector_fingerprint first.
    from opengrad.evaluation import capability

    attributes = {"strip", "search", "group"}  # str and re.Match methods, not module names
    assert set(capability.detect_refusal.__code__.co_names) == attributes | set(fs.DETECTOR_GLOBALS)
    assert fs.detector_fingerprint() == _manifest()["detector_sha256"]
    assert "src/opengrad/evaluation/capability.py" not in _manifest()["code_sha256_lf"]


def test_only_a_cache_miss_is_a_missing_input() -> None:
    assert fs._input_missing(ModuleNotFoundError("no pyarrow", name="pyarrow"))
    assert fs._input_missing(ModuleNotFoundError("no hub", name="huggingface_hub"))
    assert not fs._input_missing(FileNotFoundError("a pinned manifest is gone"))
    assert not fs._input_missing(ModuleNotFoundError("broken", name="opengrad.data.something"))
    assert not fs._input_missing(fs.FlagSetError("shard mismatch"))


def test_a_hub_cache_miss_is_a_missing_input() -> None:
    # huggingface_hub is not in the dev extra CI installs; locally it is.
    errors = pytest.importorskip("huggingface_hub.errors")
    assert fs._input_missing(errors.LocalEntryNotFoundError("not cached"))
