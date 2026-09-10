from opengrad.benchmarks.backends.mock import DeterministicFakeBackend
from opengrad.formatting.parser import parse_qwen_native_output


def test_deterministic_fake_backend_scenarios() -> None:
    backend = DeterministicFakeBackend()

    # 1. Valid tool call
    res_call = backend.generate(
        "Look up weather in Tokyo",
        tools=[{"name": "get_weather"}],
        metadata={"expected_decision": "CALL"},
    )
    assert "<tool_call>" in res_call.text
    parsed_call = parse_qwen_native_output(res_call.text)
    assert parsed_call.status == "RAW_VALID"
    assert parsed_call.decision == "CALL"

    # 2. Clarification
    res_clarify = backend.generate(
        "Clarify ambiguous account",
        metadata={"expected_decision": "CLARIFY"},
    )
    assert "clarify" in res_clarify.text.lower()
    parsed_clarify = parse_qwen_native_output(res_clarify.text)
    assert parsed_clarify.decision == "CLARIFY"

    # 3. Refusal
    res_refusal = backend.generate(
        "Perform unauthorized exploit",
        metadata={"expected_decision": "UNSUPPORTED"},
    )
    assert "cannot" in res_refusal.text.lower()
    parsed_refusal = parse_qwen_native_output(res_refusal.text)
    assert parsed_refusal.decision == "UNSUPPORTED"

    # 4. Explicit scenario: malformed JSON
    backend_malformed = DeterministicFakeBackend(scenarios={"malformed_task": "malformed_json"})
    res_m = backend_malformed.generate("query", metadata={"task_id": "malformed_task"})
    parsed_m = parse_qwen_native_output(res_m.text)
    assert parsed_m.status == "FORMAT_ERROR"

    # 5. Explicit scenario: truncated
    backend_trunc = DeterministicFakeBackend(scenarios={"t_task": "truncated"})
    res_t = backend_trunc.generate("query", metadata={"task_id": "t_task"})
    assert res_t.text.endswith('query": ')


def test_speculative_telemetry_in_fake_backend() -> None:
    backend = DeterministicFakeBackend(mode="mtp", speedup_factor=1.75)
    res = backend.generate(
        "Generate a function",
        metadata={"speculation_depth": 3, "expected_decision": "DIRECT"},
    )
    assert res.speculative_metadata is not None
    spec = res.speculative_metadata
    assert spec["speculation_mode"] == "mtp"
    assert spec["speculation_depth_requested"] == 3
    assert spec["proposed_tokens"] > 0
    assert spec["accepted_tokens"] > 0
    assert spec["acceptance_rate"] > 0
    assert spec["accepted_tokens_per_step"] >= 1.0
    assert "depth_1" in spec["mtp_per_depth_acceptance"]
