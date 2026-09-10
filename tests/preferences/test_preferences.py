from pathlib import Path
from typing import Any

import pytest

from opengrad.benchmarks.backends.mock import DeterministicFakeBackend
from opengrad.preferences.deterministic_judge import DeterministicJudge
from opengrad.preferences.generator import SyntheticPreferenceGenerator
from opengrad.preferences.openai_judge import JudgeBudget, OpenAIJudge
from opengrad.preferences.schema import PreferenceCandidate, PreferencePair


def test_preference_pair_validation_passes_valid() -> None:
    pair = PreferencePair(
        prompt_id="p1",
        canonical_id="c1",
        prompt="User query",
        chosen="Good tool call",
        rejected="Bad answer",
        preference_source="deterministic",
    )
    pair.validate()
    d = pair.to_dict()
    assert d["chosen"] == "Good tool call"
    assert d["rejected"] == "Bad answer"


def test_preference_pair_validation_rejects_identical() -> None:
    pair = PreferencePair(
        prompt_id="p2",
        canonical_id="c2",
        prompt="User query",
        chosen="Same text",
        rejected="Same text",
        preference_source="deterministic",
    )
    with pytest.raises(ValueError, match="chosen equals rejected"):
        pair.validate()


def test_preference_pair_validation_rejects_empty() -> None:
    pair = PreferencePair(
        prompt_id="p3",
        canonical_id="c3",
        prompt="User query",
        chosen="Valid chosen",
        rejected="   ",
        preference_source="deterministic",
    )
    with pytest.raises(ValueError, match="rejected response is empty"):
        pair.validate()


def test_deterministic_judge_scoring() -> None:
    judge = DeterministicJudge()

    # Tool call when expected
    valid_call = '<tool_call>{"name": "lookup", "arguments": {"q": "test"}}</tool_call>'
    score_call, reasons_call = judge.score_candidate(
        valid_call, expected_decision="CALL", expected_tool="lookup", available_tools=["lookup"]
    )
    assert score_call > 0
    assert "CORRECT_TOOL_SELECTION" in reasons_call
    assert "GROUNDED_ARGUMENTS" in reasons_call

    # Missed tool
    plain_answer = "I don't know the answer."
    score_miss, reasons_miss = judge.score_candidate(plain_answer, expected_decision="CALL")
    assert "MISSED_TOOL" in reasons_miss
    assert score_miss < score_call

    # Pair formation
    cand_a = PreferenceCandidate("c_a", valid_call)
    cand_b = PreferenceCandidate("c_b", plain_answer)
    pair = judge.judge_pair(cand_a, cand_b, "prompt", "prompt_id", expected_decision="CALL", expected_tool="lookup")
    assert pair is not None
    assert pair.chosen == valid_call
    assert pair.rejected == plain_answer
    assert pair.preference_source == "deterministic"


def test_openai_judge_budget_and_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The mock adjudication path is only selected without a key; control the
    # environment explicitly so an ambient OPENAI_API_KEY cannot turn this CPU
    # test into a live network call.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    budget = JudgeBudget(max_requests=2, max_cost_usd=0.05)
    judge = OpenAIJudge(budget=budget, cache_dir=tmp_path)

    cand_a = PreferenceCandidate("a", '<tool_call>{"name": "lookup", "arguments": {}}</tool_call>')
    cand_b = PreferenceCandidate("b", "No tool")

    # In mock mode without OPENAI_API_KEY
    pair = judge.judge_pair(cand_a, cand_b, "Search for x", "p_1")
    assert pair is not None
    assert pair.chosen == cand_a.response_text
    assert pair.rejected == cand_b.response_text

    # Verify budget exhaustion
    budget.requests_made = 2
    assert budget.is_exceeded() is True
    pair_blocked = judge.judge_pair(cand_a, cand_b, "Search for y", "p_2")
    assert pair_blocked is None


def test_synthetic_preference_generator(tmp_path: Path) -> None:
    backend = DeterministicFakeBackend()
    generator = SyntheticPreferenceGenerator(backend)
    out_file = tmp_path / "pairs.jsonl"

    prompts: list[dict[str, Any]] = [
        {"id": "p_01", "prompt": "Search order status", "expected_decision": "CALL", "tools": [{"name": "lookup"}]},
        {"id": "p_02", "prompt": "Direct answer question", "expected_decision": "ANSWER", "tools": []},
    ]

    summary = generator.generate_pairs(prompts, out_file, num_candidates_per_prompt=3)
    assert summary.total_prompts == 2
    assert summary.pairs_generated >= 1
    assert out_file.exists()
