"""Tests for the frozen P-DET validation population (Study 002 protocol ``pdet-002-v1``).

Two kinds of test:

* **unit** — the deterministic pieces (ranking, allocation, dedup, exclusions, boundary families) on
  synthetic records, so the sampling logic is checked without reading 15 parquet shards; and
* **freeze** — an end-to-end build into a temporary root, followed by tamper detection, so the verifier is
  proved to *fail* rather than merely proved to run. The real frozen artifacts under ``reports/pdet`` are
  never written to by these tests.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from opengrad.verification.accounting import PASS
from opengrad.verification.pdet import (
    CHALLENGE_SIZE,
    PREVALENCE_SIZE,
    apply_exclusions,
    challenge_families,
    deduplicate,
    family_quotas,
    keyed_rank,
    manifest_bytes,
    normalize_text,
    population_bytes,
    strata,
    verify_frozen,
    _allocate,
)

ROOT = Path(__file__).resolve().parents[2]


def _record(identifier: str, *, prompt: str = "What is the weather?", response: str = "It is sunny.", tools=None):
    return {
        "pdet_id": f"when2call-sft:{identifier}",
        "source_dataset": "when2call",
        "source_split": "train_sft",
        "source_revision": None,
        "upstream_id": f"og_{identifier[:16]}",
        "raw_record_hash": identifier,
        "canonical_hash": f"c_{identifier}",
        "prompt": prompt,
        "response": response,
        "tools": tools if tools is not None else [{"name": "get_weather"}],
        "shard": "shard-000000.parquet",
    }


# ── deterministic ranking ───────────────────────────────────────────────────────────────────────


def test_keyed_rank_is_deterministic_and_seed_dependent() -> None:
    assert keyed_rank("when2call-sft:abc") == keyed_rank("when2call-sft:abc")
    assert keyed_rank("when2call-sft:abc") != keyed_rank("when2call-sft:abd")
    assert len(keyed_rank("x")) == 64


def test_ranking_is_stable_when_input_order_changes() -> None:
    records = [_record(f"{index:04d}") for index in range(20)]
    forward = sorted(records, key=lambda item: keyed_rank(item["pdet_id"]), reverse=True)
    backward = sorted(list(reversed(records)), key=lambda item: keyed_rank(item["pdet_id"]), reverse=True)
    assert [r["pdet_id"] for r in forward] == [r["pdet_id"] for r in backward]


def test_normalize_text_collapses_whitespace_and_case() -> None:
    assert normalize_text("  Hello   World \n") == "hello world"


# ── allocation ──────────────────────────────────────────────────────────────────────────────────


def test_allocation_sums_to_the_target() -> None:
    counts = Counter({"a": 500, "b": 300, "c": 200})
    quotas = _allocate(counts, 100)
    assert sum(quotas.values()) == 100
    assert quotas["a"] > quotas["b"] > quotas["c"]


def test_allocation_is_deterministic() -> None:
    counts = Counter({"a": 7, "b": 7, "c": 7})
    assert _allocate(counts, 20) == _allocate(Counter(dict(reversed(list(counts.items())))), 20)


def test_family_quotas_sum_and_are_even() -> None:
    quotas = family_quotas(["a", "b", "c"], 10)
    assert sum(quotas.values()) == 10
    assert max(quotas.values()) - min(quotas.values()) <= 1


def test_family_quotas_of_nothing_is_empty() -> None:
    assert family_quotas([], 200) == {}


# ─ dedup and exclusions ────────────────────────────────────────────────────────────────────────


def test_dedup_by_raw_hash_and_by_response_text() -> None:
    records = [
        _record("aaaa", response="Same answer."),
        _record("bbbb", response="Same answer."),
        _record("cccc", response="Different answer."),
    ]
    survivors, stats = deduplicate(records)
    assert {r["raw_record_hash"] for r in survivors} == {"aaaa", "cccc"} or {
        r["raw_record_hash"] for r in survivors
    } == {"bbbb", "cccc"}
    assert stats["dropped_duplicate_response_text"] == 1
    assert stats["response_duplicate_groups"] == 1


def test_dedup_keeps_the_top_ranked_survivor() -> None:
    records = [_record("aaaa", response="Same."), _record("bbbb", response="Same.")]
    survivors, _ = deduplicate(records)
    expected = max(records, key=lambda item: keyed_rank(item["pdet_id"]))
    assert survivors[0]["pdet_id"] == expected["pdet_id"]


def test_dedup_records_the_group_for_reproducible_audit() -> None:
    records = [_record("aaaa", response="Same."), _record("bbbb", response="Same.")]
    survivors, _ = deduplicate(records)
    assert survivors[0]["response_duplicate_group_size"] == 2
    assert survivors[0]["response_duplicate_group"] == survivors[0]["pdet_id"]


def test_exclusions_drop_heldout_prompt_overlap() -> None:
    records = [_record("aaaa", prompt="Trending topics today?"), _record("bbbb", prompt="Other?")]
    kept, stats = apply_exclusions(
        records,
        {"evaluation_ids": set(), "quarantined_ids": set(), "heldout_prompts": {"trending topics today?"}},
    )
    assert [r["pdet_id"] for r in kept] == ["when2call-sft:bbbb"]
    assert stats["dropped_by_prompt_matching_heldout"] == 1


def test_exclusions_drop_evaluation_and_quarantined_ids() -> None:
    records = [_record("aaaa"), _record("bbbb"), _record("cccc")]
    kept, stats = apply_exclusions(
        records,
        {
            "evaluation_ids": {"og_aaaa"},
            "quarantined_ids": {"og_bbbb"},
            "heldout_prompts": set(),
        },
    )
    assert [r["pdet_id"] for r in kept] == ["when2call-sft:cccc"]
    assert stats["dropped_by_evaluation_id"] == 2


# ─ boundary families ───────────────────────────────────────────────────────────────────────────


def test_question_mark_family() -> None:
    assert "question_mark" in challenge_families(_record("a", response="Could you give the city?"))


def test_question_without_mark_family() -> None:
    """A clarification that is not closed by a question mark must still be selectable."""
    families = challenge_families(
        _record("a", response="Please provide the city name so I can check the forecast.")
    )
    assert "question_without_mark" in families
    assert "question_mark" not in families


def test_short_plain_and_question_families_are_mutually_informative() -> None:
    """A bare statement is a `short_plain` challenge, not a question family."""
    families = challenge_families(_record("a", response="The answer is 42."))
    assert "short_plain" in families
    assert "question_mark" not in families
    assert "question_without_mark" not in families


def test_refusal_families_split_on_the_question_mark_only_as_a_sampling_signal() -> None:
    refusal = "Apologies, but I'm unable to fetch live data. Which city did you mean?"
    assert "refusal_with_question" in challenge_families(_record("a", response=refusal))
    plain = "Apologies, but I'm unable to fetch live data."
    assert "refusal_plain" in challenge_families(_record("a", response=plain))


def test_tool_mentioned_without_a_payload() -> None:
    record = _record(
        "a", response="You could use get_weather for that.", tools=[{"name": "get_weather"}]
    )
    assert "tool_mentioned_no_payload" in challenge_families(record)


def test_serialized_call_shape_family() -> None:
    record = _record("a", response='{"name": "get_weather", "arguments": {}}')
    assert "serialized_call_shape" in challenge_families(record)


def test_caveat_then_content_family() -> None:
    record = _record("a", response="I can't know your exact location, but at latitude 40 it is sunny.")
    assert "caveat_then_content" in challenge_families(record)


def test_external_service_advice_family() -> None:
    record = _record(
        "a", response="I'm unable to fetch it. You may want to check a reliable news source."
    )
    assert "advice_external_service" in challenge_families(record)


def test_courtesy_followup_family() -> None:
    record = _record("a", response="The answer is 42. Does that help?")
    assert "polite_followup" in challenge_families(record)


def test_strata_are_observable_and_not_labels() -> None:
    record = _record("a", response="Apologies, but I'm unable to do that.")
    values = strata(record, Counter({normalize_text(record["response"]): 1}))
    assert set(values) == {"has_tools", "refusal_signal", "length", "response_repeated"}
    assert values["refusal_signal"] == "refusal_shaped"
    assert values["response_repeated"] == "unique"


# ── freeze and tamper detection ────────────────────────────────────────────────────────────────


@pytest.fixture()
def frozen_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a small synthetic P-DET freeze in a temp root; the real freeze is never written to."""
    import opengrad.verification.pdet as pdet

    population = [
        {**_record(f"{index:04d}", response=f"Distinct answer {index}."), "pdet_component": "prevalence"}
        for index in range(6)
    ]
    population.extend(
        {
            **_record(f"c{index:04d}", response="Apologies, but I'm unable to do that."),
            "pdet_component": "challenge",
        }
        for index in range(4)
    )
    for record in population:
        record.update({"pdet_index": 0, "gold_policy_label": None, "ambiguity_status": None})
    manifest = {
        "artifact_kind": "PDET_FROZEN_POPULATION",
        "protocol_version": pdet.PROTOCOL_VERSION,
        "sizes": {"prevalence_realized": 6, "challenge_realized": 4, "total_realized": 10},
        "exclusions": {"dropped_by_evaluation_id": 0, "dropped_by_prompt_matching_heldout": 1},
        "dedup": {"dropped_duplicate_raw_record_hash": 0, "dropped_duplicate_response_text": 0},
        "challenge_family_counts": {"refusal_plain": 4},
        "classifier_status_at_selection": pdet.CLASSIFIER_STATUS_AT_FREEZE,
        "selection_used_classifier": False,
        "gold_labels_present": False,
    }
    monkeypatch.setattr(pdet, "build_sample", lambda root: (population, manifest))
    monkeypatch.setattr(
        pdet,
        "load_exclusions",
        lambda root: {"evaluation_ids": set(), "quarantined_ids": set(), "heldout_prompts": set()},
    )
    pdet.write_frozen(tmp_path, population, manifest)
    return tmp_path


def test_verification_passes_on_an_untampered_freeze(frozen_root: Path) -> None:
    _result, summary = verify_frozen(frozen_root)
    assert summary["status"] == PASS
    assert summary["errors"] == []
    assert summary["frozen_items"] == 10


def test_verification_detects_a_tampered_population(frozen_root: Path) -> None:
    path = frozen_root / "reports/pdet/pdet-v1.population.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace("Distinct answer 0.", "A quietly edited answer.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _result, summary = verify_frozen(frozen_root)
    assert summary["status"] != PASS
    assert any(
        "FAIL_HASH" in error or "FAIL_REPRODUCIBILITY" in error for error in summary["errors"]
    )


def test_verification_detects_a_tampered_manifest(frozen_root: Path) -> None:
    path = frozen_root / "reports/pdet/pdet-v1.manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["classifier_status_at_selection"] = "IMPLEMENTED"
    path.write_bytes(manifest_bytes(manifest))
    _result, summary = verify_frozen(frozen_root)
    assert summary["status"] != PASS
    assert any("FAIL_METHOD" in error or "FAIL_HASH" in error for error in summary["errors"])


def test_verification_requires_the_manifest_to_agree_with_the_labels(frozen_root: Path) -> None:
    path = frozen_root / "reports/pdet/pdet-v1.population.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows[0]["gold_policy_label"] = "DIRECT"
    path.write_bytes(population_bytes(rows))
    _result, summary = verify_frozen(frozen_root)
    assert any("FAIL_ANNOTATION" in error for error in summary["errors"])


def test_verification_reports_missing_artifacts(tmp_path: Path) -> None:
    result, summary = verify_frozen(tmp_path)
    assert summary["status"] == "MISSING"
    assert result.errors and "FAIL_NONVACUOUS" in result.errors[0]


@pytest.mark.skipif(
    not (ROOT / "reports/pdet/pdet-v1.manifest.json").exists(),
    reason="frozen P-DET artifacts are not present in this checkout",
)
def test_the_real_frozen_population_verifies_and_has_no_labels() -> None:
    """The committed freeze must verify, must carry no labels, and must be reproducible."""
    _result, summary = verify_frozen(ROOT)
    assert summary["status"] == PASS, summary["errors"]
    assert summary["gold_labels_present"] is False
    assert summary["labelled_items"] == 0
    assert summary["prevalence"] == PREVALENCE_SIZE
    assert 0 < summary["challenge"] <= CHALLENGE_SIZE


@pytest.mark.skipif(
    not (ROOT / "reports/pdet/pdet-v1.population.jsonl").exists(),
    reason="frozen P-DET artifacts are not present in this checkout",
)
def test_frozen_population_carries_no_gold_labels_and_no_classifier_output() -> None:
    rows = [
        json.loads(line)
        for line in (ROOT / "reports/pdet/pdet-v1.population.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert all(record["gold_policy_label"] is None for record in rows)
    assert all(record["classifier_version_at_selection"] == "NOT_IMPLEMENTED" for record in rows)
    assert {record["pdet_component"] for record in rows} <= {"prevalence", "challenge"}