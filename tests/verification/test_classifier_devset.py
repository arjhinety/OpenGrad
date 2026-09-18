"""The classifier development set (33 §3): synthetic records, plus a real-data overlap check."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opengrad.data.classifier_input import normalize_prompt
from opengrad.verification import classifier_devset as devset
from opengrad.verification import pdet_coverage as coverage

ROOT = Path(__file__).resolve().parents[2]


def unit(n: int, *, source: str = "glaive", response: str | None = None, prompt: str | None = None, tools=None) -> dict:
    # Distinct words, not numbers: the dedup skeleton masks digits, so number-only variants are one template.
    word = "".join(chr(97 + (n // 26**k) % 26) for k in range(3))
    response = response if response is not None else f"Plain answer about {word} here."
    tools = tools if tools is not None else [{"name": "get_weather"}]
    return {
        "pdetcov_id": f"pdetcov:{n:064x}",
        "source_name": source,
        "record_id": f"rec-{n}",
        "upstream_id": f"up-{n}",
        "raw_record_hash": f"raw-{n}",
        "canonical_hash": f"can-{n}",
        "layer": coverage.LAYER_B,
        "user_message": prompt if prompt is not None else f"Question {n}?",
        "assistant_response": response,
        "tools": tools,
        "stratum": coverage.stratum(response, tools),
    }


def test_every_kind_of_coverage_overlap_is_removed() -> None:
    units = [unit(1), unit(2), unit(3), unit(4)]
    exclusion = devset.CoverageExclusion(
        identities=frozenset({"raw-1"}),
        prompts=frozenset({normalize_prompt("Question 2?")}),
        responses=frozenset({normalize_prompt(units[2]["assistant_response"])}),
    )
    population, counts = devset.draw(units, coverage.DrawExclusions(), exclusion)
    assert counts["pdet_coverage_v1_overlap_removed"] == 3
    assert [u["raw_record_hash"] for u in population] == ["raw-4"]


def test_pdet_v1_overlap_is_removed_through_the_construction_exclusions() -> None:
    units = [unit(1), unit(2)]
    exclusions = coverage.DrawExclusions(pdet_prompts=frozenset({normalize_prompt("Question 1?")}))
    population, counts = devset.draw(units, exclusions, devset.CoverageExclusion())
    assert [u["raw_record_hash"] for u in population] == ["raw-2"]
    assert counts["construction_exclusions"]["primary"] == {coverage.PDET_V1_OVERLAP: 1}


def test_quota_is_split_across_sources_and_shortage_is_counted_not_backfilled() -> None:
    units = [unit(n, source="glaive") for n in range(70)] + [unit(100 + n, source="toolace") for n in range(10)]
    population, counts = devset.draw(units, coverage.DrawExclusions(), devset.CoverageExclusion())
    p1 = counts["strata"]["P1"]
    assert p1["quota"] == 50 and p1["realized"] == 50
    assert p1["per_source"] == {"glaive": {"supply": 70, "realized": 40}, "toolace": {"supply": 10, "realized": 10}}
    assert counts["strata"]["X"]["shortage"] == 50  # no textual-call units: reported, never backfilled
    assert len(population) == 50


def test_the_draw_is_deterministic_and_labels_start_empty() -> None:
    units = [unit(n, source=("glaive", "toolace", "when2call")[n % 3]) for n in range(90)]
    first, _ = devset.draw([dict(u) for u in units], coverage.DrawExclusions(), devset.CoverageExclusion())
    second, _ = devset.draw([dict(u) for u in reversed(units)], coverage.DrawExclusions(), devset.CoverageExclusion())
    assert devset.population_bytes(first) == devset.population_bytes(second)
    assert all(item[name] is None for item in first for name in devset.LABEL_FIELDS)
    assert [item["dev_index"] for item in first] == list(range(len(first)))


def test_duplicate_prompts_and_responses_keep_one() -> None:
    units = [unit(1, prompt="Same?"), unit(2, prompt="Same?"), unit(3, response="Same reply."), unit(4, response="Same reply.")]
    population, counts = devset.draw(units, coverage.DrawExclusions(), devset.CoverageExclusion())
    assert len(population) == 2
    assert counts["dedup"]["user_prompt"] == 1 and counts["dedup"]["response_text"] == 1


@pytest.mark.skipif(
    not (ROOT / devset.OUTPUT_DIR / devset.POPULATION_NAME).is_file(), reason="development set not built"
)
def test_the_written_set_shares_nothing_with_either_validation_population() -> None:
    items = [json.loads(line) for line in (ROOT / devset.OUTPUT_DIR / devset.POPULATION_NAME).read_text(encoding="utf-8").splitlines()]
    # The overlap checks read the pinned §9 exclusion inputs, which are git-ignored (33 §3), so a clean
    # checkout reports the absence instead of failing on it: the BLOCKED_INPUT_MISSING rule verify_population
    # applies, and the idiom test_pdet_coverage.py uses for the same predicate.
    missing = coverage.derivation_inputs_missing(ROOT)
    if missing:
        pytest.skip(f"BLOCKED_INPUT_MISSING: {missing}")
    coverage_exclusion = devset.load_coverage_exclusion(ROOT)
    pdet = coverage.load_draw_exclusions(ROOT)
    assert not any(coverage_exclusion.hits(item) for item in items)
    assert not any(coverage.PDET_V1_OVERLAP in pdet.hits(item) for item in items)
