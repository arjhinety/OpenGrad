"""The held-out development check set (33 §5a): disjoint from the development set and both validation sets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opengrad.data.classifier_input import normalize_prompt
from opengrad.verification import classifier_devcheck as devcheck
from opengrad.verification import classifier_devset as devset
from opengrad.verification import pdet_coverage as coverage

ROOT = Path(__file__).resolve().parents[2]


def unit(n: int, *, source: str = "glaive", prompt: str | None = None) -> dict:
    word = "".join(chr(97 + (n // 26**k) % 26) for k in range(3))
    response = f"Plain answer about {word} here."
    tools = [{"name": "get_weather"}]
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


def test_development_set_items_are_removed_by_identity_prompt_and_response() -> None:
    units = [unit(1), unit(2), unit(3), unit(4)]
    development = devset.CoverageExclusion(
        identities=frozenset({"raw-1"}),
        prompts=frozenset({normalize_prompt("Question 2?")}),
        responses=frozenset({normalize_prompt(units[2]["assistant_response"])}),
    )
    population, counts = devcheck.draw(units, coverage.DrawExclusions(), devset.CoverageExclusion(), development)
    assert counts["devset_overlap_removed"] == 3
    assert [u["raw_record_hash"] for u in population] == ["raw-4"]
    assert population[0]["dev_id"].startswith("devcheck:")


def test_quota_is_25_per_stratum_and_the_seed_differs_from_the_development_set() -> None:
    units = [unit(n, source=("glaive", "toolace")[n % 2]) for n in range(80)]
    population, counts = devcheck.draw(units, coverage.DrawExclusions(), devset.CoverageExclusion(), devset.CoverageExclusion())
    assert counts["strata"]["P1"]["realized"] == 25 and len(population) == 25
    assert devcheck.SEED != devset.SEED
    assert devcheck.rank_key("P1", "x") != devset.rank_key("P1", "x")


def test_the_second_check_set_excludes_both_earlier_sets_and_has_its_own_seed() -> None:
    v1, v2 = devcheck.CHECKS["v1"], devcheck.CHECKS["v2"]
    assert {path.name for path, _sha in v2.excludes} == {devset.POPULATION_NAME, v1.population_name}
    assert v2.seed not in {v1.seed, devset.SEED} and v2.id_prefix != v1.id_prefix
    units = [unit(n) for n in range(40)]
    first, _ = devcheck.draw(units, coverage.DrawExclusions(), devset.CoverageExclusion(), devset.CoverageExclusion(), v1)
    second, _ = devcheck.draw(units, coverage.DrawExclusions(), devset.CoverageExclusion(), devset.CoverageExclusion(), v2)
    assert all(item["dev_id"].startswith("devcheck-v2:") for item in second)
    assert [u["pdetcov_id"] for u in first] != [u["pdetcov_id"] for u in second]  # different seeds, different order


@pytest.mark.skipif(
    not (ROOT / devcheck.OUTPUT_DIR / devcheck.CHECKS["v2"].population_name).is_file(), reason="check set v2 not built"
)
def test_the_written_second_check_set_shares_nothing_with_any_earlier_set() -> None:
    v2 = devcheck.CHECKS["v2"]
    items = [
        json.loads(line)
        for line in (ROOT / devcheck.OUTPUT_DIR / v2.population_name).read_text(encoding="utf-8").splitlines()
    ]
    earlier = devcheck.load_devset_exclusion(ROOT, v2)
    assert not any(earlier.hits(item) for item in items)
    assert not any(devset.load_coverage_exclusion(ROOT).hits(item) for item in items)


@pytest.mark.skipif(
    not (ROOT / devcheck.OUTPUT_DIR / devcheck.POPULATION_NAME).is_file(), reason="check set not built"
)
def test_the_written_check_set_shares_nothing_with_the_development_or_validation_sets() -> None:
    items = [
        json.loads(line)
        for line in (ROOT / devcheck.OUTPUT_DIR / devcheck.POPULATION_NAME).read_text(encoding="utf-8").splitlines()
    ]
    development = devcheck.load_devset_exclusion(ROOT)
    coverage_exclusion = devset.load_coverage_exclusion(ROOT)
    pdet = coverage.load_draw_exclusions(ROOT)
    assert len(items) == 125
    assert not any(development.hits(item) for item in items)
    assert not any(coverage_exclusion.hits(item) for item in items)
    assert not any(coverage.PDET_V1_OVERLAP in pdet.hits(item) for item in items)
