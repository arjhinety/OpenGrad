"""v2 development sets of first replies (docs/research/study-002/37 §3): the pure draw, and the written set's hash."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from opengrad.verification import classifier_devset_v2 as devset_v2
from opengrad.verification import pdet_coverage as coverage
from tests.verification.test_pdet_coverage_v2 import unit

ROOT = Path(__file__).resolve().parents[2]


def test_used_items_are_never_drawn_and_quotas_split_across_sources() -> None:
    units = [unit(i, "Q", "glaive") for i in range(40)] + [unit(100 + i, "Q", "toolace") for i in range(40)]
    used = devset_v2.coverage_v2.UsedItems(prompts=frozenset({"question 0", "question 100"}))
    spec = devset_v2.SETS["dev"]
    population, counts = devset_v2.draw(units, coverage.DrawExclusions(), used, ["glaive", "toolace"], spec)
    assert counts["used_items_removed"] == 2
    assert not any(item["user_message"] in {"question 0", "question 100"} for item in population)
    assert counts["strata"]["Q"]["per_source"]["glaive"]["realized"] == 25
    assert counts["strata"]["Q"]["per_source"]["toolace"]["realized"] == 25
    assert counts["strata"]["R"]["shortage"] == spec.quota
    assert all(item["dev_id"].startswith("dev-v2:") and item["gold_policy_label"] is None for item in population)
    assert Counter(item["dev_index"] for item in population) == Counter(range(len(population)))


def test_the_written_development_set_matches_its_manifest() -> None:
    spec = devset_v2.SETS["dev"]
    manifest = json.loads((ROOT / devset_v2.OUTPUT_DIR / spec.manifest_name).read_text(encoding="utf-8"))
    data = (ROOT / devset_v2.OUTPUT_DIR / spec.population_name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == manifest["population_sha256"]
    assert manifest["counts"]["realized"] == 300
    assert manifest["status"] == "DEVELOPMENT_ONLY_NEVER_GOLD"
