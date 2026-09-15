"""P-DET-COVERAGE-v1 supply over normalization-v3 + prose-decision-input-v1. SUPPLY ANALYSIS; NOT A DRAW.

Re-runs the supply measurement of the P-DET-COVERAGE-v1 draft (docs/research/study-002/30, section 7) on
the representation the classifier will actually receive, instead of the normalization-v1 proxy:

* the pool is every normalization-v3 record of a layer-B source that is **eligible** under the classifier
  input contract (tool-free, single exchange, not held-out, not malformed);
* the stratum predicates are the draft's frozen ones, loaded from ``scripts/audit_pdet_coverage_supply.py``
  itself so they are the same objects byte for byte (the proxy-only ``proxy_clean`` is **not** applied:
  normalization-v3 text is already the adapter's cleaned text).

Strata are sampling strata, never labels, and these counts are not gold. Nothing is ranked, selected or
copied; the output holds counts only. The draft's draw-time exclusions (section 9: QAD set, P-DET-v1
overlap, sentinel prompts) are not applied here and will reduce these pools slightly.

    python scripts/audit_pdet_coverage_supply_v3.py   # writes reports/pdet-coverage/pdet-coverage-v1.supply-v3.json
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from opengrad.data import versions
from opengrad.data.classifier_input import (
    HeldoutIndex,
    build_classifier_input,
    eligibility,
    normalize_prompt,
)
from opengrad.data.normalization_v3 import OUTPUT_DIR, iter_rows, load_source_manifest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / OUTPUT_DIR
PROXY_SCRIPT = ROOT / "scripts" / "audit_pdet_coverage_supply.py"
OUT = ROOT / "reports" / "pdet-coverage" / "pdet-coverage-v1.supply-v3.json"
QUOTAS = {"X": 60, "M": 60, "R": 60, "Q": 60, "P1": 60, "P2": 30}  # draft section 7.3 (X: at most)


def _load_predicates() -> Any:
    spec = importlib.util.spec_from_file_location("pdet_coverage_supply_proxy", PROXY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    predicates = _load_predicates()
    strata = predicates.STRATA
    heldout = HeldoutIndex.load(ROOT)
    manifest = load_source_manifest(ROOT)
    layer_b = [
        entry["name"]
        for entry in manifest["sources"]
        if str(entry["pdet_coverage_v1_sampling"]["layer_b_prose"]).startswith("eligible")
    ]
    sources: dict[str, Any] = {}
    for name in layer_b:
        eligible_total = 0
        pools = {stratum: Counter() for stratum in strata}
        responses: dict[str, set[str]] = {stratum: set() for stratum in strata}
        skeletons: dict[str, set[str]] = {stratum: set() for stratum in strata}
        prompts: dict[str, set[str]] = {stratum: set() for stratum in strata}
        for record in iter_rows(ARTIFACT, name):
            if not eligibility(record, heldout).eligible:
                continue
            eligible_total += 1
            features = build_classifier_input(record, heldout).features
            response, tools = features.assistant_response, [dict(tool) for tool in features.tools]
            stratum = predicates.stratum(response, tools)
            pools[stratum]["records"] += 1
            pools[stratum]["tools_offered" if tools else "no_tools"] += 1
            responses[stratum].add(re.sub(r"\s+", " ", response.casefold()).strip())
            skeletons[stratum].add(predicates.skeleton(response, tools))
            prompts[stratum].add(normalize_prompt(features.user_message))
        sources[name] = {
            "eligible_records": eligible_total,
            "strata": {
                stratum: {
                    "records": pools[stratum]["records"],
                    "tools_offered": pools[stratum]["tools_offered"],
                    "distinct_responses": len(responses[stratum]),
                    "distinct_skeletons": len(skeletons[stratum]),
                    "distinct_prompts": len(prompts[stratum]),
                }
                for stratum in strata
            },
        }
    coverage = {}
    for stratum in strata:
        supply = sum(sources[name]["strata"][stratum]["distinct_skeletons"] for name in layer_b)
        coverage[stratum] = {
            "quota": QUOTAS[stratum],
            "supply_distinct_skeletons_all_layer_b_sources": supply,
            "shortage": max(0, QUOTAS[stratum] - supply),
        }
    report = {
        "artifact_kind": "PDET_COVERAGE_SUPPLY_V3",
        "status": "SUPPLY_ANALYSIS_NOT_GOLD_NOT_A_DRAW",
        "statement": (
            "Eligible pools per preregistered stratum, over normalization-v3 under prose-decision-input-v1. "
            "Strata are sampling strata, not labels. Nothing was sampled. Draw-time exclusions of the draft's "
            "section 9 are not applied."
        ),
        "preregistration": "docs/research/study-002/30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md",
        "normalization_v3_fingerprint": json.loads(
            (ARTIFACT / "manifest.json").read_text(encoding="utf-8")
        )["fingerprint"],
        "classifier_input_contract": versions.CLASSIFIER_INPUT_CONTRACT_VERSION,
        "stratum_predicates": {
            "file": "scripts/audit_pdet_coverage_supply.py",
            "sha256_lf": hashlib.sha256(
                PROXY_SCRIPT.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest(),
            "priority": list(strata),
            "proxy_clean_applied": False,
        },
        "unit": "eligible normalization-v3 records (tool-free single exchanges under the contract)",
        "layer_b_sources": layer_b,
        "sources": sources,
        "coverage_against_quota": coverage,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    for name, data in sources.items():
        print(name, data["eligible_records"], {s: v["records"] for s, v in data["strata"].items()})
    print(
        {
            s: (v["supply_distinct_skeletons_all_layer_b_sources"], v["shortage"])
            for s, v in coverage.items()
        }
    )
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
