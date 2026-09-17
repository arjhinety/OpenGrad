"""How rare is DIRECT across the whole normalization-v3 corpus? Counts only (35 §4, owner decision 2026-09-17).

The supply audit (35 §4) showed the *unused* pool cannot supply 50 DIRECT items. Before choosing between a new
DIRECT-bearing source and a redesign of C1, this audit estimates DIRECT across **every** normalization-v3
record, the artifact C1 would balance, per source and by whether tools were offered.

1. **Dispositions.** Every record of every source is sorted by the input contract (30 §8), with sampling
   roles ignored: held-out, malformed, structured call, post-tool response, multi-turn, or a prose single
   exchange (a layer B unit). Only layer B units can be DIRECT in the sense of 22 §2; a structured call is never
   DIRECT, and multi-turn or post-tool records are outside what any population measured.
2. **Label-yield projection.** Layer B record counts per stratum, source and tools offered, multiplied by the
   DIRECT share that existing labels show for that stratum and source, on the two bases of the supply audit
   (the P-DET-COVERAGE-v1 consensus reference; the development and check sets' Claude labels). It assumes the
   share does not depend on tools offered within a stratum, which P1 and P2 make exact for plain responses.
3. **Frozen classifier counts.** ``prose-decision-classifier-v1`` (tag, source hash checked) is run on every
   layer B unit and its outputs are counted. These are descriptive: its DIRECT precision on the coverage
   reference was 0.80 and it over-calls DIRECT on ToolACE (33 §8), so its DIRECT count is an upper-leaning
   estimate. Counts are given twice: over every layer B unit, and without units that match P-DET-v1,
   P-DET-COVERAGE-v1 or the other 30 §9 construction exclusions by identity, prompt or response. Matching by
   response removes every record that repeats a canned reply found in those sets, so the second count drops
   most of Glaive's refusal records; the first is the corpus picture.

Nothing prints or writes item text, ids, tools or rationales. Record counts are what training sees; distinct
prompts are shown beside them because Glaive repeats prompts heavily.

    python scripts/audit_corpus_direct_prevalence.py            # print
    python scripts/audit_corpus_direct_prevalence.py --write    # also write reports/pdet-coverage-v2/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_pdet_coverage_v2_supply as supply_audit

from opengrad.data.classifier_input import ClassifierFeatures, HeldoutIndex, normalize_prompt
from opengrad.data.decision_classifier import CLASSIFIER_VERSION, LABELS, classify
from opengrad.data.normalization_v3 import load_source_manifest
from opengrad.verification import classifier_devset as devset
from opengrad.verification import pdet_coverage as coverage
from opengrad.verification import prose_classifier_oneshot as oneshot

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("reports/pdet-coverage-v2/normalization-v3.direct-prevalence.json")


def _tools_key(unit: dict[str, Any]) -> str:
    return "tools_offered" if unit["tools"] else "no_tools"


def audit(root: Path = ROOT) -> dict[str, Any]:
    oneshot.check_frozen(root)
    input_record = coverage.check_input(root)
    sources = coverage.sampling_roles(load_source_manifest(root))
    roles = {source: {coverage.LAYER_A: True, coverage.LAYER_B: True} for source in sources}
    candidates = coverage.collect_candidates(coverage._iter_input(root, roles), HeldoutIndex.load(root), roles)
    layer_b = [unit for unit in candidates.units if unit["layer"] == coverage.LAYER_B]

    records: Counter[tuple[str, str, str]] = Counter()
    prompts: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for unit in layer_b:
        key = (unit["source_name"], _tools_key(unit), unit["stratum"])
        records[key] += 1
        prompts[key].add(normalize_prompt(unit["user_message"]))
    layer_b_table = {
        source: {
            tools: {
                name: {"records": records[(source, tools, name)], "distinct_prompts": len(prompts[(source, tools, name)])}
                for name in coverage.STRATA
                if records[(source, tools, name)]
            }
            for tools in ("tools_offered", "no_tools")
        }
        for source in sorted(sources)
    }

    observed = supply_audit.yields(root)
    projection: dict[str, Any] = {}
    for basis, table in observed.items():
        per_source: dict[str, Any] = {}
        for source in sorted(sources):
            totals: Counter[str] = Counter()
            for (unit_source, tools, name), count in records.items():
                if unit_source != source:
                    continue
                cell = table.get(name, {}).get(source)
                totals[f"layer_b_records_{tools}"] += count
                if not cell or not cell["items"]:
                    totals[f"records_without_yield_{tools}"] += count
                    continue
                totals[f"expected_direct_{tools}"] += count * cell["DIRECT"] / cell["items"]
            per_source[source] = {key: round(value, 1) for key, value in sorted(totals.items())}
        projection[basis] = per_source

    validation = devset.load_coverage_exclusion(root)
    draw_exclusions = coverage.load_draw_exclusions(root)
    predictions: Counter[tuple[str, str, str, str]] = Counter()
    skipped = 0
    for unit in layer_b:
        excluded = bool(validation.hits(unit) or draw_exclusions.hits(unit))
        skipped += excluded
        features = ClassifierFeatures(
            user_message=unit["user_message"],
            assistant_response=unit["assistant_response"],
            tools=tuple(unit["tools"]),
            structured_call_present=False,
        )
        label = classify(features).label
        predictions[("all_layer_b_units", unit["source_name"], _tools_key(unit), label)] += 1
        if not excluded:
            predictions[("without_exclusion_matches", unit["source_name"], _tools_key(unit), label)] += 1
    classifier_table = {
        scope: {
            source: {
                tools: {label: predictions[(scope, source, tools, label)] for label in LABELS}
                for tools in ("tools_offered", "no_tools")
            }
            for source in sorted(sources)
            if source in {s for _scope, s, _t, _l in predictions}
        }
        for scope in ("all_layer_b_units", "without_exclusion_matches")
    }

    return {
        "artifact_kind": "NORMALIZATION_V3_DIRECT_PREVALENCE_AUDIT",
        "authorization": "docs/research/study-002/35-OWNER-DECISIONS-AFTER-CLASSIFIER-V1-TEST.md §4 (owner decision 2026-09-17)",
        "status": "COUNTS ONLY. Descriptive estimates from model labels and a frozen classifier; not gold, not a draw.",
        "input": input_record,
        "dispositions_by_source": {source: dict(sorted(counts.items())) for source, counts in sorted(candidates.dispositions.items())},
        "layer_b_records_by_source_tools_and_stratum": layer_b_table,
        "label_counts_by_stratum_and_source": observed,
        "label_yield_projection": projection,
        "frozen_classifier": {
            "version": CLASSIFIER_VERSION,
            "source_sha256_lf": oneshot.FROZEN_SOURCE_SHA256_LF,
            "layer_b_units_matching_validation_or_construction_exclusions": skipped,
            "predictions_by_source_and_tools": classifier_table,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    text = json.dumps(audit(args.root), indent=2, sort_keys=True) + "\n"
    if args.write:
        path = args.root / OUTPUT
        if path.exists():
            raise SystemExit(f"{OUTPUT.as_posix()} already exists; it is never overwritten")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
