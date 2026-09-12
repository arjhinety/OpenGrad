#!/usr/bin/env python3
"""Audit the SFT corpus for refusal text taught as correct behaviour. CPU only.

The inference finding this tests: the post-SFT checkpoints refuse 100% of bare GSM8K questions with
"Apologies, but I'm unable to perform calculations...", while answering the same questions at 55%
when exemplars are present, and refusing 0% of MMLU-Pro. That is a learned surface form, not a
capability limit, and the natural place for it to have been learned is the supervision itself.

The audit asks three separate questions and keeps them separate:

  1. How many assistant targets in the corpus ARE refusals, by the same detector used at inference?
  2. What DECISION LABEL do those records carry? A refusal labelled ANSWER is a record that teaches
     "answering" looks like declining, which is a different and worse problem than a record
     labelled CANNOT_ANSWER.
  3. Do the corpus refusal templates MATCH the ones the model emits? Shared surface forms are what
     connect the two, and without that overlap the hypothesis fails.

The output is descriptive. It establishes that refusal text is present in the supervision and in
what proportion; it does NOT prove the corpus caused the behaviour, because no ablation was run.
The ablation that would establish causation is named in the output rather than assumed.

Usage:
    python scripts/audit_sft_refusal_supervision.py
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opengrad.evaluation.capability import detect_refusal  # noqa: E402

NORM = ROOT / "data/processed/normalization-v1"
OUT = ROOT / "results/benchmarks/h200/capability_v1/sft_refusal_supervision_audit.json"
OBSERVED = ROOT / "results/benchmarks/h200/capability_v1/refusal_characterization.json"

# Sources that feed the canonical-v2 SFT corpus.
SOURCES = ["when2call-sft", "when2call-mcq", "when2call-llm-judge",
           "glaive", "toolace", "xlam", "button", "looptool"]


def normalise_template(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    t = re.sub(r"[^a-z ]", "", t)
    return " ".join(t.split()[:12])


def iter_records(source: str):
    import pyarrow.parquet as pq

    for path in sorted((NORM / source).glob("*.parquet")):
        table = pq.read_table(path)
        cols = table.to_pydict()
        n = table.num_rows
        for i in range(n):
            yield {k: cols[k][i] for k in cols}


def main() -> int:
    per_source = {}
    all_templates = collections.Counter()
    template_examples: dict[str, str] = {}
    label_breakdown = collections.Counter()
    multi_turn_refusals = collections.Counter()
    mislabelled_examples = []

    for source in SOURCES:
        if not (NORM / source).exists():
            per_source[source] = {"status": "ABSENT"}
            continue

        total = refusal_targets = 0
        by_decision = collections.Counter()
        refusal_by_decision = collections.Counter()
        tools_present_refusals = 0
        no_tools_refusals = 0

        for rec in iter_records(source):
            total += 1
            try:
                messages = json.loads(rec["messages"]) if isinstance(rec["messages"], str) else rec["messages"]
                meta = json.loads(rec["metadata"]) if isinstance(rec["metadata"], str) else rec["metadata"]
                tools = json.loads(rec["tools"]) if isinstance(rec["tools"], str) else rec["tools"]
            except Exception:
                continue

            decision = ((meta or {}).get("behavior") or {}).get("decision")
            by_decision[decision] += 1

            assistant_turns = [m for m in (messages or [])
                               if m.get("role") == "assistant" and m.get("content")]
            if not assistant_turns:
                continue
            target = assistant_turns[-1].get("content")

            # A record's decision label describes the FIRST exchange. In a multi-turn dialogue the
            # last assistant turn answers a later, different user request -- glaive records exist
            # where the model correctly calls a tool, and three turns later declines an unrelated
            # follow-up ("can you also book a flight?"). Counting that as a mislabelled CALL is a
            # measurement error, not a corpus defect. Only single-exchange records can support a
            # mislabelling claim, so the two populations are counted separately and never summed.
            user_turns = sum(1 for m in messages if m.get("role") == "user")
            single_exchange = user_turns == 1 and len(assistant_turns) == 1

            verdict = detect_refusal(target)
            if not verdict.is_refusal:
                continue

            if not single_exchange:
                multi_turn_refusals[decision] += 1
                continue

            refusal_targets += 1
            refusal_by_decision[decision] += 1
            label_breakdown[decision] += 1
            if tools:
                tools_present_refusals += 1
            else:
                no_tools_refusals += 1

            key = normalise_template(target)
            all_templates[key] += 1
            template_examples.setdefault(key, target[:240])

            # The records that matter: single-exchange, refusal target, NON-refusal label.
            if decision in ("ANSWER", "CALL") and len(mislabelled_examples) < 25:
                user = next((m.get("content") for m in messages if m.get("role") == "user"), "")
                mislabelled_examples.append({
                    "source": source,
                    "id": rec.get("id"),
                    "decision_label": decision,
                    "had_tools": bool(tools),
                    "user": (user or "")[:200],
                    "assistant_target": target[:260],
                })

        per_source[source] = {
            "status": "AUDITED",
            "records": total,
            "refusal_targets": refusal_targets,
            "refusal_rate": refusal_targets / total if total else 0.0,
            "decision_distribution": dict(by_decision),
            "refusal_targets_by_decision_label": dict(refusal_by_decision),
            "refusals_with_tools_offered": tools_present_refusals,
            "refusals_with_no_tools_offered": no_tools_refusals,
            "scope": "single-exchange records only (1 user turn, 1 assistant turn)",
        }
        print(f"  {source:<22} {refusal_targets:>6}/{total:<7} refusal targets "
              f"({refusal_targets/total*100 if total else 0:.1f}%)  "
              f"labels={dict(refusal_by_decision)}", flush=True)

    total_records = sum(v.get("records", 0) for v in per_source.values())
    total_refusals = sum(v.get("refusal_targets", 0) for v in per_source.values())

    # -- overlap with what the model actually emits ---------------------------------------------
    overlap = {"status": "observed_refusals_unavailable"}
    if OBSERVED.exists():
        obs = json.loads(OBSERVED.read_text(encoding="utf-8"))
        cur = obs["stages"].get("M1_DPO_CURRENT", {}).get("refusal_surface_forms", {})
        obs_templates = {normalise_template(t["example"]): t["count"]
                         for t in cur.get("top_templates", [])}
        shared = {k: {"corpus_count": all_templates.get(k, 0), "model_count": v}
                  for k, v in obs_templates.items() if k in all_templates}
        obs_total = cur.get("total_refusals") or 0
        covered = sum(v["model_count"] for v in shared.values())
        overlap = {
            "model_top_templates": len(obs_templates),
            "also_present_verbatim_in_corpus": len(shared),
            "model_refusals_covered_by_shared_templates": covered,
            "share_of_model_refusals": (covered / obs_total) if obs_total else None,
            "shared_templates": shared,
            "note": ("Template match is on a normalised 12-token opening. Overlap shows the model "
                     "reproduces supervision surface forms; it does not by itself establish that "
                     "these records caused the behaviour."),
        }

    payload = {
        "schema_version": 1,
        "purpose": "Test whether refusal text appears in the SFT supervision, and under which "
                   "decision labels.",
        "corpus": "normalization-v1 sources feeding ToolPolicy-Canonical-v2",
        "refusal_detector": "HEURISTIC_REGEX_v1 -- the SAME detector used to score inference, so "
                            "corpus and model refusals are counted by one definition",
        "totals": {
            "records_audited": total_records,
            "refusal_targets": total_refusals,
            "refusal_target_rate": total_refusals / total_records if total_records else 0.0,
            "refusal_targets_by_decision_label": dict(label_breakdown),
        },
        "headline": (
            "Refusal text is taught under NON-refusal decision labels. Records labelled ANSWER "
            "whose supervised target is a refusal teach the model that answering looks like "
            "declining."
            if label_breakdown.get("ANSWER") else
            "No refusal targets were found under an ANSWER label."
        ),
        "multi_turn_records_excluded": {
            "counts_by_first_exchange_label": dict(multi_turn_refusals),
            "why_excluded": (
                "These are multi-turn dialogues whose LAST assistant turn is a refusal while the "
                "decision label describes the FIRST exchange. Inspected examples are legitimate: "
                "the model calls the tool correctly, then declines an unrelated later request. "
                "Counting them as mislabelled supervision would be a measurement error. They are "
                "reported here and deliberately NOT added to the mislabelled totals."
            ),
        },
        "per_source": per_source,
        "corpus_refusal_templates": {
            "distinct_12_token_openings": len(all_templates),
            "top_15": [
                {"count": n, "example": template_examples[k]}
                for k, n in all_templates.most_common(15)
            ],
        },
        "overlap_with_model_output": overlap,
        "what_this_does_not_establish": (
            "Causation. These counts show refusal text is present in the supervision and that the "
            "model reproduces its surface forms. Establishing that these records CAUSED the "
            "behaviour requires the ablation: retrain M0 with the mislabelled records removed or "
            "relabelled, hold every other factor fixed, and re-measure GSM8K zero-shot refusal "
            "rate. That is a separate experiment with its own held-out evaluation."
        ),
        "scope_limit": "Descriptive audit. No checkpoint was modified and no data was filtered.",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print(f"\n{total_refusals} refusal targets in {total_records} audited records "
          f"({total_refusals/total_records*100 if total_records else 0:.2f}%)")
    print(f"by decision label: {dict(label_breakdown)}")
    if overlap.get("share_of_model_refusals") is not None:
        print(f"\ncorpus templates matching model output: "
              f"{overlap['also_present_verbatim_in_corpus']}/{overlap['model_top_templates']} "
              f"covering {overlap['share_of_model_refusals']*100:.1f}% of the model's refusals")
    if mislabelled_examples:
        print("\nrefusal text taught under a NON-refusal label (sample):")
        for e in mislabelled_examples[:6]:
            print(f"  [{e['decision_label']}] tools={e['had_tools']}  {e['source']}")
            print(f"      user: {e['user'][:95]!r}")
            print(f"      tgt : {e['assistant_target'][:95]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
