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

Two corpora can be audited, with one detector and one labelling rule:

  * default: the normalization-v1 source directories. This is a SUPERSET of what M0 trained on --
    it includes When2Call MCQ / LLM-judge evaluation rows, BUTTON and LoopTool -- and is kept only
    so the original audit stays reproducible.
  * --release-manifest + --shard-dir: a published release (ToolPolicy-Canonical-v2 final, the corpus
    M0 trained on). Every shard is checked against the manifest's `output_shards` sha256 and byte
    size before any record is read, and the audited record count must equal the manifest's.

Usage:
    python scripts/audit_sft_refusal_supervision.py
    python scripts/audit_sft_refusal_supervision.py \\
        --release-manifest .release/hf/toolpolicy-canonical-v2-final/release-manifest.json \\
        --shard-dir <directory holding the release's parquet shards>
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opengrad.evaluation.capability import detect_refusal

NORM = ROOT / "data/processed/normalization-v1"
OUT = ROOT / "results/benchmarks/h200/capability_v1/sft_refusal_supervision_audit.json"
OUT_RELEASE = (
    ROOT / "results/benchmarks/h200/capability_v1/sft_refusal_supervision_audit_canonical_v2.json"
)
OBSERVED = ROOT / "results/benchmarks/h200/capability_v1/refusal_characterization.json"
M0_RENDERING = ROOT / "runs/m0_sft_canonical_v2_final/rendering_report.json"

# normalization-v1 source directories. A superset of the canonical-v2 SFT corpus: it also holds
# evaluation rows (when2call-mcq, when2call-llm-judge) and BUTTON / LoopTool.
SOURCES = ["when2call-sft", "when2call-mcq", "when2call-llm-judge",
           "glaive", "toolace", "xlam", "button", "looptool"]

SHARD_NAME = re.compile(r"^(?P<source>.+)-(?P<index>\d{6})\.parquet$")


def normalise_template(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    t = re.sub(r"[^a-z ]", "", t)
    return " ".join(t.split()[:12])


def iter_parquet(paths):
    import pyarrow.parquet as pq

    for path in paths:
        table = pq.read_table(path)
        cols = table.to_pydict()
        n = table.num_rows
        for i in range(n):
            yield {k: cols[k][i] for k in cols}


def iter_records(source: str):
    yield from iter_parquet(sorted((NORM / source).glob("*.parquet")))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_release(manifest_path: Path, shard_dir: Path) -> tuple[dict, str, dict[str, list[Path]]]:
    """Verify every manifest shard in `shard_dir` and group the shards by release source.

    Fails closed: a missing shard, a size or sha256 mismatch, or a shard that maps to no declared
    source stops the audit before a single record is counted.
    """
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    declared = [s["source"] for s in manifest["sources"]]

    problems = []
    by_source: dict[str, list[Path]] = {s: [] for s in declared}
    for entry in manifest["output_shards"]:
        path = shard_dir / entry["file"]
        if not path.is_file():
            problems.append(f"missing shard {entry['file']}")
            continue
        if path.stat().st_size != entry["bytes"]:
            problems.append(f"size mismatch {entry['file']}: {path.stat().st_size} != {entry['bytes']}")
            continue
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            problems.append(f"sha256 mismatch {entry['file']}: {digest} != {entry['sha256']}")
            continue
        m = SHARD_NAME.match(entry["file"])
        if not m or m.group("source") not in by_source:
            problems.append(f"shard {entry['file']} maps to no declared source {declared}")
            continue
        by_source[m.group("source")].append(path)
    if problems:
        raise SystemExit("release verification FAILED:\n  " + "\n  ".join(problems))
    return manifest, manifest_sha256, {s: sorted(p) for s, p in by_source.items()}


def iter_release_source(source: str, paths: list[Path]):
    """Records of one release source; a record whose own source_dataset disagrees is an error."""
    for rec in iter_parquet(paths):
        if rec.get("source_dataset") != source:
            raise SystemExit(
                f"record {rec.get('opengrad_id')} in a {source!r} shard declares "
                f"source_dataset={rec.get('source_dataset')!r}"
            )
        yield rec


def audit(groups):
    """Run the refusal/label audit over (source, records-or-None) pairs. None marks ABSENT."""
    per_source = {}
    all_templates = collections.Counter()
    template_examples: dict[str, str] = {}
    label_breakdown = collections.Counter()
    multi_turn_refusals = collections.Counter()
    mislabelled_examples = []

    for source, records in groups:
        if records is None:
            per_source[source] = {"status": "ABSENT"}
            continue

        total = refusal_targets = 0
        by_decision = collections.Counter()
        refusal_by_decision = collections.Counter()
        tools_present_refusals = 0
        no_tools_refusals = 0

        for rec in records:
            total += 1
            try:
                messages = json.loads(rec["messages"]) if isinstance(rec["messages"], str) else rec["messages"]
                meta = json.loads(rec["metadata"]) if isinstance(rec["metadata"], str) else rec["metadata"]
                tools = json.loads(rec["tools"]) if isinstance(rec["tools"], str) else rec["tools"]
            except Exception:  # noqa: BLE001, S112 - an unparseable record is excluded from the audit
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
                    "id": rec["id"] if "id" in rec else rec.get("opengrad_id"),
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
        print(f"  {source:<26} {refusal_targets:>6}/{total:<7} refusal targets "
              f"({refusal_targets/total*100 if total else 0:.1f}%)  "
              f"labels={dict(refusal_by_decision)}", flush=True)

    return {
        "per_source": per_source,
        "all_templates": all_templates,
        "template_examples": template_examples,
        "label_breakdown": label_breakdown,
        "multi_turn_refusals": multi_turn_refusals,
        "mislabelled_examples": mislabelled_examples,
    }


def overlap_with_model(all_templates: collections.Counter) -> dict:
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
    return overlap


def build_payload(result: dict, corpus: str, extra: dict | None = None) -> dict:
    per_source = result["per_source"]
    label_breakdown = result["label_breakdown"]
    all_templates = result["all_templates"]
    template_examples = result["template_examples"]
    total_records = sum(v.get("records", 0) for v in per_source.values())
    total_refusals = sum(v.get("refusal_targets", 0) for v in per_source.values())

    payload = {
        "schema_version": 1,
        "purpose": "Test whether refusal text appears in the SFT supervision, and under which "
                   "decision labels.",
        "corpus": corpus,
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
            "counts_by_first_exchange_label": dict(result["multi_turn_refusals"]),
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
        "overlap_with_model_output": overlap_with_model(all_templates),
        "what_this_does_not_establish": (
            "Causation. These counts show refusal text is present in the supervision and that the "
            "model reproduces its surface forms. Establishing that these records CAUSED the "
            "behaviour requires the ablation: retrain M0 with the mislabelled records removed or "
            "relabelled, hold every other factor fixed, and re-measure GSM8K zero-shot refusal "
            "rate. That is a separate experiment with its own held-out evaluation."
        ),
        "scope_limit": "Descriptive audit. No checkpoint was modified and no data was filtered.",
    }
    payload.update(extra or {})
    return payload


def training_admission(manifest_sha256: str, per_source: dict) -> dict:
    """How the audited release relates to what M0's renderer actually admitted for training."""
    if not M0_RENDERING.exists():
        return {"status": "m0_rendering_report_unavailable"}
    report = json.loads(M0_RENDERING.read_text(encoding="utf-8"))
    if report.get("corpus", {}).get("manifest_sha256") != manifest_sha256:
        return {"status": "m0_rendering_report_is_for_a_different_corpus"}
    trainable = report.get("source_trainable", {})
    return {
        "status": "COMPARED",
        "m0_rendering_report": display(M0_RENDERING),
        "release_records": report.get("records_considered"),
        "m0_trainable_records": report.get("trainable_records"),
        "per_source": {
            s: {
                "release_records": v.get("records"),
                "m0_trainable_records": trainable.get(s),
                "refusal_targets_in_release": v.get("refusal_targets"),
                "count_applies_to_m0_training_exactly": (
                    trainable.get(s) == v.get("records") or v.get("refusal_targets") == 0
                ),
            }
            for s, v in per_source.items()
        },
        "note": (
            "Counts in this file are over every release record. M0's renderer quarantined records "
            "that fail the schema or trajectory contract, or whose target is truncated. Where a "
            "source was admitted in full the count is exactly what M0 was trained on; elsewhere it "
            "is an upper bound, because this audit does not re-run the renderer."
        ),
    }


def display(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--release-manifest", type=Path,
                    help="audit a published release: its release-manifest.json")
    ap.add_argument("--shard-dir", type=Path,
                    help="directory holding the release's parquet shards (verified against the manifest)")
    ap.add_argument("--out", type=Path, help="output path (default depends on the corpus audited)")
    args = ap.parse_args(argv)
    if (args.release_manifest is None) != (args.shard_dir is None):
        ap.error("--release-manifest and --shard-dir must be given together")

    if args.release_manifest is None:
        out = args.out or OUT
        groups = [(s, iter_records(s) if (NORM / s).exists() else None) for s in SOURCES]
        result = audit(groups)
        payload = build_payload(
            result,
            "normalization-v1 sources feeding ToolPolicy-Canonical-v2",
            {
                "superseded_by": {
                    "path": display(OUT_RELEASE),
                    "reason": "audited normalization-v1 sources, not the Canonical-v2 corpus M0 trained on",
                },
            },
        )
    else:
        out = args.out or OUT_RELEASE
        manifest, manifest_sha256, by_source = load_release(args.release_manifest, args.shard_dir)
        n_shards = sum(len(p) for p in by_source.values())
        print(f"verified {n_shards}/{len(manifest['output_shards'])} shards against "
              f"{display(args.release_manifest)} (sha256 {manifest_sha256[:12]})", flush=True)
        result = audit([(s, iter_release_source(s, p)) for s, p in by_source.items()])
        audited = sum(v.get("records", 0) for v in result["per_source"].values())
        if audited != manifest["record_count"]:
            raise SystemExit(f"audited {audited} records, manifest declares {manifest['record_count']}")
        payload = build_payload(
            result,
            f"{manifest['release_name']} {manifest['release_version']} -- the published "
            "ToolPolicy-Canonical-v2 final corpus M0 (m0_sft_canonical_v2_final) trained on",
            {
                "corpus_identity": {
                    "release_name": manifest["release_name"],
                    "release_version": manifest["release_version"],
                    "hub_repository": manifest["hub_repository"],
                    "release_manifest": display(args.release_manifest),
                    "release_manifest_sha256": manifest_sha256,
                    "release_opengrad_git_commit": manifest["opengrad_git_commit"],
                    "release_filters": manifest["release_filters"],
                    "manifest_record_count": manifest["record_count"],
                    "output_shards_verified": n_shards,
                    "shard_verification": (
                        "every shard's byte size and sha256 matched release-manifest.json "
                        "output_shards before any record was read"
                    ),
                    "per_source_shards": {s: len(p) for s, p in by_source.items()},
                },
                "m0_training_admission": training_admission(manifest_sha256, result["per_source"]),
                "supersedes": {
                    "path": display(OUT),
                    "reason": (
                        "That audit read the normalization-v1 source directories, a superset of "
                        "this corpus that also holds When2Call evaluation rows, BUTTON and "
                        "LoopTool. This file audits the release M0 trained on, with the same "
                        "detector and labelling rule."
                    ),
                },
            },
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    totals = payload["totals"]
    overlap = payload["overlap_with_model_output"]
    total_records, total_refusals = totals["records_audited"], totals["refusal_targets"]
    print(f"\nwrote {display(out)}")
    print(f"\n{total_refusals} refusal targets in {total_records} audited records "
          f"({total_refusals/total_records*100 if total_records else 0:.2f}%)")
    print(f"by decision label: {totals['refusal_targets_by_decision_label']}")
    if overlap.get("share_of_model_refusals") is not None:
        print(f"\ncorpus templates matching model output: "
              f"{overlap['also_present_verbatim_in_corpus']}/{overlap['model_top_templates']} "
              f"covering {overlap['share_of_model_refusals']*100:.1f}% of the model's refusals")
    if result["mislabelled_examples"]:
        print("\nrefusal text taught under a NON-refusal label (sample):")
        for e in result["mislabelled_examples"][:6]:
            print(f"  [{e['decision_label']}] tools={e['had_tools']}  {e['source']}")
            print(f"      user: {e['user'][:95]!r}")
            print(f"      tgt : {e['assistant_target'][:95]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
