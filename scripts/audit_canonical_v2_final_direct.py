"""DIRECT in Study 001's training corpus, canonical-v2-final. Counts only (35 §5, owner decision 2026-09-17).

35 §5 found almost no direct answers with tools offered in normalization-v3 and left one hypothesis open: that the
corpus Study 001 actually trained on lacks them too. M0 trained on the ``canonical-v2-final`` release
(``runs/m0_sft_canonical_v2_final/dataset_manifest.json``: 173,237 records, manifest sha256 ``8ced403b…``). This
audit reads that release from the Hugging Face cache, checks every shard against the repository's copy of its
release manifest, and counts.

SFT supervises **every** assistant turn (``src/opengrad/training/sft_data.py``), so turns are counted, not only
first exchanges. Each assistant turn is one of:

* ``call``: it carries structured tool calls;
* ``after_tool_result``: prose directly after a tool result (a summary of a result, outside 22 §2's DIRECT);
* ``first_exchange``: prose as the first assistant turn after exactly one user turn, the shape every P-DET
  population measures;
* ``later_turn``: any other prose turn answering a user message (multi-turn), which no population measured.

Prose turns are split by whether the record offers tools. Two estimates of DIRECT:

1. **Label yields** (first exchanges only): the stratum of 30 §7.2 times the DIRECT share existing labels show for
   that stratum and source (the two bases of ``audit_pdet_coverage_v2_supply.py``). canonical-v2 source names are
   mapped to normalization-v3's.
2. **Frozen classifier v1** on every prose turn, with the nearest preceding user message as input. For later turns
   that input drops the conversation, so those counts are rougher. Its known false-DIRECT behaviour on ToolACE and
   When2Call (33 §8, 35 §5) applies.

Trainability: M0 dropped 11,271 of 173,237 records (context overflow of supervised tokens and the like); the
per-source trainable counts from its dataset manifest are reported beside the counts, which cover all records.
Nothing prints or writes item text, ids, tools or rationales.

    python scripts/audit_canonical_v2_final_direct.py            # print
    python scripts/audit_canonical_v2_final_direct.py --write    # also write reports/pdet-coverage-v2/
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

from opengrad.data.classifier_input import ClassifierFeatures, normalize_prompt
from opengrad.data.decision_classifier import CLASSIFIER_VERSION, LABELS, classify
from opengrad.hashing import sha256_bytes as _sha256
from opengrad.verification import pdet_coverage as coverage
from opengrad.verification import prose_classifier_oneshot as oneshot

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("reports/pdet-coverage-v2/canonical-v2-final.direct-prevalence.json")
RELEASE_MANIFEST = Path(".release/hf/toolpolicy-canonical-v2-final/release-manifest.json")
DATASET_MANIFEST = Path("runs/m0_sft_canonical_v2_final/dataset_manifest.json")
REPOSITORY = "arrochi112/OpenGrad-ToolPolicy-Canonical-v2"
REVISION = "df1a1f5135cc4b08e24b7ff20582084988932c61"
SOURCE_NAMES = {
    "glaive-function-calling-v2": "glaive",
    "toolace": "toolace",
    "when2call": "when2call",
    "xlam-function-calling-60k": "xlam",
}
KINDS = ("call", "after_tool_result", "first_exchange", "later_turn")
TOOLS = ("tools_offered", "no_tools")


def load_release(root: Path) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import snapshot_download

    manifest_bytes = (root / RELEASE_MANIFEST).read_bytes()
    manifest = json.loads(manifest_bytes)
    pinned = json.loads((root / DATASET_MANIFEST).read_text(encoding="utf-8"))["corpus"][
        "manifest_sha256"
    ]
    if _sha256(manifest_bytes) != pinned:
        raise SystemExit(
            "the repository's release manifest is not the one M0's dataset manifest pins"
        )
    path = Path(
        snapshot_download(
            REPOSITORY,
            repo_type="dataset",
            revision=REVISION,
            allow_patterns=["*.parquet"],
            local_files_only=True,
        )
    )
    for shard in manifest["output_shards"]:
        data = (path / shard["file"]).read_bytes()
        if _sha256(data) != shard["sha256"] or len(data) != shard["bytes"]:
            raise SystemExit(f"{shard['file']} does not match the release manifest")
    return path, {
        "repository": REPOSITORY,
        "revision": REVISION,
        "release_manifest_sha256": _sha256(manifest_bytes),
        "shards_verified": len(manifest["output_shards"]),
        "record_count": manifest["record_count"],
    }


def _text(content: Any) -> str | None:
    return content if isinstance(content, str) else None


def audit(root: Path = ROOT) -> dict[str, Any]:
    import pyarrow.parquet as pq

    oneshot.check_frozen(root)
    path, release = load_release(root)
    manifest = json.loads((root / RELEASE_MANIFEST).read_text(encoding="utf-8"))

    turns: Counter[tuple[str, str, str]] = Counter()
    records: Counter[tuple[str, str]] = Counter()
    first_strata: Counter[tuple[str, str, str]] = Counter()
    first_prompts: dict[tuple[str, str], set[str]] = defaultdict(set)
    predictions: Counter[tuple[str, str, str, str]] = Counter()
    prediction_prompts: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    unreadable: Counter[str] = Counter()
    for shard in manifest["output_shards"]:
        for row in pq.read_table(
            path / shard["file"], columns=["source_dataset", "tools", "messages"]
        ).to_pylist():
            source = SOURCE_NAMES[row["source_dataset"]]
            tools = json.loads(row["tools"]) or []
            offered = TOOLS[0] if tools else TOOLS[1]
            records[(source, offered)] += 1
            body = [m for m in json.loads(row["messages"]) if m.get("role") != "system"]
            first_assistant = next(
                (i for i, m in enumerate(body) if m.get("role") == "assistant"), None
            )
            last_user: str | None = None
            for index, message in enumerate(body):
                role = message.get("role")
                if role == "user":
                    last_user = _text(message.get("content"))
                    continue
                if role != "assistant":
                    continue
                if message.get("tool_calls"):
                    kind = "call"
                elif index and body[index - 1].get("role") == "tool":
                    kind = "after_tool_result"
                elif index == first_assistant and [m.get("role") for m in body[:index]] == ["user"]:
                    kind = "first_exchange"
                else:
                    kind = "later_turn"
                turns[(source, offered, kind)] += 1
                if kind in ("call", "after_tool_result"):
                    continue
                response = _text(message.get("content"))
                if response is None or last_user is None:
                    unreadable[kind] += 1
                    continue
                if kind == "first_exchange":
                    first_strata[(source, offered, coverage.stratum(response, tools))] += 1
                    first_prompts[(source, offered)].add(normalize_prompt(last_user))
                label = classify(
                    ClassifierFeatures(
                        user_message=last_user,
                        assistant_response=response,
                        tools=tuple(tools),
                        structured_call_present=False,
                    )
                ).label
                predictions[(source, offered, kind, label)] += 1
                prediction_prompts[(source, offered, kind, label)].add(normalize_prompt(last_user))

    sources = sorted(set(SOURCE_NAMES.values()))
    observed = supply_audit.yields(root)
    projection: dict[str, Any] = {}
    for basis, table in observed.items():
        per_source: dict[str, Any] = {}
        for source in sources:
            totals: Counter[str] = Counter()
            for (unit_source, offered, name), count in first_strata.items():
                if unit_source != source:
                    continue
                cell = table.get(name, {}).get(source)
                if not cell or not cell["items"]:
                    totals[f"first_exchanges_without_yield_{offered}"] += count
                    continue
                totals[f"expected_direct_first_exchanges_{offered}"] += (
                    count * cell["DIRECT"] / cell["items"]
                )
            per_source[source] = {key: round(value, 1) for key, value in sorted(totals.items())}
        projection[basis] = per_source

    trainable = json.loads((root / DATASET_MANIFEST).read_text(encoding="utf-8"))[
        "source_trainable"
    ]
    return {
        "artifact_kind": "CANONICAL_V2_FINAL_DIRECT_PREVALENCE_AUDIT",
        "authorization": "docs/research/study-002/35-OWNER-DECISIONS-AFTER-CLASSIFIER-V1-TEST.md §5 (owner decision 2026-09-17)",
        "status": "COUNTS ONLY. Descriptive estimates from model labels and a frozen classifier; not gold.",
        "corpus": release,
        "m0_trainable_records_by_source": {
            SOURCE_NAMES[name]: count for name, count in sorted(trainable.items())
        },
        "records_by_source_and_tools": {s: {t: records[(s, t)] for t in TOOLS} for s in sources},
        "assistant_turns_by_source_tools_and_kind": {
            s: {t: {k: turns[(s, t, k)] for k in KINDS} for t in TOOLS} for s in sources
        },
        "prose_turns_without_text": dict(sorted(unreadable.items())),
        "first_exchange_strata": {
            s: {
                t: {
                    name: first_strata[(s, t, name)]
                    for name in coverage.STRATA
                    if first_strata[(s, t, name)]
                }
                for t in TOOLS
            }
            for s in sources
        },
        "first_exchange_distinct_prompts": {
            s: {t: len(first_prompts[(s, t)]) for t in TOOLS} for s in sources
        },
        "label_yield_projection_first_exchanges": projection,
        "frozen_classifier": {
            "version": CLASSIFIER_VERSION,
            "source_sha256_lf": oneshot.FROZEN_SOURCE_SHA256_LF,
            "predictions_by_source_tools_and_kind": {
                s: {
                    t: {
                        k: {label: predictions[(s, t, k, label)] for label in LABELS}
                        for k in ("first_exchange", "later_turn")
                    }
                    for t in TOOLS
                }
                for s in sources
            },
            "distinct_user_messages_by_source_tools_and_kind": {
                s: {
                    t: {
                        k: {label: len(prediction_prompts[(s, t, k, label)]) for label in LABELS}
                        for k in ("first_exchange", "later_turn")
                    }
                    for t in TOOLS
                }
                for s in sources
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
