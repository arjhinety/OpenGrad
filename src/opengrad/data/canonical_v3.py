"""canonical-v3: the decision-balanced training artifact (21 phase 5), built from the plan of 39.

Takes the selection plan of :mod:`opengrad.data.canonical_v3_balance`, attaches one behaviour block per record,
runs the gates 39 §4 requires, and writes an immutable artifact. It renders nothing into training text and
trains nothing: 38 §4 stands.

The behaviour block, the thing canonical-v1 and -v2 never had:

``metadata.behavior = {"decision", "capabilities": [], "confidence", "provenance"}``

* ``decision`` is the mixture vocabulary (``CALL``/``ANSWER``/``CLARIFY``/``UNSUPPORTED``), not the
  classifier's wording: the classifier's ``DIRECT`` is the mixture's ``ANSWER``.
* ``confidence`` is ``known`` for ``CALL``, whose membership is the structural fact that the first reply carries
  ``tool_calls`` (30 §4), and ``heuristic`` for the three decided by the classifier -- they rest on provisional
  ``MODEL_REFERENCE`` qualifications (37 §8), and the artifact says so rather than implying certainty.
* ``capabilities`` is always empty: no capability labeller exists (39 §1), and inventing one here would claim a
  balance nothing measured.

Gates, all run before any shard is written, each recorded with its counts:

1. **membership** -- every selected id exists once in normalization-v3 and carries the label the plan selected on;
2. **contamination** -- a record whose ``contamination_status`` is neither ``CLEAN``, ``UNASSESSED`` nor absent is
   rejected, the rule :mod:`opengrad.data.selection` already applies. Evaluation and held-out material never
   reaches here: the input contract excluded it at labelling time;
3. **supervision** -- ``metadata.supervision`` must declare a kind (``require=True``, the training boundary);
4. **semantic** -- :func:`opengrad.data.semantic.validate_training_trajectory` must return no issue under that
   contract;
5. **renderability** -- the record must render under the pinned Qwen3.5-2B renderer. Run over a seeded sample by
   default (``--render-sample``), because the tokenizer pass is the expensive part; ``--render-all`` checks
   every record. The renderer version and template hash are recorded either way.

A record failing any gate is **excluded and counted**, never silently dropped: the manifest carries every
rejection reason, and a stratum whose count falls below the plan's is reported as a shortfall rather than
back-filled from another stratum.

    python -m opengrad.data.canonical_v3 --build
    python -m opengrad.data.canonical_v3 --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from opengrad.data import behaviour_labels as labels_pass
from opengrad.data import canonical_v3_balance as balance
from opengrad.data import versions
from opengrad.data.behavior import validate_behavior
from opengrad.data.canonical import ToolConversation
from opengrad.data.normalization_v3 import iter_rows, storage_row
from opengrad.data.semantic import validate_training_trajectory
from opengrad.data.supervision import validate_supervision_block

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_KIND = "CANONICAL_V3_DECISION_BALANCED"
CORPUS_VERSION = "canonical-v3-decision-balance-v1"
CORPUS_DIR = Path("data/processed/normalization-v3")
OUTPUT_DIR = Path("data/processed/canonical-v3")
REPORT = Path("reports/canonical-v3/canonical-v3.manifest.json")
SHARD_SIZE = 10_000
RENDER_MODEL = "Qwen/Qwen3.5-2B"
RENDER_SAMPLE = 500
RENDER_SEED = "opengrad-canonical-v3-render-check-v1"

#: classifier label -> mixture decision. The classifier says DIRECT; the mixture vocabulary says ANSWER.
DECISION_OF_STRATUM = {"CALL": "CALL", "ANSWER": "ANSWER", "CLARIFY": "CLARIFY", "UNSUPPORTED": "UNSUPPORTED"}
CLEAN_CONTAMINATION = frozenset({"CLEAN", "UNASSESSED"})


class CanonicalV3Error(RuntimeError):
    """The artifact cannot be built, or what was written does not match its inputs."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def behaviour_block(stratum: str, label_row: Mapping[str, Any], classifier: Mapping[str, Any], contract: str) -> dict[str, Any]:
    """One record's behaviour block. ``CALL`` is structural and therefore ``known``; the rest are heuristic."""
    decision = DECISION_OF_STRATUM[stratum]
    confidence = "known" if stratum in balance.STRUCTURAL_STRATA else "heuristic"
    validate_behavior(decision, [], confidence)
    provenance: dict[str, Any] = {
        "stratum": stratum,
        "source_label": label_row["label"],
        "reference_status": "MODEL_REFERENCE_PROVISIONAL",
        "authorisation": "docs/research/study-002/38-BALANCING-PERMISSION-AND-C1-AUTHORISATION.md",
    }
    if confidence == "known":
        provenance["basis"] = "structural: the first assistant reply carries tool_calls (30 §4)"
    else:
        provenance.update(
            basis="classifier",
            classifier_version=classifier["version"],
            classifier_tag=classifier["tag"],
            classifier_source_sha256_lf=classifier["source_sha256_lf"],
            input_contract=contract,
            step=label_row["step"],
            unit_kind=label_row["unit_kind"],
        )
    return {"decision": decision, "capabilities": [], "confidence": confidence, "provenance": provenance}


def gate_record(record: Mapping[str, Any]) -> str | None:
    """The first gate this record fails, or None. Never raises on a bad record."""
    metadata = dict(record.get("metadata") or {})
    status = metadata.get("contamination_status")
    if status is not None and status not in CLEAN_CONTAMINATION:
        return f"CONTAMINATION:{status}"
    try:
        validate_supervision_block(metadata, require=True)
    except Exception as error:  # noqa: BLE001 - the reason is recorded, not raised
        return f"SUPERVISION:{type(error).__name__}"
    try:
        issues = validate_training_trajectory(
            ToolConversation(record["id"], record["source"], list(record["tools"]), list(record["messages"]), metadata)
        )
    except Exception as error:  # noqa: BLE001
        return f"SEMANTIC_ERROR:{type(error).__name__}"
    if issues:
        return f"SEMANTIC:{issues[0].code}"
    return None


def render_check(records: list[Mapping[str, Any]], sample: int | None, root: Path) -> dict[str, Any]:
    """Render a seeded sample (or all) under the pinned renderer, counting failures. Never writes text."""
    from opengrad.data.renderers import renderer_for

    ordered = sorted(records, key=lambda row: _sha256(f"{RENDER_SEED}|{row['id']}".encode("utf-8")))
    chosen = ordered if sample is None else ordered[:sample]
    renderer = renderer_for(RENDER_MODEL)
    failures: Counter[str] = Counter()
    rendered = 0
    identity: dict[str, Any] = {}
    for row in chosen:
        example = ToolConversation(row["id"], row["source"], list(row["tools"]), list(row["messages"]), dict(row["metadata"]))
        try:
            result = renderer.render_sft(example)
        except Exception as error:  # noqa: BLE001
            failures[type(error).__name__] += 1
            continue
        rendered += 1
        identity = {
            "renderer": result.renderer,
            "model_revision": result.model_revision,
            "tokenizer_revision": result.tokenizer_revision,
            "template_hash": result.chat_template_hash,
            "enable_thinking": result.enable_thinking,
        }
    return {
        "checked": len(chosen),
        "rendered": rendered,
        "failures": dict(sorted(failures.items())),
        "scope": "all" if sample is None else f"seeded sample of {sample}",
        "seed": RENDER_SEED,
        "identity": identity,
    }


def _write_shards(out: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from opengrad.data.normalization_v3 import _write_parquet

    shards: list[dict[str, Any]] = []
    for start in range(0, len(rows), SHARD_SIZE):
        chunk = [storage_row(row) for row in rows[start : start + SHARD_SIZE]]
        name = f"shard-{len(shards):06d}.parquet"
        _write_parquet(chunk, out / name)
        shards.append({"file": name, "records": len(chunk), "sha256": _sha256((out / name).read_bytes())})
    return shards


def build(root: Path = ROOT, *, render_sample: int | None = RENDER_SAMPLE) -> dict[str, Any]:
    out = root / OUTPUT_DIR
    if (out / "manifest.json").exists():
        raise CanonicalV3Error(f"{out / 'manifest.json'} exists: canonical-v3 is immutable, never overwritten")
    plan_path = root / balance.OUTPUT_DIR / "manifest.json"
    if not plan_path.exists():
        raise CanonicalV3Error("no selection plan: run opengrad.data.canonical_v3_balance --build")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    verification = balance.verify(root)
    if verification["status"] != "PASS":
        raise CanonicalV3Error(f"the selection plan does not verify: {verification['problems']}")
    selected = json.loads((root / balance.OUTPUT_DIR / plan["selected_ids_file"]).read_text(encoding="utf-8"))
    stratum_of_id = {record_id: stratum for stratum, ids in selected.items() for record_id in ids}
    label_rows, labels_manifest = balance.load_labels(root)
    labels_by_id = {row["id"]: row for row in label_rows}
    classifier = labels_manifest["classifier"]
    contract = labels_manifest["input_contract"]

    kept: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    per_stratum: Counter[str] = Counter()
    per_source: dict[str, Counter[str]] = defaultdict(Counter)
    seen: set[str] = set()
    for source in sorted(labels_manifest["sources"]):
        for record in iter_rows(root / CORPUS_DIR, source):
            stratum = stratum_of_id.get(record["id"])
            if stratum is None:
                continue
            if record["id"] in seen:
                raise CanonicalV3Error(f"{record['id']} appears twice in normalization-v3")
            seen.add(record["id"])
            reason = gate_record(record)
            if reason:
                rejected[reason] += 1
                continue
            row = dict(record)
            metadata = dict(row["metadata"])
            metadata["behavior"] = behaviour_block(stratum, labels_by_id[record["id"]], classifier, contract)
            row["metadata"] = metadata
            kept.append(row)
            per_stratum[stratum] += 1
            per_source[stratum][record["source"]] += 1
    missing = sorted(set(stratum_of_id) - seen)
    if missing:
        raise CanonicalV3Error(f"{len(missing)} selected records are not in normalization-v3")

    rendering = render_check(kept, render_sample, root)
    out.mkdir(parents=True, exist_ok=True)
    kept.sort(key=lambda row: row["id"])
    shards = _write_shards(out, kept)
    planned = plan["counts"]["per_stratum"]
    manifest = {
        "artifact_kind": ARTIFACT_KIND,
        "corpus_version": CORPUS_VERSION,
        "statement": (
            "The decision-balanced canonical-v3 artifact: normalization-v3 records selected by the plan of 39 "
            "and carrying one behaviour block each. Behaviour comes from the frozen prose-decision-classifier-v2 "
            "for ANSWER/CLARIFY/UNSUPPORTED (confidence heuristic, provisional MODEL_REFERENCE footing) and from "
            "record structure for CALL (confidence known). No training text is rendered here, no arm uses this "
            "corpus, and no training is authorised (38 §4)."
        ),
        "specification": balance.SPEC,
        "authorisation": "docs/research/study-002/38-BALANCING-PERMISSION-AND-C1-AUTHORISATION.md",
        "selection_plan": {"balance_version": plan["balance_version"], "selected_ids_sha256": plan["selected_ids_sha256"]},
        "labels": {"labels_version": labels_manifest["labels_version"], "classifier": classifier, "input_contract": contract},
        "versions": versions.provenance_versions(),
        "corpus_fingerprint": labels_manifest["corpus_fingerprint"],
        "counts": {
            "planned_per_stratum": planned,
            "written": len(kept),
            "per_stratum": dict(sorted(per_stratum.items())),
            "shortfall_per_stratum": {s: planned - per_stratum[s] for s in sorted(per_stratum) if per_stratum[s] < planned},
            "per_source": {s: dict(sorted(c.items())) for s, c in sorted(per_source.items())},
            "rejected_by_gate": dict(sorted(rejected.items())),
        },
        "gates": {
            "membership": "every selected id found exactly once in normalization-v3",
            "contamination": f"status in {sorted(CLEAN_CONTAMINATION)} or absent",
            "supervision": "metadata.supervision declares a kind (require=True)",
            "semantic": "validate_training_trajectory returns no issue",
            "renderability": rendering,
        },
        "shards": shards,
        "content_hash": _sha256(b"".join(bytes.fromhex(shard["sha256"]) for shard in shards)),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (out / "manifest.json").write_bytes(manifest_bytes)
    report = root / REPORT
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_bytes(manifest_bytes)
    return manifest


def verify(root: Path = ROOT) -> dict[str, Any]:
    """Re-read the written artifact and check it against its manifest. Renders nothing."""
    manifest = json.loads((root / OUTPUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    problems: list[str] = []
    for shard in manifest["shards"]:
        observed = _sha256((root / OUTPUT_DIR / shard["file"]).read_bytes())
        if observed != shard["sha256"]:
            problems.append(f"{shard['file']}: bytes do not match the manifest")
    if _sha256(b"".join(bytes.fromhex(s["sha256"]) for s in manifest["shards"])) != manifest["content_hash"]:
        problems.append("content hash does not match the shard hashes")
    written = sum(shard["records"] for shard in manifest["shards"])
    if written != manifest["counts"]["written"]:
        problems.append("shard record counts do not match the manifest")
    plan = json.loads((root / balance.OUTPUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    if plan["selected_ids_sha256"] != manifest["selection_plan"]["selected_ids_sha256"]:
        problems.append("the selection plan changed since the artifact was written")
    decisions: Counter[str] = Counter()
    behaviours_ok = True
    from opengrad.data.normalization_v3 import decode_row
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    for shard in manifest["shards"]:
        for batch in pq.ParquetFile(root / OUTPUT_DIR / shard["file"]).iter_batches(batch_size=512):
            for raw in batch.to_pylist():
                row = decode_row(raw)
                block = (row.get("metadata") or {}).get("behavior")
                if not block:
                    behaviours_ok = False
                    continue
                decisions[block["decision"]] += 1
                try:
                    validate_behavior(block["decision"], block["capabilities"], block["confidence"])
                except ValueError:
                    behaviours_ok = False
    if not behaviours_ok:
        problems.append("a record carries a missing or invalid behaviour block")
    if dict(decisions) != {DECISION_OF_STRATUM[s]: n for s, n in manifest["counts"]["per_stratum"].items()}:
        problems.append("the written decisions do not match the manifest counts")
    return {"status": "PASS" if not problems else "FAIL", "problems": problems, "decisions": dict(sorted(decisions.items()))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true")
    group.add_argument("--verify", action="store_true")
    parser.add_argument("--render-all", action="store_true", help="render every record, not a seeded sample")
    parser.add_argument("--render-sample", type=int, default=RENDER_SAMPLE)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if args.build:
        manifest = build(args.root, render_sample=None if args.render_all else args.render_sample)
        print(json.dumps({"counts": manifest["counts"], "renderability": manifest["gates"]["renderability"]}, indent=2))
        return 0
    print(json.dumps(verify(args.root), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
