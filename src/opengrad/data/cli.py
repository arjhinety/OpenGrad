from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from opengrad.data.canonical import ToolConversation
from opengrad.data.materialize import materialize_parquet
from opengrad.data.normalize import (
    build_manifest,
    iter_jsonl,
    normalize_records,
    report,
    write_jsonl,
)
from opengrad.data.real_analysis import (
    _rows,
    analyze_corpus,
    contamination_report,
    heldout_readiness,
    lineage_status,
)
from opengrad.data.release import build_release, validate_release
from opengrad.data.renderers import render_materialized_corpus, renderer_for

DATASETS = ["xlam", "when2call", "toolace", "button", "looptool", "glaive"]
DATASET_REVISIONS = {
    "xlam": "26d14ebfe18b1f7b524bd39b404b50af5dc97866",
    "when2call": "0582f7749df63a96fdc3070932e83e72396ace53",
    "toolace": "6bda777c88d21e5a204703c1ee45597a8fa4f734",
    "button": "47cb720ed223b249a2f1d0a3faf1cb1eb7175622",
    "looptool": "b6c572d442ed4f2177f23645d8e9a77522e712c3",
    "glaive": "e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac",
}


def _example(row: dict[str, Any]) -> ToolConversation:
    return ToolConversation(
        row["id"], row["source"], row["tools"], row["messages"], row["metadata"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="opengrad-data")
    sub = parser.add_subparsers(dest="command", required=True)
    normalize = sub.add_parser("normalize")
    normalize.add_argument("--dataset", choices=DATASETS, required=True)
    normalize.add_argument("--input", type=Path, required=True)
    normalize.add_argument("--output", type=Path, required=True)
    normalize.add_argument("--split", default="train")
    normalize.add_argument("--max-records", type=int)
    materialize = sub.add_parser("materialize-parquet")
    materialize.add_argument("--dataset", choices=DATASETS, required=True)
    materialize.add_argument("--input", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    materialize.add_argument("--split", required=True)
    materialize.add_argument("--mode", choices=["sft", "preference", "evaluation"], default="sft")
    materialize.add_argument(
        "--adapter",
        default=None,
        help="override the adapter by name (for a new corpus version; the default is unchanged)",
    )
    materialize.add_argument("--shard-size", type=int, default=1000)
    materialize.add_argument("--batch-size", type=int, default=128)
    materialize.add_argument("--max-records", type=int)
    audit = sub.add_parser("audit")
    audit.add_argument("input", type=Path)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("dataset", choices=DATASETS)
    validate = sub.add_parser("validate")
    validate.add_argument("input", type=Path)
    render = sub.add_parser("render")
    render.add_argument("--model", required=True)
    render.add_argument("--input", type=Path, required=True)
    render.add_argument("--max-records", type=int)
    token_stats = sub.add_parser("token-stats")
    token_stats.add_argument("--model", required=True)
    token_stats.add_argument("--input", type=Path, required=True)
    token_stats.add_argument("--max-records", type=int)
    render_materialized = sub.add_parser("render-materialized")
    render_materialized.add_argument("--model", required=True)
    render_materialized.add_argument("--input", type=Path, required=True)
    render_materialized.add_argument("--output", type=Path, required=True)
    release = sub.add_parser("build-hf-release")
    release.add_argument("--release-config", type=Path, required=True)
    release.add_argument("--output", type=Path, required=True)
    release_validate = sub.add_parser("validate-release")
    release_validate.add_argument("--input", type=Path, required=True)
    yield_report = sub.add_parser(
        "yield-report",
        help="per-source canonical/rendered/trainable yield, with collapse detection",
    )
    yield_report.add_argument("--input", type=Path, required=True)
    yield_report.add_argument(
        "--expectations",
        type=Path,
        default=Path("configs/data/yield_expectations.yaml"),
    )
    yield_report.add_argument("--model", default="Qwen/Qwen3.5-2B")
    yield_report.add_argument("--max-seq-length", type=int, default=2048)
    yield_report.add_argument("--limit", type=int)
    yield_report.add_argument("--output", type=Path)
    yield_report.add_argument("--json", action="store_true")
    real_audit = sub.add_parser(
        "real-audit", help="bounded CPU analysis of materialized JSONL corpora"
    )
    real_audit.add_argument("--input", action="append", required=True, metavar="NAME=PATH")
    real_audit.add_argument("--output", type=Path)
    real_audit.add_argument("--contamination-reference", type=Path)
    real_audit.add_argument("--heldout", type=Path)
    real_audit.add_argument("--render-manifest", type=Path)
    real_audit.add_argument("--token-manifest", type=Path)
    args = parser.parse_args()
    if args.command == "real-audit":
        groups: dict[str, Path | list[dict[str, Any]]] = {}
        for item in args.input:
            if "=" not in item:
                parser.error("--input must be NAME=PATH")
            name, path = item.split("=", 1)
            if not name or not path:
                parser.error("--input must be NAME=PATH")
            groups[name] = Path(path)
        payload = analyze_corpus(groups)
        rows = [row for path in groups.values() for row in _rows(path)]
        reference = _rows(args.contamination_reference) if args.contamination_reference else None
        payload["contamination"] = contamination_report(rows, reference)
        payload["heldout_evaluation"] = (
            heldout_readiness(_rows(args.heldout)) if args.heldout else heldout_readiness(None)
        )
        if args.render_manifest and args.token_manifest:
            payload["lineage"] = lineage_status(args.render_manifest, args.token_manifest)
        else:
            payload["lineage"] = {"status": "BLOCKED", "reasons": ["manifest_paths_not_supplied"]}
        text = json.dumps(payload, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0
    if args.command == "normalize":
        rows, counts = normalize_records(
            iter_jsonl(args.input), args.dataset, args.split, args.max_records
        )
        digest = write_jsonl(rows, args.output)
        payload = {**counts, "canonical_checksum": digest, "output": str(args.output)}
        args.output.with_suffix(".report.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip()
            or "UNKNOWN"
        )
        manifest = build_manifest(
            commit=commit,
            dataset=args.dataset,
            revision=DATASET_REVISIONS[args.dataset],
            adapter=args.dataset,
            adapter_version="1.0.0",
            config={"split": args.split, "max_records": args.max_records},
            counts=counts,
            output=str(args.output),
            checksum=digest,
        )
        args.output.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "materialize-parquet":
        result = materialize_parquet(
            args.input,
            args.output,
            dataset=args.dataset,
            split=args.split,
            mode=args.mode,
            adapter_override=args.adapter,
            shard_size=args.shard_size,
            batch_size=args.batch_size,
            max_records=args.max_records,
        )
        print(json.dumps(result["manifest"], indent=2, sort_keys=True))
        return 0
    if args.command == "audit":
        print(json.dumps(report(iter_jsonl(args.input)), indent=2, sort_keys=True))
        return 0
    if args.command == "render-materialized":
        result = render_materialized_corpus(args.input, args.output, model=args.model)
        manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
        manifest_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "build-hf-release":
        result = build_release(Path.cwd(), args.release_config, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "validate-release":
        errors = validate_release(args.input)
        if errors:
            print(json.dumps({"status": "INVALID", "errors": errors}, indent=2))
            return 1
        print(json.dumps({"status": "VALID", "input": str(args.input)}, indent=2))
        return 0
    if args.command == "yield-report":
        from opengrad.data.yield_gate import (
            evaluate_yield_gate,
            gate_is_blocking,
            load_expectations,
            measure_materialized_corpus,
            render_yield_table,
        )

        yields = measure_materialized_corpus(
            args.input,
            model=args.model,
            max_seq_length=args.max_seq_length,
            limit=args.limit,
        )
        expectations = load_expectations(args.expectations)
        findings = evaluate_yield_gate(yields, expectations)
        payload = {
            "schema_version": 1,
            "status": "COLLAPSE" if gate_is_blocking(findings) else "OK",
            "input": str(args.input),
            "expectations": str(args.expectations),
            "sources": findings,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(render_yield_table(findings))
        return 1 if gate_is_blocking(findings) else 0
    if args.command == "inspect":
        from opengrad.data.adapters import ADAPTERS

        print(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "adapter": ADAPTERS[args.dataset].__name__,
                    "status": "IMPLEMENTED",
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "validate":
        failures = 0
        records = 0
        for row in iter_jsonl(args.input):
            records += 1
            try:
                _example(row).validate()
            except (KeyError, TypeError, ValueError):
                failures += 1
        print(json.dumps({"records": records, "failures": failures}, sort_keys=True))
        return int(bool(failures))
    renderer = renderer_for(args.model)
    lengths: list[int] = []
    for count, row in enumerate(iter_jsonl(args.input)):
        if args.max_records is not None and count >= args.max_records:
            break
        rendered = renderer.render_sft(_example(row))
        if args.command == "render":
            print(
                json.dumps(
                    {"example_id": rendered.example_id, "render": rendered.__dict__},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        else:
            lengths.append(renderer.token_lengths(_example(row))["total"])
    if args.command == "token-stats":
        lengths.sort()

        def percentile(fraction: float) -> int:
            if not lengths:
                return 0
            return lengths[min(len(lengths) - 1, int((len(lengths) - 1) * fraction))]

        print(
            json.dumps(
                {
                    "count": len(lengths),
                    "mean": sum(lengths) / len(lengths) if lengths else 0,
                    "p50": percentile(0.50),
                    "p75": percentile(0.75),
                    "p90": percentile(0.90),
                    "p95": percentile(0.95),
                    "p99": percentile(0.99),
                    "max": max(lengths, default=0),
                },
                sort_keys=True,
            )
        )
    return 0
