"""Comparison logic and delta reporting across model runs and speculative modes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opengrad.benchmarks.speculative.metrics import compute_speedup


def load_run_artifacts(run_dir: Path) -> dict[str, Any]:
    """Load manifest, metrics, and failures for a given run directory."""
    manifest_file = run_dir / "manifest.json"
    metrics_file = run_dir / "metrics.json"
    if not manifest_file.exists() or not metrics_file.exists():
        raise FileNotFoundError(
            f"Run directory is missing manifest.json or metrics.json: {run_dir}"
        )

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    return {"manifest": manifest, "metrics": metrics, "path": str(run_dir)}


def compare_runs(baseline_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    """Compare a baseline run and candidate run, generating metrics deltas and speedup accounting."""
    base = load_run_artifacts(baseline_dir)
    cand = load_run_artifacts(candidate_dir)

    bm_base = base["manifest"].get("benchmark")
    bm_cand = cand["manifest"].get("benchmark")
    if bm_base != bm_cand:
        incompat = f"Warning: Comparing different benchmarks ('{bm_base}' vs '{bm_cand}')"
    else:
        incompat = None

    base_acc = float(base["metrics"].get("overall_accuracy", 0.0))
    cand_acc = float(cand["metrics"].get("overall_accuracy", 0.0))
    delta_acc = round(cand_acc - base_acc, 2)

    # Category deltas
    base_cats = base["metrics"].get("category_accuracy", {})
    cand_cats = cand["metrics"].get("category_accuracy", {})
    all_cats = sorted(set(base_cats.keys()) | set(cand_cats.keys()))

    cat_deltas: dict[str, dict[str, float]] = {}
    for cat in all_cats:
        b_val = float(base_cats.get(cat, 0.0))
        c_val = float(cand_cats.get(cat, 0.0))
        cat_deltas[cat] = {
            "baseline": b_val,
            "candidate": c_val,
            "delta": round(c_val - b_val, 2),
        }

    # Systems metrics & speedup
    base_sys = base["metrics"].get("system_metrics", {})
    cand_sys = cand["metrics"].get("system_metrics", {})

    b_tok_sec = float(base_sys.get("output_tokens_per_second", 0.0))
    c_tok_sec = float(cand_sys.get("output_tokens_per_second", 0.0))
    b_lat = float(base_sys.get("mean_latency_sec", 0.0))
    c_lat = float(cand_sys.get("mean_latency_sec", 0.0))

    speedup_record = compute_speedup(
        ar_tokens_per_sec=b_tok_sec,
        spec_tokens_per_sec=c_tok_sec,
        ar_wall_time=b_lat,
        spec_wall_time=c_lat,
    )

    comparison = {
        "baseline_run_id": base["manifest"].get("run_id"),
        "candidate_run_id": cand["manifest"].get("run_id"),
        "benchmark": bm_base,
        "warning": incompat,
        "accuracy": {
            "baseline": base_acc,
            "candidate": cand_acc,
            "delta": delta_acc,
        },
        "category_deltas": cat_deltas,
        "speedup": speedup_record.to_dict(),
    }
    return comparison


def render_comparison_markdown(comparison: dict[str, Any]) -> str:
    """Render comparison report into formatted markdown table showing deltas."""
    lines = [
        f"# Comparison Report: {comparison.get('benchmark', 'Benchmark')}",
        "",
        f"- **Baseline Run:** `{comparison.get('baseline_run_id')}`",
        f"- **Candidate Run:** `{comparison.get('candidate_run_id')}`",
        "",
    ]
    if comparison.get("warning"):
        lines.append(f"> ⚠️ **Notice:** {comparison['warning']}")
        lines.append("")

    acc = comparison.get("accuracy", {})
    lines.append("## Overall Accuracy Delta")
    lines.append("")
    lines.append("| Run | Overall Accuracy | Delta |")
    lines.append("| :--- | :---: | :---: |")
    lines.append(f"| **Baseline** | {acc.get('baseline', 0.0):.1f}% | - |")
    lines.append(
        f"| **Candidate** | {acc.get('candidate', 0.0):.1f}% | {acc.get('delta', 0.0):+.1f}% |"
    )
    lines.append("")

    cat_deltas = comparison.get("category_deltas", {})
    if cat_deltas:
        lines.append("## Category Level Deltas")
        lines.append("")
        lines.append("| Category | Baseline | Candidate | Delta |")
        lines.append("| :--- | :---: | :---: | :---: |")
        for cat, vals in sorted(cat_deltas.items()):
            lines.append(
                f"| `{cat}` | {vals['baseline']:.1f}% | {vals['candidate']:.1f}% | {vals['delta']:+.1f}% |"
            )
        lines.append("")

    speedup = comparison.get("speedup", {})
    if speedup:
        lines.append("## Systems Performance & Speedup")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| :--- | :---: |")
        lines.append(
            f"| **Throughput Speedup** | **{speedup.get('throughput_speedup', 1.0):.2f}x** |"
        )
        lines.append(f"| **Latency Speedup** | **{speedup.get('latency_speedup', 1.0):.2f}x** |")
        lines.append(f"| **TTFT Delta** | {speedup.get('ttft_delta_ms', 0.0):+.2f} ms |")
        lines.append(f"| **p95 ITL Delta** | {speedup.get('p95_itl_delta_ms', 0.0):+.2f} ms |")
        lines.append(f"| **VRAM Delta** | {speedup.get('vram_delta_mb', 0.0):+.1f} MB |")
        lines.append("")

    return "\n".join(lines)


def compare_multi_runs(run_dirs: list[Path]) -> str:
    """Compare multiple runs (e.g. BASE vs M0 vs M1) in a single unified table."""
    loaded = [load_run_artifacts(d) for d in run_dirs]
    if not loaded:
        return "No runs to compare."

    models = [l["manifest"].get("model_id", f"run_{i}") for i, l in enumerate(loaded)]

    lines = [
        "# Multi-Run Progression Report",
        "",
        "| Benchmark / Dimension | " + " | ".join(f"`{m}`" for m in models) + " |",
        "| :--- | " + " | ".join(":---:" for _ in models) + " |",
    ]

    # Accuracy row
    acc_row = ["Overall Accuracy"] + [
        f"{l['metrics'].get('overall_accuracy', 0.0):.1f}%" for l in loaded
    ]
    lines.append("| " + " | ".join(acc_row) + " |")

    # Throughput row
    tok_sec = [
        f"{l['metrics'].get('system_metrics', {}).get('output_tokens_per_second', 0.0):.1f} tok/s"
        for l in loaded
    ]
    lines.append("| Throughput | " + " | ".join(tok_sec) + " |")

    return "\n".join(lines)
