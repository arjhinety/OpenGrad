"""Report generator for benchmark runs and comparison summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_run_readme(output_dir: Path) -> str:
    """Generate human-readable README.md in reports/benchmarks/<run-id>/."""
    manifest_file = output_dir / "manifest.json"
    metrics_file = output_dir / "metrics.json"
    failures_file = output_dir / "failures.json"
    speculative_file = output_dir / "speculative.json"

    manifest: dict[str, Any] = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else {}
    metrics: dict[str, Any] = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}
    failures: dict[str, Any] = json.loads(failures_file.read_text(encoding="utf-8")) if failures_file.exists() else {}
    speculative: dict[str, Any] = json.loads(speculative_file.read_text(encoding="utf-8")) if speculative_file.exists() else {}

    lines = [
        f"# Benchmark Run Report: {manifest.get('run_id', 'unknown')}",
        "",
        "## Run Metadata",
        "",
        "| Attribute | Value |",
        "| :--- | :--- |",
        f"| **Model ID** | `{manifest.get('model_id', 'unknown')}` |",
        f"| **Model Revision** | `{manifest.get('model_revision', 'unknown')[:12]}` |",
        f"| **Benchmark** | `{manifest.get('benchmark', 'unknown')}` |",
        f"| **Benchmark Revision** | `{manifest.get('benchmark_revision', 'unknown')[:12]}` |",
        f"| **Evaluator Revision** | `{manifest.get('evaluator_revision', 'unknown')}` |",
        f"| **Timestamp** | {manifest.get('timestamp', 'unknown')} |",
        f"| **Git Commit** | `{manifest.get('git_commit', 'unknown')[:10]}` |",
        "",
        "## Capability Results",
        "",
        f"- **Total Tasks:** {metrics.get('total_tasks', 0)}",
        f"- **Successful Tasks:** {metrics.get('successful_tasks', 0)}",
        f"- **Overall Accuracy:** **{metrics.get('overall_accuracy', 0.0)}%**",
        "",
    ]

    cat_acc = metrics.get("category_accuracy", {})
    if cat_acc:
        lines.append("### Category Breakdown")
        lines.append("")
        lines.append("| Category | Accuracy |")
        lines.append("| :--- | :---: |")
        for cat, acc in sorted(cat_acc.items()):
            lines.append(f"| `{cat}` | {acc:.1f}% |")
        lines.append("")

    sys_metrics = metrics.get("system_metrics", {})
    if sys_metrics:
        lines.append("## Systems Performance")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| :--- | :---: |")
        lines.append(f"| **Mean Latency** | {sys_metrics.get('mean_latency_sec', 0.0):.4f} s |")
        lines.append(f"| **Output Throughput** | {sys_metrics.get('output_tokens_per_second', 0.0):.2f} tok/s |")
        lines.append(f"| **Total Completion Tokens** | {sys_metrics.get('total_completion_tokens', 0)} |")
        lines.append("")

    if speculative:
        lines.append("## Speculative Decoding / MTP Telemetry")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| :--- | :---: |")
        lines.append(f"| **Mean Acceptance Rate** | {speculative.get('mean_acceptance_rate', 0.0) * 100:.1f}% |")
        lines.append(f"| **Accepted Tokens / Step** | {speculative.get('mean_accepted_tokens_per_step', 0.0):.2f} |")
        lines.append(f"| **Samples Evaluated** | {speculative.get('samples_evaluated', 0)} |")
        lines.append("")

    fail_breakdown = failures.get("failure_breakdown", {})
    if fail_breakdown:
        lines.append("## Failure Taxonomy Analysis")
        lines.append("")
        lines.append("| Failure Category | Count |")
        lines.append("| :--- | :---: |")
        for f_code, cnt in sorted(fail_breakdown.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| `{f_code}` | {cnt} |")
        lines.append("")

    lines.append("## Machine-Readable Artifacts")
    lines.append("")
    lines.append("- `manifest.json`: Full execution provenance and settings")
    lines.append("- `environment.json`: Hardware, platform, runtime capture")
    lines.append("- `metrics.json`: Normalized OpenGrad metrics")
    lines.append("- `failures.json`: Structured failure records")
    lines.append("- `predictions.jsonl`: Line-by-line model predictions and parsed outputs")
    if speculative_file.exists():
        lines.append("- `speculative.json`: Speculative decoding metrics and MTP depth analysis")

    readme_text = "\n".join(lines) + "\n"
    (output_dir / "README.md").write_text(readme_text, encoding="utf-8")
    return readme_text
