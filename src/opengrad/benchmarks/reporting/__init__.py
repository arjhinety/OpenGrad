"""Reporting package."""

from opengrad.benchmarks.reporting.comparator import (
    compare_multi_runs,
    compare_runs,
    load_run_artifacts,
    render_comparison_markdown,
)
from opengrad.benchmarks.reporting.generator import generate_run_readme

__all__ = [
    "compare_multi_runs",
    "compare_runs",
    "generate_run_readme",
    "load_run_artifacts",
    "render_comparison_markdown",
]
