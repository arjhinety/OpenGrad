"""Benchmark contamination: the benchmark layer over `opengrad.contamination.levels`."""

from opengrad.benchmarks.contamination.registry import (
    ContaminationEntry,
    ContaminationRegistry,
)
from opengrad.benchmarks.contamination.scanner import (
    BenchmarkScanReport,
    load_samples,
    scan_benchmark,
)

__all__ = [
    "BenchmarkScanReport",
    "ContaminationEntry",
    "ContaminationRegistry",
    "load_samples",
    "scan_benchmark",
]
