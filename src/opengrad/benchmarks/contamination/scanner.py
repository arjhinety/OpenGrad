"""Benchmark contamination scan: the benchmark layer over :mod:`opengrad.contamination.levels`.

Levels 1-4 are the shared engine, so a level means the same thing here as in the behavioural
held-out screen: level 1 is byte-identical prompt text, level 2 identical after whitespace and case
normalisation. Level 5 is a human adjudication (`docs/evaluation/CONTAMINATION_ADJUDICATION.md`);
a scan always reports it ``NOT_RUN``, and nothing here can report a benchmark clean.

A scan reads its benchmark prompts and its training prompts from files, and records the sha256 of
each. It does not read the benchmark adapters: they synthesize placeholder tasks
(`docs/evaluation/BENCHMARK_STRATEGY.md` section 6), and the scan that recorded ``bfcl-v4`` as
``CLEAN`` compared ten of them with two fixture prompts (`reports/ERRATA.md` §25).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opengrad.contamination.levels import (
    STATUS_MEASURED,
    STATUS_REVIEW_REQUIRED,
    LevelMatches,
    Record,
    Thresholds,
    build_queue,
    level_status,
    match_levels,
    thresholds_record,
)
from opengrad.contamination.scanner import ngrams


@dataclass
class BenchmarkScanReport:
    benchmark_id: str
    benchmark_data_sha256: str
    training_corpus_sha256: str
    benchmark_samples: int
    training_samples: int
    status: str
    levels: dict[str, str]
    matches: LevelMatches
    thresholds: Thresholds
    audit_queue: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_data_sha256": self.benchmark_data_sha256,
            "training_corpus_sha256": self.training_corpus_sha256,
            "benchmark_samples": self.benchmark_samples,
            "training_samples": self.training_samples,
            "status": self.status,
            "levels": dict(self.levels),
            "counts": {
                "level_1": len(self.matches.level1),
                "level_2": len(self.matches.level2),
                "level_3": len(self.matches.level3),
                "level_4": len(self.matches.level4),
                "scored_pairs": self.matches.scored_pairs,
                "pruned_shingles": self.matches.pruned_shingles,
            },
            "thresholds": thresholds_record(self.thresholds),
            "audit_queue_size": len(self.audit_queue),
            "audit_queue": self.audit_queue,
        }

    def render_markdown(self) -> str:
        counts = self.to_dict()["counts"]
        lines = [
            f"# Contamination scan: {self.benchmark_id}",
            "",
            f"- **Status:** `{self.status}`",
            f"- **Benchmark data sha256:** `{self.benchmark_data_sha256}`",
            f"- **Training corpus sha256:** `{self.training_corpus_sha256}`",
            f"- **Benchmark samples:** {self.benchmark_samples}",
            f"- **Training samples:** {self.training_samples}",
            "",
            "| Level | Status | Matches |",
            "| :--- | :---: | ---: |",
        ]
        for index, (level, status) in enumerate(self.levels.items(), 1):
            matched = counts.get(f"level_{index}", len(self.audit_queue))
            lines.append(f"| `{level}` | {status} | {matched} |")
        lines += [
            "",
            (
                "Level 5 is a human adjudication of the audit queue and has not run. This scan "
                "makes no clean claim, even with an empty queue."
            ),
        ]
        if self.audit_queue:
            lines += ["", "## Audit queue (first 20)", ""]
            for entry in self.audit_queue[:20]:
                lines.append(f"- `{entry['heldout']}`: levels {', '.join(entry['levels'])}")
        return "\n".join(lines) + "\n"


def _record(sample: dict[str, Any], source: str) -> Record:
    text = str(sample["prompt"])
    return Record(
        record_id=str(sample["id"]),
        source=source,
        text=text,
        shingles=frozenset(ngrams(text)),
    )


def scan_benchmark(
    benchmark_id: str,
    benchmark_samples: list[dict[str, Any]],
    training_samples: list[dict[str, Any]],
    *,
    benchmark_data_sha256: str,
    training_corpus_sha256: str,
    thresholds: Thresholds | None = None,
) -> BenchmarkScanReport:
    """Run levels 1-4 of ``benchmark_samples`` against ``training_samples``.

    Each sample is ``{"id": str, "prompt": str}``. An empty side is refused: a scan that compared
    nothing would otherwise report no matches.
    """
    if not benchmark_samples:
        raise ValueError("no benchmark samples to scan")
    if not training_samples:
        raise ValueError("no training samples to scan against")
    thresholds = thresholds or Thresholds()
    benchmark = [_record(sample, benchmark_id) for sample in benchmark_samples]
    training = [_record(sample, "training") for sample in training_samples]
    matches = match_levels(benchmark, lambda: iter(training), thresholds)
    queue = build_queue(matches)
    return BenchmarkScanReport(
        benchmark_id=benchmark_id,
        benchmark_data_sha256=benchmark_data_sha256,
        training_corpus_sha256=training_corpus_sha256,
        benchmark_samples=len(benchmark),
        training_samples=len(training),
        status=STATUS_REVIEW_REQUIRED if matches.any_match else STATUS_MEASURED,
        levels=level_status(),
        matches=matches,
        thresholds=thresholds,
        audit_queue=sorted(queue.values(), key=lambda entry: entry["heldout"]),
    )


def _user_prompt(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    return " ".join(
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    )


def load_samples(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Read ``{"id", "prompt"}`` samples from a JSONL file, and the file's sha256.

    A row carries the prompt in ``prompt``, or in the user-role turns of ``messages`` (the canonical
    training schema); its id is ``id``, else ``canonical_hash``, else its line number. A row with
    neither prompt form is an error rather than a silently skipped sample.
    """
    data = path.read_bytes()
    samples: list[dict[str, Any]] = []
    for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"{path}:{number}: a sample must be a JSON object")
        prompt = row.get("prompt")
        if not isinstance(prompt, str):
            prompt = _user_prompt(row.get("messages"))
        if not prompt:
            raise ValueError(f"{path}:{number}: no `prompt` and no user turn in `messages`")
        sample_id = row.get("id") or row.get("canonical_hash") or f"line-{number}"
        samples.append({"id": str(sample_id), "prompt": prompt})
    return samples, hashlib.sha256(data).hexdigest()
