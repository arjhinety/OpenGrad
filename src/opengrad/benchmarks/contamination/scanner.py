"""Multi-level benchmark contamination detection engine (Levels 1 to 5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from opengrad.contamination.scanner import exact_hash, jaccard, normalize


@dataclass
class ContaminationMatch:
    level: int  # 1 to 5
    level_name: str
    benchmark_id: str
    benchmark_task_id: str
    training_sample_id: str
    similarity_score: float
    benchmark_text: str
    training_text: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "level_name": self.level_name,
            "benchmark_id": self.benchmark_id,
            "benchmark_task_id": self.benchmark_task_id,
            "training_sample_id": self.training_sample_id,
            "similarity_score": round(self.similarity_score, 4),
            "benchmark_text_preview": (
                self.benchmark_text[:120] + "..." if len(self.benchmark_text) > 120 else self.benchmark_text
            ),
            "training_text_preview": (
                self.training_text[:120] + "..." if len(self.training_text) > 120 else self.training_text
            ),
            "reason": self.reason,
        }


@dataclass
class ContaminationScanReport:
    benchmark_id: str
    corpus_fingerprint: str
    total_benchmark_samples: int
    total_training_samples: int
    scan_date: str
    level_status: dict[str, str]
    matches_by_level: dict[int, int]
    audit_queue: list[ContaminationMatch] = field(default_factory=list)
    verdict: str = "CLEAN"  # "CLEAN", "SUSPICIOUS_MATCHES", "CONTAMINATED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "corpus_fingerprint": self.corpus_fingerprint,
            "total_benchmark_samples": self.total_benchmark_samples,
            "total_training_samples": self.total_training_samples,
            "scan_date": self.scan_date,
            "verdict": self.verdict,
            "level_status": self.level_status,
            "matches_by_level": self.matches_by_level,
            "audit_queue_size": len(self.audit_queue),
            "audit_queue": [m.to_dict() for m in self.audit_queue],
        }

    def render_markdown(self) -> str:
        lines = [
            f"# Contamination Scan Report: {self.benchmark_id}",
            "",
            f"- **Verdict:** {self.verdict}",
            f"- **Training Corpus Fingerprint:** `{self.corpus_fingerprint}`",
            f"- **Benchmark Samples Scanned:** {self.total_benchmark_samples}",
            f"- **Training Samples Scanned:** {self.total_training_samples}",
            "",
            "## Level Progress & Match Counts",
            "",
            "| Level | Description | Status | Matches Found |",
            "| :--- | :--- | :---: | :---: |",
            f"| Level 1 | Exact Canonical Conversation Hash | {self.level_status.get('level_1', 'DONE')} | {self.matches_by_level.get(1, 0)} |",
            f"| Level 2 | Normalized Prompt / Answer Hash | {self.level_status.get('level_2', 'DONE')} | {self.matches_by_level.get(2, 0)} |",
            f"| Level 3 | N-Gram / MinHash Near-Duplicate | {self.level_status.get('level_3', 'DONE')} | {self.matches_by_level.get(3, 0)} |",
            f"| Level 4 | Semantic Similarity Candidate Generation | {self.level_status.get('level_4', 'DONE')} | {self.matches_by_level.get(4, 0)} |",
            f"| Level 5 | Manual Audit Queue Generation | {self.level_status.get('level_5', 'DONE')} | {len(self.audit_queue)} |",
            "",
        ]

        if self.audit_queue:
            lines.append("## Suspicious Matches in Audit Queue")
            lines.append("")
            lines.append(
                "> **Policy Warning:** OpenGrad does NOT silently delete benchmark examples. "
                "The matches below require manual research inspection."
            )
            lines.append("")
            for idx, match in enumerate(self.audit_queue[:20], 1):
                lines.append(f"### Match #{idx} (Level {match.level}: {match.level_name})")
                lines.append(f"- **Benchmark Task ID:** `{match.benchmark_task_id}`")
                lines.append(f"- **Training Sample ID:** `{match.training_sample_id}`")
                lines.append(f"- **Similarity Score:** {match.similarity_score:.4f}")
                lines.append(f"- **Reason:** {match.reason}")
                lines.append("```text")
                lines.append(f"Benchmark: {match.benchmark_text[:200]}")
                lines.append(f"Training:  {match.training_text[:200]}")
                lines.append("```")
                lines.append("")
        else:
            lines.append("## Result: No Contamination Detected")
            lines.append("No exact, near-duplicate, or semantic matches were found between the benchmark and training samples.")

        return "\n".join(lines)


class MultiLevelContaminationScanner:
    """Executes progressively stronger contamination checks (Levels 1 to 5)."""

    def __init__(
        self,
        ngram_threshold: float = 0.80,
        semantic_threshold: float = 0.85,
    ) -> None:
        self.ngram_threshold = ngram_threshold
        self.semantic_threshold = semantic_threshold

    def scan(
        self,
        benchmark_id: str,
        benchmark_samples: list[dict[str, Any]],  # list of {"id": str, "prompt": str, "expected": str, "canonical": str}
        training_samples: list[dict[str, Any]],   # list of {"id": str, "prompt": str, "response": str, "canonical": str}
        corpus_fingerprint: str = "unspecified",
        max_level: int = 5,
    ) -> ContaminationScanReport:
        matches_by_level: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        level_status: dict[str, str] = {}
        audit_queue: list[ContaminationMatch] = []

        # Precompute training hashes
        training_canonical_map: dict[str, str] = {}
        training_norm_prompt_map: dict[str, str] = {}

        for tr in training_samples:
            tr_id = str(tr.get("id", ""))
            canon = str(tr.get("canonical", tr.get("prompt", "")))
            norm_p = normalize(str(tr.get("prompt", "")))
            if canon:
                training_canonical_map[exact_hash(canon)] = tr_id
            if norm_p:
                training_norm_prompt_map[exact_hash(norm_p)] = tr_id

        # LEVEL 1: Exact canonical conversation hash
        level_status["level_1"] = "COMPLETED"
        for bm in benchmark_samples:
            bm_id = str(bm.get("id", ""))
            bm_canon = str(bm.get("canonical", bm.get("prompt", "")))
            bm_hash = exact_hash(bm_canon)
            if bm_hash in training_canonical_map:
                tr_id = training_canonical_map[bm_hash]
                match = ContaminationMatch(
                    level=1,
                    level_name="exact_canonical_hash",
                    benchmark_id=benchmark_id,
                    benchmark_task_id=bm_id,
                    training_sample_id=tr_id,
                    similarity_score=1.0,
                    benchmark_text=bm_canon,
                    training_text=bm_canon,
                    reason="Exact canonical conversation hash collision",
                )
                audit_queue.append(match)
                matches_by_level[1] += 1

        # LEVEL 2: Normalized prompt / answer hash
        if max_level >= 2:
            level_status["level_2"] = "COMPLETED"
            for bm in benchmark_samples:
                bm_id = str(bm.get("id", ""))
                norm_bm = normalize(str(bm.get("prompt", "")))
                bm_hash = exact_hash(norm_bm)
                if bm_hash in training_norm_prompt_map:
                    tr_id = training_norm_prompt_map[bm_hash]
                    # Avoid duplicate if already caught in L1
                    if not any(m.benchmark_task_id == bm_id and m.training_sample_id == tr_id for m in audit_queue):
                        match = ContaminationMatch(
                            level=2,
                            level_name="normalized_prompt_hash",
                            benchmark_id=benchmark_id,
                            benchmark_task_id=bm_id,
                            training_sample_id=tr_id,
                            similarity_score=1.0,
                            benchmark_text=norm_bm,
                            training_text=norm_bm,
                            reason="Exact normalized prompt hash match",
                        )
                        audit_queue.append(match)
                        matches_by_level[2] += 1

        # LEVEL 3: n-gram / MinHash near-duplicate detection
        if max_level >= 3:
            level_status["level_3"] = "COMPLETED"
            for bm in benchmark_samples:
                bm_id = str(bm.get("id", ""))
                bm_text = str(bm.get("prompt", ""))
                if len(bm_text) < 15:
                    continue
                for tr in training_samples:
                    tr_id = str(tr.get("id", ""))
                    tr_text = str(tr.get("prompt", ""))
                    score = jaccard(bm_text, tr_text, n=5)
                    if (
                        self.ngram_threshold <= score < 1.0
                        and not any(m.benchmark_task_id == bm_id and m.training_sample_id == tr_id for m in audit_queue)
                    ):
                            match = ContaminationMatch(
                                level=3,
                                level_name="ngram_minhash_near_duplicate",
                                benchmark_id=benchmark_id,
                                benchmark_task_id=bm_id,
                                training_sample_id=tr_id,
                                similarity_score=score,
                                benchmark_text=bm_text,
                                training_text=tr_text,
                                reason=f"N-gram 5-gram Jaccard overlap ({score:.2f} >= {self.ngram_threshold})",
                            )
                            audit_queue.append(match)
                            matches_by_level[3] += 1

        # LEVEL 4: Semantic similarity candidate generation
        if max_level >= 4:
            level_status["level_4"] = "COMPLETED"
            for bm in benchmark_samples:
                bm_id = str(bm.get("id", ""))
                bm_text = normalize(str(bm.get("prompt", "")))
                if len(bm_text) < 20:
                    continue
                for tr in training_samples:
                    tr_id = str(tr.get("id", ""))
                    tr_text = normalize(str(tr.get("prompt", "")))
                    ratio = SequenceMatcher(None, bm_text, tr_text).ratio()
                    if (
                        self.semantic_threshold <= ratio < 1.0
                        and not any(m.benchmark_task_id == bm_id and m.training_sample_id == tr_id for m in audit_queue)
                    ):
                            match = ContaminationMatch(
                                level=4,
                                level_name="semantic_similarity_candidate",
                                benchmark_id=benchmark_id,
                                benchmark_task_id=bm_id,
                                training_sample_id=tr_id,
                                similarity_score=ratio,
                                benchmark_text=bm_text,
                                training_text=tr_text,
                                reason=f"SequenceMatcher ratio ({ratio:.2f} >= {self.semantic_threshold})",
                            )
                            audit_queue.append(match)
                            matches_by_level[4] += 1

        # LEVEL 5: Manual audit queue verification
        level_status["level_5"] = "COMPLETED" if max_level >= 5 else "SKIPPED"
        matches_by_level[5] = len(audit_queue)

        # Determine verdict
        if matches_by_level[1] > 0 or matches_by_level[2] > 0:
            verdict = "CONTAMINATED"
        elif len(audit_queue) > 0:
            verdict = "SUSPICIOUS_MATCHES"
        else:
            verdict = "CLEAN"

        from datetime import UTC, datetime

        return ContaminationScanReport(
            benchmark_id=benchmark_id,
            corpus_fingerprint=corpus_fingerprint,
            total_benchmark_samples=len(benchmark_samples),
            total_training_samples=len(training_samples),
            scan_date=datetime.now(UTC).isoformat(),
            level_status=level_status,
            matches_by_level=matches_by_level,
            audit_queue=audit_queue,
            verdict=verdict,
        )
