"""Data validation gates verifying schema, conversation trajectory, and DPO pairs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from opengrad.data.canonical import ToolConversation, semantic_hash
from opengrad.data.semantic import validate_training_trajectory


@dataclass
class DataValidationError:
    code: str
    message: str
    sample_id: str | None = None
    line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "sample_id": self.sample_id,
            "line": self.line,
        }


@dataclass
class ValidationReport:
    dataset_name: str
    total_records: int
    valid_records: int
    invalid_records: int
    status: str  # "PASS", "WARN", "FAIL"
    reason_counts: dict[str, int]
    errors: list[DataValidationError] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "status": self.status,
            "reason_counts": self.reason_counts,
            "errors": [e.to_dict() for e in self.errors],
            "timestamp": self.timestamp,
        }


def validate_dpo_record(record: dict[str, Any], index: int) -> list[DataValidationError]:
    """Validate DPO preference pair consistency."""
    errors: list[DataValidationError] = []
    rec_id = str(record.get("id", f"dpo_{index}"))

    prompt = record.get("prompt")
    chosen = record.get("chosen")
    rejected = record.get("rejected")

    if not prompt:
        errors.append(DataValidationError("EMPTY_PROMPT", "DPO prompt is empty", rec_id, index))
    if not chosen:
        errors.append(
            DataValidationError("EMPTY_CHOSEN", "DPO chosen response is empty", rec_id, index)
        )
    if not rejected:
        errors.append(
            DataValidationError("EMPTY_REJECTED", "DPO rejected response is empty", rec_id, index)
        )

    if chosen and rejected and chosen == rejected:
        errors.append(
            DataValidationError(
                "CHOSEN_EQUALS_REJECTED",
                "DPO pair has identical chosen and rejected responses",
                rec_id,
                index,
            )
        )

    return errors


def validate_records(
    records: list[dict[str, Any]], dataset_name: str = "unnamed", mode: str = "sft"
) -> ValidationReport:
    """Execute complete dataset validation gates."""
    errors: list[DataValidationError] = []
    reason_counts: Counter[str] = Counter()
    valid_count = 0

    seen_hashes: set[str] = set()

    for idx, row in enumerate(records, 1):
        rec_id = str(row.get("id", f"rec_{idx}"))

        if mode == "dpo":
            dpo_errs = validate_dpo_record(row, idx)
            if dpo_errs:
                for err in dpo_errs:
                    reason_counts[err.code] += 1
                    errors.append(err)
            else:
                valid_count += 1
            continue

        # SFT / ToolConversation validation
        try:
            example = ToolConversation(
                rec_id,
                row.get("source", {}),
                row.get("tools", []),
                row.get("messages", []),
                row.get("metadata", {}),
            )
            example.validate()

            # Check duplicate
            canon_hash = semantic_hash({"tools": example.tools, "messages": example.messages})
            if canon_hash in seen_hashes:
                code = "DUPLICATE_CONVERSATION"
                reason_counts[code] += 1
                errors.append(DataValidationError(code, "Duplicate conversation hash", rec_id, idx))
                continue
            seen_hashes.add(canon_hash)

            # Check sequence length / extreme length
            total_chars = sum(len(str(m.get("content", ""))) for m in example.messages)
            if total_chars > 32000:
                code = "EXTREME_SEQUENCE_LENGTH"
                reason_counts[code] += 1
                errors.append(
                    DataValidationError(
                        code, f"Extreme character length: {total_chars}", rec_id, idx
                    )
                )

            # Semantic trajectory issues
            issues = validate_training_trajectory(example)
            if issues:
                for issue in issues:
                    reason_counts[issue.code] += 1
                    errors.append(DataValidationError(issue.code, issue.message, rec_id, idx))
            else:
                valid_count += 1

        except (KeyError, TypeError, ValueError) as exc:
            code = "DATASET_SCHEMA_INVALID"
            reason_counts[code] += 1
            errors.append(DataValidationError(code, str(exc), rec_id, idx))

    total = len(records)
    invalid_count = total - valid_count
    status = "FAIL" if invalid_count > 0 else "PASS"

    return ValidationReport(
        dataset_name=dataset_name,
        total_records=total,
        valid_records=valid_count,
        invalid_records=invalid_count,
        status=status,
        reason_counts=dict(sorted(reason_counts.items())),
        errors=errors,
    )
