"""The README's status for each checkpoint must agree with the registries it describes.

The README and ``results/registry.jsonl`` are edited independently, so a status change can land in
one and not the other. ``test_state_consistency.py`` checks the registries against each other but
never reads the README's status column. This module does.

The M1 row is the case that motivated it. On 2026-10-22 M1-v2's promotion was withdrawn after the v4
gate was shown to miss over-refusal (``docs/research/study-002/LOG-2026-10-22-gsm8k-refusal-recount.md``,
sections 9-12). The README, ``runs/m1_dpo_canonical_v2_final_v2/experiment.json``, the checkpoint
registry and ``results/registry.jsonl`` now all record the withdrawal.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
REGISTRY = ROOT / "results" / "registry.jsonl"
CHECKPOINT_REGISTRY = ROOT / "runs" / "checkpoint_registry.json"

M1_EXPERIMENT = "m1_dpo_canonical_v2_final_v2"
M1_CHECKPOINT = f"{M1_EXPERIMENT}::dpo-checkpoint-30"

_STATUS = re.compile(r"\*\*(NOT PROMOTED|PROMOTED)\*\*")


def _readme_m1_status() -> str:
    """The bold status in the README's M1 row, which must be PROMOTED or NOT PROMOTED."""
    rows = [
        line
        for line in README.read_text(encoding="utf-8").splitlines()
        if line.startswith("| **M1** |")
    ]
    assert len(rows) == 1, "README must have exactly one M1 row in the results table"
    match = _STATUS.search(rows[0])
    assert match, f"README M1 row has no PROMOTED / NOT PROMOTED status: {rows[0]!r}"
    return match.group(1)


def _registry_status(experiment_id: str) -> str:
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("experiment_id") == experiment_id:
            return row["status"]
    raise AssertionError(f"{experiment_id} is not in results/registry.jsonl")


def _checkpoint_status(checkpoint_id: str) -> str:
    entries = json.loads(CHECKPOINT_REGISTRY.read_text(encoding="utf-8"))["checkpoints"]
    for entry in entries:
        if entry["checkpoint_id"] == checkpoint_id:
            return entry["promotion_status"]
    raise AssertionError(f"{checkpoint_id} is not in runs/checkpoint_registry.json")


def test_readme_m1_row_has_a_recognised_status() -> None:
    assert _readme_m1_status() in {"PROMOTED", "NOT PROMOTED"}


def test_readme_m1_status_matches_the_experiment_registry() -> None:
    readme = _readme_m1_status()
    registry = _registry_status(M1_EXPERIMENT)
    assert (readme == "PROMOTED") == (registry == "PROMOTED"), (
        f"README says {readme!r} for M1-v2 but results/registry.jsonl says {registry!r}"
    )


def test_readme_m1_status_matches_the_checkpoint_registry() -> None:
    readme = _readme_m1_status()
    checkpoint = _checkpoint_status(M1_CHECKPOINT)
    assert (readme == "PROMOTED") == (checkpoint == "PROMOTED"), (
        f"README says {readme!r} for M1-v2 but runs/checkpoint_registry.json says {checkpoint!r}"
    )
