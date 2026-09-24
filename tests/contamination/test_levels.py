"""The shared contamination engine (`opengrad.contamination.levels`).

The golden test pins `heldout.screen`'s whole report on a fixture that exercises every level. The
file was produced by the code *before* the level primitives moved out of `heldout.py`, so its passing
is the evidence that the move changed nothing a reviewer or a finding fingerprint could see.
"""

from __future__ import annotations

import json
from pathlib import Path

from opengrad.contamination.heldout import screen
from opengrad.contamination.levels import (
    LEVEL_5,
    NOT_RUN,
    exact_hash,
    level_status,
    normalized_hash,
)
from tests.contamination._screen_fixture import heldout_records, training_records

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures/contamination/heldout-screen-golden.json"


def test_the_heldout_report_is_byte_identical_to_the_pre_refactor_golden(tmp_path: Path) -> None:
    report = screen(
        tmp_path, heldout_records=heldout_records(), training_records=training_records()
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    golden = GOLDEN.read_bytes().decode("utf-8").replace("\r\n", "\n")
    assert rendered == golden


def test_the_golden_fixture_exercises_every_level() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    counts = {name: len(items) for name, items in golden["findings"].items()}
    assert counts == {
        "level_1_exact_prompt_matches": 1,
        "level_2_normalized_prompt_matches": 2,
        "level_3_near_duplicates": 5,
        "level_4_semantic_matches": 5,
    }
    assert golden["audit_queue_size"] == 3


def test_level_1_is_raw_text_and_level_2_is_normalised() -> None:
    assert exact_hash("Find all users") != exact_hash("  FIND ALL USERS ")
    assert normalized_hash("Find all users") == normalized_hash("  FIND ALL USERS ")
    assert exact_hash("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_no_machine_report_marks_level_5_run() -> None:
    assert level_status()[LEVEL_5] == NOT_RUN
    assert list(level_status().values()).count("MEASURED") == 4
