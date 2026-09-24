"""A held-out/training fixture that exercises every screening level, shared by the golden test.

`heldout-screen-golden.json` beside the fixtures is the report `screen()` produced on this input
before the level primitives moved to `opengrad.contamination.levels` (2026-09-24). The golden test
requires the report to stay byte-identical, so a later change to the engine cannot silently change
what the audit queue and its finding fingerprints contain.
"""

from __future__ import annotations

from opengrad.contamination.heldout import Record
from opengrad.contamination.scanner import ngrams

LONG = (
    "Please schedule a follow up appointment with the cardiology department for "
    "patient record number four two seven one on the next available weekday morning"
)
OTHER = (
    "Summarise the quarterly revenue figures for the northern region and flag any "
    "month where the growth rate fell below two percent compared with last year"
)
SHORT = "What is the current time?"


def record(record_id: str, source: str, text: str) -> Record:
    return Record(record_id=record_id, source=source, text=text, shingles=frozenset(ngrams(text)))


def heldout_records() -> list[Record]:
    return [
        record("when2call-mcq:h1", "when2call-mcq", SHORT),
        record("when2call-mcq:h2", "when2call-mcq", LONG),
        record("when2call-llm-judge:h3", "when2call-llm-judge", OTHER),
        record("when2call-llm-judge:h4", "when2call-llm-judge", "compute the orbital decay"),
    ]


def training_records() -> list[Record]:
    return [
        record("glaive-function-calling-v2:t1", "glaive-function-calling-v2", SHORT),
        record("toolace:t2", "toolace", f"   {LONG.upper()}   "),
        record("toolace:t3", "toolace", LONG + " and confirm the parking arrangements"),
        record("looptool-23k:t4", "looptool-23k", OTHER.replace("northern", "southern")),
        record("button:t5", "button", "an unrelated prompt about the weather in lisbon"),
        record("when2call-sft:t6", "when2call-sft", SHORT),
    ]
