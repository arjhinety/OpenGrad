from pathlib import Path

import pytest

from opengrad.contamination.heldout import (
    LEVEL_1,
    LEVEL_2,
    LEVEL_3,
    LEVEL_4,
    LEVEL_5,
    MANIFEST_ID,
    TRAINING_SOURCE_IDS,
    Record,
    screen,
)

LONG = (
    "Please schedule a follow up appointment with the cardiology department for "
    "patient record number four two seven one on the next available weekday morning"
)
SHORT = "What is the current time?"


def _record(record_id: str, source: str, text: str) -> Record:
    from opengrad.contamination.scanner import ngrams

    return Record(record_id=record_id, source=source, text=text, shingles=frozenset(ngrams(text)))


def _screen(**kwargs):
    heldout = kwargs.pop("heldout")
    training = kwargs.pop("training")
    return screen(Path("."), heldout_records=heldout, training_records=training, **kwargs)


def test_exact_and_normalized_prompt_matches_are_reported():
    heldout = [_record("h1", "mcq", SHORT)]
    training = [_record("t1", "glaive-function-calling-v2", SHORT)]
    report = _screen(heldout=heldout, training=training)
    assert report["manifest_id"] == MANIFEST_ID
    assert len(report["findings"]["level_1_exact_prompt_matches"]) == 1
    assert len(report["findings"]["level_2_normalized_prompt_matches"]) == 1
    assert report["audit_queue_size"] == 1
    assert report["audit_queue"][0]["heldout"] == "h1"


def test_whitespace_and_case_only_differences_are_level_2_not_level_1():
    # Same words, different case and spacing: normalized-equal but not byte-equal.
    heldout = [_record("h1", "mcq", LONG)]
    training = [_record("t1", "toolace", f"   {LONG.upper()}   ")]
    report = _screen(heldout=heldout, training=training)
    findings = report["findings"]
    assert findings["level_1_exact_prompt_matches"] == []
    assert len(findings["level_2_normalized_prompt_matches"]) == 1


def test_unrelated_corpora_produce_an_empty_audit_queue():
    heldout = [_record("h1", "mcq", LONG)]
    training = [_record("t1", "toolace", "compute the orbital decay of a small satellite")]
    report = _screen(heldout=heldout, training=training)
    assert report["audit_queue"] == []
    assert report["audit_queue_size"] == 0
    assert report["status"] == "LEVELS_1_4_MEASURED_LEVEL_5_PENDING"


def test_short_prompts_cannot_be_flagged_by_containment_alone():
    """A 3-shingle prompt is trivially contained in anything; it must not flag."""
    heldout = [_record("h1", "mcq", SHORT)]
    training = [_record("t1", "looptool-23k", SHORT + " and also what is the weather today please")]
    report = _screen(heldout=heldout, training=training)
    near = report["findings"]["level_3_near_duplicates"]
    assert near == [], f"short prompt produced a containment false positive: {near}"


def test_long_near_duplicate_is_flagged_by_level_3():
    heldout = [_record("h1", "mcq", LONG)]
    training = [_record("t1", "toolace", LONG + " and confirm the parking arrangements")]
    report = _screen(heldout=heldout, training=training)
    near = report["findings"]["level_3_near_duplicates"]
    assert len(near) == 1
    assert 0.0 <= near[0]["jaccard"] <= 1.0
    assert 0.0 <= near[0]["containment"] <= 1.0


def test_level_4_uses_sequence_matcher_on_near_identical_text():
    heldout = [_record("h1", "mcq", LONG)]
    training = [_record("t1", "toolace", LONG.replace("cardi", "cardio"))]
    report = _screen(heldout=heldout, training=training, edit_threshold=0.8)
    assert len(report["findings"]["level_4_semantic_matches"]) >= 1


def test_report_declares_all_levels_and_pending_human_audit():
    report = _screen(heldout=[_record("h1", "mcq", LONG)], training=[])
    assert set(report["levels"]) == {LEVEL_1, LEVEL_2, LEVEL_3, LEVEL_4, LEVEL_5}
    assert report["levels"][LEVEL_5] == "NOT_RUN"
    assert set(report["levels"].values()) >= {"MEASURED", "NOT_RUN"}
    assert "methodology" in report and "thresholds" in report["methodology"]


def test_training_sources_are_reported_only_when_seen():
    heldout = [_record("h1", "mcq", LONG)]
    training = [
        _record("t1", "toolace", "alpha beta gamma delta epsilon zeta eta theta"),
        _record("t2", "not-a-training-source", "iota kappa lambda mu nu xi omicron pi"),
    ]
    report = _screen(heldout=heldout, training=training)
    assert report["training_sources_checked"] == ["toolace"]
    assert set(report["training_sources_checked"]) <= TRAINING_SOURCE_IDS


def test_impossible_overlap_is_rejected_rather_than_reported():
    """Colliding training ids must not silently produce a Jaccard above 1.

    Two rows sharing one id double-count shared shingles. The scanner treats that
    as a broken invariant (it is what produced impossible negative Jaccard values
    before the training identity was fixed) and refuses to emit a report.
    """
    heldout = [_record("h1", "mcq", LONG)]
    training = [_record("dup", "toolace", LONG), _record("dup", "toolace", LONG)]
    with pytest.raises(AssertionError, match="exceeds held-out shingle count"):
        _screen(heldout=heldout, training=training)
