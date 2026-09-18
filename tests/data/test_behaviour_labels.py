"""The C1 behaviour-labelling pass (21 phase 3; 38 §3), on synthetic records only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opengrad.data import behaviour_labels as labeller
from opengrad.data.classifier_input import HeldoutIndex
from opengrad.verification import prose_classifier_v2_oneshot as runner
from tests.data.test_classifier_input import glaive_record
from tests.data.test_classifier_input_v2 import CONTINUED_WITH_CALL
from tests.data.test_normalization_v3 import GLAIVE_CALL_CHAT

ROOT = Path(__file__).resolve().parents[2]


def test_it_is_pinned_to_the_same_freeze_as_the_one_shot_runner() -> None:
    assert labeller.FROZEN_SOURCE_SHA256_LF == runner.FROZEN_SOURCE_SHA256_LF
    assert labeller.FROZEN_TAG == runner.FROZEN_TAG
    labeller.check_frozen(ROOT)


def test_a_changed_classifier_refuses_to_label(tmp_path: Path) -> None:
    target = tmp_path / labeller.CLASSIFIER_MODULE
    target.parent.mkdir(parents=True)
    target.write_text("# not the frozen rules\n", encoding="utf-8")
    with pytest.raises(labeller.BehaviourLabelError, match="not the frozen"):
        labeller.check_frozen(tmp_path)


@pytest.mark.parametrize(
    ("label", "source", "permitted"),
    [
        ("UNSUPPORTED", "toolace", True),
        ("CLARIFY", "toolace", True),
        ("DIRECT", "glaive", True),
        ("DIRECT", "toolace", False),  # 37 §8: toolace's post-stratified DIRECT row is NOT_EVALUABLE
        ("DIRECT", "when2call", False),
        ("CALL", "glaive", False),
        (labeller.CALL_BY_STRUCTURE, "glaive", False),
        ("ABSTAIN", "glaive", False),
        (labeller.UNLABELLED, "glaive", False),
    ],
)
def test_weight_permission_follows_the_authorisation(label: str, source: str, permitted: bool) -> None:
    assert labeller.weight_permitted(label, source) is permitted


def test_a_first_reply_is_labelled_with_its_step_and_unit_kind() -> None:
    row = labeller.label_record(glaive_record(CONTINUED_WITH_CALL), "glaive", HeldoutIndex())
    assert row["label"] == "CLARIFY"
    assert row["unit_kind"] == "has_continuation"
    assert row["contract"] == "prose-decision-input-v2"
    assert row["step"] and row["reasons"] == []
    assert row["weight_permitted"] is True


def test_a_structural_first_call_is_layer_a_and_never_classified() -> None:
    row = labeller.label_record(glaive_record(GLAIVE_CALL_CHAT), "glaive", HeldoutIndex())
    assert row["label"] == labeller.CALL_BY_STRUCTURE
    assert row["step"] is None and row["unit_kind"] is None
    assert row["reasons"] == ["FIRST_REPLY_IS_STRUCTURAL_CALL"]
    assert row["weight_permitted"] is False


def test_an_ineligible_record_is_unlabelled_with_its_reasons() -> None:
    record = glaive_record(CONTINUED_WITH_CALL)
    record["messages"] = [{"role": "assistant", "content": "Hello."}]
    row = labeller.label_record(record, "glaive", HeldoutIndex())
    assert row["label"] == labeller.UNLABELLED
    assert "FIRST_TURN_NOT_ONE_USER_THEN_ASSISTANT" in row["reasons"]
    assert row["weight_permitted"] is False


def test_labelling_is_deterministic() -> None:
    record = glaive_record(CONTINUED_WITH_CALL)
    assert labeller.label_record(record, "glaive", HeldoutIndex()) == labeller.label_record(record, "glaive", HeldoutIndex())


def test_the_summary_counts_labels_permissions_and_reasons() -> None:
    rows = [
        {"label": "DIRECT", "weight_permitted": True, "unit_kind": "single_exchange", "reasons": []},
        {"label": "DIRECT", "weight_permitted": False, "unit_kind": "has_continuation", "reasons": []},
        {"label": labeller.UNLABELLED, "weight_permitted": False, "unit_kind": None, "reasons": ["MALFORMED_OR_UNRENDERABLE"]},
    ]
    assert labeller.summarise(rows) == {
        "records": 3,
        "labels": {"DIRECT": 2, labeller.UNLABELLED: 1},
        "weight_permitted": {"DIRECT": 1},
        "unit_kind": {"has_continuation": 1, "single_exchange": 1},
        "unlabelled_reasons": {"MALFORMED_OR_UNRENDERABLE": 1},
    }


def test_a_verify_catches_a_permission_that_does_not_follow_the_rule(tmp_path: Path) -> None:
    out = tmp_path / labeller.OUTPUT_DIR
    out.mkdir(parents=True)
    rows = [{"id": "a", "source": "toolace", "canonical_hash": "h", "label": "DIRECT", "step": "7_delivered",
             "unit_kind": "single_exchange", "contract": "prose-decision-input-v2", "reasons": [], "weight_permitted": True}]
    payload = labeller._rows_bytes(rows)
    (out / "toolace.labels.jsonl").write_bytes(payload)
    corpus = tmp_path / labeller.CORPUS_MANIFEST
    corpus.parent.mkdir(parents=True)
    corpus.write_text(json.dumps({"fingerprint": "f"}), encoding="utf-8")
    manifest = {
        "classifier": {"source_sha256_lf": labeller.FROZEN_SOURCE_SHA256_LF},
        "corpus_fingerprint": "f",
        "sources": {"toolace": {"file": "toolace.labels.jsonl", "sha256": labeller._sha256(payload), **labeller.summarise(rows)}},
        "totals": labeller.summarise(rows),
    }
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / labeller.CLASSIFIER_MODULE).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / labeller.CLASSIFIER_MODULE).write_bytes((ROOT / labeller.CLASSIFIER_MODULE).read_bytes())
    result = labeller.verify(tmp_path)
    assert result["status"] == "FAIL"
    assert any("38 §2" in problem for problem in result["problems"])
