"""canonical-v3, the decision-balanced artifact (21 phase 5), on synthetic records only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opengrad.data import canonical_v3 as corpus
from opengrad.data.behavior import DECISIONS
from opengrad.data.renderers import Qwen35_2BRenderer
from tests.data.test_classifier_input import glaive_record
from tests.data.test_classifier_input_v2 import CONTINUED_WITH_CALL

ROOT = Path(__file__).resolve().parents[2]
FROZEN_HELDOUT_V1 = "reports/evaluation/behavioral-heldout-v1.manifest.json"
RENDERER_SNAPSHOT = "tests/fixtures/rendered/qwen35_2b_metadata.json"
CLASSIFIER = {"version": "prose-decision-classifier-v2", "tag": "prose-decision-classifier-v2", "source_sha256_lf": "a" * 64}

# The canonical-v3 artifacts are git-ignored local builds (.gitignore:69), so the two tests that read them
# skip on a clean checkout instead of failing on a missing file -- the same rule the other artifact tests in
# this suite follow. Build order: `python -m opengrad.data.behaviour_labels --build`, then `--build` for
# opengrad.data.canonical_v3_balance and opengrad.data.canonical_v3.
BALANCE_BUILT = (ROOT / corpus.balance.OUTPUT_DIR / "manifest.json").is_file()
CORPUS_BUILT = (ROOT / corpus.OUTPUT_DIR / "manifest.json").is_file()


def label_row(label: str, source: str = "glaive") -> dict:
    return {"id": "og_1", "source": source, "label": label, "step": "7_delivered", "unit_kind": "single_exchange"}


def test_the_mixture_vocabulary_is_used_not_the_classifier_wording() -> None:
    assert set(corpus.DECISION_OF_STRATUM.values()) <= DECISIONS
    assert corpus.DECISION_OF_STRATUM["ANSWER"] == "ANSWER"  # the classifier says DIRECT
    assert "DIRECT" not in corpus.DECISION_OF_STRATUM.values()


def test_a_structural_call_is_known_and_a_classified_reply_is_heuristic() -> None:
    call = corpus.behaviour_block("CALL", label_row("CALL_BY_STRUCTURE"), CLASSIFIER, "prose-decision-input-v2")
    assert call["decision"] == "CALL" and call["confidence"] == "known"
    assert "classifier_version" not in call["provenance"] and "structural" in call["provenance"]["basis"]

    answer = corpus.behaviour_block("ANSWER", label_row("DIRECT"), CLASSIFIER, "prose-decision-input-v2")
    assert answer["decision"] == "ANSWER" and answer["confidence"] == "heuristic"
    assert answer["provenance"]["classifier_tag"] == "prose-decision-classifier-v2"
    assert answer["provenance"]["reference_status"] == "MODEL_REFERENCE_PROVISIONAL"
    assert answer["capabilities"] == [] and call["capabilities"] == []


def test_a_contaminated_record_is_rejected_by_the_gate() -> None:
    record = glaive_record(CONTINUED_WITH_CALL)
    record["metadata"]["contamination_status"] = "CONTAMINATED"
    assert corpus.gate_record(record) == "CONTAMINATION:CONTAMINATED"


def test_an_unassessed_or_clean_record_passes_contamination() -> None:
    record = glaive_record(CONTINUED_WITH_CALL)
    for status in ("UNASSESSED", "CLEAN"):
        record["metadata"]["contamination_status"] = status
        assert corpus.gate_record(record) is None


def test_a_record_without_a_declared_supervision_kind_is_rejected() -> None:
    record = glaive_record(CONTINUED_WITH_CALL)
    record["metadata"].pop("supervision", None)
    reason = corpus.gate_record(record)
    assert reason is not None and reason.startswith("SUPERVISION:")


def test_a_semantically_broken_trajectory_is_rejected() -> None:
    record = glaive_record(CONTINUED_WITH_CALL)
    record["messages"] = []
    reason = corpus.gate_record(record)
    assert reason is not None and reason.startswith("SEMANTIC")


@pytest.mark.skipif(not BALANCE_BUILT, reason="canonical-v3 balance not built in this checkout")
def test_a_written_artifact_is_never_overwritten(tmp_path: Path) -> None:
    out = tmp_path / corpus.OUTPUT_DIR
    out.mkdir(parents=True)
    (out / "manifest.json").write_text("{}", encoding="utf-8")
    plan = tmp_path / corpus.balance.OUTPUT_DIR
    plan.mkdir(parents=True)
    (plan / "manifest.json").write_bytes((ROOT / corpus.balance.OUTPUT_DIR / "manifest.json").read_bytes())
    with pytest.raises(corpus.CanonicalV3Error):
        corpus.build(tmp_path)


@pytest.mark.skipif(not CORPUS_BUILT, reason="canonical-v3 artifact not built in this checkout")
def test_the_real_artifact_matches_its_manifest_and_the_plan() -> None:
    """The built artifact verifies: hashes, counts, decisions and the plan it came from."""
    manifest = json.loads((ROOT / corpus.REPORT).read_text(encoding="utf-8"))
    assert manifest["artifact_kind"] == corpus.ARTIFACT_KIND
    assert manifest["counts"]["rejected_by_gate"] == {}
    assert manifest["counts"]["shortfall_per_stratum"] == {}
    rendering = manifest["gates"]["renderability"]
    assert rendering["checked"] == rendering["rendered"] == manifest["counts"]["written"]
    assert rendering["failures"] == {}
    assert rendering["identity"]["template_hash"] == "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
    result = corpus.verify(ROOT)
    assert result["status"] == "PASS", result["problems"]
    assert set(result["decisions"]) == DECISIONS
    assert len(set(result["decisions"].values())) == 1  # equal shares


def test_the_renderer_identity_is_unchanged_across_the_c1_intervention() -> None:
    """Phase 7 (21): the renderer is unchanged across C1 -- before and after, equality required.

    Before the intervention the pinned Study 001 contract fixed the renderer identity in the frozen
    held-out manifest and in the committed renderer snapshot; the code declares it on the renderer
    class. After it, canonical-v3 records the identity it observed while rendering all 88,056 records.
    The records are committed artifacts, so this proof runs on a clean checkout.
    """
    before = json.loads((ROOT / FROZEN_HELDOUT_V1).read_text(encoding="utf-8"))["model_renderer_contract"]
    snapshot = json.loads((ROOT / RENDERER_SNAPSHOT).read_text(encoding="utf-8"))
    after = json.loads((ROOT / corpus.REPORT).read_text(encoding="utf-8"))["gates"]["renderability"]["identity"]

    assert before["renderer"] == snapshot["renderer"] == after["renderer"] == Qwen35_2BRenderer.renderer_version
    assert before["model_revision"] == snapshot["model_revision"] == after["model_revision"]
    assert before["template_hash"] == snapshot["chat_template_hash"] == after["template_hash"]
