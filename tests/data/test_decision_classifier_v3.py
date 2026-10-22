"""prose-decision-classifier-v3: the freeze pins, and the one change from v2 (decline with delivered content).

v3 is the frozen v2 classifier with decline-with-content named DECLINE_WITH_CONTENT. Its freeze is the LF sha256
pinned in ``behaviour_labels_v3.FROZEN_SOURCE_SHA256_LF`` (the wrapper), and v2's own freeze is checked too.
v2 must stay byte-identical to its own freeze.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from opengrad.data import behaviour_labels_v3 as bl3
from opengrad.data import decision_classifier_v2 as dc2
from opengrad.data import decision_classifier_v3 as dc3
from opengrad.data.behaviour_labels import BehaviourLabelError
from opengrad.data.classifier_input import ClassifierFeatures

ROOT = Path(__file__).resolve().parents[2]
RATES = {
    "name": "get_exchange_rate",
    "description": "Exchange rate between currencies.",
    "parameters": {"type": "object"},
}
V2_RULES = "src/opengrad/data/decision_classifier_v2.py"
V3_WRAPPER = "src/opengrad/data/decision_classifier_v3.py"


def _decide(module, response: str):
    return module.classify(ClassifierFeatures("Help me, please.", response, (RATES,), False))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_v3_source_matches_its_pinned_freeze() -> None:
    assert _sha(ROOT / bl3.CLASSIFIER_MODULE) == bl3.FROZEN_SOURCE_SHA256_LF


def test_v2_source_is_unchanged_by_the_v3_work() -> None:
    raw = (ROOT / V2_RULES).read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "47436ca990bd4c1846113dd8d28e2c8755cff50837581acaa39654d0d2f2b382"
    )
    assert _sha(ROOT / V2_RULES) == bl3.RULES_SHA256_LF


def test_v3_names_decline_with_content_apart_and_v2_called_it_direct() -> None:
    response = (
        "I'm unable to look up live exchange rates. However, the euro is generally quoted near one US dollar "
        "in recent years, so a rough conversion is one to one."
    )
    v2 = _decide(dc2, response)
    v3 = _decide(dc3, response)
    assert (v2.label, v2.step) == (dc2.DIRECT, dc2.STEP_DECLINE_WITH_CONTENT)
    assert (v3.label, v3.step) == (dc3.DECLINE_WITH_CONTENT, dc3.STEP_DECLINE_WITH_CONTENT)


def test_site_534_a_but_clause_after_a_decline_is_named_apart() -> None:
    # v2 line 534: a "however"/"but" clause after the decline is itself the answer. Its evidence is the clause.
    response = (
        "I'm unable to look up live exchange rates. However, the euro is generally quoted near one US dollar "
        "in recent years, so a rough conversion is one to one."
    )
    v2 = _decide(dc2, response)
    assert v2.evidence[1].lower().startswith("the euro")
    assert _decide(dc3, response).label == dc3.DECLINE_WITH_CONTENT


def test_site_545_a_delivery_marker_after_a_decline_is_named_apart() -> None:
    # v2 line 545: "Here's the answer:" after a decline delivers the answer. Its evidence starts at that marker.
    response = (
        "I can't access live exchange rates right now. Here's the answer: the euro has recently traded close to "
        "one US dollar, so a rough conversion is one to one."
    )
    v2 = _decide(dc2, response)
    assert (v2.label, v2.step) == (dc2.DIRECT, dc2.STEP_DECLINE_WITH_CONTENT)
    assert v2.evidence[1].lower().startswith("here's")
    v3 = _decide(dc3, response)
    assert (v3.label, v3.step) == (dc3.DECLINE_WITH_CONTENT, dc3.STEP_DECLINE_WITH_CONTENT)


def test_site_555_a_long_passage_after_a_decline_is_named_apart() -> None:
    # v2 line 555: a decline followed by enough content words, with no clause or marker to point at.
    response = (
        "I can't access live exchange rates. Over the last several years the euro has traded in a fairly wide "
        "band against the US dollar, rising and falling with interest rate expectations, trade balances, energy "
        "prices and broader shifts in global risk appetite, so any single rate is only a snapshot of a moving "
        "market and should be read with that in mind."
    )
    v2 = _decide(dc2, response)
    assert (v2.label, v2.step) == (dc2.DIRECT, dc2.STEP_DECLINE_WITH_CONTENT)
    assert len(v2.evidence) == 1
    v3 = _decide(dc3, response)
    assert (v3.label, v3.step) == (dc3.DECLINE_WITH_CONTENT, dc3.STEP_DECLINE_WITH_CONTENT)


def test_v3_leaves_plain_delivered_answers_as_direct() -> None:
    response = "The euro is quoted at about 1.08 US dollars at the moment you asked."
    v3 = _decide(dc3, response)
    assert (v3.label, v3.step) == (dc3.DIRECT, dc3.STEP_DELIVERED)


def test_every_decision_carries_the_v3_classifier_version() -> None:
    # The wrapper stamps v3's version on every path, not only on the named case.
    responses = [
        "The euro is quoted at about 1.08 US dollars at the moment you asked.",
        "I can't access live exchange rates, and I cannot do that.",
        "I can't look up live rates. Which currency pair do you mean?",
        "I can't look up live rates, and I'd need the currency pair.",
    ]
    for response in responses:
        assert _decide(dc3, response).classifier_version == dc3.CLASSIFIER_VERSION
        assert _decide(dc3, response).classifier_version == "prose-decision-classifier-v3"


def test_non_step_3_decisions_match_v2_apart_from_the_version() -> None:
    response = "The euro is quoted at about 1.08 US dollars at the moment you asked."
    v2 = _decide(dc2, response)
    v3 = _decide(dc3, response)
    assert (v3.label, v3.step, v3.evidence) == (v2.label, v2.step, v2.evidence)
    assert v2.classifier_version == dc2.CLASSIFIER_VERSION
    assert v3.classifier_version == dc3.CLASSIFIER_VERSION


def test_v3_label_set_adds_one_label_and_keeps_the_rest() -> None:
    assert dc3.LABELS == (
        dc3.CALL,
        dc3.DIRECT,
        dc3.CLARIFY,
        dc3.UNSUPPORTED,
        dc3.ABSTAIN,
        dc3.DECLINE_WITH_CONTENT,
    )
    assert dc3.LABELS[:-1] == dc2.LABELS


def test_labels_v2_carries_no_weight_anywhere() -> None:
    assert bl3.weight_permitted("DECLINE_WITH_CONTENT", "glaive") is False
    assert bl3.weight_permitted("DIRECT", "glaive") is False
    assert bl3.weight_permitted("UNSUPPORTED", "toolace") is False


@pytest.mark.parametrize("changed", [V3_WRAPPER, V2_RULES])
def test_the_runtime_freeze_refuses_if_either_source_changes(tmp_path: Path, changed: str) -> None:
    # The wrapper is frozen, and so is the v2 rules it delegates to. Either one changing must stop the pass.
    for module in (V3_WRAPPER, V2_RULES):
        target = tmp_path / module
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / module, target)
    (tmp_path / changed).write_bytes((ROOT / changed).read_bytes() + b"\n# edited\n")
    with pytest.raises(BehaviourLabelError, match="not the frozen bytes"):
        bl3.check_frozen(tmp_path)


def test_the_runtime_freeze_passes_on_the_real_sources() -> None:
    bl3.check_frozen(ROOT)
