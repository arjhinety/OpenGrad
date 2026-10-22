"""prose-decision-classifier-v3: the freeze pin, and the one change from v2 (decline with delivered content).

v3 is v2 with decline-with-content named DECLINE_WITH_CONTENT. Its freeze is the LF sha256 pinned in
``behaviour_labels_v3.FROZEN_SOURCE_SHA256_LF``. v2 must stay byte-identical to its own freeze.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from opengrad.data import behaviour_labels_v3 as bl3
from opengrad.data import decision_classifier_v2 as dc2
from opengrad.data import decision_classifier_v3 as dc3
from opengrad.data.classifier_input import ClassifierFeatures

ROOT = Path(__file__).resolve().parents[2]
RATES = {"name": "get_exchange_rate", "description": "Exchange rate between currencies.", "parameters": {"type": "object"}}


def _decide(module, response: str):
    return module.classify(ClassifierFeatures("Help me, please.", response, (RATES,), False))


def test_v3_source_matches_its_pinned_freeze() -> None:
    observed = hashlib.sha256((ROOT / bl3.CLASSIFIER_MODULE).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert observed == bl3.FROZEN_SOURCE_SHA256_LF


def test_v2_source_is_unchanged_by_the_v3_work() -> None:
    raw = (ROOT / "src/opengrad/data/decision_classifier_v2.py").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "47436ca990bd4c1846113dd8d28e2c8755cff50837581acaa39654d0d2f2b382"


def test_v3_names_decline_with_content_apart_and_v2_called_it_direct() -> None:
    response = (
        "I'm unable to look up live exchange rates. However, the euro is generally quoted near one US dollar "
        "in recent years, so a rough conversion is one to one."
    )
    v2 = _decide(dc2, response)
    v3 = _decide(dc3, response)
    assert (v2.label, v2.step) == (dc2.DIRECT, dc2.STEP_DECLINE_WITH_CONTENT)
    assert (v3.label, v3.step) == (dc3.DECLINE_WITH_CONTENT, dc3.STEP_DECLINE_WITH_CONTENT)


def test_v3_leaves_plain_delivered_answers_as_direct() -> None:
    response = "The euro is quoted at about 1.08 US dollars at the moment you asked."
    v3 = _decide(dc3, response)
    assert (v3.label, v3.step) == (dc3.DIRECT, dc3.STEP_DELIVERED)


def test_v3_label_set_adds_one_label_and_keeps_the_rest() -> None:
    assert dc3.LABELS == (dc3.CALL, dc3.DIRECT, dc3.CLARIFY, dc3.UNSUPPORTED, dc3.ABSTAIN, dc3.DECLINE_WITH_CONTENT)


def test_labels_v2_carries_no_weight_anywhere() -> None:
    assert bl3.weight_permitted("DECLINE_WITH_CONTENT", "glaive") is False
    assert bl3.weight_permitted("DIRECT", "glaive") is False
    assert bl3.weight_permitted("UNSUPPORTED", "toolace") is False
