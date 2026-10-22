"""The Study 002 prose decision classifier, ``prose-decision-classifier-v3`` (a successor to v2, 37).

v3 is the frozen ``prose-decision-classifier-v2`` (``decision_classifier_v2.py``), decision for decision, with one
name changed. A decline followed by delivered content is v2's ``DIRECT`` at tree step
``3_decline_with_delivered_content``; v3 names that case ``DECLINE_WITH_CONTENT``. Every other decision is v2's
unchanged, including its tree step and its evidence. Decided by the study owner on 2026-10-22 (study-002 log,
section 16).

The rules are not copied here. v2 is read and frozen as it is, so this module is a thin wrapper. Its freeze is
the LF sha256 pinned in ``behaviour_labels_v3.FROZEN_SOURCE_SHA256_LF``, which covers this wrapper only;
``behaviour_labels_v3.check_frozen`` checks v2's hash as well.

The labels it produces are the ones described in 37 and 22: ``DIRECT``, ``CLARIFY``, ``UNSUPPORTED``,
``ABSTAIN`` and ``CALL`` as in v2, plus ``DECLINE_WITH_CONTENT``.
"""

from __future__ import annotations

from dataclasses import replace

from opengrad.data import decision_classifier_v2 as _v2
from opengrad.data.classifier_input import ClassifierFeatures
from opengrad.data.decision_classifier_v2 import (
    ABSTAIN,
    CALL,
    CLARIFY,
    DIRECT,
    STEP_DECLINE_WITH_CONTENT,
    STEP_DELIVERED,
    UNSUPPORTED,
    Decision,
)

CLASSIFIER_VERSION = "prose-decision-classifier-v3"
DECLINE_WITH_CONTENT = "DECLINE_WITH_CONTENT"
LABELS = (CALL, DIRECT, CLARIFY, UNSUPPORTED, ABSTAIN, DECLINE_WITH_CONTENT)

__all__ = [
    "ABSTAIN",
    "CALL",
    "CLARIFY",
    "CLASSIFIER_VERSION",
    "DECLINE_WITH_CONTENT",
    "DIRECT",
    "LABELS",
    "STEP_DECLINE_WITH_CONTENT",
    "STEP_DELIVERED",
    "UNSUPPORTED",
    "Decision",
    "classify",
]


def classify(features: ClassifierFeatures) -> Decision:
    """v2's decision, with the one v2 DIRECT that is a decline with delivered content named apart.

    v2 returns that case from exactly three sites, all with step ``STEP_DECLINE_WITH_CONTENT`` and label DIRECT.
    No other v2 decision carries that step, so this rule changes nothing else. Every decision returned carries
    v3's ``classifier_version``.
    """
    decision = _v2.classify(features)
    if decision.label == DIRECT and decision.step == STEP_DECLINE_WITH_CONTENT:
        return Decision(DECLINE_WITH_CONTENT, decision.step, decision.evidence, CLASSIFIER_VERSION)
    return replace(decision, classifier_version=CLASSIFIER_VERSION)
