"""Behaviour labels for normalization-v3 from the frozen ``prose-decision-classifier-v3``, labels-v2.

v3 names one case apart: a decline followed by delivered content is ``DECLINE_WITH_CONTENT``, where v2 said
``DIRECT``. Everything else is v2's decision, step for step (see ``decision_classifier_v3.py``). This module
builds a second, separate label set. It does not replace ``behaviour-labels-v1``, which stays as it is.

This module is one configuration of the shared labelling engine in ``behaviour_labels``: its own classifier,
its own freeze, its own output, and its own weight rule. It has no labelling logic of its own.

Weights: no row in labels-v2 carries a balancing weight. The v2 balancing authorisation (38) is void for any
new classifier version until it is tested on a new untouched population (38 §5 item 3, 22 §6). Until that test
passes, ``weight_permitted`` is False for every row, and the manifest says so.

    python -m opengrad.data.behaviour_labels_v3 --build
    python -m opengrad.data.behaviour_labels_v3 --verify
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from opengrad.data import behaviour_labels as _labels
from opengrad.data.behaviour_labels import LabelSet
from opengrad.data.classifier_input import HeldoutIndex
from opengrad.data.decision_classifier_v3 import CLASSIFIER_VERSION, classify

ROOT = Path(__file__).resolve().parents[3]
LABELS_VERSION = "behaviour-labels-v2"
CLASSIFIER_MODULE = Path("src/opengrad/data/decision_classifier_v3.py")
#: The freeze this pass is pinned to: the LF-normalised sha256 of ``decision_classifier_v3.py``, the wrapper.
FROZEN_SOURCE_SHA256_LF = "5c092f8e44e4bf3b534a68b089fc501a495f71b5d14201a1159439edb7328974"
#: The rules v3 delegates to. The wrapper is only as frozen as this file, so both are checked at every run.
RULES_MODULE = Path("src/opengrad/data/decision_classifier_v2.py")
RULES_SHA256_LF = "47436ca990bd4c1846113dd8d28e2c8755cff50837581acaa39654d0d2f2b382"
FROZEN_TAG = CLASSIFIER_VERSION

CORPUS_DIR = _labels.CORPUS_DIR
CORPUS_MANIFEST = _labels.CORPUS_MANIFEST
OUTPUT_DIR = Path("data/processed/behaviour-labels-v2")
REPORT = Path("reports/canonical-v3/behaviour-labels-v2.counts.json")


def weight_permitted(label: str, source: str) -> bool:
    """Always False in labels-v2: no balancing weight until the 22 §6 retest of this version passes."""
    del label, source
    return False


SPEC = LabelSet(
    labels_version=LABELS_VERSION,
    classify=classify,
    classifier_version=CLASSIFIER_VERSION,
    tag=FROZEN_TAG,
    classifier_module=CLASSIFIER_MODULE,
    classifier_sha256_lf=FROZEN_SOURCE_SHA256_LF,
    output_dir=OUTPUT_DIR,
    report=REPORT,
    weight=weight_permitted,
    weight_citation="38 §5 item 3 and 22 §6",
    statement=(
        "Behaviour labels for normalization-v3 from the frozen prose-decision-classifier-v3. Labels only: no "
        "mixture, no sampling weight, no training corpus. Identical to behaviour-labels-v1 except that a decline "
        "with delivered content is DECLINE_WITH_CONTENT, not DIRECT. weight_authorised is false for every row "
        "until the 22 §6 retest of this classifier version passes. Never human gold, never evidence of accuracy."
    ),
    policy={
        "supersedes_labels": "behaviour-labels-v1",
        "weight_authorised": False,
        "weight_authorised_reason": (
            "38 §5 item 3 and 22 §6: a classifier change voids the authorisation until retested"
        ),
    },
    rules=((RULES_MODULE, RULES_SHA256_LF),),
)


def check_frozen(root: Path) -> None:
    _labels.check_frozen(root, SPEC)


def label_record(record: dict[str, Any], source: str, heldout: HeldoutIndex) -> dict[str, Any]:
    return _labels.label_record(record, source, heldout, SPEC)


def label_source(root: Path, source: str, heldout: HeldoutIndex) -> Iterator[dict[str, Any]]:
    return _labels.label_source(root, source, heldout, SPEC)


def build(root: Path = ROOT) -> dict[str, Any]:
    return _labels.build(root, SPEC)


def verify(root: Path = ROOT) -> dict[str, Any]:
    return _labels.verify(root, SPEC)


def main(argv: list[str] | None = None) -> int:
    return _labels.main(argv, SPEC)


if __name__ == "__main__":
    sys.exit(main())
