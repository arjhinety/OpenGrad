"""Executable acceptance rules for the prose decision classifier on P-DET-v1 and P-DET-COVERAGE-v1.

This is 30 §11 as code, written before any label or classifier exists, so no choice below can be made after
the results are seen (G1). It computes metrics from gold labels and predictions it is handed; it holds no
labels, runs no classifier and reads no population.

The thresholds are the frozen 22 §6 values. What this module adds, and 22 left open, is fixed here:

* **Evaluability.** A recall row needs :data:`MIN_GOLD` usable gold of its mode on that population; a
  precision row needs :data:`MIN_PREDICTIONS` predictions of its mode there. Below that the row is
  ``NOT_EVALUABLE`` for that population, and a mode whose row is evaluable on no population does not qualify.
* **Every population gates.** A row must pass on each population where it is evaluable. "Primary" and
  "secondary" are reporting labels only.
* **Challenge rows count only hard items.** On P-DET-COVERAGE-v1 the challenge subset is strata R, Q, M and
  X, never the cue-free P1/P2 (so a classifier that copies the cues cannot pass DIRECT on easy items); on
  P-DET-v1 it is the frozen challenge component. A challenge row needs :data:`MIN_GOLD_CHALLENGE` gold.
* **False CALL includes ambiguous gold.** A CALL prediction on an item whose gold is UNKNOWN (ambiguous) is a
  false CALL in the CALL precision denominator, and in stratum M such predictions are a gated count (0).
* **Abstention.** An abstention is a miss for recall and is excluded from precision denominators.
* **Macro F1** is over the modes with at least :data:`MIN_GOLD` usable gold in that population.
* **DIRECT precision on the data the classifier will label.** Per source, each coverage stratum's DIRECT
  precision is weighted by that stratum's share of the source's eligible pool (a ratio estimator), with a
  seeded stratified bootstrap interval. DIRECT balancing is permitted only for sources where it passes.

Failure behaviour is 22 §6's: a mode that does not qualify gets no balancing permission, and C1 needs both
DIRECT and UNSUPPORTED to qualify.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

METRICS_VERSION = "pdet-coverage-metrics-v1"

CALL, DIRECT, CLARIFY, UNSUPPORTED = "CALL", "DIRECT", "CLARIFY", "UNSUPPORTED"
MODES = (CALL, DIRECT, CLARIFY, UNSUPPORTED)
UNKNOWN = "UNKNOWN"
ABSTAIN = "ABSTAIN"

PDET_V1 = "P-DET-v1"
COVERAGE = "P-DET-COVERAGE-v1"
COVERAGE_CHALLENGE_STRATA = frozenset({"R", "Q", "M", "X"})

PASS, FAIL, NOT_EVALUABLE = "PASS", "FAIL", "NOT_EVALUABLE"

#: 22 §6, unchanged.
THRESHOLDS = {
    "DIRECT.recall": 0.80,
    "DIRECT.precision": 0.80,
    "UNSUPPORTED.recall": 0.75,
    "CLARIFY.f1": 0.70,
    "CALL.precision": 0.95,
    "macro_f1": 0.75,
    "abstention_rate": 0.15,
    "challenge.recall": 0.60,
    "DIRECT.poststratified_precision": 0.80,
}
#: Accepted by the study owner on 2026-09-16 (30 §13); binding since `study_002_prereg_v4`.
MIN_GOLD = 50  # 22 §5's coverage rule
MIN_PREDICTIONS = 50
MIN_GOLD_CHALLENGE = 30
MIN_PREDICTIONS_POSTSTRATIFIED = 20
MAX_CALL_ON_AMBIGUOUS_IN_M = 0
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = "opengrad-pdet-coverage-metrics-v1"


@dataclass(frozen=True)
class Item:
    """One annotated item with the classifier's prediction. ``gold`` is a mode, ``UNKNOWN`` or ``None``."""

    population: str
    gold: str | None
    prediction: str
    stratum: str | None = None  # coverage strata; None on P-DET-v1
    source: str | None = None
    challenge: bool = False  # P-DET-v1's frozen challenge component
    layer: str = "B"
    excluded: bool = False  # e.g. EXPOSED_WORKED_EXAMPLE

    @property
    def usable(self) -> bool:
        return self.gold in MODES and not self.excluded

    @property
    def ambiguous(self) -> bool:
        return self.gold == UNKNOWN and not self.excluded

    @property
    def in_challenge(self) -> bool:
        if self.population == COVERAGE:
            return self.stratum in COVERAGE_CHALLENGE_STRATA
        return self.challenge


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (centre - half, centre + half)


def _row(
    value: float | None, threshold: float, *, evaluable: bool, n: int, k: int, low: bool = False
) -> dict[str, Any]:
    status = NOT_EVALUABLE
    if evaluable and value is not None:
        status = PASS if (value <= threshold if low else value >= threshold) else FAIL
    return {
        "status": status,
        "value": value,
        "threshold": threshold,
        "n": n,
        "k": k,
        "wilson95": wilson(k, n) if n else None,
    }


def _counts(items: list[Item], mode: str) -> dict[str, int]:
    gold = [item for item in items if item.usable and item.gold == mode]
    predicted_usable = [item for item in items if item.usable and item.prediction == mode]
    return {
        "gold": len(gold),
        "true_positive": sum(item.prediction == mode for item in gold),
        "predicted": len(predicted_usable),
        "predicted_on_ambiguous": sum(item.ambiguous and item.prediction == mode for item in items),
    }


def population_metrics(items: Iterable[Item], population: str) -> dict[str, Any]:
    """Every 22 §6 row for one population, with its evaluability and interval."""
    scoped = [item for item in items if item.population == population and item.layer == "B"]
    usable = [item for item in scoped if item.usable]
    rows: dict[str, Any] = {}
    per_mode: dict[str, Any] = {}
    f1s: dict[str, float] = {}
    for mode in MODES:
        counts = _counts(scoped, mode)
        tp, gold, predicted = counts["true_positive"], counts["gold"], counts["predicted"]
        # A CALL on ambiguous gold is a false CALL: it would inject a wrong call target.
        precision_denominator = predicted + (
            counts["predicted_on_ambiguous"] if mode == CALL else 0
        )
        recall = tp / gold if gold else None
        precision = tp / precision_denominator if precision_denominator else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else 0.0
        )
        per_mode[mode] = {
            **counts,
            "recall": recall,
            "precision": precision,
            "precision_denominator": precision_denominator,
            "f1": f1,
        }
        if gold >= MIN_GOLD:
            f1s[mode] = f1
        challenge = [
            item for item in scoped if item.in_challenge and item.usable and item.gold == mode
        ]
        challenge_tp = sum(item.prediction == mode for item in challenge)
        rows[f"{mode}.challenge_recall"] = _row(
            challenge_tp / len(challenge) if challenge else None,
            THRESHOLDS["challenge.recall"],
            evaluable=len(challenge) >= MIN_GOLD_CHALLENGE,
            n=len(challenge),
            k=challenge_tp,
        )
    for key, mode, kind in (
        ("DIRECT.recall", DIRECT, "recall"),
        ("UNSUPPORTED.recall", UNSUPPORTED, "recall"),
        ("DIRECT.precision", DIRECT, "precision"),
        ("CALL.precision", CALL, "precision"),
    ):
        values = per_mode[mode]
        if kind == "recall":
            rows[key] = _row(
                values["recall"],
                THRESHOLDS[key],
                evaluable=values["gold"] >= MIN_GOLD,
                n=values["gold"],
                k=values["true_positive"],
            )
        else:
            rows[key] = _row(
                values["precision"],
                THRESHOLDS[key],
                evaluable=values["precision_denominator"] >= MIN_PREDICTIONS,
                n=values["precision_denominator"],
                k=values["true_positive"],
            )
    clarify = per_mode[CLARIFY]
    rows["CLARIFY.f1"] = _row(
        clarify["f1"],
        THRESHOLDS["CLARIFY.f1"],
        evaluable=clarify["gold"] >= MIN_GOLD,
        n=clarify["gold"],
        k=clarify["true_positive"],
    )
    rows["macro_f1"] = {
        **_row(
            sum(f1s.values()) / len(f1s) if f1s else None,
            THRESHOLDS["macro_f1"],
            evaluable=bool(f1s),
            n=len(usable),
            k=0,
        ),
        "wilson95": None,
        "modes": sorted(f1s),
    }
    abstained = sum(item.prediction == ABSTAIN for item in usable)
    rows["abstention_rate"] = _row(
        abstained / len(usable) if usable else None,
        THRESHOLDS["abstention_rate"],
        evaluable=bool(usable),
        n=len(usable),
        k=abstained,
        low=True,
    )
    if population == COVERAGE:
        ambiguous_calls = sum(
            item.ambiguous and item.stratum == "M" and item.prediction == CALL for item in scoped
        )
        rows["CALL.on_ambiguous_in_M"] = {
            "status": PASS if ambiguous_calls <= MAX_CALL_ON_AMBIGUOUS_IN_M else FAIL,
            "value": ambiguous_calls,
            "threshold": MAX_CALL_ON_AMBIGUOUS_IN_M,
            "n": sum(item.ambiguous and item.stratum == "M" for item in scoped),
            "k": ambiguous_calls,
            "wilson95": None,
        }
    return {
        "population": population,
        "items": len(scoped),
        "usable": len(usable),
        "per_mode": per_mode,
        "rows": rows,
        "confusion": dict(
            sorted(Counter(f"{item.gold}->{item.prediction}" for item in scoped).items())
        ),
    }


def _ratio_estimate(weights: Mapping[str, float], strata: Mapping[str, list[Item]]) -> float | None:
    """``sum_h w_h*TP_h/n_h / sum_h w_h*PRED_h/n_h`` over the strata that have a sample."""
    numerator = denominator = 0.0
    for stratum, weight in weights.items():
        units = strata.get(stratum) or []
        if not units:
            continue
        numerator += (
            weight
            * sum(item.prediction == DIRECT and item.gold == DIRECT for item in units)
            / len(units)
        )
        denominator += weight * sum(item.prediction == DIRECT for item in units) / len(units)
    return numerator / denominator if denominator else None


def poststratified_direct_precision(
    items: Iterable[Item], pool_strata: Mapping[str, Mapping[str, int]]
) -> dict[str, Any]:
    """Per source: DIRECT precision on the eligible pool the classifier will label, from coverage items.

    ``pool_strata[source][stratum]`` is the number of eligible layer B records of that source in that stratum
    before any sampling (the builder records it). Estimate: ``sum_h w_h*TP_h/n_h / sum_h w_h*PRED_h/n_h``.
    """
    by_source: dict[str, dict[str, list[Item]]] = defaultdict(lambda: defaultdict(list))
    for item in items:
        if item.population == COVERAGE and item.layer == "B" and item.usable and item.source:
            by_source[item.source][str(item.stratum)].append(item)
    results: dict[str, Any] = {}
    for source, pool in sorted(pool_strata.items()):
        total = sum(pool.values())
        weights = {stratum: count / total for stratum, count in pool.items() if total and count}
        sample = by_source.get(source, {})
        uncovered = sorted(stratum for stratum in weights if not sample.get(stratum))
        predicted = sum(item.prediction == DIRECT for units in sample.values() for item in units)

        value = _ratio_estimate(weights, sample)
        rng = random.Random(f"{BOOTSTRAP_SEED}|{source}")
        replicates = sorted(
            replicate
            for replicate in (
                _ratio_estimate(
                    weights,
                    {
                        stratum: [rng.choice(units) for _ in units]
                        for stratum, units in sorted(sample.items())
                    },
                )
                for _ in range(BOOTSTRAP_REPLICATES)
            )
            if replicate is not None
        )
        interval = (
            (
                replicates[int(0.025 * (len(replicates) - 1))],
                replicates[int(0.975 * (len(replicates) - 1))],
            )
            if replicates
            else None
        )
        evaluable = predicted >= MIN_PREDICTIONS_POSTSTRATIFIED and not uncovered
        status = NOT_EVALUABLE
        if evaluable and value is not None:
            status = PASS if value >= THRESHOLDS["DIRECT.poststratified_precision"] else FAIL
        results[source] = {
            "status": status,
            "value": value,
            "threshold": THRESHOLDS["DIRECT.poststratified_precision"],
            "direct_predictions": predicted,
            "weights": dict(sorted(weights.items())),
            "strata_without_sample": uncovered,
            "bootstrap95": interval,
        }
    return results


#: The rows a mode must satisfy (22 §6), plus the whole-classifier rows every mode depends on.
MODE_ROWS = {
    DIRECT: ("DIRECT.recall", "DIRECT.precision", "DIRECT.challenge_recall"),
    UNSUPPORTED: ("UNSUPPORTED.recall", "UNSUPPORTED.challenge_recall"),
    CLARIFY: ("CLARIFY.f1", "CLARIFY.challenge_recall"),
    CALL: ("CALL.precision", "CALL.challenge_recall", "CALL.on_ambiguous_in_M"),
}
GLOBAL_ROWS = ("macro_f1", "abstention_rate")


def evaluate(items: Iterable[Item], pool_strata: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    """Every population's rows, each mode's qualification, per-source DIRECT permission and C1."""
    items = list(items)
    populations = {name: population_metrics(items, name) for name in (PDET_V1, COVERAGE)}
    poststratified = poststratified_direct_precision(items, pool_strata)

    def row_verdict(row_name: str) -> str:
        statuses = [
            metrics["rows"][row_name]["status"]
            for metrics in populations.values()
            if row_name in metrics["rows"]
        ]
        if FAIL in statuses:
            return FAIL
        return PASS if PASS in statuses else NOT_EVALUABLE

    global_ok = all(row_verdict(row) == PASS for row in GLOBAL_ROWS)
    qualification: dict[str, Any] = {}
    for mode, rows in MODE_ROWS.items():
        verdicts = {row: row_verdict(row) for row in rows}
        qualifies = global_ok and all(verdict == PASS for verdict in verdicts.values())
        qualification[mode] = {"qualifies": qualifies, "rows": verdicts}
    direct_sources = sorted(
        source for source, result in poststratified.items() if result["status"] == PASS
    )
    if qualification[DIRECT]["qualifies"] and not direct_sources:
        qualification[DIRECT]["qualifies"] = False
    qualification[DIRECT]["permitted_sources"] = (
        direct_sources if qualification[DIRECT]["qualifies"] else []
    )
    return {
        "version": METRICS_VERSION,
        "global_rows": {row: row_verdict(row) for row in GLOBAL_ROWS},
        "populations": populations,
        "direct_poststratified_precision": poststratified,
        "qualification": qualification,
        "c1_authorised": qualification[DIRECT]["qualifies"]
        and qualification[UNSUPPORTED]["qualifies"],
    }
