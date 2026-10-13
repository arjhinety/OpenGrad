"""Check 12c of ``study_002_gate_v1``: every arm was evaluated alike (04, 05).

Split from `study_002_gate.py` to keep it under the module-size limit (`tests/repo/test_module_size.py`).
The gate calls :func:`generation_identity_errors`; the constants are imported from here.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

CODE_INVALID_COMPARISON = "INVALID_COMPARISON"
#: 14: "A run that declares ``DECLARED_DETERMINISTIC`` and fails ``REP-A`` is a **defect**".
CODE_DETERMINISM = "FAIL_DETERMINISM"

#: 14's two classes. 46 §12: "One determinism mode for every arm, chosen by a rule".
DETERMINISM_MODES = frozenset({"DECLARED_DETERMINISTIC", "NON_DETERMINISTIC_KERNEL"})

#: 46 §12: "The mode is recorded in the readiness record before Tier A". Readiness check 1 reads it from the
#: GPU preflight record (``study_002_readiness.PREFLIGHT_RECORD``); the gate reads the same field there.
DETERMINISM_RECORD = ("reports/study-002/readiness/gpu-preflight.json", "determinism_mode")

#: 04 "Fixed across every arm": "Tokenizer and chat-template hash | pinned", "Evaluator version, generation
#: config, decoder | one version", "Device class for primary runs | one class, with the provider recorded".
#: Every run record carries each field as a real value (a 64-hex hash, a non-empty string, a bool, a
#: non-empty map), and all records agree on each: the check tests agreement, not particular values.
#: ``max_tokens`` is the per-benchmark budget map; ``decoding`` holds the sampling config.
RUN_IDENTITY_FIELDS = (
    "tokenizer_sha256",
    "chat_template_sha256",
    "evaluator_version",
    "engine",
    "engine_version",
    "max_tokens",
    "decoding",
    "enable_prefix_caching",
    "device_class",
)

#: 05 "Seed count": "``k = 3`` seeds per arm, identical values across all arms", the slots ``s0``, ``s1``,
#: ``s2`` "seeded ``0``, ``1``, ``2``". Every arm runs exactly these.
REQUIRED_SEEDS = frozenset({0, 1, 2})

#: 05 "What the two repeats measure": ``REP-A`` is "``C0``, seed ``s0``, re-run" (the nondeterminism floor);
#: ``REP-B`` is "``A_M0`` re-scored under the Study 002 evaluator" (configs/study_002/seed-plan.yaml). A
#: repeat's record carries ``repeat`` and is held to every factor 04 fixes, but it is not an arm's scored
#: run: it is left out of the arm set, the seed count and the one-record-per-run rule. ``None`` means the
#: seed ``A_M0`` was trained with, read from :data:`A_M0_RUN_RECORD`: a re-score trains nothing, so its seed
#: is the frozen checkpoint's, 42 in Study 001, not one of 05's 0, 1, 2 (ERRATA §38, later line).
REPEATS: dict[str, tuple[str, int | None]] = {"REP-A": ("C0", 0), "REP-B": ("A_M0", None)}

#: The Study 001 run behind ``A_M0`` (04: "the frozen M0 SFT checkpoint"; `reports/data/m0-final-freeze.json`
#: pins its config), and the field holding its training seed.
A_M0_RUN_RECORD = ("runs/m0_sft_canonical_v2_final/experiment.json", "random_seed")


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _recorded(field: str, value: Any) -> bool:
    """A held-fixed factor is recorded only as a real value, never as an empty or placeholder one."""
    if field.endswith("_sha256"):
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    if field == "enable_prefix_caching":
        return isinstance(value, bool)
    if field in ("max_tokens", "decoding"):
        # Keys are non-empty strings: mixed key types cannot be compared, and "" names nothing.
        if not isinstance(value, dict) or not value or not all(_text(k) for k in value):
            return False
        if field == "max_tokens":
            return all(type(v) is int and v > 0 for v in value.values())
        return all(v is not None for v in value.values())
    return _text(value)


def generation_identity_errors(
    runs: list[dict[str, Any]], truncation_arms: set[str], a_m0_seed: int | None
) -> list[str]:
    """The reasons the run records fail 04's held-fixed factors and 05's seeds (empty: they pass).

    ``truncation_arms`` are the arms of the bundle's truncation table: the runs must cover exactly those.
    ``a_m0_seed`` is ``A_M0``'s recorded training seed, or ``None`` when its run record cannot be read.
    Comparison rows name no arm, so their arms cannot be cross-checked here (ERRATA §38).
    """
    errors: list[str] = []
    values: dict[str, set[str]] = {field: set() for field in RUN_IDENTITY_FIELDS}
    seeds: dict[str, set[int]] = {}
    repeats_seen: set[str] = set()
    for index, run in enumerate(runs):
        run_id = f"{run.get('arm')!r}/seed {run.get('seed')!r}" if run else f"run-{index}"
        arm, seed = run.get("arm"), run.get("seed")
        # 05: seeds are the integers 0, 1, 2; 0, 0.0 and "0" are not three seeds.
        if not isinstance(arm, str) or not arm.strip() or type(seed) is not int or seed < 0:
            errors.append(f"{run_id}: {CODE_INVALID_COMPARISON}: names no arm or no integer seed")
            continue
        repeat = run.get("repeat")
        if repeat is not None:
            expected = REPEATS.get(repeat) if isinstance(repeat, str) else None
            if expected is None:
                errors.append(
                    f"{run_id}: {CODE_INVALID_COMPARISON}: {repeat!r} is not a repeat 05 names"
                )
            elif arm != expected[0] or seed != (a_m0_seed if expected[1] is None else expected[1]):
                want = a_m0_seed if expected[1] is None else expected[1]
                errors.append(
                    f"{run_id}: {CODE_INVALID_COMPARISON}: {repeat} is {expected[0]}, seed "
                    + (str(want) if want is not None else "unreadable (no A_M0 run record)")
                )
            elif repeat in repeats_seen:
                errors.append(f"{run_id}: {CODE_INVALID_COMPARISON}: a second {repeat} record")
            repeats_seen.add(str(repeat))
        else:
            if seed in seeds.setdefault(arm, set()):
                errors.append(f"{run_id}: {CODE_INVALID_COMPARISON}: a second record for one run")
            seeds[arm].add(seed)
        if not _text(run.get("provider")):
            errors.append(f"{run_id}: {CODE_INVALID_COMPARISON}: records no provider (04, 13)")
        for field in RUN_IDENTITY_FIELDS:
            if not _recorded(field, run.get(field)):
                errors.append(f"{run_id}: {CODE_INVALID_COMPARISON}: records no usable {field}")
            else:
                values[field].add(json.dumps(run[field], sort_keys=True, default=str))
    errors.extend(
        f"{field}: {CODE_INVALID_COMPARISON}: the runs disagree ({len(seen)} values); 04 holds "
        "it fixed across every arm"
        for field, seen in values.items()
        if len(seen) > 1
    )
    # An arm scored but never checked for truncation, or reported for truncation but with no run
    # record, cannot be compared (owner, 2026-10-13).
    if set(seeds) != truncation_arms:
        errors.append(
            f"arm_runs: {CODE_INVALID_COMPARISON}: the runs' arms "
            f"({', '.join(sorted(seeds)) or 'none'}) are not the truncation table's "
            f"({', '.join(sorted(truncation_arms)) or 'none'})"
        )
    # REP-A re-runs a scored arm; the residual it measures means nothing for an arm that was not scored.
    errors.extend(
        f"{repeat}: {CODE_INVALID_COMPARISON}: re-runs {REPEATS[repeat][0]}, which has no scored runs"
        for repeat in sorted(repeats_seen)
        if repeat == "REP-A" and REPEATS[repeat][0] not in seeds
    )
    if len(seeds) < 2:
        errors.append(
            f"arm_runs: {CODE_INVALID_COMPARISON}: {len(seeds)} arm(s); a comparison needs two"
        )
    errors.extend(
        f"{arm}: {CODE_INVALID_COMPARISON}: seeds {sorted(s)}, not 05's {sorted(REQUIRED_SEEDS)}"
        for arm, s in sorted(seeds.items())
        if s != REQUIRED_SEEDS
    )
    return errors


def _metrics(value: Any) -> dict[str, float] | None:
    """A non-empty map of metric name to finite number, or ``None``."""
    if not isinstance(value, dict) or not value:
        return None
    if not all(
        _text(k) and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        for k, v in value.items()
    ):
        return None
    return {str(k): float(v) for k, v in value.items()}


def determinism_errors(runs: list[dict[str, Any]], recorded_mode: str | None) -> list[str]:
    """46 §12 and 14, as written: one determinism mode for every arm, and the ``REP-A`` evidence it requires.

    14, "Evidence required": under ``DECLARED_DETERMINISTIC``, "``REP-A`` ... reproduces the metric values
    exactly"; under ``NON_DETERMINISTIC_KERNEL``, "``REP-A`` residual measured and reported". Either way a
    ``REP-A`` record is required, and it and ``C0``'s seed-0 run carry the same metrics. How large a residual
    is too large is not written (48, part (b)), so this function never judges its size.
    ``REP-B`` re-scores a Study 001 checkpoint and is not an arm, so it declares no mode.

    ``recorded_mode`` is the mode the readiness record holds (:data:`DETERMINISM_RECORD`), or ``None`` when
    there is no such record. The bundle's declaration is never trusted on its own: with no record the check
    fails, and every run must declare the recorded mode.
    """
    errors: list[str] = []
    if recorded_mode not in DETERMINISM_MODES:
        errors.append(
            f"determinism_mode: {CODE_INVALID_COMPARISON}: no mode recorded before Tier A in "
            f"{DETERMINISM_RECORD[0]} (46 §12); the bundle's declaration is not trusted"
        )
    declaring = [run for run in runs if run.get("repeat") != "REP-B"]
    modes = set()
    for index, run in enumerate(declaring):
        mode = run.get("determinism_mode")
        if isinstance(mode, str) and mode in DETERMINISM_MODES:
            modes.add(mode)
            if recorded_mode in DETERMINISM_MODES and mode != recorded_mode:
                errors.append(
                    f"{run.get('arm')!r}/seed {run.get('seed')!r}: {CODE_INVALID_COMPARISON}: declares "
                    f"{mode}, but the readiness record holds {recorded_mode} (46 §12)"
                )
        else:
            errors.append(
                f"{run.get('arm')!r}/seed {run.get('seed')!r}: {CODE_INVALID_COMPARISON}: declares no "
                f"determinism mode ({mode!r}; 14 names {', '.join(sorted(DETERMINISM_MODES))})"
                if run
                else f"run-{index}: {CODE_INVALID_COMPARISON}: declares no determinism mode"
            )
    if len(modes) > 1:
        errors.append(
            f"arm_runs: {CODE_INVALID_COMPARISON}: {len(modes)} determinism modes; 46 §12 requires one "
            "for every arm"
        )
    rep_a = [run for run in runs if run.get("repeat") == "REP-A"]
    if not rep_a:
        errors.append(
            f"REP-A: {CODE_INVALID_COMPARISON}: no REP-A record; 14 requires it in either mode"
        )
        return errors
    base = [
        run
        for run in runs
        if run.get("repeat") is None
        and run.get("arm") == "C0"
        and type(run.get("seed")) is int
        and run.get("seed") == 0
    ]
    repeated = _metrics(rep_a[0].get("metrics"))
    original = _metrics(base[0].get("metrics")) if base else None
    if repeated is None or original is None:
        errors.append(
            f"REP-A: {CODE_INVALID_COMPARISON}: REP-A and C0's seed-0 run must both carry metrics "
            "(a non-empty map of finite numbers)"
        )
        return errors
    if set(repeated) != set(original):
        errors.append(
            f"REP-A: {CODE_INVALID_COMPARISON}: REP-A and C0 seed 0 report different metrics; the "
            "residual cannot be measured"
        )
    elif modes == {"DECLARED_DETERMINISTIC"} and recorded_mode == "DECLARED_DETERMINISTIC":
        differ = sorted(name for name in original if repeated[name] != original[name])
        if differ:
            errors.append(
                f"REP-A: {CODE_DETERMINISM}: declared DECLARED_DETERMINISTIC, but REP-A does not "
                f"reproduce C0 seed 0 exactly ({', '.join(differ)}); 14: a defect, not a reclassification"
            )
    return errors
