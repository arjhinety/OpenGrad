"""Check 12c of ``study_002_gate_v1``: every arm was evaluated alike (04, 05).

Split from `study_002_gate.py` to keep it under the module-size limit (`tests/repo/test_module_size.py`).
The gate calls :func:`generation_identity_errors`; the constants are imported from here.
"""

from __future__ import annotations

import json
import re
from typing import Any

CODE_INVALID_COMPARISON = "INVALID_COMPARISON"

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
