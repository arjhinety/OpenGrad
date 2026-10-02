"""Provenance validators `V3`-`V11` (`15-PROVENANCE-VALIDATORS.md`).

`V1`, `V2` and `V12` live in :mod:`opengrad.verification.population_validators`. The nine here complete the
set of twelve, in the same shape: each returns a :class:`~opengrad.verification.accounting.ValidationResult`
whose counters must add up, each declares a population policy, an absent input is ``BLOCKED_INPUT_MISSING``
rather than a pass, and each has a fixture that makes it fail with the code 15 names (`NEGATIVE_FIXTURES`,
exercised by :func:`self_test` and by the readiness record's check 11).

* ``V3 metric_attachment`` -- every number carries the attachment of 15:41 (`FAIL_UNATTACHED_METRIC`).
* ``V4 vocabulary_lock`` -- every mode name is canonical, or an alias the artifact maps through 06's alias
  table (`FAIL_VOCABULARY`).
* ``V5 one_shot`` -- `P-CONF` and `P-UNANS` (any version) are scored once per arm, seed and protocol; a re-score
  must name the earlier run of the same key that it replaces (`FAIL_ONE_SHOT`).
* ``V6 row_key`` -- every row carries arm, partition, protocol, device class and provider (`FAIL_UNKEYED_ROW`).
* ``V7 sentinel_completeness`` -- every arm ran every registered sentinel in every required mode
  (`FAIL_MISSING_SENTINEL`).
* ``V8 census_reconciliation`` -- the `accounting.py` identities hold for an evaluation census
  (`FAIL_ACCOUNTING`).
* ``V9 fingerprint_from_artifact`` -- a fingerprint equals the value read from its artifact (`FAIL_FINGERPRINT`).
* ``V10 claim_evidence`` -- every claim's evidence path resolves and hashes to its recorded sha256
  (`FAIL_UNSUPPORTED_CLAIM`).
* ``V11 cross_study_comparability`` -- a table mixing Study 001 frozen numbers with Study 002 numbers carries
  evaluator version and partition on every row (`FAIL_INCOMPARABLE`).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from opengrad.registry.validate import result_from
from opengrad.verification.accounting import (
    ACCOUNTING_PREFIX,
    BLOCKED_INPUT_MISSING,
    CONDITIONALLY_REQUIRED,
    FAIL,
    REQUIRED_NONEMPTY,
    ValidationResult,
)
from opengrad.verification.population_validators import finite_number

CODE_UNATTACHED_METRIC = "FAIL_UNATTACHED_METRIC"
CODE_VOCABULARY = "FAIL_VOCABULARY"
CODE_ONE_SHOT = "FAIL_ONE_SHOT"
CODE_UNKEYED_ROW = "FAIL_UNKEYED_ROW"
CODE_MISSING_SENTINEL = "FAIL_MISSING_SENTINEL"
CODE_FINGERPRINT = "FAIL_FINGERPRINT"
CODE_UNSUPPORTED_CLAIM = "FAIL_UNSUPPORTED_CLAIM"
CODE_INCOMPARABLE = "FAIL_INCOMPARABLE"

#: 15:41. A number without these cannot be traced to the artifact that produced it.
ATTACHMENT_FIELDS = (
    "metric",
    "metric_version",
    "arm",
    "seed",
    "partition_id",
    "partition_fingerprint",
    "n",
    "denominator",
    "evaluator_version",
    "artifact_sha256",
)

#: 06 "Canonical enum for Study 002".
CANONICAL_MODES = ("ANSWER", "CALL", "CLARIFY", "UNSUPPORTED")

#: 06:38-41, the one alias table. An alias is accepted only where the artifact declares this mapping.
MODE_ALIASES: dict[str, str] = {
    "TOOL_CALL": "CALL",
    "REFUSE": "UNSUPPORTED",
    "request_for_info": "CLARIFY",
    "cannot_answer": "UNSUPPORTED",
    "tool_call": "CALL",
}

#: The populations scored once per arm (06 "One-shot discipline"; 15:43).
ONE_SHOT_PARTITIONS = ("P-CONF", "P-UNANS")

#: 15:44 plus 13's provider.
ROW_KEY_FIELDS = ("arm", "partition", "protocol", "device_class", "provider")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

#: Ways a table may name Study 001 or Study 002. Anything else is an unknown study, which V11 fails.
_STUDY_NAMES = {
    "001": {"1", "001", "study 001", "study-001", "study_001"},
    "002": {"2", "002", "study 002", "study-002", "study_002"},
}


def load_sentinel_registry(root: Path) -> dict[str, tuple[str, ...]]:
    """Sentinel id -> required prompt modes, from `registry/study_002_sentinels.yaml` (V7's input)."""
    import yaml

    registry = yaml.safe_load(
        (root / "registry/study_002_sentinels.yaml").read_text(encoding="utf-8")
    )
    return {
        str(s["id"]): tuple(str(m) for m in s.get("modes") or ()) for s in registry["sentinels"]
    }


def _one_shot_partition(partition: str) -> str | None:
    """The one-shot population a partition id belongs to (`P-CONF-v1` -> `P-CONF`), or None."""
    for name in ONE_SHOT_PARTITIONS:
        if partition == name or partition.startswith(name + "-"):
            return name
    return None


def _blocked(name: str, reason: str) -> ValidationResult:
    return result_from(
        name,
        CONDITIONALLY_REQUIRED,
        [name],
        [],
        blocked_ids=[name],
        blocked_status=BLOCKED_INPUT_MISSING,
        detail={"reason": reason},
    )


def _present(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and not value.strip())


def v3_metric_attachment(
    rows: Sequence[Mapping[str, Any]], *, name: str = "metric_attachment"
) -> ValidationResult:
    """Every metric row carries the full attachment, a usable `n` and denominator, and a real sha256."""
    rows = list(rows)
    if not rows:
        return _blocked(name, "no metric rows")
    ids, errors = [], []
    for index, row in enumerate(rows):
        row_id = str(row.get("id") or f"row-{index}")
        ids.append(row_id)
        missing = [field for field in ATTACHMENT_FIELDS if not _present(row.get(field))]
        if missing:
            errors.append(f"{row_id}: {CODE_UNATTACHED_METRIC}: missing {', '.join(missing)}")
            continue
        for field in ("n", "denominator"):
            count = finite_number(row[field])
            if count is None or count <= 0 or count != int(count):
                errors.append(
                    f"{row_id}: {CODE_UNATTACHED_METRIC}: {field}={row[field]!r} is not a count"
                )
        if not _SHA256.match(str(row["artifact_sha256"])):
            errors.append(f"{row_id}: {CODE_UNATTACHED_METRIC}: artifact_sha256 is not a sha256")
    return result_from(name, REQUIRED_NONEMPTY, ids, errors)


def _mode_names(record: Mapping[str, Any]) -> list[str]:
    """Every mode name a metrics record uses: its mode keys and its confusion-matrix labels."""
    names: list[str] = []
    for key in ("modes", "per_mode", "gold_counts", "predicted_counts"):
        value = record.get(key)
        if isinstance(value, (Mapping, list)):
            names.extend(str(k) for k in value)
    matrix = record.get("confusion_matrix")
    if isinstance(matrix, Mapping):
        for gold, row in matrix.items():
            names.append(str(gold))
            if isinstance(row, Mapping):
                names.extend(str(k) for k in row)
    return names


def v4_vocabulary_lock(
    records: Sequence[Mapping[str, Any]], *, name: str = "vocabulary_lock"
) -> ValidationResult:
    """Every mode name is canonical, or an alias the record maps through the one alias table."""
    records = list(records)
    if not records:
        return _blocked(name, "no metrics records")
    ids, errors = [], []
    for index, record in enumerate(records):
        record_id = str(record.get("id") or f"record-{index}")
        ids.append(record_id)
        names = _mode_names(record)
        if not names:
            errors.append(
                f"{record_id}: {CODE_VOCABULARY}: names no modes under a key the lock reads"
            )
            continue
        declared = record.get("mode_aliases") or {}
        for alias, target in dict(declared).items():
            if MODE_ALIASES.get(str(alias)) != target:
                errors.append(
                    f"{record_id}: {CODE_VOCABULARY}: declares {alias!r} -> {target!r}, "
                    "which is not 06's alias table"
                )
        for mode in sorted(set(names)):
            if mode in CANONICAL_MODES:
                continue
            if mode in MODE_ALIASES and mode in declared:
                continue
            reason = (
                "an alias used without declaring the alias table"
                if mode in MODE_ALIASES
                else "not a canonical mode and not an alias"
            )
            errors.append(f"{record_id}: {CODE_VOCABULARY}: {mode!r} is {reason}")
    return result_from(name, REQUIRED_NONEMPTY, ids, errors)


def v5_one_shot(
    artifacts: Sequence[Mapping[str, Any]], *, name: str = "one_shot"
) -> ValidationResult:
    """`P-CONF` and `P-UNANS` are scored once per (arm, seed, partition, protocol).

    A second scoring pass is accepted only as a replacement: it names, in `replacement_of`, the run id of
    the pass it replaces (06:163-165), and that run must exist.
    """
    scored = [a for a in artifacts if _one_shot_partition(str(a.get("partition"))) is not None]
    if not scored:
        return _blocked(name, "no prediction artifact on P-CONF or P-UNANS")

    def key_of(artifact: Mapping[str, Any]) -> tuple[str, ...]:
        # The full partition id: P-UNANS-v2 and P-UNANS-v2-constructed are two strata scored separately.
        return tuple(str(artifact.get(f)) for f in ("arm", "seed", "partition", "protocol"))

    by_run = {str(a.get("run_id")): a for a in scored}
    replaced: Counter[str] = Counter(
        str(a["replacement_of"]) for a in scored if a.get("replacement_of") is not None
    )

    def ends_at_an_original(run_id: str) -> bool:
        """A chain of replacements must end at a run that replaces nothing; a ring never does."""
        seen: set[str] = set()
        current: str | None = run_id
        while current is not None:
            if current in seen or current not in by_run:
                return False
            seen.add(current)
            nxt = by_run[current].get("replacement_of")
            current = None if nxt is None else str(nxt)
        return True

    keys: Counter[tuple[str, ...]] = Counter()
    ids, errors = [], []
    for index, artifact in enumerate(scored):
        artifact_id = str(artifact.get("run_id") or f"artifact-{index}")
        ids.append(artifact_id)
        key = key_of(artifact)
        replaces = artifact.get("replacement_of")
        if replaces is not None:
            target = by_run.get(str(replaces))
            if (
                target is None
                or str(replaces) == artifact_id
                or key_of(target) != key
                or str(target.get("replacement_of")) == artifact_id  # two runs replacing each other
                or replaced[str(replaces)] > 1  # one run replaced twice
                or not ends_at_an_original(artifact_id)  # a ring of replacements with no original
            ):
                errors.append(
                    f"{artifact_id}: {CODE_ONE_SHOT}: replaces {replaces!r}, which is not a recorded run of the "
                    "same arm, seed, partition and protocol that only this run replaces"
                )
            continue
        keys[key] += 1
        if keys[key] > 1:
            errors.append(
                f"{artifact_id}: {CODE_ONE_SHOT}: a second scoring pass of arm {key[0]} seed {key[1]} on "
                f"{key[2]} ({key[3]}) that names no run it replaces"
            )
    return result_from(name, REQUIRED_NONEMPTY, ids, errors)


def v6_row_key(rows: Sequence[Mapping[str, Any]], *, name: str = "row_key") -> ValidationResult:
    """Every reported row carries `(arm, partition, protocol, device_class)` and the provider (03, 13)."""
    rows = list(rows)
    if not rows:
        return _blocked(name, "no reported rows")
    ids, errors = [], []
    for index, row in enumerate(rows):
        row_id = str(row.get("id") or f"row-{index}")
        ids.append(row_id)
        missing = [field for field in ROW_KEY_FIELDS if not _present(row.get(field))]
        if missing:
            errors.append(f"{row_id}: {CODE_UNKEYED_ROW}: missing {', '.join(missing)}")
    return result_from(name, REQUIRED_NONEMPTY, ids, errors)


def v7_sentinel_completeness(
    required: Mapping[str, Sequence[str]],
    ran: Mapping[str, Mapping[str, Sequence[str]]],
    arms: Sequence[str],
    *,
    name: str = "sentinel_completeness",
) -> ValidationResult:
    """Every required arm ran every registered sentinel, in every mode the registry requires.

    `required` maps sentinel id to its modes (an empty tuple: a census sentinel with no prompt mode); read it
    with :func:`load_sentinel_registry`. `ran` maps arm to {sentinel id: modes it has an artifact for}. `arms`
    is every arm that must have run them: an arm absent from `ran` fails rather than going unchecked.
    """
    if not required:
        return _blocked(name, "no sentinel registry")
    if not arms:
        return _blocked(name, "no arm is required")
    ids, errors = [], []
    for arm in sorted(set(arms) | set(ran)):
        ids.append(arm)
        if arm not in ran:
            errors.append(f"{arm}: {CODE_MISSING_SENTINEL}: no sentinel artifact at all")
            continue
        recorded = ran[arm]
        for sentinel, modes in required.items():
            if sentinel not in recorded:
                errors.append(f"{arm}: {CODE_MISSING_SENTINEL}: no {sentinel} artifact")
                continue
            errors.extend(
                f"{arm}: {CODE_MISSING_SENTINEL}: {sentinel} has no {mode} artifact"
                for mode in modes
                if mode not in set(recorded[sentinel] or ())
            )
    return result_from(name, REQUIRED_NONEMPTY, ids, errors)


def v8_census_reconciliation(
    census: Mapping[str, Any], *, name: str = "census_reconciliation"
) -> ValidationResult:
    """The `accounting.py` identities, applied to an evaluation census read from an artifact."""
    if not census:
        return _blocked(name, "no evaluation census")
    counts = {}
    for field in ("discovered", "checked", "passed", "failed", "blocked", "skipped"):
        value = finite_number(census.get(field))
        if value is None or value < 0 or value != int(value):
            return result_from(
                name,
                REQUIRED_NONEMPTY,
                ["census"],
                [f"census: {ACCOUNTING_PREFIX}: {field}={census.get(field)!r} is not a count"],
            )
        counts[field] = int(value)
    errors = []
    if counts["discovered"] != counts["checked"] + counts["blocked"] + counts["skipped"]:
        errors.append(
            f"census: {ACCOUNTING_PREFIX}: discovered {counts['discovered']} != checked "
            f"{counts['checked']} + blocked {counts['blocked']} + skipped {counts['skipped']}"
        )
    if counts["checked"] != counts["passed"] + counts["failed"]:
        errors.append(
            f"census: {ACCOUNTING_PREFIX}: checked {counts['checked']} != passed {counts['passed']} + "
            f"failed {counts['failed']}"
        )
    return result_from(name, REQUIRED_NONEMPTY, ["census"], errors, detail=counts)


def _dotted(document: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(document, Mapping) or part not in document:
            raise KeyError(path)
        document = document[part]
    return document


def read_fingerprint(root: Path, artifact: str, field: str | None) -> str:
    """The value an artifact holds: a JSON field (dotted path), or the sha256 of its bytes."""
    path = (root / artifact).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"{artifact} is outside the repository")
    data = path.read_bytes()
    if field is None:
        return hashlib.sha256(data).hexdigest()
    return str(_dotted(json.loads(data.decode("utf-8")), field))


def v9_fingerprint_from_artifact(
    fingerprints: Sequence[Mapping[str, Any]],
    root: Path,
    *,
    name: str = "fingerprint_from_artifact",
) -> ValidationResult:
    """Each recorded fingerprint equals the value read from the artifact it names (G4).

    An entry is {name, value, artifact, field?}: without `field` the artifact's sha256 is the fingerprint.
    An artifact that does not resolve means the value was typed, not read, and fails.
    """
    fingerprints = list(fingerprints)
    if not fingerprints:
        return _blocked(name, "no fingerprints recorded")
    ids, errors = [], []
    for index, entry in enumerate(fingerprints):
        entry_id = str(entry.get("name") or f"fingerprint-{index}")
        ids.append(entry_id)
        artifact = entry.get("artifact")
        if not _present(artifact) or not _present(entry.get("value")):
            errors.append(f"{entry_id}: {CODE_FINGERPRINT}: names no artifact or no value")
            continue
        try:
            actual = read_fingerprint(root, str(artifact), entry.get("field"))
        except (OSError, ValueError, KeyError, UnicodeDecodeError) as exc:
            errors.append(
                f"{entry_id}: {CODE_FINGERPRINT}: {artifact} cannot be read ({type(exc).__name__})"
            )
            continue
        if actual != str(entry["value"]):
            errors.append(
                f"{entry_id}: {CODE_FINGERPRINT}: recorded {str(entry['value'])[:12]}… but {artifact} holds "
                f"{actual[:12]}…"
            )
    return result_from(name, REQUIRED_NONEMPTY, ids, errors)


def v10_claim_evidence(
    claims: Sequence[Mapping[str, Any]],
    root: Path,
    number_rows: Sequence[str],
    *,
    name: str = "claim_evidence",
) -> ValidationResult:
    """Every claim names a number row that exists and an evidence file that hashes to the recorded sha256."""
    known_rows = {str(row) for row in number_rows}
    claims = list(claims)
    if not claims:
        return _blocked(name, "no claims")
    ids, errors = [], []
    for index, claim in enumerate(claims):
        claim_id = str(claim.get("id") or f"claim-{index}")
        ids.append(claim_id)
        missing = [
            f
            for f in ("number_row", "evidence_path", "evidence_sha256")
            if not _present(claim.get(f))
        ]
        if missing:
            errors.append(f"{claim_id}: {CODE_UNSUPPORTED_CLAIM}: missing {', '.join(missing)}")
            continue
        if str(claim["number_row"]) not in known_rows:
            errors.append(
                f"{claim_id}: {CODE_UNSUPPORTED_CLAIM}: number row {claim['number_row']!r} does not exist"
            )
            continue
        try:
            actual = read_fingerprint(root, str(claim["evidence_path"]), None)
        except (OSError, ValueError):
            errors.append(
                f"{claim_id}: {CODE_UNSUPPORTED_CLAIM}: {claim['evidence_path']} does not resolve"
            )
            continue
        if actual != str(claim["evidence_sha256"]):
            errors.append(
                f"{claim_id}: {CODE_UNSUPPORTED_CLAIM}: {claim['evidence_path']} does not hash to the "
                "recorded sha256"
            )
    return result_from(name, REQUIRED_NONEMPTY, ids, errors)


def _study(value: Any) -> str | None:
    text = str(value).strip().casefold() if value is not None else ""
    for study, names in _STUDY_NAMES.items():
        if text in names:
            return study
    return None


def v11_cross_study_comparability(
    tables: Sequence[Mapping[str, Any]], *, name: str = "cross_study_comparability"
) -> ValidationResult:
    """A table that mixes Study 001 and Study 002 numbers carries evaluator version and partition on every row.

    Each table is {id, rows: [{study, evaluator_version, partition, ...}]}. A table holding one study only is
    checked and passes.
    """
    tables = list(tables)
    if not tables:
        return _blocked(name, "no tables")
    ids, errors = [], []
    for index, table in enumerate(tables):
        table_id = str(table.get("id") or f"table-{index}")
        ids.append(table_id)
        rows = list(table.get("rows") or [])
        studies = set()
        for row_index, row in enumerate(rows):
            study = _study(row.get("study"))
            if study is None:
                errors.append(
                    f"{table_id}: {CODE_INCOMPARABLE}: row {row_index} names no known study ({row.get('study')!r})"
                )
            studies.add(study)
        if not {"001", "002"} <= studies:
            continue
        for row_index, row in enumerate(rows):
            missing = [f for f in ("evaluator_version", "partition") if not _present(row.get(f))]
            if missing:
                errors.append(
                    f"{table_id}: {CODE_INCOMPARABLE}: row {row_index} (Study {row.get('study')}) mixes studies "
                    f"without {', '.join(missing)}"
                )
    return result_from(name, REQUIRED_NONEMPTY, ids, errors)


# ── Negative fixtures (15 "The negative tests") ────────────────────────────────────────────────────────

_SHA = "a" * 64
_ROW = {
    "id": "r",
    "metric": "answer_rate",
    "metric_version": "1",
    "arm": "C0",
    "seed": 0,
    "partition_id": "P-CONF-v1",
    "partition_fingerprint": "f" * 64,
    "n": 284,
    "denominator": 284,
    "evaluator_version": "e1",
    "artifact_sha256": _SHA,
}


def _fixtures(root: Path) -> dict[str, tuple[Callable[[], ValidationResult], str]]:
    """Each validator on the fixture 15:79-92 says must fail, with the code it must fail with."""
    missing_sha = {k: v for k, v in _ROW.items() if k != "artifact_sha256"}
    return {
        "V3": (lambda: v3_metric_attachment([missing_sha]), CODE_UNATTACHED_METRIC),
        "V4": (
            lambda: v4_vocabulary_lock([{"id": "m", "per_mode": {"request_for_info": 0.5}}]),
            CODE_VOCABULARY,
        ),
        "V5": (
            lambda: v5_one_shot(
                [
                    {
                        "run_id": "a",
                        "arm": "C0",
                        "seed": 0,
                        "partition": "P-CONF",
                        "protocol": "0-shot",
                    },
                    {
                        "run_id": "b",
                        "arm": "C0",
                        "seed": 0,
                        "partition": "P-CONF",
                        "protocol": "0-shot",
                    },
                ]
            ),
            CODE_ONE_SHOT,
        ),
        "V6": (
            lambda: v6_row_key(
                [{"arm": "C0", "partition": "P-CONF", "protocol": "0-shot", "provider": "cpu"}]
            ),
            CODE_UNKEYED_ROW,
        ),
        "V7": (
            lambda: v7_sentinel_completeness(
                {"S-REF": ("0-shot",), "S-ANS-0": ("0-shot",)},
                {"C0": {"S-ANS-0": ["0-shot"]}},
                ["C0"],
            ),
            CODE_MISSING_SENTINEL,
        ),
        "V8": (
            lambda: v8_census_reconciliation(
                {
                    "discovered": 1277,
                    "checked": 100,
                    "passed": 100,
                    "failed": 0,
                    "blocked": 0,
                    "skipped": 0,
                }
            ),
            ACCOUNTING_PREFIX,
        ),
        "V9": (
            lambda: v9_fingerprint_from_artifact(
                [{"name": "corpus", "value": "0" * 64, "artifact": "pyproject.toml"}], root
            ),
            CODE_FINGERPRINT,
        ),
        "V10": (
            lambda: v10_claim_evidence(
                [
                    {
                        "id": "c",
                        "number_row": "r",
                        "evidence_path": "reports/does-not-exist.json",
                        "evidence_sha256": _SHA,
                    }
                ],
                root,
                ["r"],
            ),
            CODE_UNSUPPORTED_CLAIM,
        ),
        "V11": (
            lambda: v11_cross_study_comparability(
                [
                    {
                        "id": "t",
                        "rows": [{"study": "001", "value": 0.555}, {"study": "002", "value": 0.6}],
                    }
                ]
            ),
            CODE_INCOMPARABLE,
        ),
    }


def self_test(root: Path) -> dict[str, dict[str, Any]]:
    """Run every validator on its failing fixture; each must FAIL and name its expected code."""
    results: dict[str, dict[str, Any]] = {}
    for validator, (run, code) in _fixtures(root).items():
        result = run()
        errors = result.all_errors()
        results[validator] = {
            "status": result.status,
            "expected_code": code,
            "matched": result.status == FAIL and any(code in error for error in errors),
        }
    return results
