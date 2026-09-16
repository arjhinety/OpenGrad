"""Build and verify P-DET-COVERAGE-v1 (Study 002, protocol ``pdet-coverage-002-v1``).

The second classifier-validation population, specified in
``docs/research/study-002/30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md`` (cited as "30" below). It
complements P-DET-v1 and never replaces it: this module imports nothing from
:mod:`opengrad.verification.pdet` and never reads or writes ``reports/pdet/``.

``--build --output-dir DIR``
    Draw the population deterministically from the recorded normalization-v3 artifact and write
    ``pdet-coverage-v1.population.jsonl``, ``pdet-coverage-v1.manifest.json`` and a ``.sha256`` sidecar.

``--verify --output-dir DIR``
    Re-hash the written artifacts, re-derive the draw, and report whether they are unchanged.

**Adoption gate.** While 30 is a draft (:data:`PREREGISTRATION_STATUS`), the build refuses
``reports/pdet-coverage/``. A draw can only be made into a scratch directory as a dry run, and nothing
drawn there is a population. Adopting 30 (amendment ``study_002_prereg_v4``) changes the status here in
the same commit.

What this module does **not** do: it labels nothing, imports or emulates no classifier, and writes no gold
label. Strata are sampling strata (30 §7.2), never labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opengrad.data import versions
from opengrad.data.classifier_input import (
    EVALUATION_ONLY_OR_HELDOUT,
    MALFORMED_OR_UNRENDERABLE,
    STRUCTURAL_CALL,
    HeldoutIndex,
    build_classifier_input,
    eligibility,
    normalize_prompt,
)
from opengrad.data.normalization_v3 import OUTPUT_DIR as NORMALIZATION_V3_DIR
from opengrad.data.normalization_v3 import (
    SOURCE_MANIFEST,
    TOP_MANIFEST,
    file_sha256,
    iter_rows,
    lf_sha256,
    load_source_manifest,
)
from opengrad.evaluation.capability import detect_refusal
from opengrad.verification import call_fidelity
from opengrad.verification.accounting import (
    BLOCKED_INPUT_MISSING,
    FAIL,
    PASS,
    REQUIRED_NONEMPTY,
    ValidationResult,
)

POPULATION_ID = "P-DET-COVERAGE-v1"
PROTOCOL_VERSION = "pdet-coverage-002-v1"
SEED = "opengrad-pdet-coverage-002-v1"
PREREGISTRATION = Path("docs/research/study-002/30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md")
#: ``DRAFT`` until the study owner adopts 30; then ``ADOPTED`` with the amendment id, in one commit.
PREREGISTRATION_STATUS = "DRAFT"
ADOPTION_AMENDMENT: str | None = None

OUTPUT_DIR = Path("reports/pdet-coverage")
FORBIDDEN_OUTPUT_DIR = Path("reports/pdet")
POPULATION_NAME = "pdet-coverage-v1.population.jsonl"
MANIFEST_NAME = "pdet-coverage-v1.manifest.json"
DRY_RUN_NAME = "pdet-coverage-v1.dry-run.json"
CLASSIFIER_STATUS_AT_SELECTION = "NOT_IMPLEMENTED"

#: The recorded canonical-v3 pre-classifier artifact (30 §12, B-1). Anything else is refused.
INPUT_FINGERPRINT = "56e8abf2f952907c0e936ac9397bde5b0a0c4a14eda3a3ccbbd47960bff2fda3"
INPUT_TOP_MANIFEST_SHA256 = "da651a46dc3848cb7cfe9755e113a815f18a078a095978a65b36f8b739a00c29"

#: The canonical-v3 source manifest (30 §12, B-1), LF-normalized. It decides which sources feed each layer.
SOURCE_MANIFEST_SHA256 = "cc40f64eaa1d6dbff618e64d2ad3f4b7fe46b053d3db31e67bcdf1d272ef0b6b"

#: Layer B quotas (30 §7.3). X takes every eligible item up to its quota; no stratum is backfilled.
#: P1 80 and P2 40 (raised from 60 and 30 before any label existed): 50 usable DIRECT needs a 42% yield.
LAYER_B_QUOTAS = {"X": 60, "M": 60, "R": 60, "Q": 60, "P1": 80, "P2": 40}
#: Layer A spot-check allocation (30 §4), without LoopTool (30 §12, U-1). Within a source it is split
#: across trajectory-gate status in proportion to supply (:func:`split_by_gate`).
LAYER_A_QUOTAS = {"glaive": 15, "toolace": 10, "xlam": 5}
GATE_VALID = "valid"
GATE_ISSUE = "trajectory_issue"
#: How stratum M matched, recorded in the manifest only (never on an annotated item).
M_INVOCATION_TALK = "invocation_talk"
M_IDENTIFIER_NAME = "identifier_shaped_tool_name"
M_WORD_NAME = "word_tool_name"
LAYER_A = "A"
LAYER_B = "B"
MIN_USABLE_GOLD = 50

# ── the preregistered stratum predicates (30 §7.2) ───────────────────────────────────────────────────
# Copied from scripts/audit_pdet_coverage_supply.py, which 30 names as their source. The constants and the
# syntax trees of offered_names, stratum and skeleton are tested equal to the script's
# (tests/verification/test_pdet_coverage.py), so the draw cannot drift from the published predicates.
STRATA_SOURCE = Path("scripts/audit_pdet_coverage_supply.py")
QUESTION_CUES = (
    "could you",
    "can you please",
    "please provide",
    "please specify",
    "please tell me",
    "which ",
    "what is the",
    "do you have",
    "would you like",
)
HEDGE_CUES = (
    "i can't",
    "i cannot",
    "i'm unable",
    "i am unable",
    "i don't know",
    "i'm not able",
    "i do not have",
)
TEXTUAL_CALL = re.compile(
    r'\{\s*"name"\s*:|<functioncall>|<tool_call>|<TOOLCALL>|"arguments"\s*:', re.IGNORECASE
)
INVOCATION_TALK = re.compile(
    r"\b(i('ll| will| can)? (use|call|run|invoke)|let me (use|call|check|run)|using the \w+ (function|tool|api))\b",
    re.IGNORECASE,
)
STRATA = ("X", "M", "R", "Q", "P1", "P2")


def offered_names(tools: list[dict[str, Any]]) -> set[str]:
    names = set()
    for tool in tools or []:
        spec = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = spec.get("name") if isinstance(spec, dict) else None
        if name and len(name) >= 4:
            names.add(str(name).casefold())
    return names


def stratum(response: str, tools: list[dict[str, Any]]) -> str:
    """First match in priority order X, M, R, Q, P1/P2. A sampling stratum, never a label."""
    low = response.casefold()
    if TEXTUAL_CALL.search(response):
        return "X"
    if any(name in low for name in offered_names(tools)) or INVOCATION_TALK.search(response):
        return "M"
    if detect_refusal(response).is_refusal or any(cue in low for cue in HEDGE_CUES):
        return "R"
    if "?" in response or any(cue in low for cue in QUESTION_CUES):
        return "Q"
    return "P1" if tools else "P2"


def skeleton(response: str, tools: list[dict[str, Any]]) -> str:
    text = response.casefold()
    for name in offered_names(tools):
        text = text.replace(name, "<tool>")
    text = re.sub(r"\"[^\"]*\"|'[^']*'", "<q>", text)
    text = re.sub(r"\d+(\.\d+)?", "<n>", text)
    return re.sub(r"\s+", " ", text).strip()


def _tool_name(tool: Any) -> str:
    spec = tool.get("function", tool) if isinstance(tool, dict) else {}
    return str(spec.get("name") or "") if isinstance(spec, dict) else ""


def m_match_kind(response: str, tools: list[dict[str, Any]]) -> str:
    """Why a stratum-M response matched: invocation talk, an identifier-shaped tool name, or a tool name
    that is an ordinary word. Recorded for metrics only; it never changes the stratum or the draw."""
    if INVOCATION_TALK.search(response):
        return M_INVOCATION_TALK
    low = response.casefold()
    matched = [
        name for name in map(_tool_name, tools or []) if len(name) >= 4 and name.casefold() in low
    ]
    if any(re.search(r"[_.\-\d]|[a-z][A-Z]", name) for name in matched):
        return M_IDENTIFIER_NAME
    return M_WORD_NAME


# ── draw-time exclusion inputs (30 §9), pinned ──────────────────────────────────────────────────────
# Classifier eligibility (held-out ids, When2Call MCQ and LLM-judge prompts, malformed records) is the
# input contract's and is applied through `eligibility`. These are the population-construction rules.

#: (path, sha256 of the file bytes). JSONL is checked out byte for byte (.gitattributes `*.jsonl -text`).
QAD_RECOVERY_SET = (
    "manifests/quantization/m1_v2_qad_recovery_v1.jsonl",
    "a26982e42d13e2e94c48aa776f12256222c91a53ce19ebda53283dac42dfa211",
)
PDET_V1_POPULATION = (
    "reports/pdet/pdet-v1.population.jsonl",
    "6ab920877ce8004056a747f36d0a9c9ae6bd8befeb249007d70a79a6ce9e781b",
)
#: (sentinel ids, path, sha256 of the file bytes as recorded in each `.meta.json`).
SENTINEL_REQUEST_FILES = (
    (
        "S-ANS-0,S-ANS-8",
        "results/benchmarks/datasets/gsm8k_v1.jsonl",
        "8081f50e43fb5e4adca928ba1c8d7b1a60dd57e47e39ee8f316ae3de52dee297",
    ),
    (
        "S-IF",
        "results/benchmarks/datasets/ifeval_v1.jsonl",
        "15b8afe856ca52535155eb7905f942a4896475ae825e26e62e306d020a949ac4",
    ),
    (
        "S-MMLU",
        "results/benchmarks/datasets/mmlu_pro_v1.jsonl",
        "f0e3e6bee2d13058e94ac8b1cd8f29b4b8a8c9a8a9aa3c7d16117d70f5755781",
    ),
)
#: (sentinel id, path, sha256 with CRLF folded to LF: a `*.json text` file).
SENTINEL_OW7 = (
    "S-OW7",
    "results/benchmarks/openweights_parity_cases_v1.json",
    "2de2e6e7c9b74811706a3fb6a4f85e777b34fe0a2c75944ae673913b5127a82f",
)
#: Later evaluation sets 30 §9 excludes once each is frozen. None is frozen at this code version; a set
#: that freezes before this population is drawn is added here, pinned, before the draw.
LATER_EVALUATION_SETS: dict[str, str] = {
    "P-UNANS": "NOT_FROZEN",
    "P-SEALED": "NOT_FROZEN",
    "P-CONF ANSWER": "NOT_FROZEN (whichever of it and this population freezes second excludes the other)",
}

SENTINEL_PROMPT = "SENTINEL_PROMPT"
QAD_RECOVERY = "QAD_RECOVERY_SET"
PDET_V1_OVERLAP = "PDET_V1_OVERLAP"
DRAW_EXCLUSION_ORDER = (SENTINEL_PROMPT, QAD_RECOVERY, PDET_V1_OVERLAP)

# Unit dispositions beyond the input contract's reasons (30 §8.4).
LAYER_B_CANDIDATE = "LAYER_B_CANDIDATE"
LAYER_A_CANDIDATE = "LAYER_A_CANDIDATE"
SOURCE_NOT_LAYER_B = "TOOL_FREE_SINGLE_EXCHANGE_SOURCE_NOT_LAYER_B"
SOURCE_NOT_LAYER_A = "STRUCTURAL_CALL_SOURCE_NOT_LAYER_A"
CALL_NOT_IN_FIRST_ASSISTANT_TURN = "STRUCTURAL_CALL_NOT_IN_FIRST_ASSISTANT_TURN"
CALL_PREFIX_NOT_ONE_USER_TURN = "STRUCTURAL_CALL_PREFIX_NOT_ONE_USER_TURN"

_RENDERED_USER_TURN = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)

ANNOTATION_FIELDS = (
    "gold_policy_label",
    "ambiguity_status",
    "annotator_id",
    "annotator_rationale",
    "boundary_rule_cited",
    "annotation_version",
)


class CoverageInputError(ValueError):
    """An input is not the recorded one, so nothing may be drawn from it."""


class CoverageOutputError(ValueError):
    """The requested output location is not allowed for this draw."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def coverage_id(source_dataset: str, raw_record_hash: str) -> str:
    """Neutral id (30 §8.5): reveals no source."""
    return "pdetcov:" + _sha256_bytes(f"{source_dataset}:{raw_record_hash}".encode())


def rank_key(unit: str, item_id: str) -> str:
    """``sha256(seed | stratum | id)`` (30 §8.8); layer A uses the unit ``A``. Sorted descending."""
    return _sha256_bytes(f"{SEED}|{unit}|{item_id}".encode())


def order_key(item_id: str) -> str:
    """Annotation presentation order (30 §8.11), interleaving strata, layers and sources."""
    return _sha256_bytes(f"{SEED}:order:{item_id}".encode())


# ── exclusions ───────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DrawExclusions:
    """The 30 §9 construction exclusions, as normalized prompts, responses and identities."""

    sentinel_prompts: frozenset[str] = frozenset()
    qad_ids: frozenset[str] = frozenset()
    qad_prompts: frozenset[str] = frozenset()
    pdet_identities: frozenset[str] = frozenset()
    pdet_prompts: frozenset[str] = frozenset()
    pdet_responses: frozenset[str] = frozenset()
    inputs: tuple[tuple[str, str], ...] = ()

    def hits(self, candidate: Mapping[str, Any]) -> list[str]:
        """Every rule the candidate matches, in :data:`DRAW_EXCLUSION_ORDER`."""
        prompt = normalize_prompt(candidate["user_message"])
        identities = {
            candidate["record_id"],
            candidate["upstream_id"],
            candidate["raw_record_hash"],
            candidate["canonical_hash"],
        }
        found: list[str] = []
        if prompt in self.sentinel_prompts:
            found.append(SENTINEL_PROMPT)
        if identities & self.qad_ids or prompt in self.qad_prompts:
            found.append(QAD_RECOVERY)
        response = candidate.get("assistant_response")
        if (
            identities & self.pdet_identities
            or prompt in self.pdet_prompts
            or (candidate["layer"] == LAYER_B and normalize_prompt(response) in self.pdet_responses)
        ):
            found.append(PDET_V1_OVERLAP)
        return found


def _pinned_bytes(root: Path, relative: str, expected: str, *, lf: bool = False) -> bytes:
    path = root / relative
    if not path.is_file():
        raise CoverageInputError(f"exclusion input missing: {relative}")
    data = path.read_bytes()
    observed = _sha256_bytes(data.replace(b"\r\n", b"\n") if lf else data)
    if observed != expected:
        raise CoverageInputError(f"exclusion input changed: {relative} ({observed})")
    return data


def rendered_user_turns(prompt: str) -> list[str]:
    """User turns of a chat-template-rendered prompt (the QAD set stores prompts rendered)."""
    return _RENDERED_USER_TURN.findall(prompt)


def load_draw_exclusions(root: Path) -> DrawExclusions:
    sentinel: set[str] = set()
    inputs: list[tuple[str, str]] = []
    for _ids, relative, expected in SENTINEL_REQUEST_FILES:
        for line in _pinned_bytes(root, relative, expected).decode("utf-8").splitlines():
            if line.strip():
                for message in json.loads(line)["messages"]:
                    if message.get("role") == "user":
                        sentinel.add(normalize_prompt(message["content"]))
        inputs.append((relative, expected))
    _ids, relative, expected = SENTINEL_OW7
    cases = json.loads(_pinned_bytes(root, relative, expected, lf=True))
    for case in cases["cases"]:
        for turn in case["turns"]:
            if turn.get("role") == "user" and isinstance(turn.get("content"), str):
                sentinel.add(normalize_prompt(turn["content"]))
    inputs.append((relative, expected))

    qad_ids: set[str] = set()
    qad_prompts: set[str] = set()
    relative, expected = QAD_RECOVERY_SET
    for line in _pinned_bytes(root, relative, expected).decode("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            qad_ids.add(str(row["canonical_id"]))
            qad_prompts.update(
                normalize_prompt(turn) for turn in rendered_user_turns(row["prompt"])
            )
    inputs.append((relative, expected))

    identities: set[str] = set()
    prompts: set[str] = set()
    responses: set[str] = set()
    relative, expected = PDET_V1_POPULATION
    for line in _pinned_bytes(root, relative, expected).decode("utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            identities.update(
                str(item[key])
                for key in ("pdet_id", "raw_record_hash", "upstream_id", "canonical_hash")
                if item.get(key)
            )
            prompts.add(normalize_prompt(item["prompt"]))
            responses.add(normalize_prompt(item["response"]))
    inputs.append((relative, expected))
    return DrawExclusions(
        frozenset(sentinel),
        frozenset(qad_ids),
        frozenset(qad_prompts),
        frozenset(identities),
        frozenset(prompts),
        frozenset(responses),
        tuple(inputs),
    )


# ── the input artifact ──────────────────────────────────────────────────────────────────────────────


def check_input(root: Path, input_dir: Path = NORMALIZATION_V3_DIR) -> dict[str, Any]:
    """Refuse any input that is not the recorded normalization-v3 artifact (30 §5, §8.1)."""
    if "normalization-v1" in input_dir.as_posix():
        raise CoverageInputError("normalization-v1 is not a representation of canonical-v3 (30 §5)")
    directory = root / input_dir
    top_path = directory / TOP_MANIFEST
    if not top_path.is_file():
        raise CoverageInputError(f"normalization-v3 artifact missing: {input_dir.as_posix()}")
    top_sha = lf_sha256(top_path)
    if top_sha != INPUT_TOP_MANIFEST_SHA256:
        raise CoverageInputError(
            f"normalization-v3 top manifest is not the recorded one ({top_sha})"
        )
    top = json.loads(top_path.read_text(encoding="utf-8"))
    if top.get("fingerprint") != INPUT_FINGERPRINT:
        raise CoverageInputError("normalization-v3 fingerprint is not the recorded one")
    source_manifest = root / SOURCE_MANIFEST
    if not source_manifest.is_file() or lf_sha256(source_manifest) != SOURCE_MANIFEST_SHA256:
        raise CoverageInputError(
            f"{SOURCE_MANIFEST.as_posix()} is not the recorded canonical-v3 source manifest"
        )
    for name, summary in sorted(top["sources"].items()):
        manifest_path = directory / name / "manifest.json"
        if lf_sha256(manifest_path) != summary["manifest_sha256"]:
            raise CoverageInputError(f"normalization-v3 {name} manifest changed")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for shard in manifest["shards"]:
            if file_sha256(directory / name / shard["file"]) != shard["sha256"]:
                raise CoverageInputError(f"normalization-v3 {name}/{shard['file']} changed")
    return {
        "path": input_dir.as_posix(),
        "fingerprint": top["fingerprint"],
        "top_manifest_sha256_lf": top_sha,
        "source_manifests_sha256_lf": {
            name: summary["manifest_sha256"] for name, summary in sorted(top["sources"].items())
        },
        "upstream_disposition_counts": {
            name: summary["counts"] for name, summary in sorted(top["sources"].items())
        },
    }


def sampling_roles(source_manifest: Mapping[str, Any]) -> dict[str, dict[str, bool]]:
    """Which layers each source may supply, from the canonical-v3 source manifest."""
    roles: dict[str, dict[str, bool]] = {}
    for entry in source_manifest["sources"]:
        sampling = entry["pdet_coverage_v1_sampling"]
        roles[entry["name"]] = {
            LAYER_A: str(sampling["layer_a_structural_call"]).startswith("eligible"),
            LAYER_B: str(sampling["layer_b_prose"]).startswith("eligible"),
        }
    return roles


def _base_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    metadata = record["metadata"]
    source = metadata["source"]
    dataset = str(source["dataset_id"])
    raw_hash = str(metadata["raw_record_hash"])
    return {
        "pdetcov_id": coverage_id(dataset, raw_hash),
        "source_name": str(source["source_name"]),
        "source_dataset": dataset,
        "source_revision": source.get("upstream_revision"),
        "record_id": str(record["id"]),
        "upstream_id": str(source.get("upstream_id")),
        "raw_record_hash": raw_hash,
        "canonical_hash": str(record["canonical_hash"]),
    }


def classify_unit(
    record: Mapping[str, Any], heldout: HeldoutIndex, roles: Mapping[str, bool]
) -> tuple[str, dict[str, Any] | None]:
    """The record's disposition (30 §8.2-§8.4) and, for a candidate, its unit.

    Held-out records are excluded from both layers, and malformed records from layer B. A tool-free single
    exchange is a layer B unit. A record whose first assistant turn carries a structured call, after
    exactly one user turn, is a layer A unit, including when its only defect is a trajectory issue (for
    example ``MISSING_TOOL_RESULT``): routing must be audited on the records the gate rejects as well, and
    the unit records its ``trajectory_gate``. Every other record keeps the input contract's primary reason.
    """
    result = eligibility(record, heldout)
    if EVALUATION_ONLY_OR_HELDOUT in result.reasons:
        return EVALUATION_ONLY_OR_HELDOUT, None
    gate = GATE_VALID
    if MALFORMED_OR_UNRENDERABLE in result.reasons:
        defects = [detail for detail in result.details if not detail.startswith("shape:")]
        if STRUCTURAL_CALL not in result.reasons or not all(
            detail.startswith("trajectory:") for detail in defects
        ):
            return MALFORMED_OR_UNRENDERABLE, None
        gate = GATE_ISSUE
    if result.eligible:
        if not roles[LAYER_B]:
            return SOURCE_NOT_LAYER_B, None
        features = build_classifier_input(record, heldout)
        unit = {
            **_base_fields(record),
            "layer": LAYER_B,
            "user_message": features.features.user_message,
            "assistant_response": features.features.assistant_response,
            "tools": [dict(tool) for tool in features.features.tools],
            "features_sha256": features.features_sha256(),
            "classifier_input_contract": features.contract_version,
        }
        unit["stratum"] = stratum(unit["assistant_response"], unit["tools"])
        return LAYER_B_CANDIDATE, unit
    if STRUCTURAL_CALL not in result.reasons:
        return str(result.primary_reason), None
    body = [message for message in record["messages"] if message.get("role") != "system"]
    first = next(index for index, message in enumerate(body) if message.get("role") == "assistant")
    if not body[first].get("tool_calls"):
        return CALL_NOT_IN_FIRST_ASSISTANT_TURN, None
    if [message.get("role") for message in body[:first]] != ["user"]:
        return CALL_PREFIX_NOT_ONE_USER_TURN, None
    if not roles[LAYER_A]:
        return SOURCE_NOT_LAYER_A, None
    turn = body[first]
    return LAYER_A_CANDIDATE, {
        **_base_fields(record),
        "layer": LAYER_A,
        "stratum": None,
        "trajectory_gate": gate,
        "user_message": body[0]["content"],
        "assistant_response": turn.get("content") if isinstance(turn.get("content"), str) else None,
        "tools": [dict(tool) for tool in record["tools"]],
        "structured_calls": [dict(call) for call in turn["tool_calls"]],
    }


@dataclass
class Candidates:
    """Every layer A and layer B unit, with each record's disposition counted per source."""

    units: list[dict[str, Any]] = field(default_factory=list)
    dispositions: dict[str, Counter[str]] = field(default_factory=dict)


def collect_candidates(
    records: Iterable[tuple[str, Mapping[str, Any]]],
    heldout: HeldoutIndex,
    roles: Mapping[str, Mapping[str, bool]],
) -> Candidates:
    collected = Candidates()
    for source, record in records:
        disposition, unit = classify_unit(record, heldout, roles[source])
        collected.dispositions.setdefault(source, Counter())[disposition] += 1
        if unit is not None:
            collected.units.append(unit)
    return collected


def _iter_input(root: Path, roles: Mapping[str, Mapping[str, bool]]) -> Iterator[tuple[str, Any]]:
    for source in sorted(roles):
        if roles[source][LAYER_A] or roles[source][LAYER_B]:
            for record in iter_rows(root / NORMALIZATION_V3_DIR, source):
                yield source, record


# ── the draw ────────────────────────────────────────────────────────────────────────────────────────


def _unit_key(unit: Mapping[str, Any]) -> str:
    return unit["stratum"] if unit["layer"] == LAYER_B else LAYER_A


def _ranked(units: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        units,
        key=lambda unit: (rank_key(_unit_key(unit), unit["pdetcov_id"]), unit["pdetcov_id"]),
        reverse=True,
    )


def apply_draw_exclusions(
    units: Iterable[dict[str, Any]], exclusions: DrawExclusions
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Drop 30 §9 matches, counted per layer: the primary (first) rule, every rule matched, and the
    distinct normalized prompts behind the primary counts. Templated sources repeat a prompt thousands of
    times, so a record count alone overstates how much material a rule removes.
    """
    kept: list[dict[str, Any]] = []
    stats: dict[str, dict[str, Counter[str]]] = {
        layer: {"primary": Counter(), "matched": Counter()} for layer in (LAYER_A, LAYER_B)
    }
    prompts: dict[str, dict[str, set[str]]] = {LAYER_A: defaultdict(set), LAYER_B: defaultdict(set)}
    for unit in units:
        hits = exclusions.hits(unit)
        if hits:
            stats[unit["layer"]]["primary"][hits[0]] += 1
            stats[unit["layer"]]["matched"].update(hits)
            prompts[unit["layer"]][hits[0]].add(normalize_prompt(unit["user_message"]))
            continue
        kept.append(unit)
    return kept, {
        layer: {
            **{kind: dict(sorted(counter.items())) for kind, counter in counts.items()},
            "distinct_prompts_matched": {
                rule: len(values) for rule, values in sorted(prompts[layer].items())
            },
        }
        for layer, counts in stats.items()
    }


def dedup_order(units: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Layer A in rank order, then layer B strata scarcest first, each in rank order."""
    ranked = _ranked(units)
    supply = Counter(unit["stratum"] for unit in ranked if unit["layer"] == LAYER_B)
    strata = sorted(supply, key=lambda name: (supply[name], STRATA.index(name)))
    return [unit for unit in ranked if unit["layer"] == LAYER_A] + [
        unit
        for name in strata
        for unit in ranked
        if unit["layer"] == LAYER_B and unit["stratum"] == name
    ]


def deduplicate(
    units: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """30 §9 dedup: raw_record_hash, then normalized response, then normalized user prompt (one per prompt
    across the whole layer), then one per response skeleton within a stratum. Layer A: raw_record_hash,
    then normalized user prompt.

    Order decides which member of a duplicate group survives. Layer B strata are processed scarcest first
    (fewest candidates after the §9 exclusions, ties in :data:`STRATA` order), each in rank order, so a
    prompt shared across strata stays in the stratum that has least to spare. Layer A is in rank order.
    """
    stats = {
        LAYER_B: Counter(
            {"raw_record_hash": 0, "response_text": 0, "user_prompt": 0, "skeleton": 0}
        ),
        LAYER_A: Counter({"raw_record_hash": 0, "user_prompt": 0}),
    }
    seen: dict[str, set[Any]] = defaultdict(set)
    survivors: list[dict[str, Any]] = []
    for unit in dedup_order(units):
        layer = unit["layer"]
        keys = [("raw_record_hash", unit["raw_record_hash"])]
        if layer == LAYER_B:
            keys.append(("response_text", normalize_prompt(unit["assistant_response"])))
        keys.append(("user_prompt", normalize_prompt(unit["user_message"])))
        if layer == LAYER_B:
            keys.append(
                ("skeleton", (unit["stratum"], skeleton(unit["assistant_response"], unit["tools"])))
            )
        duplicate = next((name for name, key in keys if key in seen[f"{layer}:{name}"]), None)
        if duplicate is not None:
            stats[layer][duplicate] += 1
            continue
        for name, key in keys:
            seen[f"{layer}:{name}"].add(key)
        survivors.append(unit)
    return survivors, {layer: dict(counter) for layer, counter in stats.items()}


def allocate(quota: int, supply: Mapping[str, int]) -> dict[str, int]:
    """Split a quota across sources (30 §7.3). Allocation within a stratum, never backfill across strata.

    Rounds of equal shares: every source with supply left takes an equal share of what is left, capped at
    its supply, so a short source's remainder is split equally among the others. Units too few to share
    equally go one each to the sources with the largest remaining supply, ties by name. Never more than the
    stratum's total supply. On the proxy supplies of 30 §7.3 this gives M 19/26/15 and P1 28/29/3.
    """
    allocation = {name: 0 for name in sorted(supply)}
    left = max(quota, 0)
    while left > 0:
        open_sources = sorted(
            (name for name in supply if supply[name] > allocation[name]),
            key=lambda name: (-(supply[name] - allocation[name]), name),
        )
        if not open_sources:
            break
        share = left // len(open_sources)
        for name in open_sources if share else open_sources[:left]:
            taken = min(share or 1, supply[name] - allocation[name])
            allocation[name] += taken
            left -= taken
    return allocation


def split_by_gate(quota: int, supply: Mapping[str, int]) -> dict[str, int]:
    """Split a source's layer A quota across trajectory-gate status in proportion to supply.

    Every status with supply gets at least one item when the quota allows; the rest is proportional,
    remainders to the largest fractional part (ties by name); never more than a status's supply, and any
    unplaceable remainder goes to a status with supply left.
    """
    split = {gate: 0 for gate in sorted(supply)}
    present = sorted(gate for gate, count in supply.items() if count > 0)
    total = sum(supply[gate] for gate in present)
    if not present or quota <= 0:
        return split
    left = quota
    for gate in present[:left]:
        split[gate] = 1
    left -= min(left, len(present))
    if left:
        remaining = {gate: supply[gate] - split[gate] for gate in present}
        exact = {gate: left * supply[gate] / total for gate in present}
        for gate in present:
            split[gate] += min(int(exact[gate]), remaining[gate])
        left = quota - sum(split.values())
        order = sorted(present, key=lambda gate: (-(exact[gate] - int(exact[gate])), gate))
        while left > 0:
            open_gates = [gate for gate in order if supply[gate] > split[gate]]
            if not open_gates:
                break
            for gate in open_gates[:left]:
                split[gate] += 1
                left -= 1
    return split


def select(
    survivors: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Take each unit's quota per source, in rank order (30 §8.9)."""
    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for unit in _ranked(survivors):
        pools[(_unit_key(unit), unit["source_name"])].append(unit)
    chosen: list[dict[str, Any]] = []

    def take(unit_key: str, quota: int, sources: Iterable[str]) -> dict[str, Any]:
        supply = {source: len(pools.get((unit_key, source), [])) for source in sorted(sources)}
        allocation = allocate(quota, supply)
        for source, count in allocation.items():
            chosen.extend(pools.get((unit_key, source), [])[:count])
        realized = sum(allocation.values())
        return {
            "quota": quota,
            "supply": sum(supply.values()),
            "realized": realized,
            "shortage": quota - realized,
            "per_source": {
                source: {"supply": supply[source], "realized": allocation[source]}
                for source in sorted(supply)
            },
        }

    layer_b_sources = sorted({source for key, source in pools if key != LAYER_A})
    layer_b = {name: take(name, LAYER_B_QUOTAS[name], layer_b_sources) for name in STRATA}
    layer_a_sources: dict[str, Any] = {}
    for source, quota in sorted(LAYER_A_QUOTAS.items()):
        # A fixed per-source spot-check allocation (30 §4): a short source is not topped up from another.
        pool = pools.get((LAYER_A, source), [])
        by_gate = {
            gate: [unit for unit in pool if unit["trajectory_gate"] == gate]
            for gate in (GATE_ISSUE, GATE_VALID)
        }
        split = split_by_gate(quota, {gate: len(units) for gate, units in by_gate.items()})
        for gate, count in split.items():
            chosen.extend(by_gate[gate][:count])
        realized = sum(split.values())
        layer_a_sources[source] = {
            "quota": quota,
            "supply": len(pool),
            "realized": realized,
            "shortage": quota - realized,
            "per_gate": {
                gate: {"supply": len(by_gate[gate]), "realized": split[gate]}
                for gate in sorted(split)
            },
        }
    layer_a = {
        "quota": sum(LAYER_A_QUOTAS.values()),
        "realized": sum(item["realized"] for item in layer_a_sources.values()),
        "per_source": layer_a_sources,
    }
    return chosen, layer_b, layer_a


def finalize(chosen: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Presentation order (30 §8.11), index, and empty annotation fields."""
    population = sorted(
        chosen, key=lambda unit: (order_key(unit["pdetcov_id"]), unit["pdetcov_id"])
    )
    for index, unit in enumerate(population):
        unit["pdetcov_index"] = index
        unit.setdefault("structured_calls", None)
        unit.setdefault("trajectory_gate", None)
        unit.setdefault("features_sha256", None)
        unit.setdefault("classifier_input_contract", None)
        for name in ANNOTATION_FIELDS:
            unit[name] = None
        unit["classifier_version_at_selection"] = CLASSIFIER_STATUS_AT_SELECTION
    return population


def _pool_strata(units: Iterable[Mapping[str, Any]]) -> dict[str, Counter[str]]:
    pool: dict[str, Counter[str]] = defaultdict(Counter)
    for unit in units:
        pool[unit["source_name"]][unit["stratum"]] += 1
    return pool


def draw(
    candidates: Candidates, exclusions: DrawExclusions
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The deterministic draw over already-collected candidates. Pure: no IO, no clock, no RNG."""
    kept, exclusion_stats = apply_draw_exclusions(candidates.units, exclusions)
    survivors, dedup_stats = deduplicate(kept)
    chosen, layer_b, layer_a = select(survivors)
    population = finalize(chosen)
    m_kinds = {
        unit["pdetcov_id"]: m_match_kind(unit["assistant_response"], unit["tools"])
        for unit in population
        if unit["layer"] == LAYER_B and unit["stratum"] == "M"
    }
    counts = {
        "dispositions": {
            source: dict(sorted(counter.items()))
            for source, counter in sorted(candidates.dispositions.items())
        },
        "candidates": dict(sorted(Counter(unit["layer"] for unit in candidates.units).items())),
        # The eligible pool the classifier will label, per source and stratum, before any sampling: the
        # weights of the post-stratified DIRECT precision (pdet_coverage_metrics).
        "pool_strata": {
            source: dict(sorted(counter.items()))
            for source, counter in sorted(
                _pool_strata(unit for unit in candidates.units if unit["layer"] == LAYER_B).items()
            )
        },
        "draw_exclusions": exclusion_stats,
        "dedup": dedup_stats,
        "after_dedup": dict(sorted(Counter(unit["layer"] for unit in survivors).items())),
        "layer_b": layer_b,
        "layer_a": layer_a,
        "m_match_kind": dict(sorted(Counter(m_kinds.values()).items())),
        "m_match_kind_by_id": dict(sorted(m_kinds.items())),
        "realized": {
            LAYER_B: sum(item["realized"] for item in layer_b.values()),
            LAYER_A: layer_a["realized"],
            "total": len(population),
        },
        "shortages": {
            "layer_b": {
                name: item["shortage"] for name, item in layer_b.items() if item["shortage"]
            },
            "layer_a": {
                source: item["shortage"]
                for source, item in layer_a["per_source"].items()
                if item["shortage"]
            },
        },
    }
    return population, counts


def code_sha256_lf(root: Path) -> dict[str, str]:
    modules = (
        "src/opengrad/verification/pdet_coverage.py",
        "src/opengrad/verification/call_fidelity.py",
        "src/opengrad/data/classifier_input.py",
        STRATA_SOURCE.as_posix(),
    )
    return {module: lf_sha256(root / module) for module in modules}


def build_population(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Draw from the recorded artifact. A pure function of the pinned inputs and these constants."""
    input_record = check_input(root)
    source_manifest = load_source_manifest(root)
    roles = sampling_roles(source_manifest)
    heldout = HeldoutIndex.load(root)
    exclusions = load_draw_exclusions(root)
    candidates = collect_candidates(_iter_input(root, roles), heldout, roles)
    population, counts = draw(candidates, exclusions)
    structural_call_evidence = call_fidelity.measure(root)
    manifest: dict[str, Any] = {
        "artifact_kind": "PDET_COVERAGE_POPULATION",
        "schema_version": 1,
        "population_id": POPULATION_ID,
        "protocol_version": PROTOCOL_VERSION,
        "preregistration": {
            "document": PREREGISTRATION.as_posix(),
            "status_at_build": PREREGISTRATION_STATUS,
            "adoption_amendment": ADOPTION_AMENDMENT,
        },
        "seed": SEED,
        "ranking": "sha256(seed | stratum | pdetcov_id) descending; layer A uses stratum 'A'",
        "presentation_order": "sha256(seed :order: pdetcov_id) ascending",
        "id_rule": "pdetcov: + sha256(source_dataset : raw_record_hash)",
        "input": {**input_record, "source_manifest_sha256_lf": SOURCE_MANIFEST_SHA256},
        "classifier_input_contract": versions.CLASSIFIER_INPUT_CONTRACT_VERSION,
        "heldout_inputs": [list(item) for item in heldout.inputs],
        "draw_exclusion_inputs": [list(item) for item in exclusions.inputs],
        "draw_exclusion_order": list(DRAW_EXCLUSION_ORDER),
        "later_evaluation_sets": LATER_EVALUATION_SETS,
        "sampling_roles": roles,
        "stratum_predicates": {
            "source": STRATA_SOURCE.as_posix(),
            "priority": list(STRATA),
        },
        "quotas": {"layer_b": LAYER_B_QUOTAS, "layer_a": LAYER_A_QUOTAS},
        "backfill": "none: an undersupplied stratum takes all it has and reports its shortage",
        "min_usable_gold_per_mode_or_boundary": MIN_USABLE_GOLD,
        "counts": counts,
        "dedup_order": "layer B strata scarcest first (candidates after exclusions), each in rank order",
        "structural_call_evidence": structural_call_evidence,
        "call_permission_basis": (
            "Layer B textual CALL is NOT_EVALUABLE unless stratum X yields enough usable gold. Structural "
            "CALL rests on structural_call_evidence: every call re-read from the raw upstream row by a parser "
            "independent of the adapters must equal the structured call (count, name, arguments), on every "
            "accepted row whatever its trajectory-gate status. The human layer A spot check audits "
            "plausibility only, with error bounds per source, never pooled."
        ),
        "code_sha256_lf": code_sha256_lf(root),
        "classifier_status_at_selection": CLASSIFIER_STATUS_AT_SELECTION,
        "selection_used_classifier": False,
        "gold_labels_present": False,
        "annotation_fields": list(ANNOTATION_FIELDS),
        "statement": (
            "A constructed boundary-coverage population. It estimates no natural prevalence and its class "
            "mix is not a property of any corpus. Items were selected from the preregistered stratum "
            "predicates and structural facts only; no decision classifier existed or was consulted. Strata "
            "are sampling strata, not labels."
        ),
    }
    return population, manifest


# ── writing and verifying ───────────────────────────────────────────────────────────────────────────


def population_bytes(population: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for item in population
    )


def manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def resolve_output_dir(root: Path, output_dir: Path, *, dry_run: bool = False) -> Path:
    """Where a build may write. ``reports/pdet/`` never; a population in ``reports/pdet-coverage/`` only
    once 30 is adopted. A counts-only dry run holds no item, so it may be written there before adoption."""
    resolved = (output_dir if output_dir.is_absolute() else root / output_dir).resolve()
    forbidden = (root / FORBIDDEN_OUTPUT_DIR).resolve()
    if resolved == forbidden or forbidden in resolved.parents:
        raise CoverageOutputError("P-DET-COVERAGE-v1 is never written under reports/pdet/ (30)")
    official = (root / OUTPUT_DIR).resolve()
    if (
        not dry_run
        and (resolved == official or official in resolved.parents)
        and PREREGISTRATION_STATUS != "ADOPTED"
    ):
        raise CoverageOutputError(
            f"{PREREGISTRATION.as_posix()} is {PREREGISTRATION_STATUS}: no population may be written to "
            f"{OUTPUT_DIR.as_posix()}/ before adoption. Build into a scratch directory for a dry run."
        )
    return resolved


def write_population(
    root: Path, output_dir: Path, population: list[dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Write the artifacts, stamping hashes over the exact bytes. Never overwrites a drawn population."""
    directory = resolve_output_dir(root, output_dir)
    if (directory / POPULATION_NAME).exists() or (directory / MANIFEST_NAME).exists():
        raise CoverageOutputError(
            f"a population already exists in {directory}; it is never overwritten"
        )
    directory.mkdir(parents=True, exist_ok=True)
    payload = population_bytes(population)
    stamped_manifest = {
        **manifest,
        "population_file": POPULATION_NAME,
        "population_records": len(population),
        "population_sha256": _sha256_bytes(payload),
    }
    stamped = manifest_bytes(stamped_manifest)
    (directory / POPULATION_NAME).write_bytes(payload)
    (directory / MANIFEST_NAME).write_bytes(stamped)
    (directory / (MANIFEST_NAME + ".sha256")).write_bytes((_sha256_bytes(stamped) + "\n").encode())
    return stamped_manifest


def write_dry_run(
    root: Path, output_dir: Path, population: list[dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Write the manifest of a draw without its population: counts and hashes, no item and no item id.

    The draw is byte-reproducible, so a written population *is* the future blind sample. Its annotator must
    never see it before annotation, so a dry run records only what the draw would be, never what it holds.
    """
    directory = resolve_output_dir(root, output_dir, dry_run=True)
    directory.mkdir(parents=True, exist_ok=True)
    counts = {
        key: value for key, value in manifest["counts"].items() if key != "m_match_kind_by_id"
    }
    record = {
        **manifest,
        "artifact_kind": "PDET_COVERAGE_DRY_RUN",
        "counts": counts,
        "population_written": False,
        "population_records": len(population),
        "population_sha256": _sha256_bytes(population_bytes(population)),
        "statement": (
            "Counts-only dry run. No population file was written, and no item text, item id, source or "
            "stratum of an individual item is recorded here. " + manifest["statement"]
        ),
    }
    (directory / DRY_RUN_NAME).write_bytes(manifest_bytes(record))
    return record


def derivation_inputs_missing(root: Path) -> list[str]:
    """Git-ignored inputs a re-derivation reads that this checkout lacks (a clean clone has none)."""
    from opengrad.data.classifier_input import HELDOUT_PROMPT_FILES

    paths = [
        (NORMALIZATION_V3_DIR / TOP_MANIFEST).as_posix(),
        *(relative for relative, _sha in HELDOUT_PROMPT_FILES),
        *(relative for _ids, relative, _sha in SENTINEL_REQUEST_FILES),
    ]
    if (root / SOURCE_MANIFEST).is_file():
        # The raw upstream rows the structural call evidence re-reads.
        paths.extend(
            str(entry["raw_artifact"]["path"])
            for entry in load_source_manifest(root)["sources"]
            if entry["name"] in call_fidelity.SOURCES
        )
    else:
        paths.append(SOURCE_MANIFEST.as_posix())
    return [path for path in paths if not (root / path).is_file()]


#: Manifest fields a fresh derivation must reproduce exactly.
REPRODUCED_FIELDS = (
    "counts",
    "structural_call_evidence",
    "input",
    "heldout_inputs",
    "draw_exclusion_inputs",
    "quotas",
    "seed",
    "protocol_version",
)


def verify_population(root: Path, output_dir: Path) -> tuple[ValidationResult, dict[str, Any]]:
    """Prove a drawn population is unchanged and reproducible.

    Blocking checks: both artifacts exist; the population matches ``population_sha256``; the manifest
    matches its sidecar; a fresh draw reproduces the bytes and the counts; no item matches a 30 §9
    exclusion; no item carries an annotation; no classifier was used. The re-draw and contamination checks
    need git-ignored inputs. Without them the result is ``BLOCKED_INPUT_MISSING``, never ``PASS``.
    """
    directory = output_dir if output_dir.is_absolute() else root / output_dir
    population_path = directory / POPULATION_NAME
    manifest_path = directory / MANIFEST_NAME
    sidecar = directory / (MANIFEST_NAME + ".sha256")
    if not population_path.is_file() or not manifest_path.is_file():
        return (
            ValidationResult(
                name="pdet-coverage population",
                policy=REQUIRED_NONEMPTY,
                discovered=0,
                checked=0,
                failed=1,
                errors=["FAIL_NONVACUOUS: P-DET-COVERAGE-v1 artifacts are missing"],
            ),
            {"status": "MISSING"},
        )
    populated = population_path.read_bytes()
    manifest_text = manifest_path.read_bytes()
    manifest = json.loads(manifest_text)
    items = [json.loads(line) for line in populated.decode("utf-8").splitlines() if line.strip()]
    errors: list[str] = []
    if manifest.get("population_sha256") != _sha256_bytes(populated):
        errors.append("FAIL_HASH: population bytes do not match manifest population_sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != _sha256_bytes(
        manifest_text
    ):
        errors.append("FAIL_HASH: manifest bytes do not match the .sha256 sidecar")

    blocked_reasons: list[str] = []
    missing = derivation_inputs_missing(root)
    if missing:
        blocked_reasons.append(
            f"{BLOCKED_INPUT_MISSING}: derivation inputs absent from this checkout {missing}; "
            "reproducibility and contamination were not verified"
        )
    else:
        rebuilt, rebuilt_manifest = build_population(root)
        if population_bytes(rebuilt) != populated:
            errors.append(
                "FAIL_REPRODUCIBILITY: a fresh draw did not reproduce the population bytes"
            )
        for key in REPRODUCED_FIELDS:
            if manifest.get(key) != rebuilt_manifest.get(key):
                errors.append(f"FAIL_PROVENANCE: manifest field {key!r} differs from a fresh draw")
        exclusions = load_draw_exclusions(root)
        heldout = HeldoutIndex.load(root)
        contaminated = [
            item["pdetcov_id"]
            for item in items
            if exclusions.hits(item)
            or normalize_prompt(item["user_message"]) in heldout.prompts
            or {item["record_id"], item["upstream_id"]} & heldout.record_ids
        ]
        if contaminated:
            errors.append(f"FAIL_CONTAMINATION: {len(contaminated)} items match excluded material")

    labelled = [
        item["pdetcov_id"] for item in items if item.get("gold_policy_label") not in (None, "")
    ]
    if labelled and not manifest.get("gold_labels_present"):
        errors.append(
            f"FAIL_ANNOTATION: {len(labelled)} items carry a gold label in a population file"
        )
    if manifest.get("selection_used_classifier") or (
        manifest.get("classifier_status_at_selection") != CLASSIFIER_STATUS_AT_SELECTION
    ):
        errors.append("FAIL_METHOD: manifest does not record that no classifier was used to select")
    layers = Counter(item.get("layer") for item in items)
    if set(layers) - {LAYER_A, LAYER_B}:
        errors.append(f"FAIL_LAYER: unexpected layer values {sorted(map(str, layers))}")

    is_blocked = bool(blocked_reasons) and not errors
    result = ValidationResult(
        name="pdet-coverage population",
        policy=REQUIRED_NONEMPTY,
        discovered=len(items),
        checked=0 if is_blocked else len(items),
        passed=len(items) if not errors and not is_blocked else 0,
        failed=len(items) if errors else 0,
        blocked=len(items) if is_blocked else 0,
        errors=errors,
        blocked_reasons=blocked_reasons,
        detail={"layer_a": layers.get(LAYER_A, 0), "layer_b": layers.get(LAYER_B, 0)},
        blocked_status=BLOCKED_INPUT_MISSING if is_blocked else None,
    )
    if errors or result.accounting_errors():
        status = FAIL
    elif is_blocked:
        status = BLOCKED_INPUT_MISSING
    else:
        status = PASS
    return result, {
        "status": status,
        "population_id": manifest.get("population_id"),
        "preregistration_status_at_build": (manifest.get("preregistration") or {}).get(
            "status_at_build"
        ),
        "population_sha256": manifest.get("population_sha256"),
        "items": len(items),
        "layer_a": layers.get(LAYER_A, 0),
        "layer_b": layers.get(LAYER_B, 0),
        "labelled_items": len(labelled),
        "errors": errors + result.accounting_errors(),
        "blocked": blocked_reasons,
    }


def summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Counts only, no record text: what a dry run reports."""
    counts = manifest["counts"]
    return {
        "preregistration_status_at_build": manifest["preregistration"]["status_at_build"],
        "population_sha256": manifest.get("population_sha256"),
        "realized": counts["realized"],
        "layer_b": {
            name: {
                "quota": item["quota"],
                "supply": item["supply"],
                "realized": item["realized"],
                "shortage": item["shortage"],
                "per_source": {src: v["realized"] for src, v in item["per_source"].items()},
            }
            for name, item in counts["layer_b"].items()
        },
        "layer_a": counts["layer_a"],
        "draw_exclusions": counts["draw_exclusions"],
        "dedup": counts["dedup"],
        "shortages": counts["shortages"],
        "m_match_kind": counts["m_match_kind"],
        "structural_call_evidence_status": manifest["structural_call_evidence"]["status"],
    }


DRY_RUN_TABLE_START = (
    "<!-- dry-run-table:start (generated by pdet_coverage.render_dry_run_table) -->"
)
DRY_RUN_TABLE_END = "<!-- dry-run-table:end -->"


def render_dry_run_table(record: Mapping[str, Any]) -> str:
    """The dry-run counts as the markdown 30 embeds, so its numbers are generated, never retyped (G14)."""
    counts = record["counts"]
    sources = sorted(
        {source for item in counts["layer_b"].values() for source in item["per_source"]}
    )
    lines = [
        "| Stratum | Quota | Supply after dedup | Realized | Shortage | "
        + " | ".join(f"{source}" for source in sources)
        + " |",
        "|---|---:|---:|---:|---:|" + "---:|" * len(sources),
    ]
    for name in STRATA:
        item = counts["layer_b"][name]
        per_source = " | ".join(str(item["per_source"][source]["realized"]) for source in sources)
        lines.append(
            f"| {name} | {item['quota']} | {item['supply']} | {item['realized']} | "
            f"{item['shortage']} | {per_source} |"
        )
    lines += [
        "",
        "| Layer A source | Quota | Realized | Gate-rejected realized / supply | Valid realized / supply |",
    ]
    lines.append("|---|---:|---:|---:|---:|")
    for source, item in sorted(counts["layer_a"]["per_source"].items()):
        gates = item["per_gate"]
        lines.append(
            f"| {source} | {item['quota']} | {item['realized']} | "
            f"{gates[GATE_ISSUE]['realized']} / {gates[GATE_ISSUE]['supply']} | "
            f"{gates[GATE_VALID]['realized']} / {gates[GATE_VALID]['supply']} |"
        )
    evidence = record["structural_call_evidence"]
    kinds = ", ".join(f"{kind} {count}" for kind, count in counts["m_match_kind"].items())
    lines += [
        "",
        (
            f"- Realized: layer B {counts['realized'][LAYER_B]}, "
            f"layer A {counts['realized'][LAYER_A]}, total {counts['realized']['total']}."
        ),
        f"- Population sha256 (not written): `{record['population_sha256']}`.",
        f"- Stratum M match kinds: {kinds}.",
        f"- Structural call evidence: **{evidence['status']}**; mismatched rows "
        + ", ".join(
            f"{source} {item['mismatched_rows']}" for source, item in evidence["sources"].items()
        )
        + "; rows the independent readers could not parse (unverified) "
        + ", ".join(
            f"{source} {item['unverified_rows']}" for source, item in evidence["sources"].items()
        )
        + "; rows read only by the format-grammar fallback (weaker evidence) "
        + ", ".join(
            f"{source} "
            f"{sum(cell['rows_read_by_format_grammar'] for cell in item['accepted_rows'].values())}"
            for source, item in evidence["sources"].items()
        )
        + ".",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="where the population is written or verified; reports/pdet-coverage only once 30 is adopted",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true", help="draw and write the population")
    action.add_argument(
        "--dry-run",
        action="store_true",
        help="draw, then write only counts and hashes (never the population)",
    )
    action.add_argument("--verify", action="store_true", help="verify a written population")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir)
    if args.dry_run:
        resolve_output_dir(root, output_dir, dry_run=True)
        population, manifest = build_population(root)
        record = write_dry_run(root, output_dir, population, manifest)
        print(json.dumps(summary(record), indent=2, sort_keys=True))
        return 0
    if args.build:
        resolve_output_dir(root, output_dir)  # refuse before spending a minute on the draw
        population, manifest = build_population(root)
        written = write_population(root, output_dir, population, manifest)
        print(json.dumps(summary(written), indent=2, sort_keys=True))
        return 0
    _result, report = verify_population(root, output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
