"""P-DET-COVERAGE-v1 builder (docs/research/study-002/30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md, "30").

Everything here runs on synthetic records or real normalization-v3 rows built in memory, so no test reads
the git-ignored artifact. Nothing writes ``reports/pdet-coverage/`` or ``reports/pdet/``: the builder
refuses both while 30 is a draft, and that refusal is itself tested.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import random
import re
from pathlib import Path
from typing import Any

import pytest

from opengrad.data.classifier_input import (
    EVALUATION_ONLY_OR_HELDOUT,
    MALFORMED_OR_UNRENDERABLE,
    MULTI_TURN_UNSUPPORTED,
    HeldoutIndex,
    normalize_prompt,
)
from opengrad.data.normalization_v3 import lf_sha256, load_source_manifest, normalize_row
from opengrad.verification import pdet_coverage as coverage
from opengrad.verification.accounting import BLOCKED_INPUT_MISSING, FAIL, PASS
from opengrad.verification.pdet_coverage import (
    CALL_NOT_IN_FIRST_ASSISTANT_TURN,
    LAYER_A,
    LAYER_A_CANDIDATE,
    LAYER_B,
    LAYER_B_CANDIDATE,
    PDET_V1_OVERLAP,
    QAD_RECOVERY,
    SENTINEL_PROMPT,
    SOURCE_NOT_LAYER_B,
    Candidates,
    CoverageInputError,
    CoverageOutputError,
    DrawExclusions,
    allocate,
    check_input,
    classify_unit,
    coverage_id,
    deduplicate,
    draw,
    population_bytes,
    rank_key,
    rendered_user_turns,
    resolve_output_dir,
    sampling_roles,
    verify_population,
    write_population,
)
from tests.data.test_normalization_v3 import (
    GLAIVE_CALL_CHAT,
    GLAIVE_SYSTEM,
    glaive_spec,
    toolace_raw,
    toolace_spec,
)

ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = ROOT / "docs/research/study-002/30-PDET-COVERAGE-PREREGISTRATION-DRAFT.md"
BOTH = {LAYER_A: True, LAYER_B: True}


# ── the predicates are the published ones ─────────────────────────────────────────────────────────


def _script_module() -> Any:
    spec = importlib.util.spec_from_file_location("supply_proxy", ROOT / coverage.STRATA_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _function_trees(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: ast.dump(node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"offered_names", "stratum", "skeleton"}
    }


def test_stratum_predicates_are_the_scripts_byte_for_byte():
    script = _script_module()
    assert coverage.STRATA == script.STRATA
    assert coverage.QUESTION_CUES == script.QUESTION_CUES
    assert coverage.HEDGE_CUES == script.HEDGE_CUES
    for name in ("TEXTUAL_CALL", "INVOCATION_TALK"):
        ours, theirs = getattr(coverage, name), getattr(script, name)
        assert (ours.pattern, ours.flags) == (theirs.pattern, theirs.flags)
    ours = _function_trees(ROOT / "src/opengrad/verification/pdet_coverage.py")
    theirs = _function_trees(ROOT / coverage.STRATA_SOURCE)
    assert set(ours) == {"offered_names", "stratum", "skeleton"}
    assert ours == theirs


@pytest.mark.parametrize(
    ("response", "tools", "expected"),
    [
        ('Here: {"name": "get_weather"}', [], "X"),
        ("I can use the get_weather tool for that.", [{"name": "get_weather"}], "M"),
        ("I'm sorry, but I cannot book flights.", [], "R"),
        ("Which city do you mean?", [], "Q"),
        ("Paris is the capital of France.", [{"name": "get_weather"}], "P1"),
        ("Paris is the capital of France.", [], "P2"),
    ],
)
def test_strata_follow_the_priority_order(response, tools, expected):
    assert coverage.stratum(response, tools) == expected


# ── ids, ranking, allocation ───────────────────────────────────────────────────────────────────────


def test_coverage_id_is_neutral_and_stable():
    identifier = coverage_id("glaive-function-calling-v2", "ab" * 32)
    assert re.fullmatch(r"pdetcov:[0-9a-f]{64}", identifier)
    assert "glaive" not in identifier
    assert identifier == coverage_id("glaive-function-calling-v2", "ab" * 32)
    assert identifier != coverage_id("toolace", "ab" * 32)


def test_rank_key_depends_on_the_stratum():
    assert rank_key("R", "pdetcov:1") == rank_key("R", "pdetcov:1")
    assert rank_key("R", "pdetcov:1") != rank_key("Q", "pdetcov:1")


@pytest.mark.parametrize(
    ("quota", "supply", "expected"),
    [
        # 30 §7.3 "With LoopTool", on proxy supply (glaive / toolace / looptool).
        (
            60,
            {"glaive": 0, "toolace": 6, "looptool": 0},
            {"glaive": 0, "toolace": 6, "looptool": 0},
        ),
        (
            60,
            {"glaive": 19, "toolace": 722, "looptool": 15},
            {"glaive": 19, "toolace": 26, "looptool": 15},
        ),
        (
            60,
            {"glaive": 10694, "toolace": 228, "looptool": 145},
            {"glaive": 20, "toolace": 20, "looptool": 20},
        ),
        (
            60,
            {"glaive": 58, "toolace": 375, "looptool": 3},
            {"glaive": 28, "toolace": 29, "looptool": 3},
        ),
        (
            30,
            {"glaive": 65, "toolace": 491, "looptool": 0},
            {"glaive": 15, "toolace": 15, "looptool": 0},
        ),
        # 30 §7.3 "Without LoopTool" and the measured v3 allocation.
        (60, {"glaive": 19, "toolace": 722}, {"glaive": 19, "toolace": 41}),
        (60, {"glaive": 14, "toolace": 706}, {"glaive": 14, "toolace": 46}),
        (60, {"glaive": 58, "toolace": 375}, {"glaive": 30, "toolace": 30}),
        # A stratum whose whole supply is short takes everything and nothing more.
        (60, {"glaive": 3, "toolace": 4}, {"glaive": 3, "toolace": 4}),
        (60, {}, {}),
    ],
)
def test_allocation_reproduces_the_preregistered_tables(quota, supply, expected):
    assert allocate(quota, supply) == expected


def test_the_recorded_input_hashes_are_the_ones_in_the_preregistration():
    text = PREREGISTRATION.read_text(encoding="utf-8")
    assert coverage.INPUT_FINGERPRINT in text
    assert coverage.INPUT_TOP_MANIFEST_SHA256 in text
    assert coverage.SEED in text and coverage.PROTOCOL_VERSION in text


def test_layer_roles_come_from_the_canonical_v3_source_manifest():
    roles = sampling_roles(load_source_manifest(ROOT))
    assert roles == {
        "xlam": {LAYER_A: True, LAYER_B: False},
        "glaive": {LAYER_A: True, LAYER_B: True},
        "toolace": {LAYER_A: True, LAYER_B: True},
        "when2call": {LAYER_A: False, LAYER_B: False},
    }
    assert set(coverage.LAYER_A_QUOTAS) == {name for name, role in roles.items() if role[LAYER_A]}


# ── units from real normalization-v3 rows ─────────────────────────────────────────────────────────


def _glaive(chat: str) -> dict[str, Any]:
    return normalize_row(glaive_spec(), {"system": GLAIVE_SYSTEM, "chat": chat}, 0)


def test_a_tool_free_single_exchange_is_a_layer_b_unit_with_its_stratum():
    record = _glaive(
        "USER: Weather in Paris?\n\n\nASSISTANT: Which date do you mean? <|endoftext|>\n"
    )
    disposition, unit = classify_unit(record, HeldoutIndex(), BOTH)
    assert disposition == LAYER_B_CANDIDATE
    assert unit is not None
    assert (unit["layer"], unit["stratum"]) == (LAYER_B, "Q")
    assert unit["user_message"] == "Weather in Paris?"
    assert unit["pdetcov_id"] == coverage_id(unit["source_dataset"], unit["raw_record_hash"])
    assert (
        unit["features_sha256"] and unit["classifier_input_contract"] == "prose-decision-input-v1"
    )


def test_a_call_in_the_first_assistant_turn_is_a_layer_a_unit_even_after_a_tool_result():
    disposition, unit = classify_unit(_glaive(GLAIVE_CALL_CHAT), HeldoutIndex(), BOTH)
    assert disposition == LAYER_A_CANDIDATE
    assert unit is not None and unit["layer"] == LAYER_A and unit["stratum"] is None
    assert [call["name"] for call in unit["structured_calls"]] == ["get_weather"]
    assert "assistant_response" in unit


def test_toolace_bracket_call_is_a_layer_a_unit():
    raw = toolace_raw('[get_weather(city="Paris")]')
    raw["conversations"] += [
        {"from": "tool", "value": '{"temperature": 20}'},
        {"from": "assistant", "value": "It is 20 degrees in Paris."},
    ]
    disposition, unit = classify_unit(normalize_row(toolace_spec(), raw, 0), HeldoutIndex(), BOTH)
    assert disposition == LAYER_A_CANDIDATE
    assert unit is not None and unit["assistant_response"] is None
    assert unit["trajectory_gate"] == coverage.GATE_VALID


def test_a_call_with_no_tool_result_is_still_a_layer_a_unit_marked_by_its_gate():
    # MISSING_TOOL_RESULT fails the input contract's trajectory gate: never layer B, but routing is audited
    # on gate-rejected records too, so it stays a layer A unit that records its gate status.
    record = normalize_row(toolace_spec(), toolace_raw('[get_weather(city="Paris")]'), 0)
    disposition, unit = classify_unit(record, HeldoutIndex(), BOTH)
    assert disposition == LAYER_A_CANDIDATE
    assert unit is not None and unit["trajectory_gate"] == coverage.GATE_ISSUE


def test_a_malformed_record_with_a_non_trajectory_defect_is_in_neither_layer():
    record = normalize_row(toolace_spec(), toolace_raw('[get_weather(city="Paris")]'), 0)
    record["metadata"]["parse_status"] = "PARTIAL"
    assert classify_unit(record, HeldoutIndex(), BOTH) == (MALFORMED_OR_UNRENDERABLE, None)


def test_prose_before_a_later_call_is_in_neither_layer():
    record = _glaive(
        "USER: Hi\n\n\nASSISTANT: Hello! <|endoftext|>\n\n\n"
        "USER: Weather in Paris?\n\n\n"
        'ASSISTANT: <functioncall> {"name": "get_weather", "arguments": \'{"city": "Paris"}\'}\n\n\n'
        'FUNCTION RESPONSE: {"temperature": 20}\n\n\n'
        "ASSISTANT: It is 20 degrees. <|endoftext|>\n"
    )
    assert classify_unit(record, HeldoutIndex(), BOTH) == (CALL_NOT_IN_FIRST_ASSISTANT_TURN, None)


def test_multi_turn_and_held_out_records_keep_their_contract_reason():
    multi = _glaive(
        "USER: Hi\n\n\nASSISTANT: Hello!\n\n\nUSER: A joke?\n\n\nASSISTANT: Knock knock.\n"
    )
    assert classify_unit(multi, HeldoutIndex(), BOTH) == (MULTI_TURN_UNSUPPORTED, None)
    heldout = HeldoutIndex(prompts=frozenset({normalize_prompt("What is the weather in Paris?")}))
    # Held out wins over the structural call: excluded from layer A too.
    assert classify_unit(_glaive(GLAIVE_CALL_CHAT), heldout, BOTH) == (
        EVALUATION_ONLY_OR_HELDOUT,
        None,
    )


def test_a_source_not_sampled_for_a_layer_supplies_no_unit_of_it():
    record = _glaive("USER: Hi\n\n\nASSISTANT: Paris is lovely. <|endoftext|>\n")
    roles = {LAYER_A: True, LAYER_B: False}
    assert classify_unit(record, HeldoutIndex(), roles) == (SOURCE_NOT_LAYER_B, None)


# ── exclusions and dedup ──────────────────────────────────────────────────────────────────────────


def _unit(
    number: int,
    *,
    stratum: str = "P1",
    source: str = "toolace",
    prompt: str | None = None,
    response: str | None = None,
    layer: str = LAYER_B,
) -> dict[str, Any]:
    raw = f"{number:064x}"
    return {
        "pdetcov_id": coverage_id(source, raw),
        "layer": layer,
        "stratum": stratum if layer == LAYER_B else None,
        "source_name": source,
        "source_dataset": source,
        "source_revision": None,
        "record_id": f"og_{raw[:16]}",
        "upstream_id": f"og_{raw[:16]}",
        "raw_record_hash": raw,
        "canonical_hash": f"c{raw[1:]}",
        "user_message": prompt if prompt is not None else f"prompt number {number}",
        "assistant_response": response if response is not None else f"answer {'x' * number}",
        "tools": [{"name": "get_weather"}] if stratum != "P2" else [],
    }


def test_rendered_prompt_user_turns_are_extracted():
    rendered = (
        "<|im_start|>system\nTools<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert rendered_user_turns(rendered) == ["What is 2+2?"]


def test_each_draw_exclusion_rule_matches_what_it_should():
    exclusions = DrawExclusions(
        sentinel_prompts=frozenset({"how many sheep?"}),
        qad_ids=frozenset({_unit(2)["canonical_hash"]}),
        qad_prompts=frozenset({"qad prompt"}),
        pdet_identities=frozenset({_unit(4)["raw_record_hash"]}),
        pdet_prompts=frozenset({"pdet prompt"}),
        pdet_responses=frozenset({"pdet response"}),
    )
    assert exclusions.hits(_unit(1, prompt="How many  SHEEP?")) == [SENTINEL_PROMPT]
    assert exclusions.hits(_unit(2)) == [QAD_RECOVERY]
    assert exclusions.hits(_unit(3, prompt="QAD prompt")) == [QAD_RECOVERY]
    assert exclusions.hits(_unit(4)) == [PDET_V1_OVERLAP]
    assert exclusions.hits(_unit(5, prompt="pdet prompt")) == [PDET_V1_OVERLAP]
    assert exclusions.hits(_unit(6, response="PDET response")) == [PDET_V1_OVERLAP]
    # A layer A unit has no prose response to match.
    layer_a = {**_unit(7, layer=LAYER_A), "assistant_response": None}
    assert exclusions.hits(layer_a) == []
    both = _unit(8, prompt="how many sheep?", response="pdet response")
    assert exclusions.hits(both) == [SENTINEL_PROMPT, PDET_V1_OVERLAP]


def test_dedup_keeps_the_highest_ranked_member_of_each_group():
    same_response = [_unit(n, response="Same answer.") for n in (1, 2, 3)]
    survivors, stats = deduplicate(same_response)
    expected = max(same_response, key=lambda unit: rank_key(unit["stratum"], unit["pdetcov_id"]))
    assert [unit["pdetcov_id"] for unit in survivors] == [expected["pdetcov_id"]]
    assert stats[LAYER_B]["response_text"] == 2


def test_dedup_rules_raw_hash_prompt_and_skeleton():
    units = [
        _unit(1, prompt="Same prompt", response="first"),
        _unit(2, prompt="same   PROMPT", response="second"),
        _unit(3, stratum="R", response="I cannot book 3 flights."),
        _unit(4, stratum="R", response="I cannot book 7 flights."),
        _unit(5, stratum="Q", response="I cannot book 9 flights?"),
    ]
    duplicate_hash = {
        **_unit(6, prompt="Same prompt", response="first"),
        "raw_record_hash": units[0]["raw_record_hash"],
    }
    survivors, stats = deduplicate([*units, duplicate_hash])
    assert stats[LAYER_B] == {
        "raw_record_hash": 1,
        "response_text": 0,
        "user_prompt": 1,
        "skeleton": 1,
    }
    # The skeleton cap is per stratum: the Q item shares no stratum with the R items.
    assert sum(1 for unit in survivors if unit["stratum"] == "Q") == 1
    assert len(survivors) == 3


# ── the draw ─────────────────────────────────────────────────────────────────────────────────────


def _pool() -> Candidates:
    units = []
    number = 1
    supply = {
        ("X", "toolace"): 6,
        ("M", "glaive"): 14,
        ("M", "toolace"): 80,
        ("R", "glaive"): 70,
        ("R", "toolace"): 70,
        ("Q", "glaive"): 70,
        ("Q", "toolace"): 70,
        ("P1", "glaive"): 14,
        ("P1", "toolace"): 80,
        ("P2", "glaive"): 40,
        ("P2", "toolace"): 40,
    }
    for (stratum, source), count in supply.items():
        for _ in range(count):
            units.append(_unit(number, stratum=stratum, source=source))
            number += 1
    layer_a = {("glaive", "valid"): 16, ("glaive", "trajectory_issue"): 4}
    layer_a |= {("toolace", "trajectory_issue"): 3, ("toolace", "valid"): 1, ("xlam", "valid"): 20}
    for (source, gate), count in layer_a.items():
        for _ in range(count):
            unit = _unit(number, source=source, layer=LAYER_A)
            units.append({**unit, "structured_calls": [], "trajectory_gate": gate})
            number += 1
    return Candidates(units=units)


def test_draw_meets_quotas_reports_shortages_and_never_backfills():
    population, counts = draw(_pool(), DrawExclusions())
    layer_b = counts["layer_b"]
    assert layer_b["X"] == {
        "quota": 60,
        "supply": 6,
        "realized": 6,
        "shortage": 54,
        "per_source": {
            "glaive": {"supply": 0, "realized": 0},
            "toolace": {"supply": 6, "realized": 6},
        },
    }
    assert {name: layer_b[name]["per_source"]["glaive"]["realized"] for name in ("M", "P1")} == {
        "M": 14,
        "P1": 14,
    }
    assert {name: item["realized"] for name, item in layer_b.items()} == {
        "X": 6,
        "M": 60,
        "R": 60,
        "Q": 60,
        "P1": 80,
        "P2": 40,
    }
    # ToolACE supplies only 4 layer A items; its shortfall is reported, not taken from another source.
    assert counts["layer_a"]["per_source"]["toolace"] == {
        "quota": 10,
        "supply": 4,
        "realized": 4,
        "shortage": 6,
        "per_gate": {
            "trajectory_issue": {"supply": 3, "realized": 3},
            "valid": {"supply": 1, "realized": 1},
        },
    }
    # Glaive 15 from 16 valid + 4 gate-rejected: one each, then proportional (2.6 / 10.4), remainder to .6.
    assert counts["layer_a"]["per_source"]["glaive"]["per_gate"] == {
        "trajectory_issue": {"supply": 4, "realized": 4},
        "valid": {"supply": 16, "realized": 11},
    }
    assert counts["realized"] == {LAYER_B: 306, LAYER_A: 24, "total": 330}
    assert counts["shortages"] == {"layer_b": {"X": 54}, "layer_a": {"toolace": 6}}
    assert len(population) == 330
    assert [item["pdetcov_index"] for item in population] == list(range(330))
    assert all(item["gold_policy_label"] is None for item in population)
    assert all(item["classifier_version_at_selection"] == "NOT_IMPLEMENTED" for item in population)


def test_draw_is_byte_reproducible_whatever_the_input_order():
    first, _ = draw(_pool(), DrawExclusions())
    shuffled = _pool()
    random.Random(7).shuffle(shuffled.units)
    second, _ = draw(shuffled, DrawExclusions())
    assert population_bytes(first) == population_bytes(second)


def test_draw_exclusions_are_counted_per_layer_and_rule():
    pool = _pool()
    excluded = pool.units[6]
    exclusions = DrawExclusions(
        sentinel_prompts=frozenset({normalize_prompt(excluded["user_message"])})
    )
    population, counts = draw(pool, exclusions)
    assert counts["draw_exclusions"][LAYER_B]["primary"] == {SENTINEL_PROMPT: 1}
    assert counts["draw_exclusions"][LAYER_B]["distinct_prompts_matched"] == {SENTINEL_PROMPT: 1}
    assert excluded["pdetcov_id"] not in {item["pdetcov_id"] for item in population}


# ── adoption gate, writing and verification ────────────────────────────────────────────────────────


def test_output_is_refused_under_reports_pdet_and_before_adoption(tmp_path):
    assert coverage.PREREGISTRATION_STATUS == "DRAFT"
    for refused in (
        "reports/pdet",
        "reports/pdet/sub",
        "reports/pdet-coverage",
        "reports/pdet-coverage/x",
    ):
        with pytest.raises(CoverageOutputError):
            resolve_output_dir(ROOT, Path(refused))
    assert resolve_output_dir(ROOT, tmp_path / "dry-run") == (tmp_path / "dry-run").resolve()


def test_input_other_than_the_recorded_artifact_is_refused(tmp_path):
    with pytest.raises(CoverageInputError, match="normalization-v1"):
        check_input(ROOT, Path("data/processed/normalization-v1"))
    with pytest.raises(CoverageInputError, match="missing"):
        check_input(tmp_path)
    fake = tmp_path / "data/processed/normalization-v3"
    fake.mkdir(parents=True)
    (fake / "manifest.json").write_text('{"fingerprint": "x", "sources": {}}\n', encoding="utf-8")
    with pytest.raises(CoverageInputError, match="not the recorded one"):
        check_input(tmp_path)


def _written(tmp_path: Path) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    population, counts = draw(_pool(), DrawExclusions())
    manifest = {
        "population_id": coverage.POPULATION_ID,
        "preregistration": {"status_at_build": "DRAFT"},
        "counts": counts,
        "classifier_status_at_selection": "NOT_IMPLEMENTED",
        "selection_used_classifier": False,
        "gold_labels_present": False,
    }
    output = tmp_path / "dry-run"
    write_population(tmp_path, output, population, manifest)
    return output, population, manifest


def test_written_population_verifies_and_is_never_overwritten(tmp_path, monkeypatch):
    output, population, manifest = _written(tmp_path)
    with pytest.raises(CoverageOutputError, match="never overwritten"):
        write_population(tmp_path, output, population, manifest)
    monkeypatch.setattr(coverage, "derivation_inputs_missing", lambda root: [])
    monkeypatch.setattr(coverage, "build_population", lambda root: (population, manifest))
    monkeypatch.setattr(coverage, "load_draw_exclusions", lambda root: DrawExclusions())
    monkeypatch.setattr(coverage.HeldoutIndex, "load", classmethod(lambda cls, root: cls()))
    _result, report = verify_population(tmp_path, output)
    assert report["status"] == PASS, report
    assert (report["layer_a"], report["layer_b"]) == (24, 306)


def test_verification_without_inputs_is_blocked_never_passed(tmp_path, monkeypatch):
    output, _population, _manifest = _written(tmp_path)
    monkeypatch.setattr(coverage, "derivation_inputs_missing", lambda root: ["data/processed/x"])
    result, report = verify_population(tmp_path, output)
    assert report["status"] == BLOCKED_INPUT_MISSING
    assert result.passed == 0 and result.blocked == 330


def test_tampering_fails_verification(tmp_path, monkeypatch):
    output, _population, _manifest = _written(tmp_path)
    monkeypatch.setattr(coverage, "derivation_inputs_missing", lambda root: ["data/processed/x"])
    path = output / coverage.POPULATION_NAME
    lines = path.read_bytes().splitlines(keepends=True)
    item = json.loads(lines[0])
    item["gold_policy_label"] = "DIRECT"
    lines[0] = (json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(b"".join(lines))
    _result, report = verify_population(tmp_path, output)
    assert report["status"] == FAIL
    assert any(error.startswith("FAIL_HASH") for error in report["errors"])
    assert any(error.startswith("FAIL_ANNOTATION") for error in report["errors"])


# ── dedup order, gate split, M match kinds (review decisions D1, D8, D16) ─────────────────────────────


def test_a_prompt_shared_across_strata_stays_in_the_scarcest_stratum():
    plentiful = [_unit(n, stratum="R", response=f"I cannot do task {n}.") for n in range(1, 6)]
    shared = plentiful[0]["user_message"]
    scarce = _unit(99, stratum="P1", prompt=shared, response="Paris is the capital.")
    survivors, stats = deduplicate([*plentiful, scarce])
    kept = {unit["pdetcov_id"] for unit in survivors}
    assert scarce["pdetcov_id"] in kept
    assert plentiful[0]["pdetcov_id"] not in kept
    assert stats[LAYER_B]["user_prompt"] == 1


def test_dedup_order_puts_layer_a_first_then_strata_by_supply_then_strata_order():
    units = [_unit(1, stratum="Q"), _unit(2, stratum="R"), _unit(3, stratum="R")]
    units.append({**_unit(4, layer=LAYER_A), "trajectory_gate": coverage.GATE_VALID})
    order = [(unit["layer"], unit["stratum"]) for unit in coverage.dedup_order(units)]
    assert order == [(LAYER_A, None), (LAYER_B, "Q"), (LAYER_B, "R"), (LAYER_B, "R")]
    tie = [_unit(5, stratum="P2"), _unit(6, stratum="M")]
    assert [unit["stratum"] for unit in coverage.dedup_order(tie)] == ["M", "P2"]


@pytest.mark.parametrize(
    ("quota", "supply", "expected"),
    [
        (10, {"trajectory_issue": 900, "valid": 60}, {"trajectory_issue": 9, "valid": 1}),
        (15, {"trajectory_issue": 4, "valid": 16}, {"trajectory_issue": 4, "valid": 11}),
        (5, {"trajectory_issue": 0, "valid": 20}, {"trajectory_issue": 0, "valid": 5}),
        (10, {"trajectory_issue": 3, "valid": 1}, {"trajectory_issue": 3, "valid": 1}),
        (1, {"trajectory_issue": 5, "valid": 5}, {"trajectory_issue": 1, "valid": 0}),
        (0, {"trajectory_issue": 5, "valid": 5}, {"trajectory_issue": 0, "valid": 0}),
    ],
)
def test_layer_a_quota_is_split_across_gate_status_in_proportion(quota, supply, expected):
    assert coverage.split_by_gate(quota, supply) == expected


@pytest.mark.parametrize(
    ("response", "tools", "expected"),
    [
        ("I will use the lookup tool.", [{"name": "lookup"}], coverage.M_INVOCATION_TALK),
        ("The get_weather result is ready.", [{"name": "get_weather"}], coverage.M_IDENTIFIER_NAME),
        ("Try searchFlights next.", [{"name": "searchFlights"}], coverage.M_IDENTIFIER_NAME),
        ("Here is the weather today.", [{"name": "weather"}], coverage.M_WORD_NAME),
    ],
)
def test_m_match_kind_separates_real_mentions_from_ordinary_words(response, tools, expected):
    assert coverage.stratum(response, tools) == "M"
    assert coverage.m_match_kind(response, tools) == expected


def test_realized_m_items_get_a_match_kind_in_the_counts_only():
    population, counts = draw(_pool(), DrawExclusions())
    m_items = {item["pdetcov_id"] for item in population if item["stratum"] == "M"}
    assert set(counts["m_match_kind_by_id"]) == m_items
    assert sum(counts["m_match_kind"].values()) == len(m_items) == 60
    assert all("m_match_kind" not in item for item in population)


# ── remaining dispositions ────────────────────────────────────────────────────────────────────────


def test_a_call_after_two_user_turns_is_not_a_layer_a_unit():
    record = _glaive(GLAIVE_CALL_CHAT)
    body_start = 1 if record["messages"][0]["role"] == "system" else 0
    record["messages"].insert(body_start, {"role": "user", "content": "Hello."})
    disposition, unit = classify_unit(record, HeldoutIndex(), BOTH)
    assert (disposition, unit) == (coverage.CALL_PREFIX_NOT_ONE_USER_TURN, None)


def test_a_source_not_sampled_for_layer_a_supplies_no_call_unit():
    roles = {LAYER_A: False, LAYER_B: True}
    assert classify_unit(_glaive(GLAIVE_CALL_CHAT), HeldoutIndex(), roles) == (
        coverage.SOURCE_NOT_LAYER_A,
        None,
    )


def test_collect_candidates_counts_every_record_by_source_and_disposition():
    plain = _glaive("USER: Hi\n\n\nASSISTANT: Paris is lovely. <|endoftext|>\n")
    collected = coverage.collect_candidates(
        [("glaive", plain), ("glaive", _glaive(GLAIVE_CALL_CHAT)), ("toolace", plain)],
        HeldoutIndex(),
        {"glaive": BOTH, "toolace": {LAYER_A: True, LAYER_B: False}},
    )
    assert {source: dict(counter) for source, counter in collected.dispositions.items()} == {
        "glaive": {LAYER_B_CANDIDATE: 1, LAYER_A_CANDIDATE: 1},
        "toolace": {SOURCE_NOT_LAYER_B: 1},
    }
    assert [unit["layer"] for unit in collected.units] == [LAYER_B, LAYER_A]


def test_rendered_prompt_user_turns_include_every_turn_of_a_multi_turn_prompt():
    rendered = (
        "<|im_start|>user\nFirst?<|im_end|>\n<|im_start|>assistant\nOk<|im_end|>\n"
        "<|im_start|>user\nSecond\nline?<|im_end|>\n<|im_start|>assistant\n"
    )
    assert rendered_user_turns(rendered) == ["First?", "Second\nline?"]


# ── pinned inputs ─────────────────────────────────────────────────────────────────────────────────────


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fake_artifact(root: Path, monkeypatch) -> Path:
    artifact = root / "data/processed/normalization-v3"
    (artifact / "glaive").mkdir(parents=True)
    (artifact / "glaive/shard-000000.parquet").write_bytes(b"shard bytes")
    source_manifest = json.dumps(
        {"shards": [{"file": "shard-000000.parquet", "sha256": _sha(b"shard bytes")}]}
    ).encode()
    (artifact / "glaive/manifest.json").write_bytes(source_manifest)
    top = json.dumps(
        {
            "fingerprint": "fake-fingerprint",
            "sources": {"glaive": {"manifest_sha256": _sha(source_manifest), "counts": {"a": 1}}},
        }
    ).encode()
    (artifact / "manifest.json").write_bytes(top)
    config = root / "configs/releases/toolpolicy_canonical_v3_sources.yaml"
    config.parent.mkdir(parents=True)
    config.write_bytes(b"sources: []\n")
    monkeypatch.setattr(coverage, "INPUT_TOP_MANIFEST_SHA256", _sha(top))
    monkeypatch.setattr(coverage, "INPUT_FINGERPRINT", "fake-fingerprint")
    monkeypatch.setattr(coverage, "SOURCE_MANIFEST_SHA256", _sha(b"sources: []\n"))
    return artifact


def test_the_recorded_input_is_accepted_and_described(tmp_path, monkeypatch):
    _fake_artifact(tmp_path, monkeypatch)
    record = check_input(tmp_path)
    assert record["fingerprint"] == "fake-fingerprint"
    assert record["upstream_disposition_counts"] == {"glaive": {"a": 1}}


def test_a_changed_shard_is_refused(tmp_path, monkeypatch):
    artifact = _fake_artifact(tmp_path, monkeypatch)
    (artifact / "glaive/shard-000000.parquet").write_bytes(b"other bytes")
    with pytest.raises(CoverageInputError, match="shard-000000.parquet changed"):
        check_input(tmp_path)


def test_a_changed_source_manifest_of_the_artifact_is_refused(tmp_path, monkeypatch):
    artifact = _fake_artifact(tmp_path, monkeypatch)
    (artifact / "glaive/manifest.json").write_bytes(b'{"shards": []}')
    with pytest.raises(CoverageInputError, match="glaive manifest changed"):
        check_input(tmp_path)


def test_an_edited_canonical_v3_source_manifest_is_refused(tmp_path, monkeypatch):
    _fake_artifact(tmp_path, monkeypatch)
    config = tmp_path / "configs/releases/toolpolicy_canonical_v3_sources.yaml"
    config.write_bytes(b"sources: [edited]\n")
    with pytest.raises(CoverageInputError, match="not the recorded canonical-v3 source manifest"):
        check_input(tmp_path)


def test_the_committed_source_manifest_is_the_pinned_one():
    assert lf_sha256(ROOT / "configs/releases/toolpolicy_canonical_v3_sources.yaml") == (
        coverage.SOURCE_MANIFEST_SHA256
    )


def _fake_exclusion_inputs(root: Path, monkeypatch) -> None:
    qad_prompt = (
        "<|im_start|>user\nFirst?<|im_end|>\n"
        "<|im_start|>assistant\nOk<|im_end|>\n<|im_start|>user\nBook a flight?<|im_end|>\n"
    )
    pdet_item = {
        "pdet_id": "when2call-sft:h",
        "raw_record_hash": "h",
        "prompt": "P",
        "response": "R",
    }
    files = {
        "sentinel.jsonl": (
            json.dumps({"messages": [{"role": "user", "content": "How many  SHEEP?"}]}) + "\n"
        ).encode(),
        "ow7.json": json.dumps(
            {"cases": [{"turns": [{"role": "user", "content": "Weather in Manila?"}]}]}
        ).encode(),
        "qad.jsonl": (json.dumps({"canonical_id": "qad-id", "prompt": qad_prompt}) + "\n").encode(),
        "pdet.jsonl": (json.dumps(pdet_item) + "\n").encode(),
    }
    for name, data in files.items():
        (root / name).write_bytes(data)
    sentinel = (("S", "sentinel.jsonl", _sha(files["sentinel.jsonl"])),)
    monkeypatch.setattr(coverage, "SENTINEL_REQUEST_FILES", sentinel)
    monkeypatch.setattr(coverage, "SENTINEL_OW7", ("S-OW7", "ow7.json", _sha(files["ow7.json"])))
    monkeypatch.setattr(coverage, "QAD_RECOVERY_SET", ("qad.jsonl", _sha(files["qad.jsonl"])))
    monkeypatch.setattr(coverage, "PDET_V1_POPULATION", ("pdet.jsonl", _sha(files["pdet.jsonl"])))


def test_exclusion_inputs_are_parsed_into_their_rules(tmp_path, monkeypatch):
    _fake_exclusion_inputs(tmp_path, monkeypatch)
    loaded = coverage.load_draw_exclusions(tmp_path)
    assert loaded.sentinel_prompts == {"how many sheep?", "weather in manila?"}
    assert loaded.qad_ids == {"qad-id"}
    assert loaded.qad_prompts == {"first?", "book a flight?"}
    assert loaded.pdet_identities == {"when2call-sft:h", "h"}
    assert (loaded.pdet_prompts, loaded.pdet_responses) == ({"p"}, {"r"})
    assert len(loaded.inputs) == 4


def test_a_changed_or_missing_exclusion_input_is_refused(tmp_path, monkeypatch):
    _fake_exclusion_inputs(tmp_path, monkeypatch)
    (tmp_path / "qad.jsonl").write_bytes(b"{}\n")
    with pytest.raises(CoverageInputError, match=r"exclusion input changed: qad\.jsonl"):
        coverage.load_draw_exclusions(tmp_path)
    (tmp_path / "qad.jsonl").unlink()
    with pytest.raises(CoverageInputError, match=r"exclusion input missing: qad\.jsonl"):
        coverage.load_draw_exclusions(tmp_path)


def test_derivation_inputs_include_the_source_manifest_when_absent(tmp_path):
    missing = coverage.derivation_inputs_missing(tmp_path)
    assert "data/processed/normalization-v3/manifest.json" in missing
    assert "configs/releases/toolpolicy_canonical_v3_sources.yaml" in missing


def test_derivation_inputs_include_the_raw_upstream_rows():
    manifest = load_source_manifest(ROOT)
    raw = {
        entry["raw_artifact"]["path"]
        for entry in manifest["sources"]
        if entry["name"] in {"glaive", "toolace", "xlam"}
    }
    missing = set(coverage.derivation_inputs_missing(ROOT))
    assert all((ROOT / path).is_file() or path in missing for path in raw)


# ── verification failures ─────────────────────────────────────────────────────────────────────────────


def _verifiable(tmp_path, monkeypatch, rebuilt=None, exclusions=None):
    output, population, manifest = _written(tmp_path)
    monkeypatch.setattr(coverage, "derivation_inputs_missing", lambda root: [])
    redraw = rebuilt if rebuilt is not None else (population, manifest)
    monkeypatch.setattr(coverage, "build_population", lambda root: redraw)
    excluded = exclusions if exclusions is not None else DrawExclusions()
    monkeypatch.setattr(coverage, "load_draw_exclusions", lambda root: excluded)
    monkeypatch.setattr(coverage.HeldoutIndex, "load", classmethod(lambda cls, root: cls()))
    return output, population, manifest


def test_a_redraw_that_differs_fails_reproducibility_and_provenance(tmp_path, monkeypatch):
    population, counts = draw(_pool(), DrawExclusions())
    other = [{**population[0], "user_message": "changed"}, *population[1:]]
    manifest = {"counts": {**counts, "realized": {}}}
    output, _population, _manifest = _verifiable(tmp_path, monkeypatch, rebuilt=(other, manifest))
    _result, report = verify_population(tmp_path, output)
    assert report["status"] == FAIL
    assert any(error.startswith("FAIL_REPRODUCIBILITY") for error in report["errors"])
    assert any("'counts'" in error for error in report["errors"])


def test_an_item_matching_an_exclusion_fails_contamination(tmp_path, monkeypatch):
    population, _counts = draw(_pool(), DrawExclusions())
    leaked = frozenset({normalize_prompt(population[0]["user_message"])})
    exclusions = DrawExclusions(sentinel_prompts=leaked)
    output, _population, _manifest = _verifiable(tmp_path, monkeypatch, exclusions=exclusions)
    _result, report = verify_population(tmp_path, output)
    assert report["status"] == FAIL
    assert any(error.startswith("FAIL_CONTAMINATION: 1 items") for error in report["errors"])


def test_a_manifest_claiming_classifier_use_fails_method(tmp_path, monkeypatch):
    monkeypatch.setattr(coverage, "derivation_inputs_missing", lambda root: ["x"])
    population, counts = draw(_pool(), DrawExclusions())
    manifest = {"counts": counts, "selection_used_classifier": True}
    write_population(tmp_path, tmp_path / "out", population, manifest)
    _result, report = verify_population(tmp_path, tmp_path / "out")
    assert report["status"] == FAIL
    assert any(error.startswith("FAIL_METHOD") for error in report["errors"])


def test_a_missing_sidecar_fails_and_missing_artifacts_are_not_vacuous(tmp_path, monkeypatch):
    output, _population, _manifest = _written(tmp_path)
    monkeypatch.setattr(coverage, "derivation_inputs_missing", lambda root: ["x"])
    (output / (coverage.MANIFEST_NAME + ".sha256")).unlink()
    _result, report = verify_population(tmp_path, output)
    assert any("sidecar" in error for error in report["errors"])
    result, report = verify_population(tmp_path, tmp_path / "nothing-here")
    assert report["status"] == "MISSING" and result.failed == 1


# ── counts-only dry run and the command line ────────────────────────────────────────────────────────


def test_a_dry_run_writes_counts_and_hashes_but_no_item(tmp_path):
    population, counts = draw(_pool(), DrawExclusions())
    manifest = {"counts": counts, "statement": "s."}
    record = coverage.write_dry_run(tmp_path, Path("reports/pdet-coverage"), population, manifest)
    directory = tmp_path / "reports/pdet-coverage"
    assert sorted(path.name for path in directory.iterdir()) == [coverage.DRY_RUN_NAME]
    text = (directory / coverage.DRY_RUN_NAME).read_text(encoding="utf-8")
    assert not re.search(r"pdetcov:[0-9a-f]{64}", text)
    assert population[0]["user_message"] not in text
    assert record["population_written"] is False
    assert record["population_sha256"] == _sha(population_bytes(population))
    with pytest.raises(CoverageOutputError):
        coverage.write_dry_run(tmp_path, Path("reports/pdet"), population, manifest)


def test_build_into_the_official_directory_is_refused_before_any_draw(monkeypatch):
    def forbidden(root):
        raise AssertionError("the draw ran before the output directory was checked")

    monkeypatch.setattr(coverage, "build_population", forbidden)
    with pytest.raises(CoverageOutputError, match="DRAFT"):
        coverage.main(["--root", str(ROOT), "--build", "--output-dir", "reports/pdet-coverage"])


def test_command_line_dry_run_and_verify_exit_codes(tmp_path, monkeypatch, capsys):
    population, counts = draw(_pool(), DrawExclusions())
    manifest = {
        "counts": counts,
        "statement": "s.",
        "preregistration": {"status_at_build": "DRAFT"},
        "structural_call_evidence": {"status": "PASS"},
    }
    monkeypatch.setattr(coverage, "build_population", lambda root: (population, manifest))
    code = coverage.main(["--root", str(tmp_path), "--dry-run", "--output-dir", "dry"])
    assert code == 0
    assert (tmp_path / "dry" / coverage.DRY_RUN_NAME).is_file()
    assert '"realized"' in capsys.readouterr().out
    assert coverage.main(["--root", str(tmp_path), "--verify", "--output-dir", "dry"]) == 1


# ── the real artifact (skipped, never passed, where the git-ignored inputs are absent) ─────────────────

DRY_RUN_RECORD = ROOT / "reports/pdet-coverage" / coverage.DRY_RUN_NAME


def test_the_real_draw_reproduces_the_recorded_dry_run():
    missing = coverage.derivation_inputs_missing(ROOT)
    if missing:
        pytest.skip(f"BLOCKED_INPUT_MISSING: {missing}")
    if not DRY_RUN_RECORD.is_file():
        pytest.skip("no recorded dry run to reproduce")
    recorded = json.loads(DRY_RUN_RECORD.read_text(encoding="utf-8"))
    population, manifest = coverage.build_population(ROOT)
    assert _sha(population_bytes(population)) == recorded["population_sha256"]
    counts = {
        key: value for key, value in manifest["counts"].items() if key != "m_match_kind_by_id"
    }
    assert counts == recorded["counts"]
    assert manifest["structural_call_evidence"] == recorded["structural_call_evidence"]


# ── the preregistration's numbers and exposure ──────────────────────────────────────────────────────


def test_dry_run_table_in_the_preregistration_is_generated():
    """30 §13's dry-run block is rendered from the recorded dry run, never retyped (G14)."""
    text = PREREGISTRATION.read_text(encoding="utf-8")
    start = text.index(coverage.DRY_RUN_TABLE_START) + len(coverage.DRY_RUN_TABLE_START)
    end = text.index(coverage.DRY_RUN_TABLE_END)
    recorded = json.loads(DRY_RUN_RECORD.read_text(encoding="utf-8"))
    assert text[start:end].strip() == coverage.render_dry_run_table(recorded).strip()


def test_the_recorded_dry_run_holds_no_item():
    text = DRY_RUN_RECORD.read_text(encoding="utf-8")
    record = json.loads(text)
    assert record["artifact_kind"] == "PDET_COVERAGE_DRY_RUN"
    assert record["population_written"] is False
    assert not re.search(r"pdetcov:[0-9a-f]{64}", text)
    assert "m_match_kind_by_id" not in record["counts"]


def _quoted_spans(text: str) -> list[str]:
    spans = re.findall(r'"([^"\n]{12,})"|“([^”\n]{12,})”', text)
    return [
        normalize_prompt(first or second)
        for first, second in spans
        if len((first or second).split()) >= 3
    ]


def test_the_preregistration_quotes_no_text_from_the_eligible_pools():
    """30 §9: this document is annotator-facing, so no quoted span may come from a sampled pool."""
    from opengrad.data.normalization_v3 import iter_rows

    artifact = ROOT / "data/processed/normalization-v3"
    if not (artifact / "manifest.json").is_file():
        pytest.skip("BLOCKED_INPUT_MISSING: normalization-v3 is not in this checkout")
    spans = _quoted_spans(PREREGISTRATION.read_text(encoding="utf-8"))
    assert spans, "the extraction found no quoted span; it broke"
    found: set[str] = set()
    for source in ("glaive", "toolace"):
        for record in iter_rows(artifact, source):
            for message in record["messages"]:
                if message.get("role") == "assistant" and isinstance(message.get("content"), str):
                    content = normalize_prompt(message["content"])
                    found.update(span for span in spans if span in content)
    assert not found, f"30 quotes pool text in {len(found)} spans"
