"""Draft 46 (`study_002_prereg_v14`): every number and code fact it quotes comes from an artifact (G14)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import yaml

from opengrad.data.canonical import ToolConversation
from opengrad.data.renderers import Qwen35_2BRenderer, _qwen_messages, _qwen_tools
from opengrad.formatting.parser import parse_qwen_native_output
from opengrad.verification.resolvability import resolvable_margin

ROOT = Path(__file__).parents[2]
DRAFT = ROOT / "docs/research/study-002/46-READINESS-DESIGN-AMENDMENT-DRAFT.md"


def _flat() -> str:
    return " ".join(DRAFT.read_text(encoding="utf-8").split())


def _json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_the_draft_is_recorded_in_neither_03_nor_errata() -> None:
    assert "> **Status: DRAFT, not adopted.**" in DRAFT.read_text(encoding="utf-8")
    for record in ("docs/research/study-002/03-PREREGISTRATION.md", "reports/ERRATA.md"):
        assert "study_002_prereg_v14" not in (ROOT / record).read_text(encoding="utf-8"), record


def test_the_flag_counts_come_from_the_canonical_v2_audit() -> None:
    audit = _json(
        "results/benchmarks/h200/capability_v1/sft_refusal_supervision_audit_canonical_v2.json"
    )
    flat = _flat()
    by_source = {name: row["refusal_targets"] for name, row in audit["per_source"].items()}
    assert audit["totals"]["refusal_targets"] == sum(by_source.values())
    assert f"The {audit['totals']['refusal_targets']:,} is `HEURISTIC_REGEX_v1`'s count" in flat
    assert (
        f"Glaive {by_source['glaive-function-calling-v2']:,}, When2Call {by_source['when2call']:,}, "
        f"ToolACE {by_source['toolace']}, xLAM {by_source['xlam-function-calling-60k']}"
    ) in flat
    multi = audit["multi_turn_records_excluded"]["counts_by_first_exchange_label"]
    assert f"({multi['ANSWER']} labelled `ANSWER` and {multi['CALL']:,} labelled `CALL`" in flat
    xlam = audit["per_source"]["xlam-function-calling-60k"]
    assert f"no refusal target ({xlam['refusal_targets']} of {xlam['records']:,})" in flat
    # §15: the triage is about 24 times the 750 items of 45.
    assert round(audit["totals"]["refusal_targets"] / 750) == 24
    assert "about 24 times the 750 items" in flat


def test_the_exposure_budget_is_m0s_logged_tally() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/experiments/m0_sft_canonical_v2_final.yaml").read_text(encoding="utf-8")
    )["trainer"]
    lines = (
        (ROOT / "runs/m0_sft_canonical_v2_final/metrics/train_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    last = json.loads([line for line in lines if line.strip()][-1])
    rendering = _json("runs/m0_sft_canonical_v2_final/rendering_report.json")
    flat = _flat()
    assert last["optimizer_step"] == config["max_steps"]
    assert (
        f"filled up to {config['micro_batch_tokens']:,} tokens or {config['micro_batch_size']} "
        f"sequences (Study 001's M0 config, with {config['gradient_accumulation_steps']} micro-batches "
        f"per optimizer step and {config['max_steps']:,} steps)"
    ) in flat
    passes = last["examples_seen"] / rendering["trainable_records"]
    assert (
        f"M0 stopped at step {last['optimizer_step']:,} having seen {last['examples_seen']:,} "
        f"examples, {passes:.3f} passes over its {rendering['trainable_records']:,} trainable records, "
        f"and {last['supervised_tokens_seen']:,} supervised tokens of the corpus's "
        f"{rendering['supervised_tokens']:,}"
    ) in flat
    assert f"**The budget `T` is {last['supervised_tokens_seen']:,} supervised tokens**" in flat
    assert f"supervised tokens fixed at M0's {last['supervised_tokens_seen']:,}" in flat
    assert f"warmup stays at {config['warmup_steps']} steps" in flat


def test_the_margin_table_is_computed_from_p_conf() -> None:
    partition = _json("reports/study-002/pconf-v1/pconf-v1.partition.json")
    gold, strata = partition["gold_counts"], partition["answer_strata"]
    rows = {
        "`CALL`": gold["CALL"],
        "`UNSUPPORTED`": gold["UNSUPPORTED"],
        "`CLARIFY`": gold["CLARIFY"],
        "`ANSWER`, pooled": gold["ANSWER"],
        "`ANSWER-constructed`": strata["ANSWER-constructed"],
        "`ANSWER-natural`": strata["ANSWER-natural"],
    }
    text = DRAFT.read_text(encoding="utf-8")
    for name, n in rows.items():
        assert f"| {name} | {n:,} | {100 * resolvable_margin(n):.2f}pp |" in text, name
    over = [name for name, n in rows.items() if resolvable_margin(n) > 0.10]
    assert over == ["`CLARIFY`", "`ANSWER-natural`"]
    assert "Two rows resolve more than 10 points." in text


def test_the_classifier_figures_come_from_its_one_test() -> None:
    result = _json("reports/prose-classifier/test-v2/prose-decision-classifier-v2.test-result.json")
    row = result["evaluation"]["by_unit_kind"]["single_exchange"]["per_mode"]["UNSUPPORTED"]
    assert f"with precision {row['precision']:.3f} and recall {row['recall']:.3f}" in _flat()


def test_the_evaluators_prose_rule_decides_as_quoted() -> None:
    # §8's table: the rule Study 001's evaluator applies to every reply that is not a tool call.
    quoted = {
        "I'm sorry, but I can't assist with that.": "ANSWER",
        "I am sorry, I cannot provide real-time information.": "ANSWER",
        "The answer is 42. Did you mean the total?": "CLARIFY",
    }
    text = DRAFT.read_text(encoding="utf-8")
    for reply, decision in quoted.items():
        assert parse_qwen_native_output(reply).decision == decision, reply
        assert f'| "{reply}" | `{decision}` |' in text, reply


def _conversation(decision: str) -> ToolConversation:
    return ToolConversation(
        id="x",
        source="s",
        tools=[{"name": "lookup", "description": "d", "parameters": {"type": "object"}}],
        messages=[
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ],
        metadata={"split": "train", "behavior": {"decision": decision}},
    )


def test_relabelling_changes_no_rendered_input() -> None:
    # §3: R2 trains on C0's tokens, because the renderer never reads the decision label.
    answer, unsupported = _conversation("ANSWER"), _conversation("UNSUPPORTED")
    assert _qwen_messages(answer) == _qwen_messages(unsupported)
    assert _qwen_tools(answer) == _qwen_tools(unsupported)
    assert "metadata" not in inspect.getsource(Qwen35_2BRenderer.render_sft)


def test_the_run_count_and_s2_size() -> None:
    tier_a, tier_b, tier_c, repeats = (
        ["C0", "R1"],
        ["R3", "C2", "X1"],
        ["D25", "D50", "S1", "S2"],
        2,
    )
    runs = 3 * (len(tier_a) + len(tier_b) + len(tier_c)) + repeats
    flat = _flat()
    assert runs == 29
    assert (
        f"Tier B becomes `R3`, `C2`, `X1` ({3 * len(tier_b)} runs), and the set {runs} runs" in flat
    )
    manifest = _json("reports/canonical-v3/canonical-v3.manifest.json")
    assert f"It holds {manifest['counts']['written']:,} records" in flat
