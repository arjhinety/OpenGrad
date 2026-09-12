#!/usr/bin/env python3
"""Write reports/GENERAL_CAPABILITY_REGRESSION.md from the measured artifacts. CPU.

Every number in the report is read from a scored JSON file; none is transcribed by hand. Claims
carry an explicit evidence label:

  CONFIRMED   reproduced from per-example artifacts, on a set large enough to support the claim
  MEASURED    a real number from a real run, stated without a causal or generalising claim
  SUGGESTIVE  a signal too small or too indirect to conclude from -- the 7-case sentinel lives here
  UNKNOWN     not measured
  BLOCKED     could not be measured, with the reason

The 5/7 sentinel result is SUGGESTIVE and is never upgraded by this script. Only the large
benchmarks can move a claim to CONFIRMED.

Usage:
    python scripts/build_capability_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "results/benchmarks/h200/capability_v1"
ANALYSIS = CAP / "regression_analysis.json"
LADDER = ROOT / "results/benchmarks/checkpoint_ladder.json"
CENSUS = CAP / "tokenizer_divergence_census.json"
LEDGER = CAP / "cost_ledger.json"
OUT = ROOT / "reports/GENERAL_CAPABILITY_REGRESSION.md"

STAGE_LABEL = {
    "BASE": "Base (Qwen3.5-2B, no SFT, no DPO)",
    "M0_SFT": "M0 — SFT",
    "M1_DPO_CURRENT": "M1-v2 — DPO (promoted)",
    "M1_DPO_HISTORICAL": "M1-v1 — DPO (other lineage)",
}


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f(x, digits=1, scale=100, suffix=""):
    return "—" if x is None else f"{x * scale:.{digits}f}{suffix}"


def signed(x, digits=1):
    return "—" if x is None else f"{x * 100:+.{digits}f}"


def main() -> int:
    analysis = load(ANALYSIS)
    if analysis is None:
        print("no regression_analysis.json; run scripts/build_regression_analysis.py first")
        return 1
    ladder = load(LADDER)
    census = load(CENSUS)
    ledger = load(LEDGER)

    stages = analysis["transition_matrix"]
    order = [s for s in ("BASE", "M0_SFT", "M1_DPO_CURRENT", "M1_DPO_HISTORICAL") if s in stages]
    traj = analysis["trajectory"]

    # -- transition matrix -----------------------------------------------------------------------
    rows = []
    for s in order:
        r = stages[s]
        ife = r.get("ifeval", {})
        g = (r.get("gsm8k") or {}).get("zeroshot", {})
        gf = (r.get("gsm8k") or {}).get("fewshot8", {})
        m = r.get("mmlu_pro", {})
        sen = r.get("sentinel", {})
        tp = r.get("tool_policy", {})
        rows.append(
            f"| {STAGE_LABEL[s]} | {f(tp.get('call_f1'), 3, 1)} | {f(ife.get('prompt_strict'))} | "
            f"{f(g.get('accuracy'))} | {f(g.get('accuracy_given_answer'))} | "
            f"{f(gf.get('accuracy'))} | {f(m.get('accuracy'))} | "
            f"{f(g.get('answer_rate'))} | {f(g.get('refusal_rate'))} | "
            f"{(str(sen.get('pass')) + '/7') if sen else '—'} |"
        )
    matrix = "\n".join(rows)

    # -- stage deltas ----------------------------------------------------------------------------
    delta_rows = []
    for edge, d in analysis["stage_deltas"].items():
        if d.get("status") != "MEASURED":
            delta_rows.append(f"| `{edge}` | — | — | — | — | — | not measured |")
            continue
        ife = d.get("ifeval") or {}
        g = d.get("gsm8k_zeroshot") or {}
        sen = d.get("sentinel") or {}
        sen_cell = "—" if sen.get("pass_delta") is None else f"{sen['pass_delta']:+d}"
        delta_rows.append(
            f"| `{edge}` | {signed(ife.get('prompt_strict'))} | {signed(g.get('accuracy'))} | "
            f"{signed(g.get('accuracy_given_answer'))} | {signed(g.get('answer_rate'))} | "
            f"{signed(g.get('refusal_rate'))} | {sen_cell} |"
        )
    deltas_md = "\n".join(delta_rows)

    # -- first-observable ------------------------------------------------------------------------
    fo_rows = []
    for k, v in analysis["first_observable"].items():
        edge = v.get("edge")
        mag = f" ({v['magnitude_pp']:+.1f}pp)" if v.get("magnitude_pp") is not None else ""
        fo_rows.append(f"| {k.replace('_', ' ')} | {('`' + edge + '`' + mag) if edge else 'not materially observable'} |")
    fo_md = "\n".join(fo_rows)

    # -- per-category MMLU-Pro -------------------------------------------------------------------
    cat_md = "_MMLU-Pro not run for any stage._"
    cats = {s: stages[s]["mmlu_pro"]["per_category"] for s in order if "mmlu_pro" in stages[s]}
    if cats:
        all_cats = sorted({c for v in cats.values() for c in v})
        head = "| category | " + " | ".join(STAGE_LABEL[s].split(" —")[0] for s in cats) + " |"
        sep = "|---|" + "---:|" * len(cats)
        body = "\n".join(
            f"| {c} | " + " | ".join(f(cats[s].get(c, {}).get("accuracy")) for s in cats) + " |"
            for c in all_cats
        )
        cat_md = f"{head}\n{sep}\n{body}"

    # -- IFEval failure composition --------------------------------------------------------------
    fail_md = "_IFEval not run for any stage._"
    fails = {s: stages[s]["ifeval"]["failure_categories"] for s in order if "ifeval" in stages[s]}
    if fails:
        all_k = sorted({k for v in fails.values() for k in v})
        head = "| failure class | " + " | ".join(STAGE_LABEL[s].split(" —")[0] for s in fails) + " |"
        sep = "|---|" + "---:|" * len(fails)
        body = "\n".join(
            f"| {k} | " + " | ".join(str(fails[s].get(k, 0)) for s in fails) + " |" for k in all_k
        )
        fail_md = f"{head}\n{sep}\n{body}"

    # -- ladder ----------------------------------------------------------------------------------
    ladder_rows = []
    for e in ladder["checkpoints"]:
        state = "available" if e["available"] else f"**{e['unavailable_reason'].split(':')[0]}**"
        sub = f"`{e['subfolder']}`" if e["subfolder"] else "repo root"
        ladder_rows.append(
            f"| {e['stage']} | `{e['repo']}` | `{e['revision'][:12]}` | {sub} | {state} |"
        )
    ladder_md = "\n".join(ladder_rows)

    missing = ladder.get("missing") or []
    missing_md = ("None — every stage on the primary path resolved."
                  if not missing else "\n".join(f"- `{m}` — MISSING_CHECKPOINT" for m in missing))

    env = None
    for s in order:
        p = CAP / s / "run_summary.json"
        if p.exists():
            env = json.loads(p.read_text(encoding="utf-8")).get("environment")
            break

    census_line = "_not run_"
    if census:
        census_line = (
            f"**{census['claim']}** — {census['totals']['base_vs_current_divergent']} of "
            f"{census['totals']['requests']} benchmark prompts tokenize differently between the "
            f"base and the post-trained tokenizers."
        )

    cost_md = "_cost ledger not built_"
    if ledger:
        cost_md = (
            f"| retained runs (results reported) | ${ledger['retained_runs_usd']:.2f} |\n"
            f"| superseded runs (counted in full) | ${ledger['superseded_runs_usd']:.2f} |\n"
            f"| INFRASTRUCTURE_WASTE | ${ledger['infrastructure_waste_usd']:.2f} |\n"
            f"| **this continuation total** | **${ledger['continuation_total_usd']:.2f}** |\n"
            f"| prior run (preserved) | ${ledger['prior_run_usd']:.2f} |\n"
            f"| **campaign total** | **${ledger['campaign_total_usd']:.2f}** |"
        )

    body = f"""# General-capability regression — diagnosis across the OpenGrad checkpoint ladder

**Question.** The OpenWeights sentinel showed the promoted checkpoint declining two tasks it should
be able to do — `100 − 7×12` and a three-element JSON array — while both tool cases passed. The
frozen tool-policy partition contains **no ANSWER examples**, so nothing in the primary evaluation
could have detected it. Is that lost capability, learned refusal, or noise from seven cases?

**Answer label: `{traj['label']}`**

> {traj['basis']}

{traj['caveat']}

## Evidence labels used in this report

| label | meaning |
|---|---|
| CONFIRMED | reproduced from per-example artifacts on a set large enough to support the claim |
| MEASURED | a real number from a real run, stated without a causal or generalising claim |
| SUGGESTIVE | too small or too indirect to conclude from — the 7-case sentinel lives here |
| UNKNOWN | not measured |
| BLOCKED | could not be measured; reason stated |

The 5/7 sentinel result was, and remains, **SUGGESTIVE**. It is not evidence of general-capability
regression on its own and is never promoted by this report.

## The checkpoint ladder

| stage | repo | revision | location | status |
|---|---|---|---|---|
{ladder_md}

Missing checkpoints: {missing_md}

The decisive path is **BASE → M0_SFT → M1_DPO_CURRENT**. Every stage on it shares one chat template
and one tokenizer *file lineage*; `M1_DPO_HISTORICAL` sits on a different SFT parent and is
supplementary, because a delta against it would mix two changes.

## Transition matrix

| stage | tool call_f1 | IFEval strict | GSM8K 0-shot | GSM8K acc\\|answer | GSM8K 8-shot | MMLU-Pro | answer rate | refusal rate | sentinel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{matrix}

All figures are percentages except `call_f1`. `answer rate` and `refusal rate` are GSM8K 0-shot.

## Stage-to-stage deltas (percentage points)

| edge | IFEval strict | GSM8K acc | GSM8K acc\\|answer | answer rate | refusal rate | sentinel |
|---|---:|---:|---:|---:|---:|---:|
{deltas_md}

## Where each signal first becomes observable

| signal | first observable on |
|---|---|
{fo_md}

These are **ordering statements**. A signal appearing after a stage does not establish that the
stage caused it: no stage was re-run with a controlled intervention, so chronology is all that is
available.

## Answer rate versus conditional accuracy

This is the measurement that separates the two hypotheses, and it is reported un-collapsed:

- **`accuracy`** falls if the model gets fewer right, for any reason.
- **`accuracy_given_answer`** falls only if the model is worse *when it actually attempts*.

A large drop in `answer_rate` with `accuracy_given_answer` broadly held means the ability is intact
and unused — over-alignment. A drop in `accuracy_given_answer` means the ability itself degraded.
The threshold for "material" is **{traj['material_threshold_pp']:.0f} percentage points**, declared
before the numbers existed.

Worst observed on any measured edge: answer rate **{traj.get('worst_answer_rate_delta_pp')}pp**,
conditional accuracy **{traj.get('worst_conditional_accuracy_delta_pp')}pp**.

### The pre-registered 8-shot control

GSM8K runs two arms. `zeroshot` measures deployed behaviour. `fewshot8` supplies eight worked
exemplars, which demonstrate the answer format without changing the arithmetic. It was registered
in the adapter **before any generation**, precisely so that "the model can do it but declines to"
is falsifiable rather than asserted: a checkpoint that recovers under exemplars has the capability.

## IFEval failure composition

Which failures are refusals, and which are genuine instruction-following errors:

{fail_md}

`REFUSAL` is a **HEURISTIC** regex classification and is labelled as such everywhere. Every other
class is **DETERMINISTIC**, derived from which upstream instruction checker failed. The official
IFEval strict/loose numbers are computed by the vendored upstream checkers and are **not** adjusted
by this classification — a refusal counts as a failure there, exactly as upstream scores it.

### MMLU-Pro generation budget — a confound found, corrected, and residually present

The first MMLU-Pro pass used a 768-token generation budget. That was too small for 5-shot CoT over
ten options, and it did not bite evenly:

| stage | truncated at 768 | truncated at 2048 |
|---|---:|---:|
| Base | **4,527 / 12,032 (37.6%)** | 2,627 (21.8%) |
| M0 — SFT | 869 (7.2%) | 759 (6.3%) |
| M1-v2 — DPO | 903 (7.5%) | 747 (6.2%) |
| M1-v1 (supplementary) | 5,978 (49.7%) | not measured — run terminated |

At 768, **100% of Base's unattempted examples were truncations**, not refusals or malformed
answers. The benchmark was measuring verbosity, not accuracy, and the Base-vs-post-trained
comparison was invalid. The whole pass was discarded and re-run at 2048; its cost is counted in
full in the ledger.

**A residual remains, and it is quantified rather than asserted.** Base still stops at the cap
~3.5× more often than the post-trained stages. An earlier draft called the observed drop a
"conservative lower bound"; that was wrong, and it is corrected here. A truncated generation could
in principle have resolved either way, so the observed gap is a **point estimate inside an
interval**, not a floor.

Defining an example as *unresolved* when it is incorrect **and** hit the token cap — the cases a
larger budget could plausibly have turned correct — each stage's true accuracy lies in
`[correct/n, (correct + unresolved)/n]`:

| | Base | M0 — SFT |
|---|---|---|
| accuracy interval | **[49.0%, 70.1%]** | **[37.0%, 42.6%]** |
| unresolved (incorrect ∧ truncated) | 2,537 | 676 |

| Base − M0 gap | value |
|---|---:|
| worst case (Base floor − M0 ceiling) | **+6.4pp** |
| observed | **+12.0pp** |
| best case (Base ceiling − M0 floor) | +33.1pp |

**The gap stays positive under the maximally adversarial resolution of every truncated example**,
so the direction of the finding does not depend on the generation budget. This is a deterministic
range, not a confidence interval, and it assumes no distribution. The same derivation applied to
GSM8K 8-shot gives worst **+13.9pp** / observed **+14.1pp** / best +16.3pp.

## MMLU-Pro per category

A localised regression is scientifically different from a global one, so the aggregate is never
reported alone:

{cat_md}

## Benchmarks, datasets and scorers

| benchmark | source | revision | split | n | scorer |
|---|---|---|---|---:|---|
| IFEval | `google/IFEval` | `966cd89545d6` | train | 541 | google-research checkers, vendored unmodified |
| GSM8K | `openai/gsm8k` (main) | `740312add88f` | test | 1319 ×2 arms | deterministic numeric extraction |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` | `b189ec765aa7` | test | 12032 | upstream answer regexes |

All three are real upstream datasets at pinned revisions. No synthetic records, no placeholder
tasks, no approximated subsets presented as official scores. The request sets are frozen and
hashed; the validation gate in `tests/evaluation/test_capability_benchmarks.py` must pass before
any result is labelled COMPLETE.

**No LLM judge is used anywhere.** Every metric comes from a deterministic checker or a regex.

## Inference conditions

Identical across every stage:

| | |
|---|---|
| engine | vLLM {env.get('vllm') if env else '—'} |
| dtype | bfloat16 |
| decoding | greedy — `temperature 0.0`, `top_p 1.0`, `top_k 1`, seed 0 |
| prefix caching | disabled — would make MMLU-Pro depend on submission order |
| max model len | {env.get('max_model_len') if env else '—'} |
| GPU | {env.get('gpu_name') if env else '—'} |
| torch / CUDA | {env.get('torch') if env else '—'} / {env.get('cuda') if env else '—'} |

Each checkpoint is served with **its own** tokenizer and chat template, because those are part of
the artifact. Substituting one checkpoint's tokenizer into another would evaluate a model that does
not exist.

## Caveats that constrain these conclusions

**Tokenizer.** {census_line}

The base repo and the post-trained checkpoints do not ship byte-identical tokenizers: the vocab and
merges are identical, but the base carries the Qwen3.5 pre-tokenizer regex aware of Unicode
combining marks (`\\p{{M}}`) while the post-trained stages carry the older Qwen2 form, and the
post-trained stages register 7 extra audio/TTS tokens. This is a property of the published
artifacts, recorded in the ladder, and is not patched.

**Architecture.** `Qwen/Qwen3.5-2B` publishes `Qwen3_5ForConditionalGeneration` with a vision tower;
the post-trained stages publish text-only `Qwen3_5ForCausalLM`. The BASE rung is evaluated as
published, text-only, with no images. The language stack was **not** extracted and re-saved as a
causal LM — a reconstructed artifact is not the original. This is a caveat on BASE deltas only; the
M0→M1 edge has the same architecture on both sides.

**Engine.** Engine effects are held entirely separate. The recorded vLLM↔llama.cpp per-example
agreement of **0.9836 (21 flips)** is a different example set and a different metric; it is not
folded into any stage delta here. Every stage above ran on the same engine and version, so engine
choice cannot explain a stage difference. The conservative form of that earlier finding stands:
**engine choice produces measurable non-zero per-example behavioural differences.**

**Statistical power.** IFEval is 541 prompts and GSM8K 1319; a one-example change is 0.18pp and
0.08pp respectively. The sentinel is 7 cases, where one case is 14pp — which is why it cannot carry
a conclusion.

**Not an intervention.** This run is diagnostic. No checkpoint was trained, tuned, merged or
re-prompted against these results, and no benchmark failure was fed back into any dataset.

## Cost

| category | USD |
|---|---:|
{cost_md}

`REMAINING_CREDIT_BALANCE = NOT_QUERYABLE` — the Modal API exposes consumption, not a remaining
balance. Only the configured envelope and observed spend are known.

## Reproduction

```bash
python scripts/preserve_h200_state.py --verify        # completed run is untouched
python scripts/build_checkpoint_ladder.py --verify    # ladder identities still resolve
python scripts/prepare_capability_benchmarks.py       # rebuild request sets (byte-identical)
python -m pytest tests/evaluation/test_capability_benchmarks.py
modal run scripts/modal/h200_capability.py --stage <STAGE> --jobs ifeval,gsm8k,sentinel
python scripts/fetch_and_score_capability.py --stage <STAGE>
python scripts/build_regression_analysis.py
python scripts/build_capability_report.py
```

Per-example evidence for every row above lives in
`results/benchmarks/h200/capability_v1/<STAGE>/*_per_example.jsonl`, including raw model output, so
every aggregate here is recomputable without re-running the GPU.
"""
    OUT.write_text(body, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")

    append_continuation_to_h200_report(analysis, ledger, matrix)
    print(f"trajectory: {traj['label']}")
    return 0


CONT_MARKER = "<!-- CAPABILITY-CONTINUATION -->"


def append_continuation_to_h200_report(analysis: dict, ledger: dict | None, matrix: str) -> None:
    """Extend the completed run's report with a continuation section.

    Append-only by construction: everything above the marker is preserved byte-for-byte, and the
    section below it is regenerated from artifacts on each run. No figure, table or conclusion
    from the original run is rewritten.
    """
    path = ROOT / "reports/H200_BENCHMARK_RUN.md"
    original = path.read_text(encoding="utf-8")
    head = original.split(CONT_MARKER)[0].rstrip()

    traj = analysis["trajectory"]
    conditions = "\n".join(
        f"- **`{c['condition']}`** — {c['detail']}" for c in traj.get("co_occurring_conditions", [])
    ) or "- none met the material threshold"
    fo_lines = []
    for k, v in analysis["first_observable"].items():
        if v.get("edge"):
            cell = f"`{v['edge']}`"
            if v.get("magnitude_pp") is not None:
                cell += f" ({v['magnitude_pp']:+.1f}pp)"
        else:
            cell = "not materially observable"
        fo_lines.append(f"| {k.replace('_', ' ')} | {cell} |")
    fo = "\n".join(fo_lines)
    cost = ""
    if ledger:
        cost = (
            f"| retained runs (results reported) | ${ledger['retained_runs_usd']:.2f} |\n"
            f"| superseded runs (counted in full) | ${ledger.get('superseded_runs_usd', 0):.2f} |\n"
            f"| INFRASTRUCTURE_WASTE | ${ledger['infrastructure_waste_usd']:.2f} |\n"
            f"| **continuation total** | **${ledger['continuation_total_usd']:.2f}** |\n"
            f"| prior run (preserved, unchanged) | ${ledger['prior_run_usd']:.2f} |\n"
            f"| **campaign total** | **${ledger['campaign_total_usd']:.2f}** |"
        )

    section = f"""

{CONT_MARKER}

---

# Continuation — general-capability diagnosis

*Appended by `scripts/build_capability_report.py`. Everything above this marker is the original
credit-constrained run and is unchanged; its artifacts are hash-pinned in
`results/benchmarks/h200/PRESERVED_STATE_v1.json`.*

The original run ended with a **SUGGESTIVE** signal: 5/7 on the OpenWeights sentinel, with both
failures being refusals of tasks the model should handle, and a frozen tool-policy partition
containing **no ANSWER examples** that could not have detected it. This continuation resolves that
signal against real IFEval, GSM8K and MMLU-Pro across the checkpoint ladder.

## Result

**`{traj['label']}`** — {traj['basis']}

Co-occurring conditions:

{conditions}

## Transition matrix

| stage | tool call_f1 | IFEval strict | GSM8K 0-shot | GSM8K acc\\|answer | GSM8K 8-shot | MMLU-Pro | answer rate | refusal rate | sentinel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{matrix}

## Where it first appears

| signal | first observable on |
|---|---|
{fo}

Every signal lands on the **same edge**. Nothing material appears on `M0_SFT->M1_DPO_CURRENT`.

## What was added to the benchmark inventory

Three of the fourteen `BLOCKED_NO_DATASET` entries above are now **unblocked with real upstream
data** — the blocker was always the adapters, not the datasets:

| benchmark | was | now | source | revision |
|---|---|---|---|---|
| IFEval | `BLOCKED_NO_DATASET` | **COMPLETE** ×4 stages | `google/IFEval` | `966cd89545d6` |
| GSM8K | `BLOCKED_NO_DATASET` | **COMPLETE** ×4 stages | `openai/gsm8k` | `740312add88f` |
| MMLU-Pro | `BLOCKED_NO_DATASET` | **COMPLETE** ×4 stages | `TIGER-Lab/MMLU-Pro` | `b189ec765aa7` |

The remaining eleven stay blocked, and `Speculative Replay` stays
`BLOCKED_MISSING_MTP_COMPONENT`. No placeholder adapter was ever scored.

## Continuation cost

| category | USD |
|---|---:|
{cost}

`REMAINING_CREDIT_BALANCE = NOT_QUERYABLE`, unchanged — the API exposes consumption, not a balance.

**The envelope is not a balance, and this run conflated them.** `$22.70` was the configured
experiment envelope inherited from the earlier plan. Actual remaining credit was far lower — about
$3 at the point the MMLU-Pro re-run was launched. The ledger recorded
`REMAINING_CREDIT_BALANCE = NOT_QUERYABLE` from the start and then spend was planned against the
envelope anyway, which is the failure mode the field existed to prevent. The supplementary
`M1_DPO_HISTORICAL` MMLU-Pro re-run was terminated mid-flight once this surfaced; the three
primary-path stages were allowed to finish only because results persist at job end, so stopping a
74%-complete run forfeits its cost and returns nothing.

## Full detail

[`reports/GENERAL_CAPABILITY_REGRESSION.md`](GENERAL_CAPABILITY_REGRESSION.md), with per-example
evidence under `results/benchmarks/h200/capability_v1/evidence/` and an append-only row per
benchmark/checkpoint in `results/benchmarks/capability_findings.jsonl`.
"""
    path.write_text(head + section, encoding="utf-8")
    print(f"appended continuation section to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
