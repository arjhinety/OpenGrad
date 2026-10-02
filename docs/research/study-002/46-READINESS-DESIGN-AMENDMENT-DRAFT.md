# 46 — Readiness design: arms, exposure, flags and instruments (`study_002_prereg_v14`)

> **Status: ADOPTED 2026-10-02, as drafted.** The owner chose the proposal on each of the eight §14 choices:
> drop `R2` and H4; supervised tokens fixed, steps planned; classifier v2's flag set; the base model's own
> answers; drop `C1`; `S2` is canonical-v3; classifier v2 as the measured output instrument; `S-ANS-E` on GSM8K.
> The other proposals come in with it. It is recorded in [03](03-PREREGISTRATION.md) and `reports/ERRATA.md`
> §31 in the same commit. The owner authorised the readiness engineering that follows, with no labelling run,
> GPU time or training. The text below is kept as drafted, as 40 and 43–45 were.
>
> **As drafted:** *Status: DRAFT, not adopted. It is recorded in neither 03 nor `reports/ERRATA.md` until the
> owner adopts it. Nothing is built, labelled, generated or trained until then.*
>
> - **What the owner decided on 2026-10-01,** after an audit of the fourteen checks of
>   [16](16-GPU-READINESS-GATE.md): *"Draft v14 first"*. The design questions the audit found are settled
>   before any readiness engineering, so nothing is built around a design that may change. The primary GPU
>   is left for later, so check 1 stays `BLOCKED`.
> - **What this document adds:** a proposal for each of those questions, marked as Claude's. The eight
>   choices that most change the study are listed in §14 for the owner.

## 1. Why

The audit found that no check of [16](16-GPU-READINESS-GATE.md) can be recorded `READY` today. For most
checks the missing piece is engineering. For the ones below it is the design itself: the design is
undefined, contradicts itself, or would spend money on a run that cannot test what it claims.

| Problem | Evidence | Section |
|---|---|---|
| Arm `R2` ("relabelled only") trains on exactly the tokens `C0` trains on | the decision label is never rendered (`src/opengrad/data/renderers.py`, `_qwen_messages`) | §3 |
| [04](04-ARM-MATRIX.md) fixes both the step budget and the supervised-token budget; with token-budgeted batches only one can hold | `04:13`, `04:14`, `04:101-103`; `deterministic_batches` in `src/opengrad/training/sft_runner.py` | §4 |
| `04:110` cites a token-matching threshold in [11](11-THRESHOLDS.md) that does not exist | 11 has no exposure tolerance | §4 |
| The flagged records are known only as counts, from a detector whose precision was never measured | `results/benchmarks/h200/capability_v1/sft_refusal_supervision_audit_canonical_v2.json`; `03:60` | §5 |
| No rule says what a corrected record's target text is | ROADMAP item 3, never settled | §6 |
| Arm `C1` and arm `S2` are undefined | "synthetic/back-translated" appears only at `02:48` and `04:32`; `04:38` | §7 |
| Stop rule 1 and `07:153` point to a detector precision floor that 11 does not contain | `03:91-94`; `HEURISTIC_REGEX_v2` not implemented (`04:142`) | §8 |
| The evaluator decides answer, clarify or decline for prose replies by a keyword rule that has never been measured | `src/opengrad/formatting/parser.py`, `parse_qwen_native_output` | §8 |
| [08](08-SENTINEL-SPEC.md) puts `S-ANS-E` on `ANSWER`-gold items, [09](09-BENCHMARK-PLAN.md) on GSM8K; its instruction text is not written | `08:51`, `09:28` | §9 |
| Check 3's "the margin it is used to test" has no value, and two populations resolve more than 10 points | `pconf-v1.partition.json` | §10 |
| Checks 2 and 7 and stop rule 4 are worded against superseded objects | `16:17`, `16:22`, `03:100-102` | §11 |
| No rule fixes the determinism mode, or what check 12's "real run" is | `05:112-118`, `16:27` | §12 |

## 2. Items changed

- **Arms** *(every item here is a proposal; §14)*: `R2` and `C1` are dropped, with H4 (§3, §7). `S2` is defined (§7). `R1`, `D25`, `D50` and `R3`
  are defined through a triage of the flagged records (§5, §6). The verdict vocabulary follows (§3).
- **Exposure:** supervised tokens are held fixed, steps vary, and 11 gains the tolerance (§4).
- **The flag set** is defined by the frozen `prose-decision-classifier-v2`; stop rules 1 and 2 apply to its
  triage (§5).
- **One measured instrument** judges every model reply as answer, clarify or decline, with precision floors in
  11 (§8).
- **`S-ANS-E`** is fixed (§9). Check 3 is given a value (§10). Checks 2 and 7 and stop rule 4 are reworded
  (§11). The determinism mode and the CPU smoke run are defined (§12).
- **Unchanged:** §13.

## 3. `R2` and H4: the decision label is not a training input

**What the code does.** The SFT renderer builds each training example from a record's messages and tools
only. `_qwen_messages` keeps each message's role, content, tool-call id, name and tool calls; `render_sft`
reads nothing else. The decision label lives in the record's metadata (`metadata.behavior.decision`). The
training data layer carries it as `behavior_decision` only to count it (`training/preprocess.py`,
`training/dataset.py`). No trained token depends on it.

**So relabelling changes nothing the model sees.** `R2`, defined as flagged records "relabelled only, target
text untouched" (`04:30`), trains on the same tokens as `C0`, in the same order under the same seed. Its
three seeds would be three paid repeats of `C0`. H4 ("relabelling alone restores direct answering", `02:64-70`)
cannot be tested in this trainer: `R2` can only reproduce `C0`, so the trainer's design, not data, decides it.

**What this says about the mechanism.** `ROADMAP.md` puts the defect in the label ("The defect is the LABEL,
not the refusal"). For supervised training, what the model learns from is the target text: a decline written in
reply to a question it could have answered. The label matters to how the corpus is described, balanced and
scored, not to what SFT teaches. `R1`'s training effect therefore comes entirely from its corrected text (§6).

**Options.**
- **A. Drop `R2` and H4** *(proposed)*. Record H4 as untestable here, and why. Three runs fewer.
- **B. Render the label into every training example** (for instance as a line of the system turn). That
  changes every arm, `C0` included, so `C0` no longer reproduces Study 001's training. It is a different study.
- **C. Keep `R2` as a `C0` replicate.** `REP-A` already measures that, and the name would mislead.

**Verdict vocabulary under A** ([02](02-RESEARCH-QUESTIONS.md)):
- `MECHANISM_SUPPORTED_RELABEL_RECOMMENDED` becomes `MECHANISM_SUPPORTED_CORRECTION_RECOMMENDED`: H1 holds and
  `R1` restores direct answering.
- `MECHANISM_SUPPORTED_CORRECTION_INSUFFICIENT` reads "`R1` does not restore direct answering" in place of
  "neither `R1` nor `R2`".

## 4. Exposure: hold supervised tokens fixed

**The contradiction.** [04](04-ARM-MATRIX.md) holds the step budget fixed (`04:13`) and also matches supervised
tokens (`04:14`); for `R3`, `C1` and `C2` it holds both (`04:101-103`). Batches are token-budgeted: a
micro-batch is filled up to 4,096 tokens or 8 sequences (Study 001's M0 config, with 2 micro-batches per
optimizer step and 2,400 steps). Change the corpus, and the tokens a step holds change with it. Study 001's
"matched" arm saw 1.19× its reference's supervised tokens for this reason (finding `#6`, `reports/ERRATA.md`
§2). The trainer stops only at `max_steps`, or on interruption.

**How much of its corpus an arm sees.** M0 stopped at step 2,400 having seen 27,372 examples, 0.169 passes over
its 161,966 trainable records, and 5,678,531 supervised tokens of the corpus's 33,565,721
(`runs/m0_sft_canonical_v2_final/metrics/train_log.jsonl`, `rendering_report.json`). Every arm sees about a
sixth of its corpus, so which records a seed reaches is part of what exposure means.

**Proposal.**
- **The budget `T` is 5,678,531 supervised tokens**, M0's logged tally at step 2,400, for every arm and seed.
  `X1` trains to half of `T`.
- **Steps are planned on CPU.** Batch order is a pure function of sample lengths, seed, epoch and the two
  budgets (`deterministic_batches`). A planner replays it with each sample's supervised-token count, and the
  first optimizer step at which the cumulative tally reaches `T` becomes that run's `max_steps`. The cosine
  schedule therefore ends where training ends; warmup stays at 120 steps.
- **Checkpoints at equal exposure.** Checkpoints are saved at the planned steps where the tally reaches 25%,
  50%, 75% and 100% of `T`, not every 600 steps. Selection on DEV then compares arms at equal exposure. Study
  001's ablation compared checkpoints that had seen different amounts (`reports/ERRATA.md` §2).
- **Tolerance, added to 11:** at every saved checkpoint the logged tally is within 1% of its target. Batching is
  deterministic, so the logged tally must also equal the plan exactly; a mismatch is a defect, not noise. An arm
  outside 1% is unmatched, and its comparison is descriptive (`04:110-111`).
- **Batch-composition log:** for every optimizer step, the record ids, their sources and their dispositions
  (§5), so "which corrected records a seed saw" is measured rather than inferred.
- **Varies by arm and is reported:** steps, examples seen, passes. **Fixed:** learning rate, schedule shape,
  warmup, both micro-batch budgets, gradient accumulation.

**Alternative.** Hold steps at 2,400 and report each arm's tally. Any arm more than 1% from `C0` becomes
descriptive. `R1` replaces short declines with longer answers and `R3` removes records, so both may cross it;
by how much is unknown until the planner runs.

## 5. The flag set, and the triage of flagged records

**What exists.** The 18,114 is `HEURISTIC_REGEX_v1`'s count of single-exchange records labelled `ANSWER` whose
target is a refusal: Glaive 14,066, When2Call 4,038, ToolACE 10, xLAM 0. Multi-turn records are outside its
scope (5 labelled `ANSWER` and 9,124 labelled `CALL` end in a refusal). It is a count: no list of the records
exists, and the detector's precision was never measured (`03:60`).

The frozen `prose-decision-classifier-v2` reads the first reply of any record, multi-turn included (contract
`prose-decision-input-v2`). On single-exchange items of P-DET-COVERAGE-v2 it predicted `UNSUPPORTED` (a
decline) with precision 0.978 and recall 0.964
([test result](../../../reports/prose-classifier/test-v2/prose-decision-classifier-v2.test-result.json)). That
population is Glaive and ToolACE. When2Call, the source of 4,038 of the 18,114, was measured only on P-DET-v1,
which is development-exposed for v2.

**Proposal: the flag set `F`** is every canonical-v2-final record labelled `ANSWER` whose first reply the
classifier predicts `UNSUPPORTED`.
- `ANSWER`-labelled first replies predicted `CLARIFY`, and abstentions, are counted and reported, not flagged:
  H1 concerns declines.
- The 18,114 stays the prior of [03](03-PREREGISTRATION.md). It is reported beside the size of `F`, with their
  overlap.
- The size of `F` is not yet counted.

*Alternative:* `F` is the v1 regex's 18,114, recomputed with record ids.

**The triage.** Every record in `F` is labelled blind by Gemini 3.8 Flash (High) and deepseek-v4.1-flash with
one of:
- `NOT_A_DECLINE`: the reply does not decline;
- `DECLINE_JUSTIFIED`: it declines, and the request needs something the assistant lacks (current or private
  data, an action in the world, a tool not offered);
- `DECLINE_UNJUSTIFIED`: it declines, but a capable assistant could answer correctly from general knowledge or
  reasoning, without the tools;
- `UNKNOWN`.

Each item shows the offered tools, the user's first message and the first reply. Source, record id and label
are hidden. A trial of 100 records comes first, and the procedure may be revised once after it, as 44 §9
allowed; the floors may not. No Claude model labels: Claude designed the intervention and built the classifier.

**Floors.** Stop rules 1 and 2 return to their original object, the flagged training records (ROADMAP item 2).
- **Stop rule 2, agreement:** raw agreement ≥ 0.80 over `F`, with κ beside it. Below it, the corpus
  intervention has no defined target and the study does not proceed to training.
- **Stop rule 1, flag precision:** among items both models label alike, the share labelled `DECLINE_*` is at
  least 0.90. Below it, the corpus intervention does not proceed.
- **Recall, reported:** 400 `ANSWER`-labelled first replies outside `F`, drawn by seed and labelled the same way.
  The share of declines among them is reported, with no floor. A missed decline stays in every arm, which
  biases `R1` towards a null result; that direction is stated.

**Dispositions, one per record in `F`** (check 7's disposition map):

| Reference label | Disposition |
|---|---|
| `NOT_A_DECLINE` | unchanged in every arm |
| `DECLINE_JUSTIFIED` | text unchanged; label becomes `UNSUPPORTED` (bookkeeping: no trained token changes) |
| `DECLINE_UNJUSTIFIED` | corrected in `R1` (§6) |
| disagreement or `UNKNOWN` | unchanged, counted |

## 6. Corrected target text

`R1` replaces the target of every `DECLINE_UNJUSTIFIED` record with an answer. Where the answer comes from
decides what `R1` tests.

- **A. The base model's own answer** *(proposed)*. Qwen3.5-2B at the pinned revision answers the record's own
  rendered prompt, tools included, with thinking off, greedy decoding and at most 512 new tokens. A reply is
  not used if the §8 instrument does not judge it an answer, if it calls a tool, or if it is truncated. That
  record keeps its decline, is disposed `CORRECTION_FAILED` and is counted.
  - *For:* it adds nothing the base model did not already know, so `R1` differs from `C0` in what the model is
    taught to do, not in the quality of a teacher.
  - *Against:* some answers will be wrong.
  - *Cost:* GPU inference of about an hour. It is not a training run, but it is spend the owner authorises.
- **B. A stronger external model writes the answers** (for instance one of the labelling models). The answers
  are better, but `R1` becomes partly a distillation arm: against `C0` it would change both the decision taught
  and the teacher, which confounds the capability endpoints.
- **C. Claude writes them.** The same confound as B, and Claude designed the study.

**Arms built from the dispositions:**
- `R1`: every `DECLINE_UNJUSTIFIED` record corrected.
- `D25`, `D50`: nested subsets of `R1`'s corrections (`D25` ⊂ `D50` ⊂ `R1`), taken in ascending order of
  `sha256("opengrad-dose-v14" | record id)`, so each dose contains the one below it.
- `R3`: every record disposed `DECLINE_JUSTIFIED` or `DECLINE_UNJUSTIFIED` removed. It measures what deletion
  does, including deleting correct declines, the trade ROADMAP warns of.

## 7. `C1`, `S1`, `S2` and a name

**`C1` cannot be built as written.** No field or document in the repository marks a record
"synthetic/back-translated". The phrase appears only at `02:48` and `04:32`. Each of Canonical-v2's four sources
is itself generated with language models, so the arm removes either everything or nothing.
- **A. Drop `C1`** *(proposed)*. The question it was for ("does any narrow tool-policy SFT collapse direct
  answering?") is partly answered by `R3` (the same narrow corpus without the declines) and by `C2` (a source
  removed).
- **B. Redefine it as `N1`, xLAM only:** the one source with no refusal target (0 of 57,342), trained to `T`.
  It tests narrow SFT with no decline text at all, but confounded with one source and call-only supervision.

Either way the arm id `C1` is retired. "C1" also names the canonical-v3 workstream ([21](21-C1-IMPLEMENTATION-STATUS.md),
[38](38-BALANCING-PERMISSION-AND-C1-AUTHORISATION.md)) and 06's coverage rule.

**`S2` (proposed): canonical-v3**, trained like `C0` to `T`. It holds 88,056 records, decision-balanced, from
normalization-v3. Its registration said it could enter a study only as its own numbered amendment and its own
factor (03, "Registration of the C1 corpus"). It differs from canonical-v2 in sources, normalisation version,
balance and labels, so it tests only whether the collapse appears under a second construction, not the
mechanism. It stays exploratory, Tier C. *Alternative:* drop `S2`.

**`S1`** needs a second model named and revision-pinned. Tier C runs only after a decisive Tier A, so its
design may not follow Tier A's result. **Rule:** every Tier C arm is defined in an amendment dated before any
Tier A score; `S1`, if still unnamed then, is dropped.

**Run count under A:** Tier B becomes `R3`, `C2`, `X1` (9 runs), and the set 29 runs with the two repeats,
down from 35. The cost table of [16](16-GPU-READINESS-GATE.md) is recomputed at adoption.

## 8. One measured instrument for model replies (check 6)

**Three instruments decide whether a reply answers or declines,** and none is measured on model output:
1. the corpus flagger (§5);
2. `HEURISTIC_REGEX_v1` on the benchmark sentinels (`src/opengrad/evaluation/capability.py`): GSM8K's
   `answer_rate` and `refusal_rate`;
3. the evaluator's own rule for prose replies, in `parse_qwen_native_output`. It decides answer, clarify or
   decline for every reply on `P-CONF` and `P-UNANS` that is not a tool call, so it sits under every behavioural
   floor of the gate.

**The third is a keyword rule.** A reply counts as a decline only if it contains "cannot help", "can't help",
"unable to", "not supported" or "unsupported". Any question mark makes it a clarification. Everything else is an
answer. On its own terms:

| Reply | Decision it assigns |
|---|---|
| "I'm sorry, but I can't assist with that." | `ANSWER` |
| "I am sorry, I cannot provide real-time information." | `ANSWER` |
| "The answer is 42. Did you mean the total?" | `CLARIFY` |

A model that declined every `ANSWER`-gold question in the first two phrasings would score as answering them.
It is Study 001's instrument and stays as it is. Study 002 needs a measured one.

**Proposal.**
- **One instrument** judges every prose reply: `prose-decision-classifier-v2`, frozen, applied after the existing
  parser has extracted any tool call. Tool calls stay the parser's. The parser's rule and the v1 regex are
  reported beside it on the sentinels, for continuity with Study 001, and never mixed with it (G9).
  `HEURISTIC_REGEX_v2` is not built.
- **Measured before any arm, on `P-DET-OUT`:** at least 300 replies from Study 001's four checkpoints (Base,
  M0, M1-v2, M1-v1). They are the "≥ 3 arms" of `07:144` that can exist before the gate. The replies come from:
  - the GSM8K 0-shot and 8-shot, IFEval and MMLU-Pro generations;
  - B0's held-out predictions on the DEV split (tool context).

  No reply to a `P-CONF`, `P-UNANS` or sealed item is used. The draw is stratified by checkpoint, prompt mode
  and the instrument's decision. Two non-Claude models label it blind, as in §5.
- **Floors, added to 11 (stop rule 1 for model output):** precision ≥ 0.90 for the answer class and for the
  decline class, on items both labellers agree on. Recall is reported and carried into H1's interval as
  `07:126-134` requires. Below either floor, the study stops at the detector.
- **After Tier A,** a second sample from `C0` and `R1` replies is labelled the same way and reported. If its
  precision falls below 0.90, every refusal-derived row carries the measured figure and its claims are
  descriptive. This rule is fixed now, before any reply exists.

**Alternatives.**
- Keep the parser's rule and measure it. The examples above suggest it would fail the floor.
- Build `HEURISTIC_REGEX_v2` as 07 planned: a new instrument, needing its own development and test, while
  classifier v2 already exists, frozen and tested.

## 9. `S-ANS-E` and the elicit instruction (check 10)

**Proposal:** `S-ANS-E` runs on GSM8K, the same 1,319 questions as `S-ANS-0` and `S-ANS-8`, so H2's three
points ([08](08-SENTINEL-SPEC.md), "Interaction with H2") sit on one set of items. Endpoints: `answer_rate`,
`accuracy`, `accuracy_given_answer`. `no_call_accuracy` leaves `S-ANS-E`, because GSM8K offers no tools.

The prompt is the 0-shot user turn with this instruction before it, separated by one blank line:

> Answer the question below directly if you can. If you cannot answer it, you may say so.

The text is fixed here, verbatim, and is never tuned against outputs.

*Alternative:* 08's version, on the `ANSWER` strata. It would add a second protocol on `P-CONF`, and its items
differ from the other two points of the curve.

The nine sentinels of 08 are also registered in a machine-readable file with mode and blocking status; that
is engineering.

## 10. Check 3: what "the margin it is used to test" means

| Population | n | Resolvable margin |
|---|---|---|
| `CALL` | 453 | 9.21pp |
| `UNSUPPORTED` | 453 | 9.21pp |
| `CLARIFY` | 371 | 10.18pp |
| `ANSWER`, pooled | 1,055 | 6.03pp |
| `ANSWER-constructed` | 771 | 7.06pp |
| `ANSWER-natural` | 284 | 11.63pp |

Two rows resolve more than 10 points. Both are already declared: `CLARIFY` is not adjudicable at 10 points
(`study_002_prereg_v8` item D), and `ANSWER-natural` meets the floor only (41 §10).

**Proposal.**
- The margin a population is used to test is 10 points, for every comparison between arms (`06:132-133`).
- A population that resolves more than 10 points passes check 3 only if an amendment dated before any score
  declares its comparisons descriptive. `CLARIFY` and `ANSWER-natural` are so declared.
- Floors stay blocking and fail closed: a measured value below a floor fails, however close (11, "Failing
  closed"). A between-arm difference on those two populations is `WITHIN_NOISE` and supports nothing.

*Alternative:* add items. Neither has a source: `CLARIFY` is the frozen confirmatory side, kept for continuity
with Study 001, and 41 §10 forbids topping up `ANSWER-natural`.

## 11. Wording

- **Check 2** reads: `study_002_gate_v1` at contract 3, wrapping `tool_use_promotion_v6` with
  `ADOPTED_PARAMETERS` (`study_002_prereg_v8` A and B), under the current `study_002_prereg_vN`; no amendment
  dated after a candidate's score; the commit recorded in the readiness record.
- **Stop rule 4** reads "returns PASS on an empty population, or on a synthetic population built to be
  vacuous". The self-test's healthy synthetic bundle is meant to pass (`16:55-58`).
- **Check 7** reads "a disposition for every record in the flag set of 46 §5", in place of "all 18,114".
- [04](04-ARM-MATRIX.md)'s scaffold table is out of date: the `ANSWER`-mode population is built, and §8
  replaces `HEURISTIC_REGEX_v2`. It gains a dated note, not an edit in place.

## 12. Determinism and the CPU smoke run (checks 9 and 12)

- **One determinism mode for every arm, chosen by a rule.** The GPU preflight runs `C0`'s config twice for 50
  steps under `DECLARED_DETERMINISTIC`. If both complete with identical logged losses, every arm declares
  `DECLARED_DETERMINISTIC`. If a kernel refuses deterministic mode or the losses differ, every arm declares
  `NON_DETERMINISTIC_KERNEL`. The mode is recorded in the readiness record before Tier A. `REP-A` measures
  the residual either way.
- **Check 12's "real run":** the real SFT code path on provider `cpu`, with a randomly initialised miniature of
  the Qwen3.5 architecture (as `tests/training/test_mtp_components.py` builds), a few steps on a `C0` subset,
  the real artifact writers, and the real evaluator on DEV items and synthetic fixtures.
  - It never reads `P-CONF`, `P-UNANS` or `P-SEALED` (03, tripwire 2).
  - Its outputs are plumbing evidence, never a result.
  - The pinned 2B model is not used. Full fine-tuning in fp32 with AdamW needs about 16 bytes per parameter
    (weights, gradients, two moments), about 32 GB, twice the development machine's memory.

## 13. What does not change

- Every threshold value in 11, the 0.80 agreement floor, the `n ≥ 200` mode floor and `P-UNANS`'s `n ≥ 385`.
- `P-CONF-v1`, `P-UNANS` and every other population.
- Arms `C0`, `R1`, `R3`, `C2`, `X1`, `D25`, `D50` and `S1`, each still varying one factor.
- Seeds 0, 1 and 2, `REP-A` and `REP-B`.
- The corpus every arm derives from: canonical-v2-final, release manifest sha `8ced403b…`.
- Training stays unauthorised. Checks 1 (the primary GPU) and 14 (available credit) need the owner's input,
  not a design decision, and are outside this amendment.

## 14. For the owner: the choices that most change the study

1. **`R2` and H4.** *Proposed:* drop both (§3). *Alternatives:* render the label in every arm; keep `R2` as a
   replicate.
2. **Exposure.** *Proposed:* supervised tokens fixed at M0's 5,678,531, steps planned per run (§4).
   *Alternative:* steps fixed at 2,400.
3. **The flag set.** *Proposed:* classifier v2's declines under `ANSWER` (§5). *Alternative:* the v1 regex's
   18,114.
4. **Corrected text.** *Proposed:* the base model's own answers (§6). *Alternative:* a stronger model's.
5. **`C1`.** *Proposed:* drop (§7). *Alternative:* `N1`, xLAM only.
6. **`S2`.** *Proposed:* canonical-v3 (§7). *Alternative:* drop.
7. **The output instrument.** *Proposed:* classifier v2, measured on `P-DET-OUT` with 0.90 precision floors
   (§8). *Alternatives:* measure the parser's rule; build regex v2.
8. **`S-ANS-E`.** *Proposed:* GSM8K with the §9 instruction. *Alternative:* the `ANSWER` strata.

Everything else is a proposal that is adopted with the amendment unless the owner objects. That covers:
- the 1% tolerance and checkpoints at equal exposure;
- the triage labels, its trial and the 0.90 flag-precision floor;
- the check 3 reading and the wording of §11;
- the determinism rule and the smoke run.

## 15. Limitations and cost

- **Written after population results, before any model result.** `P-CONF` and `P-UNANS` are built and
  labelled. No arm is trained or scored, and no threshold changes.
- **New floors are set before their measurements.** The classifier's precision on P-DET-COVERAGE-v2 is known
  (§5). The new floors apply to new measurements, the triage of `F` and `P-DET-OUT`, not to that result.
- **Labelling.** The triage covers every record in `F`. If `F` is near the v1 count, that is about 24 times the
  750 items of [45](45-PUNANS-CONSTRUCTED-AMENDMENT-DRAFT.md), so days of labelling runs. `P-DET-OUT` adds at
  least 300.
- **GPU before training:** the base model's answers (§6) and the determinism preflight (§12), each about an
  hour, each needing the owner's authorisation.
- **Model reference, not human gold**, for the triage and `P-DET-OUT` (`MODEL_REFERENCE`).
- **Dropping arms narrows the study.** Without `R2`, the study says nothing about labels as a training signal.
  Without `C1`, the "any narrow SFT" explanation rests on `R3` and `C2`.

## If adopted

In the adoption commit:
- this status line changes;
- the amendment is recorded in 03 and `reports/ERRATA.md`;
- 02, 04, 11 and 16 gain dated notes pointing here (they are preregistered, so they are not edited in place);
- the cost table of 16 is recomputed for the new run count;
- the status surfaces, including the public site, record it.

The engineering of [16](16-GPU-READINESS-GATE.md) follows, in the order the owner chooses.
