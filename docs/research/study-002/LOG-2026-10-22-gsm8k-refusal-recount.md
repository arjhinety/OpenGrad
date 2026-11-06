# Study 002 working log: GSM8K refusal recount and When2Call ANSWER check (2026-10-22)

Status: working log, not a frozen study file. Nothing here changes a promotion status, a registry row, or a frozen Study 001 file. No GPU or training job was started. All numbers come from existing committed or generated files.

Reason for the entry: before any tool-use gain is reported, check (1) whether the When2Call test split contains any ANSWER-gold item, (2) what `no_call_accuracy` returns on an empty ANSWER row and what the gates do with it, and (3) how many zero-shot GSM8K answers are refusals per checkpoint, using only existing raw responses.

## 1. When2Call test split: ANSWER-gold items

Files (found by `grep -ril when2call data manifests configs` and `find data/raw/when2call -name "*test*"`):

- `data/raw/when2call/test/when2call_test_mcq.jsonl`
- `data/raw/when2call/test/when2call_test_llm_judge.jsonl`

Commands and counts, reproduced:

```
grep -c '"correct_answer": "direct"' data/raw/when2call/test/when2call_test_mcq.jsonl          -> 0
grep -c '"correct_answer": "answer"' data/raw/when2call/test/when2call_test_mcq.jsonl          -> 0
grep -c '"correct_answer": "direct"' data/raw/when2call/test/when2call_test_llm_judge.jsonl    -> 0
grep -c '"correct_answer": "answer"' data/raw/when2call/test/when2call_test_llm_judge.jsonl    -> 0
grep -o '"correct_answer": "[a-z_]*"' <mcq file> | sort | uniq -c
```

| file | lines | cannot_answer (UNSUPPORTED) | request_for_info (CLARIFY) | tool_call (CALL) | direct / answer (ANSWER) |
|---|---|---|---|---|---|
| mcq | 3652 | 1295 | 1062 | 1295 | **0** |
| llm_judge | 300 | 100 | 100 | 100 | **0** |

Mapping that would produce ANSWER: `src/opengrad/evaluation/backends.py:67` (`"direct": "ANSWER"`). No raw When2Call test record carries `direct`, so the mapping never fires on this split.

Caveat, not resolved here: the normalized file `data/processed/behaviour-labels-v1/when2call.labels.jsonl` has 0 `"label": "ANSWER"` rows but 177 `"label": "DIRECT"` rows (steps `3_decline_with_delivered_content` 109, `4_optional_request_after_content` 20, `7_delivered` 48). `src/opengrad/data/behaviour_labels.py` treats `DIRECT` as a separate label. These rows carry `og_` hash ids and no split name, so I have not established whether they belong to the test population. Open: which split the 177 `DIRECT` rows come from, and whether they enter any evaluation as ANSWER gold.

## 2. What `no_call_accuracy` returns on an empty ANSWER row, and what the gates do

- `src/opengrad/evaluation/routing.py:28`: `"no_call_accuracy": matrix["ANSWER"]["ANSWER"] / sum(matrix["ANSWER"].values()) if sum(...) else 0.0`. An empty ANSWER row returns **0.0**, not "missing".
- `src/opengrad/promotion/tool_use_policy.py:173-174` (policy v3, `tool_use_promotion_v3`): `if name in unmeasurable: continue`. When the ANSWER row is empty, the no_call floor is **skipped**, so the verdict does not see the 0.0.
- `src/opengrad/promotion/tool_use_policy.py:37-39`: the v3 header says the same thing, and the code comment at `:71-75` says the empty row "is 0.0 for every model".
- `src/opengrad/promotion/m1_calibration.py:118` (policy v4, `tool_use_promotion_v4`): the detail string reads `no_call_accuracy is omitted when the evaluation population has no ANSWER examples.` v4 has no no-call / ANSWER floor at all.
- `src/opengrad/promotion/tool_use_policy.py:237-284` (v5) and `:395-471` (v6): v5 checks `ANSWER`-gold answer rate, and v6 returns `NOT_EVALUABLE` on an empty population. Neither is the policy under which M1-v2 was promoted.

**Conclusion: the gate cannot detect over-refusal on answerable items under the promoted policy (v4) or under v3.** Over-refusal on ANSWER gold would appear in `routing.py` only as ANSWER→UNSUPPORTED/CLARIFY cells, and `no_call_accuracy` would count those as misses. With no ANSWER gold, that path is empty. `over_call_rate` (`routing.py`, the last entry in the returned dict) counts calls on non-CALL gold, which is the opposite error. The gate never reads GSM8K either: the GSM8K outputs are scored only by `scripts/score_gsm8k.py`, which is not wired into any promotion verdict.

## 3. Zero-shot GSM8K refusals per checkpoint (existing raw responses only)

Source: `results/benchmarks/h200/capability_v1/{BASE,M0_SFT,M1_DPO_CURRENT,M1_DPO_HISTORICAL}/generations_gsm8k.jsonl`. Each file has 2,638 records: 1,319 with `example_id` ending `zeroshot` and 1,319 ending `fewshot8`. Only the `zeroshot` records were counted. Response text is the `output` field.

Three counts requested, plus the production classifier:

| checkpoint | n | exact phrase `unable to perform calculations` | broader pattern (`BROAD`, see below) | outputs with no digit | production `detect_refusal` (flag) | repo scorer `refused` (bucket) |
|---|---|---|---|---|---|---|
| BASE | 1319 | 0 | 2 | 0 | 0 | 0 |
| M0_SFT | 1319 | 1154 | 1319 | 1318 | 1319 | 1319 |
| M1_DPO_CURRENT (M1-v2) | 1319 | 1154 | 1319 | 1317 | 1319 | 1319 |
| M1_DPO_HISTORICAL (M1-v1, checkpoint-300) | 1319 | 202 | 955 | 871 | 937 | 933 |

Pattern definitions:
- **exact**: `unable to perform calculations` (case-insensitive).
- **BROAD**: `\b(apologi[sz]e|unable to|i can(?:no|')t|i cannot|i am unable|i'm unable|i'm sorry|cannot (?:help|assist|solve|perform))\b`, run on the whole output. Written in `scratchpad/gsm_refusal.py`.
- **no digit**: output contains no `\d`.
- **production `detect_refusal`**: `src/opengrad/evaluation/capability.py:33-53` (patterns), `:112-119` (400-character window, anchored near the start). Loaded by file path and run unchanged.
- **repo scorer**: `scripts/score_gsm8k.py` run unchanged on each generations file with `--stage <ck> --out <scratchpad>/rescore_<ck>.json`. Its `refusals` field counts `is_refusal and not attempted` (`capability.py:139-142`, `score_gsm8k.py:120-126`).

Result for the 1,319-item figure: **confirmed**. M0_SFT and M1_DPO_CURRENT have 1,319 zero-shot refusals under the production detector and under the repo scorer.

Wording check: the exact sentence "Apologies, but I'm unable to perform calculations…" covers only 1,154 of the 1,319 M0 items, not all of them. See section 4.

Historical M1-v1 caveat: `results/benchmarks/h200/capability_v1/M1_DPO_HISTORICAL/run_summary.json` records `checkpoint-300`. The repo scorer gives **933** refused (0.7074), which matches `docs/EXPERIMENT_RESULTS.md:59`, `reports/FINAL_CAMPAIGN_AUDIT.md:247`, `reports/ERRATA.md:501`, `docs/research/study-002/HANDOFF.md:115`. The production detector gives **937** on the same outputs, and `results/benchmarks/h200/capability_v1/evidence/gsm8k__M1_DPO_HISTORICAL.jsonl` records 937 `refusal` flags. The 4-item gap is the scorer's rule: an output that the detector flags but that also yields an extractable number is counted as an attempt, not a refusal. Both figures are reproducible under their stated definitions. Neither is a transcription error, so no ERRATA edit was made. The docs should say "933 refusals under the scorer's bucket rule (937 detector flags)".

Expected naming: `M1_DPO_CURRENT` is M1-v2 (promoted), per `scripts/build_capability_report.py:36`. `M1_DPO_HISTORICAL` is M1-v1, DPO applied directly to the base, per `scripts/audit_campaign_final.py:227`.

## 4. Hand classification of the 165 non-exact M0_SFT zero-shot outputs

Command: `scratchpad/gsm_recount.py` (prints the grouped list) and `scratchpad/gsm_classify.py` (keyword-assisted grouping, then checked by reading the 92 distinct texts).

Every non-exact output opens `Apologies, but I'm unable to …` or `Apologies, I'm unable to …` and gives no number as an answer. Verbs seen: solve (≈65), calculate (≈55), provide (≈45), answer (1). Grouped by hand:

- 148: capability disclaimer only (`I can't perform calculations / arithmetic / math`).
- 11: claims missing or insufficient information (e.g. "I don't have information about the price of a quarter"). GSM8K items are fully specified, so these are false claims of missing information.
- 6: redirects to a calculator, timer or professional.
- 0: clarifying questions to the user.

The single M0 output containing a digit is `gsm8k-1001-zeroshot`: "Apologies, but I'm unable to provide a specific time. I recommend using a timer or clock to ensure the cake is ready to serve at 5:00 pm." It is a refusal; the digit is a time in the text.

## 5. Commands run (all read-only with respect to the repo)

```
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I <scratch>/gsm_recount.py
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I <scratch>/gsm_classify.py
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I <scratch>/gsm_prod.py          # production detect_refusal, by file path
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/Scripts/python.exe -I scripts/score_gsm8k.py <gen.jsonl> --stage <ck> --out <scratch>/rescore_<ck>.json   # x4
git status --short -- results scripts src docs reports      # empty: no tracked or repo file changed by the scorer runs
```

Scratch files are in `C:\Users\arro\AppData\Local\Temp\claude\C--Users-arro-Downloads\c50bc2d7-32e8-4105-9629-f8425da6478d\scratchpad\`.

Scorer note: `scripts/score_gsm8k.py` writes its output only to the path given with `--out`, and the git status above shows no repo change.

## 6. Decisions left open (no action taken)

1. **Promotion status.** M1-v2 is still listed as PROMOTED under `tool_use_promotion_v4`. Not changed here. Changing or reinterpreting it needs your decision.
2. **177 `DIRECT` rows in `behaviour-labels-v1/when2call.labels.jsonl`.** If they enter an evaluation as ANSWER gold, the over-refusal path in section 2 would become measurable. Which split they come from is not yet established.
3. **933 versus 937 wording** in `docs/EXPERIMENT_RESULTS.md:59`, `reports/ERRATA.md:501`, `reports/FINAL_CAMPAIGN_AUDIT.md:247`, `docs/research/study-002/HANDOFF.md:115`. The numbers are reproducible; the docs do not state the bucket rule. Whether to add that note to ERRATA is your call.
4. **Over-refusal gate.** No promoted policy has an ANSWER-gold refusal check that can fire on this data. Adding one, or treating an empty ANSWER row as NOT_EVALUABLE in v4, would change a gate and needs your decision.

## 7. DIRECT label ids mapped to splits (dated 2026-10-22, added after sections 1-6)

Question: does any `DIRECT` (ANSWER-type) label id fall in the held-out When2Call split used for evaluation?

Method, all read-only, no training or GPU:
- Held-out split: `reports/evaluation/behavioral-heldout-v2-partition.json` (`algorithm: stratified_by_expected_decision`, `frozen: true` in `reports/evaluation/behavioral-heldout-v2.manifest.json`). Its ids are `confirmatory` 1277 + `dev` 2373 = 3650.
- Study 001 eval wiring: `reports/data/m0-final-freeze.json` and `configs/experiments/m0_sft_canonical_v2_final.yaml` point at this partition (`scripts/evaluate_sft_checkpoints.py ... --partition reports/evaluation/behavioral-heldout-v2-partition.json --partition-side confirmatory|dev`). The M0 freeze file has 0 mentions of `when2call`, so the When2Call linkage runs through the partition, not through a named split in the freeze.
- Label-to-split join: each `DIRECT` label id was looked up in the normalized When2Call shards `data/processed/normalization-v1/when2call-sft/shard-*.parquet` (columns `id`, `canonical_hash`, `metadata`). `metadata.source.original_split` gives the split of origin.

Commands (run from `C:\Users\arro\Downloads\OpenGrad`, `PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I`):
```
# 1. partition vs When2Call test uuids
ids = partition.example_ids.confirmatory U partition.example_ids.dev   -> 3650
test uuids = uuid field of data/raw/when2call/test/when2call_test_mcq.jsonl (3652) and when2call_test_llm_judge.jsonl (300)
overlap(partition, test uuids) = 3650

# 2. DIRECT ids -> split
data/processed/behaviour-labels-v1/when2call.labels.jsonl: when2call rows 14701; DIRECT rows 177 (177 distinct ids)
join DIRECT ids to when2call-sft shard ids: found 177, not found 0
original_split: train_sft 177
normalized metadata.behavior.decision: ANSWER 177
DIRECT ids equal to a held-out test uuid: 0
```

Counts per split (DIRECT ids):

| split | DIRECT ids | notes |
|---|---|---|
| train_sft (When2Call train, used for SFT) | **177** | all 177 normalized rows carry `behavior.decision = ANSWER` |
| held-out (behavioral-heldout-v2 = When2Call test MCQ, 3650 ids) | **0** | no DIRECT id matches a partition id |
| When2Call test llm_judge (300) | **0** | no DIRECT id in the raw file |
| train_pref, when2call-preference | not in the label file | no ids to compare |

Stop condition check: no DIRECT id falls in the held-out split, so the stop condition in the instruction does not fire.

Two findings to record:
1. **The held-out set is the When2Call test split.** All 3650 partition ids are When2Call test MCQ uuids, so the Study 001 held-out evaluation is the test split, and it has 0 raw `direct` items (section 1).
2. **ANSWER-decision examples do exist, in the training split.** 177 `train_sft` rows are ANSWER (DIRECT) and enter the SFT corpus. The held-out eval has none, so the gate's no-call floor cannot test them. The asymmetry is the one that matters for over-refusal: the model is trained on direct answers and evaluated on a split with none to test.

Data-integrity note (not resolved): for 148 of the 177 DIRECT ids, the `canonical_hash` in the labels file differs from the `canonical_hash` of the same id in the SFT shard (id match is 177/177). The labels file and the SFT shard were built with different canonical hashes. Whether the labels or the shard hash is the intended one has not been established. Recorded, not repaired.

Commit context: not recorded here. No frozen Study 001 file, no `results/registry.jsonl`, and no promotion status was modified.

## 8. Linkage of the 177 DIRECT ids to held-out uuids (dated 2026-10-22, follow-up to section 7)

Purpose: section 7 showed that the DIRECT ids are train_sft rows. This section tries to link each DIRECT id to a held-out uuid directly, and says where that link cannot be made.

Method 1, reproduce the id from raw rows (no assumption about the id):
- The id is built in `src/opengrad/data/adapters.py:122-126`: `raw_hash = sha256(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str))`, and `upstream_id = "og_" + raw_hash[:16]`.
- Applied to the raw train source `.cache/normalization/raw/when2call_train_sft_train.parquet` (15000 rows, columns `tools`, `messages`), the same expression reproduces **177 of 177** DIRECT ids. The map from id to raw record is therefore exact.
- 178 raw rows hash to those 177 ids. One id is held by two rows with identical content (the same sha256 prefix). This is the duplicate that `data/processed/normalization-v1/when2call-sft/manifest.json` counts as `canonical_duplicates: 124`.

Method 2, content match of each DIRECT train record against the held-out test records:
- Held-out candidates: the 3650 partition ids in `reports/evaluation/behavioral-heldout-v2-partition.json`, all of which are uuids of `data/raw/when2call/test/when2call_test_mcq.jsonl` (3652 rows) or `when2call_test_llm_judge.jsonl` (300 rows). Test question texts: 2299 distinct.
- Each DIRECT train record's user-turn text (`messages`, role `user`) was compared with every test `question`: exact match, case- and whitespace-normalized match, and substring match of any test question of 40 or more characters.

Results:

| check | DIRECT train records (178 rows / 177 ids) | matches to held-out test |
|---|---|---|
| exact user-text = test question | 178 | **0** |
| normalized user-text = test question | 178 | **0** |
| test question (>=40 chars) contained in user text | 178 | **0** |

Counts per split, final:

| split | DIRECT ids | how established |
|---|---|---|
| train_sft (When2Call train, SFT corpus) | **177** (178 raw rows) | `original_split = train_sft` in the normalized metadata (177/177); id reproduced from the raw train parquet (177/177) |
| held-out (`behavioral-heldout-v2`, = When2Call test MCQ, 3650 partition uuids) | **0** | no content match to any held-out question (0/178) |
| When2Call test llm_judge (300) | **0** | as above |
| not linkable | **0** | every DIRECT id was linked to a train record; no DIRECT id was left unlinked |

**What is and is not established.**
- Established: each DIRECT id maps to exactly one raw train record (method 1), and none of those records shares its question text with a held-out item (method 2). That is strong evidence that none is a held-out item.
- Not established: a uuid-level link. The raw train records carry no `uuid` field, so there is no uuid to compare with the partition. The non-membership result rests on content, not on id. A held-out item whose question was paraphrased would not be caught by the exact, normalized or 40-character substring test. Only a uuid or a canonical-hash join would close that gap, and none exists in the repo for these rows.
- Stop condition: no DIRECT id is in the held-out split under any test used, so the instruction's stop condition does not fire. The 2 test uuids outside the partition (3652 minus 3650) were not examined for DIRECT membership; they are not in the label file in any case.

**Data-integrity note carried forward from section 7.** For 148 of the 177 DIRECT ids, `canonical_hash` in the labels file differs from the SFT shard value for the same id (id match 177/177). Still not resolved.

**Consequence, for the decision on the gate.** The 177 DIRECT (ANSWER-decision) examples are in the SFT corpus, and the held-out split used for Study 001 evaluation contains no ANSWER-gold items (section 1). An over-refusal check on answerable items therefore cannot be run on this split with the current gate (section 2).

No frozen Study 001 file, no `results/registry.jsonl`, no `reports/ERRATA.md` and no promotion status was edited. No GPU or training job was started.

## 9. Over-refusal on answerable items, existing outputs only (dated 2026-10-22, requested by Arjhine: "can we check it now?")

Question: does M1-v2 refuse answerable items beyond GSM8K, and is that visible to the promotion gate?

Method: the production refusal detector `src/opengrad/evaluation/capability.py` (`detect_refusal`, patterns at lines 33-53, 400-character anchored window), loaded by file path and run unchanged on the existing generations files in `results/benchmarks/h200/capability_v1/<ckpt>/generations_<bench>.jsonl`. No model was loaded and no job was run.

Command: `PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -I <scratch>/over_refusal.py` (loader registers the module in `sys.modules` before `exec_module`, because `@dataclass` needs it).

| benchmark (answerable items) | BASE | M0_SFT | M1_DPO_CURRENT (M1-v2) |
|---|---|---|---|
| gsm8k (rows = zero-shot + fewshot8, 2638) | 0 / 2638 (0.00%) | 1319 / 2638 (50.00%) | 1319 / 2638 (50.00%) |
| gsm8k zero-shot only (1319) | 0 (0.00%) | 1319 (100%) | 1319 (100%) |
| mmlu_pro (12032) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| ifeval (541) | 2 (0.37%) | 103 (19.04%) | 107 (19.78%) |

The gsm8k 50% row is the zero-shot 100% plus the few-shot 0%, so the zero-shot line is the one to read.

IFEval check (because a 20% rate could be false positives): the 107 M1-v2 matches are 104 `apology_inability`, 1 `leading_cannot`, 1 `no_capability`, 1 `leading_unable`. Six matches were read directly. They are real refusals of benign instruction-following tasks, for example "I'm unable to create a resume for you", "I'm sorry, but I'm unable to write lyrics or generate content in all capital letters", "I'm unable to create an advertisement in JSON format". So the rate reflects refusals, not detector noise. Caveat: the detector is regex-based. I read six matches, not all 107, so the 19.78% figure is approximate to the extent that some matches could be borderline.

Findings:
1. **Over-refusal on answerable items is real and not confined to GSM8K.** IFEval refusals rise from 0.37% (BASE) to 19.78% (M1-v2), on instruction-following items that are all answerable.
2. **MMLU-Pro shows no refusals for any checkpoint.** So the over-refusal is not a general behaviour of the tuned model.
3. **The promotion gate cannot see any of this.** v4 has no check that reads GSM8K or IFEval, and no no-call / ANSWER floor (`m1_calibration.py:118`). v3 skips the same floor on an empty ANSWER row (`tool_use_policy.py:173-174`). The IFEval rise therefore does not affect M1-v2's PROMOTED status under the current policy.

What this does not show: that M1-v2 is worse overall. The gate measures tool-routing behaviour; these refusals are on a different capability. Whether they should change the status is a decision for Arjhine. No promotion status was changed, and no frozen file, registry row, or ERRATA entry was edited.

## 10. Status change: M1-v2 PROMOTED -> NOT PROMOTED (dated 2026-10-22, on Arjhine's instruction "change the status")

Decision: Arjhine asked for the status to be changed after §9 showed that the v4 gate cannot see over-refusal on answerable items. Target status: `NOT PROMOTED`, the same term the repo uses for M0 (`reports/M0_CANONICAL_V2_FINAL_EXECUTION_REPORT.md`: `COMPLETED / NOT PROMOTED`). The reason is the one stated in §9 and §2, not a new measurement.

Edited (not frozen, not pinned by any freeze manifest):
- `README.md`: the M1 row now reads NOT PROMOTED, and a dated amendment paragraph sits above the "PROMOTED is a gate change" paragraph. Original sentences are kept.
- `reports/M1_DPO_EVALUATION.md`: a dated "Status amendment" line was added under the original status line. The original line is kept.
- `reports/M1_DPO_EXECUTION_REPORT.md`: same, a dated amendment line above the original status line.

Not edited, and why:
- `results/registry.jsonl`: frozen (pinned by `reports/data/m0-final-freeze.json` and `reports/data/m0-v2-minus-xlam-freeze.json`), and excluded by instruction. Still says PROMOTED.
- `runs/checkpoint_registry.json` (line 20, `promotion_status: PROMOTED`): pinned by both freeze manifests. Still PROMOTED.
- `docs/EXPERIMENT_STATUS.md`: generated from `runs/checkpoint_registry.json`. `tests/results/test_state_consistency.py` requires byte-identical regeneration, so a hand edit would break the test. Still PROMOTED until the registry changes.
- `reports/releases/hf-publication-2026-09-11-m1-canonical-v2-final-v2.json` (line 36, `"status": "PROMOTED"`) and the published Hugging Face model card `release/huggingface/qwen35-2b-m1-dpo-canonicalv2-final-v2/README.md` (line 73): external publication. Changing it is outward-facing and needs approval; not changed.
- `reports/ERRATA.md`: no stated number is wrong, so it was not touched. The status is a decision, not a correction.
- Code: `opengrad promote` records PROMOTED without evaluation (`reports/ERRATA.md:1471`). Not changed.

Commands and checks:
```
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/results -p no:cacheprovider -q   -> all passed, exit 0
git status --short -- README.md reports docs runs results registry  -> tracked changes: README.md, reports/M1_DPO_EVALUATION.md, reports/M1_DPO_EXECUTION_REPORT.md only
```
The test that checks README wording (`tests/results/test_state_consistency.py`, `test_removed_stale_claims_do_not_return`) still passes. It requires "promoted" and "M1-v2" to appear in the README, and both still do.

Observation: the consistency tests pass with the README saying NOT PROMOTED and the frozen registry saying PROMOTED. So the test suite does not check the README status against the registry. The disagreement is real and visible only from the records listed above.

Open items for Arjhine:
1. Whether to re-state the frozen registry entries (needs a freeze-amendment path; not done here).
2. Whether to update or withdraw the published Hugging Face model card (external).
3. Whether to add a README-vs-registry status check to the test suite.

## 11. Follow-up on the status change (dated 2026-10-22, on Arjhine's instruction: restate, update card, add check)

**Side-agent note checked.** A note said the M1 model card might be generated, so a hand edit could be overwritten. Checked: no generator writes `release/huggingface/qwen35-2b-m1-dpo-canonicalv2-final-v2/README.md`. `scripts/build_gguf_card.py:18` writes only the GGUF card (`...-gguf/README.md`). `scripts/publish_card_corrections.py` writes only an append-only publication record under `reports/releases/` (line 362) and reads the committed card. The M1 card is hand-written.

**3. Status check (done).** `tests/results/test_readme_registry_status.py` compares the README M1 row status with `results/registry.jsonl` and with `runs/checkpoint_registry.json::m1_dpo_canonical_v2_final_v2::dpo-checkpoint-30`. Both checks are strict `xfail` with the reason, because the two registries still say PROMOTED. A passing test checks that the README row has a recognised status.
```
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/results/test_readme_registry_status.py -p no:cacheprovider -q -rxX  -> 1 passed, 2 xfailed, exit 0
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/results -p no:cacheprovider -q  -> all passed (2 xfailed), exit 0
freeze_m0_final.py --verify -> VERIFY PASSED
```

**2. Model card (local only, not committed, not published).** `release/huggingface/qwen35-2b-m1-dpo-canonicalv2-final-v2/README.md`: the sentence "Checkpoint 30 is PROMOTED" now says it was promoted on 2026-09-11 and that the promotion is withdrawn on 2026-10-22, so checkpoint 30 is NOT PROMOTED. A paragraph gives the reason, with the measured figures (GSM8K zero-shot refusal 100%, base 0%; IFEval 19.8%, base 0.4%; MMLU-Pro 0%). Publishing needs a commit first, because `publish_card_corrections.py` works from committed bytes, and it pushes to the Hugging Face repository. That is outward-facing and needs explicit approval. Not done.

**1. Restating the frozen registries (not done, needs a decision).** `results/registry.jsonl` and `runs/checkpoint_registry.json` are pinned by `reports/data/m0-final-freeze.json` and `reports/data/m0-v2-minus-xlam-freeze.json`. Changing their bytes makes `freeze_m0_final.py --verify` fail until the manifests are re-built. Three guard assertions in `tests/results/test_state_consistency.py` also encode PROMOTED for M1: lines 124 and 163 check the registry row, and `test_every_valid_promotion_has_a_promoted_registered_checkpoint` (lines 134-154) needs a promoted M1 checkpoint. Restating means re-freezing evidence manifests whose purpose is to pin the numbers behind reports, and rewriting guards that exist to stop a regression to "no promoted model". Options are in the message to Arjhine; nothing was changed.

**Repo state at the end of this section.** Tracked changes: `README.md`, `reports/M1_DPO_EVALUATION.md`, `reports/M1_DPO_EXECUTION_REPORT.md`, `release/huggingface/qwen35-2b-m1-dpo-canonicalv2-final-v2/README.md`. Untracked: this log, and `tests/results/test_readme_registry_status.py`. No registry, freeze manifest, or ERRATA file was changed.

## 12. Card committed and published; registry restatement attempted and reverted (dated 2026-10-22)

**Model card (done).** Committed alone as `d74fae2` ("docs(release): M1-v2 card states that the promotion is withdrawn"). Dry run of `scripts/publish_card_corrections.py`: 6 cards examined, 1 changed (the M1 card), 0 unrecorded Hub commits, and the only removed line is the sentence rewritten to say the promotion is withdrawn. `--upload` published it at Hub commit `e4e1a31c98d0884e67cd18b83fcd68085620f79a`; the script re-fetched it and the bytes match. Publication record: `reports/releases/hf-publication-2026-10-22-card-license-correction.json` (untracked; the script names every record "card-license-correction", whatever the content). The side-agent note about a generated card was checked: no generator writes the M1 card (`build_gguf_card.py` writes only the GGUF card).

**Registry restatement (attempted, reverted, not done).** Three steps, each checked:
1. `runs/checkpoint_registry.json` (checkpoint `m1_dpo_canonical_v2_final_v2::dpo-checkpoint-30`, line 458) `promotion_status` PROMOTED -> REJECTED. `results/registry.jsonl` M1 row `status` PROMOTED -> REJECTED. Byte counts unchanged (both words have 8 letters); only the SHA-256 values change.
2. Rebuilding with `scripts/freeze_m0_final.py` (build) **dropped every `unversioned_by_design` pin** (weights and predictions for the M0 checkpoints, 14 keys), because those files are absent from this checkout and the builder records them as `None`. That erases frozen evidence. **Defect in the builder** (`reports/data/m0-final-freeze.json` before the rebuild had those pins). The manifest was restored from backup and only the two registry SHA-256 values were changed by exact string replacement. `freeze_m0_final.py --verify` then passes.
3. `scripts/freeze_minus_xlam_ablation.py` cannot be rebuilt here (missing `.release/hf/toolpolicy-canonical-v2-minus-xlam/release-manifest.json` and two `environment.json` files). Its `--verify` **already fails before this change**: missing files and `changed: runs/central_ledger.jsonl`, which was not touched here. Its manifest was not edited.
4. The test `tests/results/test_experiment_index.py::test_current_repository_registry_is_not_the_empty_bootstrap_placeholder`, plus `test_state_consistency.py::test_registry_projection_is_not_stale`, showed the registry is a **projection of the authoritative record** `runs/m1_dpo_canonical_v2_final_v2/experiment.json` (line 38, `"status": "PROMOTED"`). A registry-only restatement therefore fails `FIELD_MISMATCH (DRIFT)`. A complete restatement also has to change that record and regenerate `results/registry.jsonl` through `src/opengrad/results/registry.py`, which the freeze also pins.

**Reverted.** `results/registry.jsonl`, `runs/checkpoint_registry.json`, `reports/data/m0-final-freeze.json`, and `tests/results/test_state_consistency.py` were restored from backup (`scratchpad/pre-restate/`). This also reverted my own edits to the last two files, which the environment reported as changed on disk; the revert was deliberate and is disclosed here. The guard edits (renamed M1 publication test asserting REJECTED; "no valid promotion remains" assertion) were reverted with them.

**Check (done, kept as strict xfail).** `tests/results/test_readme_registry_status.py` is restored to two strict `xfail` agreement checks (README vs `results/registry.jsonl`, README vs `runs/checkpoint_registry.json`) and one passing test (README M1 row has a recognised status). The reason string names the frozen registries.

**Verification at the end of this section.**
```
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/results -p no:cacheprovider -q -rxX  -> all passed, 2 xfailed, exit 0
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts .venv/Scripts/python.exe -I scripts/freeze_m0_final.py --verify  -> VERIFY PASSED
```

**Open for Arjhine:** restating the M1 status needs the authoritative `experiment.json` and the registry projection changed as well. That is a larger change than the approved option, so it is not done. The publication record `reports/releases/hf-publication-2026-10-22-card-license-correction.json` is not committed.

## 13. Full restatement of M1-v2 as REJECTED (dated 2026-10-22, on Arjhine's choice "Full restatement")

**What changed (uncommitted; the card commit `d74fae2` and publication are the only commits and publications today).**
- `runs/m1_dpo_canonical_v2_final_v2/experiment.json` line 38: `"status": "PROMOTED"` -> `"REJECTED"`. This is the authoritative record. The registry projection is derived from it (`src/opengrad/results/registry.py:477`). The `promotion_checkpoint` history and the central ledger are unchanged (the central ledger is pinned by the Study 001 freeze and was not appended to).
- `runs/checkpoint_registry.json`: the entry for `m1_dpo_canonical_v2_final_v2::dpo-checkpoint-30` (`tokens_seen` 10327): `promotion_status` `PROMOTED` -> `REJECTED`. The historical `status_note` (the v4 pass at the time) is kept.
- `results/registry.jsonl`: rebuilt through `rebuild_registry` (the builder API). Only the M1 row changed, and only `status`. The 16 rows and the two integrity findings are unchanged.
- `reports/data/m0-final-freeze.json`: two SHA-256 pins re-pinned (`results/registry.jsonl`, `runs/checkpoint_registry.json`). Nothing else in the manifest changed (2 lines changed in git).
- `tests/results/test_state_consistency.py` (CRLF preserved): the "every valid promotion" test asserts that no valid promotion remains, with the reason; the M1 publication test is renamed `test_withdrawn_dpo_is_still_traceable_to_its_release_record` and asserts `REJECTED`; the generated status count assertion changes from `Promoted models: 1.` to `Promoted models: 0.`.
- `tests/results/test_readme_registry_status.py`: the expected-failure markers are removed. The two agreement checks pass, and so does the row-status check.
- `docs/EXPERIMENT_STATUS.md`: regenerated with `scripts/reporting/generate_experiment_status.py`, not hand-edited.

**Not changed, with reasons.**
- `reports/data/m0-v2-minus-xlam-freeze.json`: cannot be rebuilt here (missing release manifest and two environment files). Its `--verify` already failed before this section (missing files; `changed: runs/central_ledger.jsonl`, which was not touched here). Its pins for `results/registry.jsonl` and `runs/checkpoint_registry.json` are also stale. Left as is; reported.
- The published Hugging Face card and publication record (section 12) say PROMOTED or describe the withdrawal, as published. The card now states the withdrawal (`e4e1a31`).

**Defects found in the tooling (not fixed here).**
1. `scripts/freeze_m0_final.py build` drops every `unversioned_by_design` pin (weights and predictions) when those files are absent, recording `None`. Running it here would have erased frozen evidence; it was not used for the final state.
2. `python -m opengrad.results.cli rebuild-registry` silently does nothing: `src/opengrad/results/cli.py` has no `__main__` entry point. The registry was rebuilt through the `rebuild_registry` function instead.
3. Registry status is read from `runs/<id>/experiment.json` (`registry.py:477`), not from the ledger. A registry-only edit fails the projection drift test (`FIELD_MISMATCH`).

**Verification.**
```
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/results -p no:cacheprovider -q -rxX   -> all passed, exit 0 (no xfail)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts .venv/Scripts/python.exe -I scripts/freeze_m0_final.py --verify  -> VERIFY PASSED
git diff --numstat: results/registry.jsonl 1/1, runs/checkpoint_registry.json 1/1, runs/m1_.../experiment.json 1/1, reports/data/m0-final-freeze.json 2/2
```

**Open.** Commit the restatement set (not done; no instruction to commit it). The publication record `reports/releases/hf-publication-2026-10-22-card-license-correction.json` is untracked. The minus-xlam freeze needs its own decision.

## 14. Commit-order check: v4 gate versus M0 results (dated 2026-10-22, read-only)

Question: was the v4 gate (`tool_use_promotion_v4`, the parent-relative gate that promoted M1-v2) committed after the M0 confirmatory results?

Commands (read-only, `git log` and `git show`):
```
git log -1 --format='%h  author %ai  commit %ci  %s' 26234c2
git log -1 --format='%h  author %ai  commit %ci  %s' 92ca2b9
git merge-base --is-ancestor 26234c2 92ca2b9   -> true
git log --format='%h %ci %s' -S "tool_use_promotion_v4" -- src/opengrad/promotion
git log --format='%h %ci %s' -- runs/m0_sft_canonical_v2_final/eval/confirmatory
```

| commit | UTC commit time | what it adds |
|---|---|---|
| `26234c2` | 2026-09-11 08:58:54 | the definitive M0 SFT run and its evaluation, including `runs/m0_sft_canonical_v2_final/eval/confirmatory` (first appearance) |
| `92ca2b9` | 2026-09-11 17:10:36 | `tool_use_promotion_v4` first appears in `src/opengrad/promotion` (`-S` finds this commit); `dpo_live.py`, `test_m1_calibration_policy.py` |
| `078c6fe` | 2026-09-12 23:50:06 (+0800) | a later change to the v4 code path (diagnosis of an SFT-induced regression) |

Result: `26234c2` is an ancestor of `92ca2b9`. The v4 gate was committed about 8 hours 12 minutes after the M0 confirmatory results. The M1-v2 run launched at 17:29:53 (`runs/m1_dpo_canonical_v2_final_v2/experiment.json`, `launch_timestamp`), after `92ca2b9`. The order matches what the card and README say.

What each rule admits (read from the code):
- v3 (`src/opengrad/promotion/tool_use_policy.py`, `PromotionPolicyV2`, lines 100-234): call_f1 retention >= 0.90 of baseline; macro recall over measurable classes >= 0.40; no-call accuracy >= 0.40 **only when the ANSWER row is non-empty** (skipped otherwise, lines 173-174); unsupported >= 0.30; clarification >= 0.50; over-call rate <= 0.20; parse validity >= 0.99; regression allowance 0.10 on call precision, call recall, clarification and unsupported.
- v4 (`src/opengrad/promotion/m1_calibration.py`, `M1CalibrationPolicy`): call precision >= 0.65; call recall >= 0.60; macro recall >= 0.60; over-call rate <= 0.20; clarification >= 0.60; unsupported >= 0.40; parse validity >= 0.99; regression allowances against the M0 parent: precision 0.05, recall 0.10, clarification 0.10, unsupported 0.10. **No no-call / ANSWER check at all** (line 118 omits it when the population has no ANSWER examples).

Implication, unchanged from section 2: under v4 an answerable item that is refused is not checked. The v4 thresholds are not derived in the repo (no source found in section 2 or in the v4 code path).

No promotion status, frozen file or registry was changed by this section.

## 15. Corrections and the minus-xlam fix (dated 2026-10-22)

**Side-agent note on normalization-v3 (checked; the premise is wrong).** The note said the main agent wrote new `canonical_hash` values into the normalization-v3 files, and that the split of the DIRECT ids had been traced only against v1. Checked:
- `data/processed/normalization-v3/when2call/` shard mtimes are 2026-09-16T22:40. `git status --short -- data/processed/normalization-v3` is empty, and no file there is newer than this log. Nothing in normalization-v3 was written in this session.
- The v3 split was checked in this session (section 14 work). Results for the 177 DIRECT ids: stored `canonical_hash` == recomputed `semantic_hash({tools, messages})` in 177/177; v3 hash == labels hash in 177/177; v3 `metadata.source.original_split` = `train` for 177/177. v1 names the same split `train_sft`. So the split statement is the same training split under two names, and it holds under both versions.
- The 148 mismatched hashes are a normalization difference in `tools`, not a hashing fault. 148/148 mismatched rows differ in `tools` (`parameters` in 403 tool entries, a top-level `required` key in 384). v1 writes Python-style types (`float`, `List[float]`), and v3 writes JSON Schema types (`number`, `array`), adds `nullable`, and moves requiredness into `parameters`. I did not test whether a transform of v1 reproduces the v3 hash; that claim is not made here.
- Which normalization the training runs consumed is **not established** by these files. That is the open question for the SFT provenance.

**Minus-xlam ablation: superseding manifest (done, uncommitted).** `reports/data/m0-v2-minus-xlam-freeze.json` (v1) is left byte-identical. Its verify fails for three reasons, each checked:
- 3 pinned items are missing and not recoverable: `.release/hf/toolpolicy-canonical-v2-minus-xlam/release-manifest.json` (its parent release has 176 shards, none present anywhere in the repo or caches), and `environment.json` for the fixed-compute and matched-exposure arms (written at run time; no copy exists).
- 8 weights files (`unversioned_by_design`) are absent from this checkout.
- 3 pins changed on disk: `results/registry.jsonl`, `runs/checkpoint_registry.json`, `runs/central_ledger.jsonl`. All three match their git blobs at the manifest's commit `449161ab` (checked with `git show`), so the pinned state is recoverable from git. The ledger is append-only and has grown by 5 lines since, which is by design, so an on-disk pin on it can never stay current.

New manifest `reports/data/m0-v2-minus-xlam-freeze-v2.json`, built by `scripts/freeze_minus_xlam_v2.py`:
- `present`: 49 pins, verified against the working tree.
- `git`: 3 pins, verified against `git show 449161ab:<path>`.
- `unrecoverable`: 11 pins, reported by name with their v1 hashes, and not counted as verified.
```
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I scripts/freeze_minus_xlam_v2.py            -> wrote v2: present 49, git 3, unrecoverable 11
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -I scripts/freeze_minus_xlam_v2.py --verify   -> checked present 49, git 3; VERIFY PASSED for every recoverable pin
git status --short -- reports/data/m0-v2-minus-xlam-freeze.json   -> empty (v1 untouched)
```
Not done: the 11 unrecoverable pins cannot be restored from this checkout. Whether the environment and release files exist on the training host is not something this repository can answer.

**Also established in this section.** The commit-order check in section 14 stands. The freeze builder fix and the registry CLI fix from section 12 are committed (`c8f13b6`). The v2 manifest and its script are not yet committed.

**Training corpus provenance (checked; open).** `configs/experiments/m0_sft_canonical_v2_final.yaml` names the training corpus `canonical_v2_final` with hash `8ced403b996e563d6e279aee7fdb346fc829fe5ff6af9daf8ef47c0a4007e161`. That hash is the SHA-256 of `.release/hf/toolpolicy-canonical-v2-final/release-manifest.json`, which matches its own pin. The corpus shards are not present in this checkout (the release manifest lists 176 of them, none found on disk or in caches). So which normalization version the When2Call rows in the trained corpus used cannot be checked here. The labels file is built from normalization-v3 (`src/opengrad/data/behaviour_labels.py:64`). The 177 DIRECT rows are in the train split under both versions (v1 `train_sft`, v3 `train`).

**Small Mind re-freeze (done, uncommitted).** `scripts/freeze_study_001.py` `REFREEZE_LOG` and `reports/data/study-001-freeze.json` `refreeze_log` gained the same dated entry for README.md, and the README pin in the manifest now matches the committed README (18090 bytes, sha256 `fd6e8aaa…`). `scripts/freeze_study_001.py --check`: OK, 76 artifacts match. `scripts/validate.py`: exit 0, 143 checks, 0 failed, 1 acknowledged drift, 14 manual.

## 16. Canonical-v2 When2Call rows: normalization, DIRECT ids, held-out overlap (dated 2026-10-22)

This section closes the open question in section 15. The release was downloaded from the Hub repo `arrochi112/OpenGrad-ToolPolicy-Canonical-v2` at the pinned revision `66470c07ed0a79941f49a5cf67c1b3b1a7d8196e`. All 176 shards match the SHA-256 values in `.release/hf/toolpolicy-canonical-v2-final/release-manifest.json`. The copies live in the scratchpad, not in the repo. The When2Call part is 7 shards with 6,505 rows. Nothing was trained, and the repo was not edited.

**Which normalization the training corpus used (directly shown for the tool schema; hash identity is strong evidence).**
- Parameter type names are JSON Schema names. In the first 300 rows the counts are `string` 418, `integer` 164, `array` 8, `number` 3, `boolean` 1. The v1 normalization writes Python-style names such as `float` and `List[float]`, so the release is not v1 form.
- Each row's stored `canonical_hash` equals `semantic_hash({tools, messages})` recomputed from that row's own content, in 6,505 of 6,505 rows.
- The same `canonical_hash` equals the one in `data/processed/behaviour-labels-v1/when2call.labels.jsonl` in 6,505 of 6,505 rows. That labels file is built from normalization-v3 (`src/opengrad/data/behaviour_labels.py`, docstring lines 1-3).
- The release manifest hash is the one named in `configs/experiments/m0_sft_canonical_v2_final.yaml` (`8ced403b…`). So the M0 training corpus is this release, and its When2Call rows use normalization-v3.
- Not checked: whether the copy on the training host is byte-identical to the Hub copy. The release-manifest hash ties the two together, and the per-shard hashes match.

**DIRECT ids against the release (direct).**
- 80 of the 177 DIRECT ids appear in the release (matched by `opengrad_id`). All 80 have the same `canonical_hash` as the labels file.
- 97 DIRECT ids do not appear in the release. The release holds 6,505 of the 14,701 labelled When2Call rows. Its selection rule was not read, so why these 97 are absent is **unknown**.
- None of the 177 DIRECT ids is in the held-out split. This is the content check from section 14, and it still holds. The ids cannot be matched at the uuid level, because training rows carry no uuid.

**The release's own label field carries no information (direct).** Every one of the 6,505 rows has `behavior_decision` = `ANSWER`. That is the adapter default (`src/opengrad/data/adapters.py` `adapt_when2call`). Every row ends in an assistant prose turn, and zero rows contain a tool call. The actual labels exist only in the labels file: UNSUPPORTED 3,979, CLARIFY 2,445, DIRECT 80, UNLABELLED 1. Anyone reading `behavior_decision` from the release would get the wrong distribution.

**What the DIRECT training targets say (observed; the interpretation is open).** A rough refusal-wording count (`unable`, `apolog`, `cannot`, `can't`, `not able`, `don't have`, `do not have`) on the last assistant turn:

| Label | Refusal-style wording | Rows |
|---|---|---|
| UNSUPPORTED | 3,979 | 3,979 |
| CLARIFY | 29 | 2,445 |
| DIRECT | 54 | 80 |

- The CLARIFY count is low because clarifying replies are usually phrased as "could you please provide...", which the pattern does not match. So the table is not a clarify count.
- Two DIRECT examples: "apologies, i'm unable to determine the timezone for a specific date and time. however, cape town, south africa is generally in the south african standard time"; and "apologies, i'm unable to filter profanity from the review. however, i can confirm that the review does contain profanity."
- So for most DIRECT rows the target is a partial refusal, not a plain answer. This is by design, not a parser error. `src/opengrad/data/decision_classifier.py` (line 9 and the step at lines 409-412) defines DIRECT as "answers or performs the request", and it has an explicit rule `3_decline_with_delivered_content`: a decline followed by a "however ..." clause with content is DIRECT. The labels file records the step for each row. Across the 177 DIRECT labels: `3_decline_with_delivered_content` 109, `4_optional_request_after_content` 20, `7_delivered` 48. The definition is a choice the classifier made, and the mixture inherits it. Whether that choice fits a "can the model answer directly" behaviour is a research decision, not a bug to fix here.
- By rule, DIRECT rows from When2Call carry no balancing weight (`DIRECT_PERMITTED_SOURCES = {glaive}` in `behaviour_labels.py`).

**Held-out overlap (direct).**
- The raw held-out MCQ file has 3,652 rows and 2,295 distinct normalized question texts.
- One release prompt matches a held-out question exactly: "what is the current weather?" (held-out uuid `77821b1c-2320-4070-95dd-f8e274536495`). The release row is labelled UNLABELLED, and its target is "to provide you with the current weather information, i'll need the location. could you please specify the location?"
- So the claim that the held-out set is disjoint in content needs one qualification: an exact-text overlap of one generic prompt. Its effect is small, but it is not zero. Section 14 was about DIRECT ids only, and that claim still holds for them.

**Still open.**
- Why 97 DIRECT ids are absent from the release, and what its selection rule was.
- Whether the classifier's DIRECT rule (decline followed by content counts as DIRECT, decision_classifier.py line 9 and lines 409-412) is the intended meaning for the mixture. It is a definitional choice, now recorded.
- Whether the overlapping held-out uuid should be excluded from any comparison that depends on the held-out split.

No frozen file, registry, promotion status or held-out file was changed. The log append is uncommitted.

**Decision and derived split (dated 2026-10-22).** The study owner chose to split partial refusals into their own label. Done as a derived file, not a change to the source:
- Source `data/processed/behaviour-labels-v1/when2call.labels.jsonl` is untouched. The frozen classifier (`src/opengrad/data/decision_classifier.py`) and `behaviour_labels.py` are untouched, because `behaviour_labels.py` refuses to run unless the classifier source is byte-identical to the frozen one. A change to the pipeline would reopen that freeze, which is a separate decision.
- Derived file `scratchpad/when2call.labels.split-v1.jsonl` (sha256 `f090ab23b4dc81c4cee9b8c3efb7f6d6cbed8e0789fa58d404eaa74397b71495`). Rule: rows labelled DIRECT whose recorded `step` is `3_decline_with_delivered_content` become `DECLINE_WITH_CONTENT`, with `split_from: DIRECT` added. No other row changes. The split uses the classifier's own step field, not the wording regex.
- Counts, all 14,701 rows: before UNSUPPORTED 7,334, CLARIFY 7,189, DIRECT 177, UNLABELLED 1. After UNSUPPORTED 7,334, CLARIFY 7,189, DECLINE_WITH_CONTENT 109, DIRECT 68, UNLABELLED 1. The 68 remaining DIRECT rows are steps 4 (20) and 7 (48).
- Release subset (6,505 rows): DECLINE_WITH_CONTENT 54, DIRECT 26. The 54 happens to match the wording-count figure above, but the two were not checked row by row, so the match is not claimed as the same rows.
- Consequence: the M0 mixture and its weights still use the original labels. Any count that uses the new label is a derived view until the pipeline is changed under a new contract.

## 17. Classifier v3 and behaviour-labels-v2 (dated 2026-10-22)

The study owner chose to carry the DIRECT split into the pipeline (section 16, decision). This section records what was built and what was not.

**Built (new files only; nothing frozen was edited).**
- `src/opengrad/data/decision_classifier_v3.py`, tag `prose-decision-classifier-v3`. It is v2 with one change: the three places where v2 returns DIRECT at tree step `3_decline_with_delivered_content` now return `DECLINE_WITH_CONTENT`. The `LABELS` tuple gains that label. The step names, the rules and the evidence are unchanged. Its LF sha256 is `b3ea0e12b49c984c00f73d5dfaa343912308c59d9fd55cdb58f55b8721d31fb3`.
- `src/opengrad/data/decision_classifier_v2.py` is byte-identical to its freeze, sha256 `47436ca9…`. The v2 freeze and the authorisation in doc 38 are unchanged.
- `src/opengrad/data/behaviour_labels_v3.py` builds `data/processed/behaviour-labels-v2/` (one file per source, plus `manifest.json`). It imports the counting and serialising helpers from `behaviour_labels.py`. It does not replace `behaviour-labels-v1`.
- `tests/data/test_decision_classifier_v3.py`: freeze pin, v2 byte check, the decline-with-content case (v2 DIRECT at step 3, v3 DECLINE_WITH_CONTENT at step 3), a plain answer that stays DIRECT at step 7, the label set, and zero weight from the labels-v2 weight rule.

**Verification (direct).**
- `python -m opengrad.data.behaviour_labels_v3 --verify`: PASS, no problems. Exit 0.
- Row-by-row comparison with `behaviour-labels-v1` over all sources (180,211 + 1,222 = 181,433 rows): every row has the same `id`, `step` and `canonical_hash`. The only label differences are 1,222 rows that were DIRECT at `3_decline_with_delivered_content` and are now DECLINE_WITH_CONTENT: glaive 1,087, toolace 26, when2call 109. No other label changed. v1 DIRECT was therefore 32,968 + 1,222 = 34,190 rows, and v2 DIRECT is 32,968.
- Labels-v2 counts: DECLINE_WITH_CONTENT 1,222; DIRECT 32,968; CLARIFY 22,014; UNSUPPORTED 22,419; ABSTAIN 7,791; CALL 12; CALL_BY_STRUCTURE 92,851; UNLABELLED 2,156.
- `pytest tests/data/test_decision_classifier_v3.py tests/data/test_decision_classifier_v2.py tests/data/test_behaviour_labels.py`: 87 passed.

**Weight: none in labels-v2 (direct, by construction).** Every row has `weight_permitted: False`, and the manifest says `weight_authorised: false`. The reason is doc 38 §5 item 3 and 22 §6: a classifier change voids the balancing authorisation until the new version is tested on a new, untouched population. That retest has not been done, and it needs new labelling work and model calls.

**Consequence of the change, stated plainly.** Under v1, glaive DIRECT rows carry weight (38 §2), so the 1,087 glaive rows that move to DECLINE_WITH_CONTENT currently count in the training mix. Under v2 they would not, until the retest passes.

**Decision (the study owner, 2026-10-22).** The training mix keeps the v1 labels and their weights. Labels-v2 is for counts and comparisons only. No retest is scheduled.

**Not done.**
- `src/opengrad/versions.py` still names `prose-decision-classifier-v1` for `DECISION_CLASSIFIER_VERSION`, and it is not updated here. Any canonical-v3 or balance manifest that is rebuilt on labels-v2 would fail the provenance gate (`FAIL_CLASSIFIER_VERSION`, as in `reports/canonical-v3/provenance-gate-v1.json`). No manifest references labels-v2 at present.
- No mixture, no balance, no training corpus, and no M0 or M1 result was rebuilt or rerun. No GPU or training job was started.
- Nothing is committed. The v3 files, the labels-v2 output and the test file are uncommitted.

**Decisions taken after section 17 (the study owner, 2026-10-22).**
- v3 retest: not now. The training mix stays on v1 labels, with v1 weights. Labels-v2 is for counts only.
- `src/opengrad/versions.py` is left as it is, naming v1. The owner wants to propose a different fix later.
- Commit: all work recorded in sections 16 and 17 is committed as one change.
- Held-out overlap: the training prompt "what is the current weather?" matches held-out uuid `77821b1c-2320-4070-95dd-f8e274536495` exactly. Any comparison that uses the training overlap must exclude that uuid. The held-out partition file is frozen and was not edited to do this. The exclusion is recorded here and applies to any future comparison.

## 18. The 97 DIRECT ids missing from the release (dated 2026-10-22, read-only)

**Question.** Section 16 found 97 of the 177 DIRECT ids absent from the canonical-v2 final release. Why?

**Answer (direct).** They were quarantined by the release's source-schema step, not dropped at random and not leaked.
- The release manifest (`.release/hf/toolpolicy-canonical-v2-final/release-manifest.json`, `sources[3]` = when2call) reports 15,000 raw rows: accepted 6,505, duplicates 38, rejected 8,457, of which `rejected_SCH_UNSUPPORTED_TYPE` 8,445 and `rejected_duplicate tool name` 12.
- Raw file: `.cache/normalization/raw/when2call_train_sft_train.parquet` (15,000 rows). Each raw row's id is `og_` + the first 16 hex characters of sha256 of `json.dumps(record, sort_keys=True, ensure_ascii=False)`, the same rule as `src/opengrad/data/adapters.py` lines 122-126. This reproduces all 177 DIRECT ids and the release's opengrad ids.
- The rule that separates them: a row is rejected when any tool property type is outside the JSON Schema names and outside the bare Python names `str`, `int`, `float`, `bool`, `list`. Forms such as `str, optional`, `int, optional`, `List[int]`, `Dict`, `Tuple[...]` and `set` are rejected. The rule predicts 8,445 rejections and 6,543 kept rows (of which 38 are duplicates, leaving 6,505). It does not predict the 12 duplicate-tool-name rejections, which are a separate check.
- All 97 missing DIRECT rows are predicted rejections. The rule is stated in `src/opengrad/data/source_schema.py` (docstring): forms that cannot be represented safely are quarantined with a reason, not coerced.

**Consequence.** The 97 rows never entered M0 training, because the M0 corpus is this release. The labels file includes them only because labelling reads normalization-v3, which translates these forms at the adapter boundary (`source_schema.py`). The labels file is therefore wider than the training corpus by 8,457 rows of When2Call.

**Not established.**
- 175 raw rows match no label id. They are all in the rejected group, so the labeller did not see them. The reason was not checked.
- The rule was tested against the release manifest's counts, not against the release-building code path itself.

No files were written in this section beyond this log entry. Nothing is committed.

**Decisions after section 18 (the study owner, 2026-10-22).**
- The 11 unrecoverable pins in `reports/data/m0-v2-minus-xlam-freeze-v2.json` are accepted as unverified. The log does not claim they match, and no further search is planned.
- The 97 missing DIRECT ids were investigated read-only (section 18). No change was made to the release or the labels.

## 19. Wrapper build of classifier v3 and labels-v2 (dated 2026-10-22)

**What changed (uncommitted).**
- `src/opengrad/data/decision_classifier_v3.py` is now a wrapper over the frozen v2 rules. v2 is byte-identical (sha256 `47436ca9…`). The wrapper turns v2's DIRECT at step `3_decline_with_delivered_content` into `DECLINE_WITH_CONTENT`, and returns every other v2 decision unchanged, apart from v3's `classifier_version`. Its LF sha256 is now `5c092f8e…`. It was `b3ea0e12…` for the 626-line copy it replaces.
- `src/opengrad/data/behaviour_labels.py` is one labelling engine, parameterised by a `LabelSet`. `V1` keeps v1's defaults, and the v1 module-level names are unchanged. `src/opengrad/data/behaviour_labels_v3.py` is a `LabelSet` over that engine, with its own pin.
- `check_frozen` now checks the wrapper and the v2 rules on every run. `verify` now checks that the labelled sources are the corpus's sources, and that the stored totals equal the sum of the rows.

**Checks.**
- labels-v2 rebuilt over 181,433 rows. The four label files are byte-identical to the output before the wrapper. The manifest and the counts report differ from that output only in `classifier.source_sha256_lf` (`b3ea0e12…` to `5c092f8e…`).
- labels-v1 rebuilt through the shared engine. The four label files, the manifest and the counts report are byte-identical to the snapshot taken before any change.
- `verify` passes for labels-v1 and for labels-v2.
- Tests: all of `tests/data`, plus `test_flag_set.py` and `test_prose_classifier_v2_oneshot.py`, pass. `ruff check` is clean on the changed files.
- `tests/data/test_behaviour_labels.py`: the corpus fixture of one test now lists its source, because the source-set check is new.

**Migration counts, v1 to labels-v2 (field by field, 181,433 rows in both sets).**
- 1,222 rows change `label`. All of them are DIRECT to DECLINE_WITH_CONTENT.
- 78,360 rows change `weight_permitted`. v2 sets every row to False. v1 gave weight to every UNSUPPORTED and CLARIFY row and to the glaive DIRECT rows.
- 78,495 rows differ in at least one field: 135 in the label only, 77,273 in the weight only, and 1,087 in both.
- The phrase "1,222 rows differ" in earlier notes counted label changes only. Quote all three numbers together.

**Not done.** Nothing is committed. `src/opengrad/versions.py` is untouched. Dia-Grad is untouched. The retest pass condition for labels-v2 is still the owner's to set (38 §5 item 3 gives none). The Small Mind root-script check is still open. The training mix stays on v1 labels, with v1 weights.

## 20. Reconciling the ANSWER count and the over-refusal claim (dated 2026-10-22, read-only)

**Question.** Before any claim that the When2Call tool-use test catches over-refusal: (a) how many held-out items have a correct action of answering directly, from the raw test files and the held-out uuid list; (b) do any of the 177 DIRECT train ids fall in the held-out split?

**Counts.**

| check | result |
|---|---|
| `correct_answer = direct` (or `answer`), `when2call_test_mcq.jsonl` (3652 rows) | **0** |
| `correct_answer = direct` (or `answer`), `when2call_test_llm_judge.jsonl` (300 rows) | **0** |
| the same, restricted to the 3650 held-out partition uuids | **0** |
| `correct_answer` values in the held-out split | `cannot_answer` 1295, `tool_call` 1295, `request_for_info` 1060 (sum 3650) |
| the 2 MCQ uuids outside the partition | both `request_for_info` (`6a9903a1…` "What is the current time?", `77821b1c…` "What is the current weather?") |
| DIRECT ids reproduced from the raw train parquet (`og_` + sha256 prefix, section 8 rule) | **177 of 177**, behind 178 raw rows |
| DIRECT train user texts matching any test question (exact, normalized, 40-character substring) | **0**, against 3952 test rows (2299 distinct raw strings; 2295 after whitespace and case normalization) |
| DIRECT ids in the held-out split | **0** by content. No uuid join is possible, because the train rows carry no uuid. |

**Positive control.** The same content matcher, run over all 15,000 raw train rows, finds the weather prompt in exactly one row (`og_15ed4821407a8b5b`). That row is labelled UNLABELLED with reason `EVALUATION_ONLY_OR_HELDOUT`, so the labeller already excluded it. The matcher works. The overlap the decision section names is a row the labeller excluded, and it is not one of the 177 DIRECT ids.

**Reconciliation.** The two counts agree with each other and with sections 1, 7 and 8. Both say zero. The 2299 and 2295 figures differ only by normalization, not by content. Nothing disagrees, so no stop condition applies.

**The over-refusal claim, narrowed.** The When2Call test has no item whose correct action is a direct answer. Its three gold classes are "call a tool", "ask a follow-up question" and "say the question cannot be answered with these tools". So the test can register a model that refuses when a tool call was correct, or that answers directly when it should not have. It cannot register a model that refuses a question it should answer without a tool, because no such item exists in the gold. Any statement that the tool-use test "catches over-refusal" must say which kind.

**Literature check (pwc, abstract only, not cross-checked).** `pwc search` returns When2Call (arXiv 2504.18851, Ross, Mahabaleshwarkar and Suhara, 2025). Its abstract describes three decisions: when to generate a tool call, when to ask a follow-up, and when to admit the question cannot be answered with the tools. It names no direct-answer gold class. That matches the raw test files. The dataset card and the repository were not read, so whether the released data ever had direct-answer items is not established here.

**Not established.**
- The content match does not catch a paraphrase of a held-out question, as section 8 already said.
- The `pwc` result rests on the abstract alone.
- The two uuids outside the partition were not joined to any train record by uuid.

No frozen file, no `results/registry.jsonl`, no `reports/ERRATA.md`, no manifest and no label file was edited. This section is the only change.

## 21. canonical_hash mismatch for the DIRECT ids, and the direct-answer test source (dated 2026-10-22, read-only investigation, owner decisions)

**Finding (read-only).** For the 177 DIRECT ids in `behaviour-labels-v1/when2call.labels.jsonl`:
- The labels' `canonical_hash` equals the normalization-v3 corpus hash for 177 of 177 ids.
- The SFT shard hash (`normalization-v1/when2call-sft`) equals the v3 hash for only 29 of 177 ids. For 148 ids the two differ. Each id has one shard hash, so the shards are not internally inconsistent.
- The cause, checked on one id (`og_f9fdd50763d21b16`): the v1 shard keeps Python-style tool types (`"str"`, `"int"`, `"int, optional"`). The v3 corpus translates them to JSON Schema types (`"string"`, `"integer"`). The content is the same, so the hash changes. The labels therefore agree with the v3 pipeline, and the shard agrees with v1. Neither hash is corrupt.
- The release rule in section 18 rejected Python-style forms. So the training data follows the v1 types, and the labels follow v3. The labels describe 8,457 When2Call rows that the training release does not contain.

**Owner decision.** Keep the labels on v3 hashes. Record the difference here. Nothing is rebuilt, and no label or shard file is changed.

**Direct-answer test set, decisions.**
- Source: OR-Bench benign subset (arXiv 2405.20947, Cui et al., 2024). Its over-refusal prompts are benign, so the correct action is to answer. Its labels were generated automatically and have not been checked by a human. It is not about tool use, so it tests direct answering only.
- Size: 300 items.
- Precision floor for each label before weight can be given: 90%, on a fresh, untouched, hand-labelled sample.

**Not established.** The OR-Bench licence and whether its prompts overlap the When2Call or training data have not been checked. No item has been sampled or read. This section is the only change; the log is not committed.

## 22. Naming decision, and the retest sample size is not yet set (dated 2026-10-22, owner decisions)

**Naming (owner decision).** One scheme for classifier and label versions: a name says what the module does, and the version lives in its manifest and in `src/opengrad/data/versions.py`. Pinned names stay, because renaming a frozen file breaks its freeze: `decision_classifier_v2.py`, and the v3 wrapper file for now. New names follow the rule. No file is renamed in this section, and no code changes, because no new name is pending.

**Retest sample size (open).** A 90% precision floor is checked on items the classifier puts under each label, not on the whole sample. That makes rare labels the constraint. Using the labels-v2 totals (181,433 rows):

| label | rows | share | fresh random pool needed for 100 items of that label |
|---|---|---|---|
| CALL | 12 | 0.007% | about 1,500,000 |
| DECLINE_WITH_CONTENT | 1,222 | 0.674% | about 14,800 |
| ABSTAIN | 7,791 | 4.29% | about 2,300 |
| CLARIFY | 22,014 | 12.1% | about 820 |
| UNSUPPORTED | 22,419 | 12.4% | about 810 |
| DIRECT | 32,968 | 18.2% | about 550 |

A random sample of 300 would contain about 2 items the classifier calls DECLINE_WITH_CONTENT, so a pass or fail on that label would mostly reflect chance. The 300 figure in section 21 was for the direct-answer test set, not for the retest.

Pass rule and interval width (one-sided 95% Wilson lower bound, with at least 90% required):
- 50 items per label: 49 of 50 correct needed. Width of the 95% interval at 90%: about +/- 8 pp.
- 100 items per label: 95 of 100 correct needed. About +/- 6 pp.
- 150 items per label: 142 of 150 correct needed. About +/- 5 pp.
- 200 items per label: 187 of 200 correct needed. About +/- 4 pp.

**Options under consideration (not decided).** Sampling stratified by classifier label, with a fixed number per label, so that each label's precision is estimated from its own items. Requires a fresh pool with enough items under each label. CALL has 12 items in the whole labelled set, so a fresh pool can't supply 100 of them without a targeted source.

**Not decided.** The sample design, the per-label count, and how CALL is handled.

This section changes no label, manifest, freeze or registry file. The log is not committed.

## 23. Retest sampling decisions (dated 2026-10-22, owner decisions)

- **Sample design (owner decision).** 100 fresh items per label, stratified by the label the classifier assigns. Each label is checked on the items the classifier put under it. A label passes its 90% precision floor only if at least 95 of its 100 items are correct (one-sided 95% Wilson lower bound at or above 0.90). Items are drawn from a fresh, untouched population, never from the frozen 181,433-row labelled set.
- **CALL (owner decision).** CALL has 12 items in the labelled set, so it cannot be checked from the existing data. The owner chose to build a targeted CALL pool, meaning a source that contains textual tool calls, from which fresh items can be sampled. That collection is not yet planned or approved, and no data has been collected.
- **Direct-answer set.** Still pending. The source was chosen as the OR-Bench benign subset, with 300 items (section 21). Its licence has not been checked, and nothing has been sampled.

**Consequence.** The labels-v2 retest needs a fresh pool with at least 100 items under each label. The DECLINE_WITH_CONTENT label alone needs about 14,800 random rows to reach 100 items, so the pool must be built deliberately, not sampled at random. No weight is given until that retest passes.

This section changes no label, manifest, freeze or registry file. The log is not committed.

## 24. Source checks for the direct-answer set and the CALL pool (dated 2026-10-22, read-only, no data collected)

**Threshold conflict (flagged, not resolved).** Doc 22 §6, the preregistered acceptance criteria, sets `CALL` precision at ≥ 0.95, `DIRECT` precision at ≥ 0.80, `DIRECT` recall at ≥ 0.80, and `UNSUPPORTED` recall at ≥ 0.75. The owner's answer in section 21 chose a 90% per-label precision floor for the retest. That floor is looser than the CALL threshold and does not match the DIRECT thresholds. The retest rule needs the owner's decision before any sample is drawn.

**Direct-answer set: OR-Bench (arXiv 2405.20947).**
- Dataset card (Hugging Face `bench-llms/or-bench`): licence `cc-by-4.0`. Subsets: or-bench-80k (80.4k rows), or-bench-hard-1k (1.32k rows), or-bench-toxic (655 rows).
- The card and repository describe an automated pipeline. The toxic prompts were moderated by a model (`gpt4`, per the README). No human verification of labels is mentioned.
- The 80k subset is treated as answerable because it was generated as benign over-refusal prompts. That is an assumption of the generation process, not a human judgement of each item.
- Repository licence: Apache-2.0 (sidebar only; the LICENSE text was not read).

**CALL pool: candidates checked.**
- Already in the labelled corpus, so not fresh: APIGen (the xLAM source), ToolACE, Glaive (adapters in `normalization-v3`).
- ACEBench (arXiv 2501.12851): the paper page shows a CC BY-NC-ND 4.0 licence badge. That licence covers the paper, not a dataset. No data-release URL was found, and no human verification is stated. Not usable as a pool until the data release is confirmed.
- Not checked: FunctionChat-Bench (Korean, language mismatch), mind_call (narrow domain), 2509.26553 (contamination-free generator; release unknown).

**Not established.** Textual-call sources with a clear licence and a documented release have not been found. No data has been downloaded or sampled.

This section changes no label, manifest, freeze or registry file. The log is not committed.

## 25. Owner decisions on the retest threshold and the direct-answer set; sample drawn (dated 2026-10-22)

**Retest threshold (owner decision).** The retest uses the preregistered criteria in doc 22 §6, not a flat 90% per label. Those are: `CALL` precision ≥ 0.95; `DIRECT` precision and recall ≥ 0.80; `UNSUPPORTED` recall ≥ 0.75; `CLARIFY` F1 ≥ 0.70; macro F1 ≥ 0.75; abstention rate ≤ 0.15; and per-mode challenge-subset recall ≥ 0.60. The 90% floor from section 21 is superseded for the retest. The superseding change is recorded here, so that no threshold is chosen after seeing retest data. The sample-size analysis in section 22 still applies: each mode needs about 100 items under the classifier's own label, and the 95-of-100 rule was derived for a 90% floor, so it does not carry over unchanged to the 0.95 CALL threshold. That needs its own calculation before sampling.

**Direct-answer set: OR-Bench 80k benign subset.**
- Source: `bench-llms/or-bench`, file `or-bench-80k.csv`, licence `cc-by-4.0`. Downloaded 80,359 rows. The SHA-256 of the downloaded file is `22e95602…`.
- Drawn: 300 items, `random.Random(20261022).sample`, saved to `data/raw/or-bench/sample-300-seed20261022.csv` (SHA-256 prefix `427cf209…`). Category counts in the sample: deception 35, harassment 11, harmful 63, hate 18, illegal 28, privacy 37, self-harm 11, sexual 12, unethical 55, violence 30. Prompt text was not printed.
- Caveat for review: in some categories (self-harm in particular) a direct answer may not be the right action. The review must judge the correct action item by item, not assume a direct answer.
- Hand-check: 50 items chosen with `random.Random(7)`, written to `data/raw/or-bench/review-50-blind.csv` with no category column. The category key is in `review-50-key.csv`, which must not be shown to the reviewer. No reviewer has been assigned, and no review has been done.

**CALL pool: ACEBench (arXiv 2501.12851).**
- Repository `chenchen0103/ACEBench`, MIT licence. The data are in `data_all/` (English and Chinese). The README states no separate data licence and no human verification of the data. A 2025-10-29 correction changed the possible answers in three datasets.
- Limit: the items are user queries with gold call strings. The classifier's `CALL` label needs assistant responses that contain a textual call. Using the gold strings as responses would make the label true by construction, so ACEBench would need model responses generated for each query before it can supply a fresh `CALL` pool.

**Not decided.** The CALL pool approach, the reviewer for the 50-item check, and the sample-size calculation for the 0.95 CALL threshold.

This section changes no label, manifest, freeze or registry file. The log is not committed.

## 26. Sample size for the doc 22 §6 precision thresholds (dated 2026-10-22, calculation only)

Thresholds are the doc 22 §6 values in section 25. Intervals are one-sided 95% Wilson lower bounds, and a label passes when that bound is at or above its threshold.

| threshold | n | correct needed to pass | observed rate |
|---|---|---|---|
| CALL precision ≥ 0.95 | 100 | 99 | 99.0% |
| CALL precision ≥ 0.95 | 200 | 196 | 98.0% |
| CALL precision ≥ 0.95 | 300 | 292 | 97.3% |
| DIRECT precision ≥ 0.80 | 100 | 87 | 87.0% |
| DIRECT precision ≥ 0.80 | 200 | 170 | 85.0% |

Power, meaning the chance of passing when the true precision is above the threshold:
- CALL, true precision 0.98: about 0.40 at n = 100; 0.63 at n = 200; 0.85 at n = 300.
- DIRECT, true precision 0.85: about 0.35 at n = 100; 0.55 at n = 200.

**Consequence.** At n = 100, a CALL label that is truly 98% precise fails most of the time. The 100-per-label design from section 23 does not test the doc 22 thresholds reliably. The CALL sample needs about 300 items under the classifier's own CALL label, and each label needs its own count. Section 23's 95-of-100 rule should be replaced by these counts once the owner confirms the design.

This section changes no label, manifest, freeze or registry file. The log is not committed.

## 27. Audit of collected data, and a correction to sections 20, 21 and 25 (dated 2026-10-22, read-only)

**What was checked.**
- **When2Call test, all six copies.** Two raw jsonl files, two raw parquet files, and two parquet files in `.cache/normalization/raw`. Each copy has 3,652 MCQ rows and 300 llm_judge rows, and the same `correct_answer` values: `cannot_answer`, `tool_call`, `request_for_info`. None is `direct`. Section 20 stands.
- **OR-Bench (new this session).** `data/raw/or-bench/or-bench-80k.csv`, 80,359 rows, licence cc-by-4.0, SHA-256 `22e95602…`. The 300-item sample and the 50-item blind review sheet are written. The category key is in a separate file. No review has been done.
- **`.cache/answer-strata/`.** BFCL irrelevance (240 rows), BFCL live irrelevance (884), BFCL simple python (400), and an NQ-open validation parquet (3,610 rows). These are the sources of the answer-strata population below.
- **`.annotation/` (20 SQLite annotation databases).** Each passes `quick_check`. Their `-shm` files have modification times of 22 October 12:54 or 13:00, before this session's work. No data was written. The main file of each database is unchanged.
- **`reports/study-002/answer-strata-v1/`.** Population of 1,767 items: NQ-open validation 850 (pool K, constructed), BFCL irrelevance 226 and BFCL live irrelevance 691 (pool N, natural). Each item was labelled by two models, DeepSeek v4.1 flash and Gemini 3.8 flash high, and 1,565 items are unanimous. The reference file has 1,055 `ANSWER`, 80 `CALL`, 148 `CLARIFY`, 197 `UNSUPPORTED`, 85 `UNKNOWN`, and 202 with no consensus. The reference status is `MODEL_REFERENCE_PROVISIONAL`, and the reference statement says these are model judgments, not human gold.
- **Overlap.** No answer-strata user message equals a When2Call test question.

**Correction.** Sections 20, 21 and 25 said the study had no ANSWER-labelled set outside When2Call that could test over-refusal, and proposed building one from OR-Bench. That was incomplete. The `answer-strata-v1` set already provides 1,055 consensus `ANSWER` items from two models, documented under pre-registration amendments 41 and 42. Section 20's statement about the When2Call test is still correct. The claim that the gate cannot test over-refusal for lack of ANSWER items must be restated, because this set exists.

**Caveats on the answer-strata set.**
- Its labels are model judgments. No human has checked them, and no adjudication has been recorded.
- Its population carries `user_message` and `tools` only. It has no assistant responses. Its `CALL` labels are therefore expected-action labels, not classifier output on responses. It cannot serve as the CALL pool for the classifier's textual-call label without response generation, the same limit as ACEBench in section 25.

**Not established.** Whether the study's gate has been run on this set. The `.cache/normalization/raw` files other than the When2Call copies (button, glaive, looptool, toolace, xlam) were inventoried by size and date only.

This section changes no label, manifest, freeze, registry or annotation file. The log is not committed.

## 28. Owner decisions after the audit (dated 2026-10-22)

- **Direct-answer set.** The owner chose to use the existing `answer-strata-v1` consensus `ANSWER` items (reference, 1,055 items, model judgments, provisional) and to drop the OR-Bench plan. The OR-Bench files written this session stay on disk, unused: `data/raw/or-bench/`. The results of the earlier OR-Bench steps (section 25) are superseded for this purpose.
- **CALL responses.** The owner chose to generate assistant responses for the CALL pool with a named model. No model has been named yet, so nothing has been generated.

This section changes no label, manifest, freeze or registry file. The log is not committed.

## 29. CALL response model named; generation blocked (dated 2026-10-22)

- **Model (owner decision).** `claude-haiku-5-5` (Haiku 5.5) generates the assistant responses for the CALL pool. It is not one of the three models that labelled `answer-strata-v1` (Gemini 3.8 flash high, DeepSeek v4.1 flash, GPT 5.6 sol), so it is an independent choice for response generation.
- **Blocked.** This environment has no `ANTHROPIC_API_KEY` and no `anthropic` Python package. Nothing has been generated.
- **Pending.** A generation budget and an approval to run it, and API access in the environment. Nothing is sent to any API until both are given.

This section changes no label, manifest, freeze or registry file. The log is not committed.

## 30. Agreed and disputed split of answer-strata-v1, and the exclusion rule (dated 2026-10-22, read-only)

**Exclusion rule.** The reference label is the label that both annotators give (amendment 42 §2, which amends 41 §9). When the two differ, the item is `NO_CONSENSUS`, is excluded, and is never relabelled. The reference manifest records this as `metric_eligible = False` for those items. Items are not dropped from the population or the annotation database.

**Split.** 1,767 items: 1,565 agreed (`metric_eligible = True`) and 202 disputed (`metric_eligible = False`). The agreed set equals the eligible set exactly.

**Agreed (1,565), reference labels:** `ANSWER` 1,055; `CLARIFY` 148; `UNSUPPORTED` 197; `CALL` 80; `UNKNOWN` 85. By pool: NQ-open 771 `ANSWER`, 19 `UNSUPPORTED`, 2 `UNKNOWN`; BFCL irrelevance 143 `ANSWER`, 4 `CLARIFY`, 30 `UNSUPPORTED`, 15 `CALL`; BFCL live irrelevance 141 `ANSWER`, 144 `CLARIFY`, 148 `UNSUPPORTED`, 65 `CALL`, 83 `UNKNOWN`.

**Disputed (202), individual annotator labels.**
- DeepSeek: `CLARIFY` 58, `UNSUPPORTED` 67, `UNKNOWN` 42, `CALL` 31, `ANSWER` 4.
- Gemini: `ANSWER` 132, `UNSUPPORTED` 35, `CLARIFY` 13, `UNKNOWN` 14, `CALL` 8.
- By pool: BFCL live irrelevance 110; NQ-open 58; BFCL irrelevance 34.
- The most common disagreement is DeepSeek `UNSUPPORTED` with Gemini `ANSWER` (63 items), followed by DeepSeek `UNKNOWN` with Gemini `ANSWER` (31) and DeepSeek `CLARIFY` with Gemini `ANSWER` (29).

**Selection effect (not yet resolved).** On the agreed set, `ANSWER` is 67.4% of items. DeepSeek labels `ANSWER` for only 2.0% of the disputed items, and Gemini for 65.3%. The disputes are therefore mostly cases where the stricter annotator (DeepSeek) withheld `ANSWER`. Because agreement requires both annotators, the agreed `ANSWER` set is conditioned on DeepSeek's judgement. Any answer rate computed on the agreed set is a rate among items the stricter annotator also called `ANSWER`, not a rate over the population.

**No model responses.** Neither the population, the reference, the annotation databases nor `reports/study-002/pconf-v1/pconf-v1.answer-records.jsonl` holds any model response. That file has 1,055 gold records with empty `candidates`. An over-refusal or answer rate for a model under test cannot be computed from these files. The only figures available are the annotators' label distributions above.

**Reporting restrictions (41 §9, amended by 42).** A pooled figure may be printed only beside the two strata reported separately, never instead of them. The consensus is model-model agreement, `MODEL_REFERENCE` and provisional, never human gold. No figure from this section goes into a summary until the owner has confirmed it.

This section changes no label, manifest, freeze, registry or annotation file. The log is not committed.

**Owner decision on summary reporting (dated 2026-10-22).** Summaries may report counts and ANSWER rates from section 30 for the agreed set (1,565) and the disputed set (202), each rate carrying the selection caveat. These are annotator label distributions, not model over-refusal rates, and they must be labelled that way.

## 31. Status update (dated 2026-11-06, owner instruction "commit and fix everything so it's up to date")

This section records what changed after the entries above were written. The earlier entries are not edited, because they describe the state on 2026-10-22.

- **Section 13, "Open. Commit the restatement set (not done)."** Closed. Commit `ac5b93e` ("docs(results): M1-v2 promotion withdrawn; registries, freeze pins and guards restated") committed the restatement set, including `reports/releases/hf-publication-2026-10-22-card-license-correction.json`, which section 13 listed as untracked.
- **"The log is not committed" in sections 21 to 30.** Those sentences described the working tree on the day each section was written. Sections 21 to 30 are committed by the commit that adds this section.
- **Section 24, threshold conflict (flagged, not resolved).** Resolved by the owner's decision in section 25: the retest uses the preregistered thresholds in doc 22 §6.
- **README.** On `master`, commit `c8de767` relabels the M1-v2 row as a withdrawn promotion (status REJECTED, withdrawn 2026-10-22). It also corrects the status paragraph. The registries it describes already record REJECTED: `results/registry.jsonl` and `docs/EXPERIMENT_STATUS.md` for M1-v2, and `runs/checkpoint_registry.json` for checkpoint 30.
- **Still open:** the threshold sample-size question (section 26) and the reviewer and approval for the CALL pool (section 29). Nothing in this section decides either.

This section changes no label, manifest, freeze, registry or annotation file.
