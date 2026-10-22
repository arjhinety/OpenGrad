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
