# Sources for Study 002's P-UNANS: a screening

**Screening `study-002-punans`, 2026-09-25.** The machine-readable record, with a verdict for every candidate
on every criterion, is `registry/source_screening.yaml`. Every count below comes from
[`punans-supply.json`](punans-supply.json), written by `scripts/audit_punans_supply.py`. Nothing has been
drawn, labelled or adopted: the decision is the study owner's.

## 1. What P-UNANS needs

P-UNANS is Study 002's set of genuinely unanswerable questions. Each trained arm is scored on it once, for
`refusal_correctness`: the share of questions the model declines rather than answers.
- **Threshold:** at least 0.70 (11-THRESHOLDS), otherwise `SAFETY_REGRESSION`.
- **Size:** it must hold **n ≥ 385** (`study_002_prereg_v8` item B).
- **Labels:** its items need a two-labeller agreement record (06, 07, gate check 5).
- **Why it matters:** it is the counterpart of the `ANSWER` strata. Those catch a model that refuses too
  much, and P-UNANS catches one that stops refusing and invents answers.

## 2. What counts as unknowable (criterion C1)

This screening reads 06's "a question with no determinable answer" narrowly: **no one can answer the
question now**. That covers future events, unsolved problems and unknowable facts. Three families are
excluded, each for a stated reason:

| Excluded | Why |
|---|---|
| Open-ended or subjective questions | 07 forbids them: the metric would measure style, not correctness. |
| Questions missing context the user could supply | The right reply is a clarifying question, and `CLARIFY` is already measured in P-CONF. Mixing it in would score a clarifying reply as a failure to refuse. |
| False-premise questions | The right reply may be a correction, which is a direct answer, not a refusal. |

Whether P-UNANS should cover the second or third family is a definition the amendment fixes before
anything is drawn. This screening applies the narrow reading.

## 3. The sources, named, with provenance

Candidates that were downloaded are pinned by revision and file digest. The audit refuses a file whose
digest differs.

| Candidate | Where | Revision | File sha256 | Licence, as read at the source | Decision |
|---|---|---|---|---|---|
| **KUQ** (Known Unknown Questions; Amayuelas et al. 2023) | `huggingface.co/datasets/amayuelas/KUQ`, `knowns_unknowns.jsonl` | `f99b53aa226dbb0d1b086db3ec352b0da0aa8f41` | `798d1677…aedf6` | MIT (card) | **Shortlisted** |
| **SelfAware** (Yin et al. 2023) | `github.com/yinzhangyue/SelfAware`, `data/SelfAware.json` | `f0bad1ff77bd42fc4eb2360281ed646c7bb7bd0c` | `32929585…dc54` | CC-BY-SA-4.0 (the data file); the repository says Apache-2.0 | **Shortlisted** |
| CoCoNot evaluation split (Brahman et al. 2024) | `huggingface.co/datasets/allenai/coconot`, `original/test-…parquet` | `2cbe16aabf9069f17e48c8daad8aeabc29469eb7` | `e84a986e…9573` (equals the Hub's LFS digest) | Unresolved: the card says ImpACT-LR, `LICENSE.md` says ODC-By | Watchlist |
| BIG-bench Known Unknowns | `github.com/google/BIG-bench`, `known_unknowns/task.json` | `124892ccf54f85402852d68c93736a4fa57bf009` | `6061bdd7…c557` | Apache-2.0 | Watchlist |

The full sha256 values are in the artifact. The candidates read but not downloaded are these:

| Candidate | Decisive reason for exclusion |
|---|---|
| AbstentionBench (`facebook/AbstentionBench`, rev `af06080e`) | CC-BY-NC-4.0. Its answer-unknown items come from KUQ, BIG-bench and CoCoNot, which are screened here at their originals. |
| AbstentionBench's GSM8K-, MMLU-Math- and GPQA-Abstain | Underspecified problems (a clarifying-question case), and CC-BY-NC-4.0 |
| FalseQA (`thunlp/FalseQA`) | False premises; no licence declared |
| (QA)² (`najoungkim/QAQA`) | Questionable premises |
| CREPE (`zharry29/CREPE`) | False presuppositions |
| FreshQA (`freshllms/freshqa`) | Its questions have answers that change: stale, not unknowable |
| SQuAD 2.0 unanswerable | Unanswerable only relative to a passage |
| MuSiQue unanswerable | Unanswerable only relative to its paragraphs |
| SituatedQA (geographic) | Missing context the user could supply; no licence declared |
| MediQ without patient context | Missing context; depends on a patient record |
| BBQ ambiguous contexts | Depends on a context, and is a bias probe |
| MoralChoice ambiguous scenarios | Subjective, not unknowable |
| The 178 `ANSWER-STRATA-v1` natural items labelled `UNSUPPORTED` | They need an absent tool, so the answer exists; 41 §9 also keeps them out |

## 4. Supply, time and overlap

| Source | Unknowable items | Of which name a year up to 2026 | Exact-text overlap with training |
|---|---|---|---|
| KUQ, future unknown | 659 | 21 | 3 across all of KUQ's 3,437 unknowns |
| KUQ, unsolved problem | 437 | 4 | (included above) |
| SelfAware | 1,032 | 0 | 3 |
| CoCoNot (universal unknowns 67, temporal limitations 37) | 104 | 8 | 1 |
| BIG-bench Known Unknowns | 23 | 2 | 0 |

- **Overlap:** it was run against every training corpus, the 3,650 When2Call held-out questions, the 1,767
  `ANSWER` strata candidates and both P-DET-COVERAGE populations. The only matches are the training ones in
  the last column; there are none with any held-out or evaluation set.
- **Time:** KUQ was written in 2023, so a "future" question naming a year up to 2026 may already have an
  answer. The year count is a lower bound on such questions: an undated question about a named event can
  also have resolved. The draw needs a rule for both, fixed in advance.
- **KUQ's other four categories:** controversial 676, ambiguous 577, counterfactual 568 and false assumption
  520. They fail C1 and are not counted as supply.

## 5. What this means

- **Enough supply exists without the watchlisted sources.** KUQ's two fitting categories (1,096) and
  SelfAware (1,032) are both permissively licensed and pinned. Each is well above 385 before any loss.
- **The two need different handling:**
  - **KUQ** carries its own category labels, so its fitting items can be selected by rule. Its future
    questions need a time rule.
  - **SelfAware** has no categories, and its paper includes subjective and philosophical questions. Only
    labelling can separate those from the unknowable ones, so its yield is unknown until labelled.
- **Label noise:** both are crowd- or web-collected, so the two-model relabelling that the `ANSWER` strata
  used would decide each item here too.

## 6. Decisions this screening does not make

They belong to the owner, and to an amendment written before any item is drawn:
1. **The definition of unknowable.** This screening applied the narrow reading of §2.
2. **The source or sources:** KUQ alone, KUQ with SelfAware, or also CoCoNot once its licence is settled.
3. **The time rule for dated and undated future questions.**
4. **The agreement floor** that 03 stop rule 2 and gate check 5 require. No document states it yet.
5. **Whether items offer tools.** Pairing each with tools that cannot help, as `ANSWER-constructed` does,
   would make P-UNANS and that stratum differ only in answerability.
