# Sources for Study 002's second P-UNANS attempt: a screening

**Screening `study-002-punans-v2`, 2026-09-25.** The machine-readable record, with a verdict for every candidate
on every criterion, is `registry/source_screening.yaml`. Every count below comes from
[`punans-v2-supply.json`](punans-v2-supply.json), written by `scripts/audit_punans_v2_supply.py`. Nothing has
been drawn or labelled. The design that uses these sources is the draft amendment
[44](../../../docs/research/study-002/44-PUNANS-V2-AMENDMENT-DRAFT.md), which binds only once the owner adopts it.

## 1. Why a second screening

- **The first attempt stopped.** P-UNANS-v1's two labelling models agreed on too few items (0.770 against a
  0.80 floor), and SelfAware turned out to be mostly not unknowable
  ([negative result](../../study-002/punans-v1/NEGATIVE-RESULT.md)).
- **What the write-up asked of a retry:** a source screening for unknowable questions beyond KUQ. This is that
  screening.
- **What it must also respect:** a question the first attempt drew already has labels, so it is not fresh.
  The audit removes all 1,063 of them by normalised text.

## 2. The sources, named, with provenance

| Candidate | Where | Revision | File sha256 | Licence | Decision |
|---|---|---|---|---|---|
| **KUQ, `unknowns_all.jsonl`** (Amayuelas et al. 2023) | `huggingface.co/datasets/amayuelas/KUQ` | `f99b53aa226dbb0d1b086db3ec352b0da0aa8f41` (the first attempt's revision) | `8469ab01…0855` | MIT | **Shortlisted** |
| **KUQP, future questions** (Deng et al. 2024) | `github.com/zhaoy777/kuqp-dataset`, `KUQP Dataset/future_questions.json` | `596472f31f73acfdcb95741c277413fd500f8b35` | `25ee426e…40d1` | MIT | **Shortlisted** |
| **BIG-bench Known Unknowns** | `github.com/google/BIG-bench`, `known_unknowns/task.json` | `124892ccf54f85402852d68c93736a4fa57bf009` | `6061bdd7…c557` | Apache-2.0 | **Shortlisted** (on the first screening's watchlist until the first attempt fell short) |
| CoCoNot, universal unknowns and temporal limitations | `huggingface.co/datasets/allenai/coconot` | `2cbe16aa…` | as the first screening | Unresolved | Watchlist, unchanged |

The full sha256 values are in the artifact. The audit refuses a file whose digest differs.

**Read and excluded:**

| Candidate | Decisive reason |
|---|---|
| SelfAware | The first attempt measured it: mostly not unknowable. |
| BeHonest Unknowns (`GAIR/BeHonest`, CC-BY-SA-4.0) | Its paper builds the split from SelfAware and UnknownBench, whose questions concern non-existent concepts or false premises. |
| KUQP ambiguous, incomplete and incorrect questions | Clarifying-question cases and false premises. |
| HoneSet (HonestLLM) | Limits of the assistant, not questions no one can answer; no licence. |
| Idk dataset (Say-I-Dont-Know) | Unknown to one model, not to anyone; no licence. |
| Temporal QA abstention (When Silence Is Golden) | Answerable once the date is fixed; no data released. |

The first screening's other exclusions (AbstentionBench, FalseQA, (QA)², CREPE, FreshQA, SQuAD 2.0, MuSiQue,
SituatedQA, MediQ, BBQ, MoralChoice) still hold and are not repeated.

## 3. Fresh supply

| Source | Distinct questions | Drawn by the first attempt | Name a year up to 2026 | Collide with training | **Fresh** |
|---|---|---|---|---|---|
| KUQ `unknowns_all.jsonl`, future and unsolved | 1,784 | 358 | 19 | 2 | **1,405** |
| KUQP future | 40 | 0 | 0 | 0 | **40** |
| BIG-bench Known Unknowns | 23 | 0 | 2 | 0 | **21** |
| **Total** | | | | | **1,466** |

- **Who wrote KUQ's fresh questions:** GPT 904, crowdworkers 433, the web 68. The first attempt's file held no
  GPT-written questions, so their agreement rate is unknown.
- **By KUQ category:** future unknown 792, unsolved problem 613.
- **Overlap** was checked against every training corpus, the When2Call held-out questions, the ANSWER strata
  candidates and both P-DET-COVERAGE populations. There are no matches outside the training column.

## 4. What this means

- **Beyond KUQ, the public supply is small:** 61 fresh questions (KUQP 40, BIG-bench 21). The screening that the
  write-up asked for was done, and it found that a second attempt still depends mostly on KUQ.
- **So the source is not what changes.** The write-up expected the first attempt's instrument (six labels) to
  fail again on a redraw. A second attempt therefore has to change what the labellers are asked, before any
  label exists. That is the subject of 44, not of this screening.
