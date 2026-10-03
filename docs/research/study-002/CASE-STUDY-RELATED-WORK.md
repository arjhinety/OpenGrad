# Related work for a possible case study: model-labelled triage in a one-person lab

**Status: notes, not a paper.** Gathered 2026-10-03 at the owner's request, in case the flag-set triage of Study
002 is written up as an arXiv case study. Every paper below was found and checked with the Papers With Code CLI
(`pwc search`, `pwc paper info`). Only its **abstract** has been read, so each line says no more than the abstract
does. The full records are in [`docs/references/papers.yaml`](../../references/papers.yaml) (category
`model_annotation` or `over_refusal`, status `abstract_verified_full_text_unread`).

## The method the case study would describe

- **The task.** Two non-Claude models (Gemini 3.8 Flash (High), deepseek-v4.1-flash) each label every one of the
  19,051 training replies a classifier flagged as declines. Labels are `NOT_A_DECLINE`, `DECLINE_JUSTIFIED`,
  `DECLINE_UNJUSTIFIED` or `UNKNOWN`. Each model labels blind: it sees no source, no record id and no sign of how
  the items were chosen ([46](46-READINESS-DESIGN-AMENDMENT-DRAFT.md) §5).
- **Decided before any label exists:**
  - agreement and precision floors;
  - floors per source as well as pooled, with sources that can stop the study
    ([47](47-PER-SOURCE-FLAG-PRECISION-DRAFT.md));
  - a 100-record trial, with one allowed revision of the procedure;
  - a 400-reply sample of unflagged replies, to estimate what the classifier missed.
- **Audited.** Every prompt, answer and label is hashed and archived, and the rules that decide the outcome are
  code with tests.
- **Why models:** the study has one person and no annotation budget. No Claude model labels, because Claude built
  the classifier under test.

## What the literature says, by question

**Can model labels replace human ones?** Sometimes, and it has to be shown, not assumed.
- Gilardi et al. (2023, arXiv 2303.15056): ChatGPT's zero-shot accuracy beat crowd-workers on four of five
  annotation tasks, and its intercoder agreement beat crowd-workers' and trained annotators' on all of them.
- Pangakis et al. (2023, arXiv 2306.00176): LLM annotation performance is "highly contingent" on dataset and
  task, so every automated annotation must be validated against human labels, task by task.
- Calderon et al. (2025, arXiv 2501.10970): the Alternative Annotator Test justifies using LLM annotations from
  "a modest subset" of human-annotated examples. LLMs can sometimes replace humans, and closed models did better
  than the open models they examined.

**Does two models agreeing mean they are right?** Not on its own: models' errors are correlated.
- Kim et al. (2025, arXiv 2506.07962): across over 350 LLMs, errors are substantially correlated (on one
  leaderboard, models agree 60% of the time when both err), most strongly among larger, more accurate models,
  even across providers.
- Kuai et al. (2026, arXiv 2604.07650): shared pretraining data, distillation and alignment entangle models'
  behaviour, so in multi-model judging "apparent agreement" can reflect shared error modes.
- Gorbett and Jana (2026, arXiv 2603.25450): disagreement between models is a useful signal that an answer is
  wrong.

**Where do model judges fail?**
- Ye et al. (2024, arXiv 2410.02736): twelve potential biases of LLM judges; significant biases persist in some
  tasks.
- Krumdick et al. (2025, arXiv 2503.05061): a judge grading correctness struggles on the questions it cannot answer
  itself, even when its aggregate agreement with humans looks high. This bears directly on
  `DECLINE_UNJUSTIFIED`, which asks whether a capable assistant could have answered.
- Qian et al. (2026, arXiv 2602.16610): judges differ in reliability across tasks, and their reliability can be
  estimated from comparisons alone.

**How could a few human labels make the conclusions valid?**
- Gligorić et al. (2024, arXiv 2408.15204): Confidence-Driven Inference chooses which human annotations to collect
  and gives valid confidence intervals with over 25% fewer of them.
- Feng et al. (2026, arXiv 2601.20913): a small human-labelled calibration set, used to estimate a judge's true- and
  false-positive rates, keeps a test on a large judge-labelled set statistically valid.
- Li et al. (2023, arXiv 2310.15638): allocating annotation between humans and LLMs by the LLM's uncertainty.

**How should it be reported?** Kunilovskaya et al. (2026, arXiv 2606.02255) find that NLP papers often omit what is
needed to assess annotation validity, including adjudication and agreement values, and give minimum reporting
recommendations.

**The behaviour under study.**
- XSTest (Röttger et al., 2023, arXiv 2308.01263) and OR-Bench (Cui et al., 2024, arXiv 2405.20947) measure models
  refusing safe prompts.
- Study 002 asks where such refusals come from in the supervised training data.

## What a reviewer will ask, and the honest answer today

- **"How do you know the model labels are right?"** We don't, beyond agreement. The triage has no human validation
  set: the study owner annotates nothing, by a standing decision of 2026-09-17. The literature above treats a
  modest human subset as the way to justify model labels. The case study should say so plainly, or report that
  one was added.
- **"Are the two labellers independent?"** They are from different providers and label blind, but errors are known
  to correlate across providers. Agreement is evidence of consistency, not of correctness.
- **Wording.** Results are "replies both models judged unjustified declines", never "unjustified declines".

## Gaps in this search

- **Prediction-Powered Inference.** The original paper is not in the catalog: `pwc paper info 2301.09633`
  returned 404 on 2026-10-03. It is cited here only through Feng et al., who position their method against it.
- **Preregistration in ML.** No ML or NLP preregistration paper turned up in the catalog under the queries tried.
- **Novelty.** No claim of novelty is made. The search was a few queries, not a systematic review.

## Searches run (2026-10-03)

`pwc search` with:
- "large language models as annotators replace human annotation";
- "LLM annotation agreement multiple models consensus labels";
- "LLM-as-a-judge bias reliability agreement with humans";
- "can we trust LLM annotations validity human anchor sample";
- "ensemble of LLM annotators inter-annotator agreement";
- "preregistration machine learning research";
- "correlated errors language models agree when wrong";
- "prediction-powered inference small labeled sample model predictions";
- "pre-registration NLP experiments hypothesis before results";
- "ChatGPT outperforms crowd workers text annotation";
- "over-refusal benchmark exaggerated safety refusals";
- "SFT data quality filtering mislabeled training examples refusal";
- "Prediction-Powered Inference";
- "XSTest exaggerated safety behaviours".
