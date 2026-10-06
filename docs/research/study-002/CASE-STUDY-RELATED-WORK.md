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
  - floors per source as well as pooled, where a failing source can stop the corpus intervention
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
  leaderboard, models agree 60% of the time when both err); larger, more accurate models have highly
  correlated errors, even across architectures and providers.
- Kuai et al. (2026, arXiv 2604.07650): shared pretraining data, distillation and alignment can entangle
  models' behaviour, so in multi-model judging "apparent agreement" can reflect shared error modes.
- Gorbett and Jana (2026, arXiv 2603.25450): a second model's surprise at reading the first model's answer is a
  label-free signal that the answer is wrong. This is a different disagreement from two labellers'.

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

## Statistics and the study's direction (added 2026-10-06)

Found and checked the same way (`pwc search`, `pwc paper info`, abstracts only), for the formulas in
[EVALUATION-FORMULAS-DRAFT.md](EVALUATION-FORMULAS-DRAFT.md) and for where Study 002 is going.

**How the numbers are reported.**
- Miller (2024, arXiv 2411.00640) gives formulas for analysing language-model evaluations, comparing two models and
  planning an evaluation.
- Bowyer et al. (2025, arXiv 2503.01747): normal-approximation error bars are too small on evaluations of fewer than
  a few hundred items; one reason this study uses the Wilson interval.
- Card et al. (2020, arXiv 2010.06595): underpowered NLP comparisons are common; background for printing every
  row's resolvable margin (which is not a power analysis).
- Dodge et al. (2020, arXiv 2002.06305) and Madaan et al. (2024, arXiv 2406.10229): fine-tuning seeds and benchmarks
  vary substantially; background for training each arm with several seeds. Neither analyses three seeds or a sign
  check: those are the study's own choices (10), and their weakness is open question 6 of the formulas draft.
- Bestgen (2022, arXiv 2205.11134) argues for bootstrap intervals of differences over significance alone, as 10
  does.
- Lee et al. (2025, arXiv 2511.21140): a model judge's accuracy estimate is biased, and correcting it needs a
  calibration set with known answers, which the triage does not have.

**Why supervised data can teach over-refusal (the study's premise).**
- Bianchi et al. (2023, arXiv 2309.07875): adding about 3% safety examples improves safety, but too much
  safety-tuning makes models refuse safe prompts that resemble unsafe ones.
- Kim et al. (2026, arXiv 2609.04714): boilerplate refusal statements in safety-tuning data induce reliance on
  superficial cues and false refusals; training on the rationale alone reduces them.
- Li et al. (2024, arXiv 2402.00530): a small model can select instruction-tuning data for a larger one, improving
  efficiency and performance; an example of intervening on the supervised corpus, as Study 002 does, though its aim
  is efficiency, not behaviour.

**Abstention and unanswerable questions (H6, `P-UNANS`).**
- Wen et al. (2024, arXiv 2407.18418) survey abstention; Kirichenko et al. (2025, arXiv 2506.09038) benchmark it over
  20 datasets, including unknown answers and false premises.
- Zhang et al. (2023, arXiv 2311.09677) teach a model to say it does not know (R-Tuning), the other side of
  over-refusal.
- `P-UNANS` draws on KUQ (Amayuelas et al., 2023, arXiv 2305.13712) and SelfAware (Yin et al., 2023, arXiv
  2305.18153).

**When to call a tool (the decision modes).**
- ToolBeHonest (Zhang et al., 2024, arXiv 2406.20015) covers recognising missing or limited tools, close to
  `UNSUPPORTED`.
- When2Tool (Sun et al., 2026, arXiv 2605.09252) studies when a tool call is needed at all, close to `DIRECT`
  versus `CALL`.

**Not found in the catalog.** The classic statistics papers behind the formulas (Wilson 1927, Cohen 1960, Holm 1979,
Newcombe 1998, Efron 1979) predate arXiv. Four were checked on Crossref instead; Holm was not verified. Nothing in
the catalog turned up on turning a bootstrap interval into a p-value or a multiplicity-adjusted interval.

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

## Searches run (2026-10-06)

`pwc search` with:
- "adding error bars to evals clustered standard errors language model evaluation";
- "statistical significance testing NLP bootstrap hitchhiker";
- "variance random seeds fine-tuning pretrained language models early stopping";
- "accounting for variance in machine learning benchmarks";
- "statistical power NLP experiments little power";
- "confidence intervals binomial proportion evaluation small test sets";
- "bootstrap confidence interval p-value test inversion multiple comparisons adjusted";
- "simultaneous confidence intervals multiple testing machine learning model comparison";
- "safety tuning exaggerated safety refusals fine-tuning data safety-tuned llamas";
- "refusal behavior learned from supervised fine-tuning data instruction tuning";
- "abstention large language models survey know your limits";
- "refusal-aware instruction tuning unanswerable questions R-Tuning";
- "known unknown questions dataset LLM uncertainty" (no results) and "Knowledge of Knowledge known-unknowns
  uncertainty large language models KUQ";
- "do large language models know what they don't know SelfAware";
- "instruction tuning data quality filtering low quality examples improves";
- "label noise in instruction tuning data effect";
- "tool use LLM refusal when tools unavailable hallucinated tool calls benchmark";
- for the classics, which the catalog does not hold: "Holm sequentially rejective multiple test procedure", "Wilson
  score interval probable inference", "Cohen coefficient of agreement nominal scales kappa", "Newcombe interval
  estimation difference between independent proportions", "Efron bootstrap methods another look at the jackknife",
  "hitchhiker's guide to testing statistical significance in natural language processing", "empirical investigation
  of statistical significance in NLP Berg-Kirkpatrick", "multiple comparisons NLP multiple datasets replicability
  analysis".
