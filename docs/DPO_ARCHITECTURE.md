# Direct Preference Optimization (DPO) Architecture

**Building in Public.** OpenGrad treats DPO as a precision instrument to correct measured residual tool policy errors, not as a general-purpose replacement for SFT.

---

## 1. DPO Experimental Scope

DPO in OpenGrad is conditional and hypothesis-driven:
$$\text{Baseline (B0)} \longrightarrow \text{SFT (M0)} \longrightarrow \text{Evaluation} \longrightarrow \text{Residual Diagnosis} \longrightarrow \text{DPO (M1) IF Justified}$$

DPO is invoked only when post-SFT evaluation identifies specific residual decision-boundary errors, such as:
- Calling tools when direct answers were appropriate (over-calling).
- Omitting tools on ambiguous queries instead of requesting clarification.
- Emitting conversational prose before the tool markup envelope.

---

## 2. Frozen Reference Policy Semantics

In DPO:
$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

### Critical Grounding Rule:
The intended rule is that the reference policy $\pi_{\text{ref}}$ corresponds to the frozen SFT policy (e.g. `checkpoints/qwen35_2b_m0_sft`), not the original untouched Qwen base model. **The code does not enforce that, and one executed run violated it.** What is actually implemented:
1. `trainer.reference` must be declared explicitly as `initial_policy` or `explicit_checkpoint` (`src/opengrad/training/dpo_runner.py`); `explicit_checkpoint` requires `reference_checkpoint`.
2. In the live M1 path (`dpo_live.py`), the initial checkpoint's hash must match the config, and an explicit reference must hash-match that initial checkpoint.
3. Nothing rejects a base-model reference. `initial_policy` on a base-model start makes the base the reference.

M1-v1 (`qwen35_2b_m1_dpo_v1`) ran exactly that way: `parent_experiment_id: null`, `reference: initial_policy`, DPO directly on the base model. M1-v2 (`m1_dpo_canonical_v2_final_v2`) follows the rule: `reference: explicit_checkpoint` at M0-final-v2 checkpoint 1800.

---

## 3. Two Preference Sources

OpenGrad constructs DPO mixtures from two distinct sources:

1. **Curated Historical Preferences (`When2Call`)**:
   - High-quality human-annotated preference pairs targeting call/no-call decision boundaries.
   - Pinned revision: `0582f7749df63a96fdc3070932e83e72396ace53`.
2. **Synthetic Residual Preferences**:
   - Generated from the active student checkpoint ($N=4$ candidates).
   - Adjudicated via deterministic schema/tool validators first, with OpenAI adjudication used strictly for ambiguous semantic comparisons.

**What the executed runs used.** Neither run used synthetic residual preferences. M1-v1 used 1,741
When2Call training preference pairs (`when2call_pref_v1`). M1-v2 used 481 pairs
(`m1_calibration_preference_pairs_v1`): 240 curated When2Call pairs plus 241 deterministic
disagreement pairs between the base model and the selected M0 checkpoint, generated locally with no
external API (`reports/data/m1-calibration-preference-pairs-v1.json`).

---

## 4. Model Components (`full-model-components-v1`, from 2026-09-16)

The policy model carries every component its base declares: for Qwen3.5-2B, the vision encoder and the native
MTP layer as well as the language model. An initial checkpoint from before the policy is text-only. Its
missing components are grafted from the pinned base revision, and the lineage's
`model_components.initialized_from` names the source.

MTP trains on the **chosen** completion only. The default `gradient_scope: head_only` keeps the gradient on the
`mtp.*` parameters, so the DPO objective itself is unchanged. `mtp_loss` is logged beside `loss`.

The reference model stays a text-only frozen scorer.

See [`MODEL_COMPONENT_POLICY.md`](MODEL_COMPONENT_POLICY.md).

---

## 5. Diagnostics & Telemetry

During DPO execution, `DPOTrainerBackend` tracks:
- `reward_margin`: $\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$
- `chosen_reward` & `rejected_reward`
- `preference_accuracy`: Fraction of pairs where implicit reward of chosen exceeds rejected.
- `response_length_diff`: Monitoring against length exploitation.
