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
The reference policy $\pi_{\text{ref}}$ **must** correspond to the frozen SFT policy (e.g. `checkpoints/qwen35_2b_m0_sft`), **never** the original untouched Qwen base model. OpenGrad enforces this by:
1. Recording `reference_checkpoint` explicitly in the experiment configuration.
2. Freezing the SFT policy weights or merging the SFT adapter before initializing the DPO model.
3. Rejecting runs where the reference model hash diverges from the SFT checkpoint.

---

## 3. Two Preference Sources

OpenGrad constructs DPO mixtures from two distinct sources:

1. **Curated Historical Preferences (`When2Call`)**:
   - High-quality human-annotated preference pairs targeting call/no-call decision boundaries.
   - Pinned revision: `0582f7749df63a96fdc3070932e83e72396ace53`.
2. **Synthetic Residual Preferences**:
   - Generated from the active student checkpoint ($N=4$ candidates).
   - Adjudicated via deterministic schema/tool validators first, with OpenAI adjudication used strictly for ambiguous semantic comparisons.

---

## 4. Diagnostics & Telemetry

During DPO execution, `DPOTrainerBackend` tracks:
- `reward_margin`: $\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$
- `chosen_reward` & `rejected_reward`
- `preference_accuracy`: Fraction of pairs where implicit reward of chosen exceeds rejected.
- `response_length_diff`: Monitoring against length exploitation.
