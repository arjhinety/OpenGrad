# Training configuration namespaces

| Namespace | Status |
|---|---|
| [sft/](sft/) | **Implemented and executed** — the M0 runs. See [`docs/SFT_TRAINING.md`](../../docs/SFT_TRAINING.md) |
| [preference/](preference/) | **Implemented and executed** — M1 DPO. Preference configs live in [`configs/experiments/`](../experiments/) |
| [distillation/](distillation/) | Reserved. On-policy distillation was deliberately not attempted in Study 001, by decision rather than omission. It is in scope for [Study 002](../../docs/research/STUDIES.md) |
| [rl/](rl/) | Reserved. A future extension boundary; nothing is implemented |

**Model components.** From 2026-09-16 every trainer carries and trains all components the base checkpoint
declares: vision and native MTP as well as the language model. `trainer.model_components` and `trainer.mtp`
are optional; absent means the policy default, which is recorded in the run. Existing configs describe runs
made before the policy. Rerunning one now carries vision and MTP unless it declares
`model_components: {vision: exclude, mtp: exclude}`. See
[`docs/MODEL_COMPONENT_POLICY.md`](../../docs/MODEL_COMPONENT_POLICY.md).

Executable experiment configs live in [`configs/experiments/`](../experiments/) rather than in these
namespaces, which only reserve the interface. The definitive M0 is
[`configs/experiments/m0_sft_canonical_v2_final.yaml`](../experiments/m0_sft_canonical_v2_final.yaml).
