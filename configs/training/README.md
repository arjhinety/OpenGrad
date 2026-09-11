# Training configuration namespaces

| Namespace | Status |
|---|---|
| [sft/](sft/) | **Implemented and executed** — the M0 runs. See [`docs/SFT_TRAINING.md`](../../docs/SFT_TRAINING.md) |
| [preference/](preference/) | **Implemented and executed** — M1 DPO. Preference configs live in [`configs/experiments/`](../experiments/) |
| [distillation/](distillation/) | Reserved. On-policy distillation was deliberately not attempted and is out of scope by decision, not by omission |
| [rl/](rl/) | Reserved. A future extension boundary; nothing is implemented |

Executable experiment configs live in [`configs/experiments/`](../experiments/) rather than in these
namespaces, which only reserve the interface. The definitive M0 is
[`configs/experiments/m0_sft_canonical_v2_final.yaml`](../experiments/m0_sft_canonical_v2_final.yaml).
