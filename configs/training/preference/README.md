# Preference-optimization configuration namespace

An empty namespace, kept so the path exists. Direct preference optimization is implemented and has
been executed twice: M1-v1 (DPO applied directly to the base) and M1-v2 (DPO from the M0 parent; it passed v4
on 2026-09-11 and was promoted, and the promotion was withdrawn on 2026-10-22). Experiment configs are in [`configs/experiments/`](../../experiments/); the DPO
training config is [`configs/training/dpo/`](../dpo/).
