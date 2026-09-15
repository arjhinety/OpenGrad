# 13 — Hardware-agnostic execution

The study must be runnable by someone who does not own the hardware the author used. The repository already
has the foundation, and Study 002 adopts it rather than inventing a portability layer.

## The existing foundation

`registry/gpu_preflight.schema.json` (schema_version 1) already models a provider-agnostic preflight:

- `requested.provider` and `compatibility.provider` are both `enum: ["nvidia", "amd", "cpu", "unknown"]`;
- `requested` carries `device_count`, `min_vram_gib` and an optional `backend`;
- `observed` carries `device_count`, `devices[]`, `driver`, and **both** `cuda_version` and `rocm_version`,
  each nullable — so a record can say "this ran on AMD" without a CUDA field pretending to be populated;
- `compatibility.result` is `enum: ["UNKNOWN", "COMPATIBLE", "INCOMPATIBLE", "NOT_TESTED"]` with a required
  `basis` string;
- `status` is `enum: ["NOT_RUN", "READY", "BLOCKED", "INCOMPLETE"]`;
- `limitations: []` is a first-class field, so what a record cannot establish is recorded beside what it can.

`registry/provenance.schema.json` carries `run_id`, `status`, nullable `compute_provider` and a free-form
`hardware` object. `src/opengrad/experiments/preflight.py` and `src/opengrad/readiness.py` implement the
checks.

## Rules

1. **A preflight record with `status: READY` is a precondition of every training run.** `NOT_RUN`,
   `BLOCKED` or `INCOMPLETE` blocks the launch. This is check 1 of the pre-GPU gate
   ([16](16-GPU-READINESS-GATE.md)).
2. **`compatibility.result` may not be `UNKNOWN` or `NOT_TESTED` for a primary arm.** "Unknown" means the
   runtime question is open, and running anyway converts an open question into an unreported confound.
3. **`device_class` is part of the reporting key** `(arm, partition, protocol, device_class)`. Two runs of
   the same arm on two providers are **two rows**, never one averaged row. Averaging across device classes
   would hide exactly the variation the field exists to expose.
4. **No cross-provider bit-exactness claim.** The study's claims are about corpus-level mechanism, which
   does not require identical bits. Where a claim would require it, the claim is out of scope and says so.
5. **Primary runs use one device class.** Cross-provider work is a heterogeneity check
   ([14](14-HETEROGENEITY-POLICY.md)), not a second set of arms.
6. **A missing capability is a blocker, not a downgrade.** If a required operation is unavailable on a
   provider, the arm's status is `BLOCKED` with a reason — it is not silently run with a substitute kernel
   or precision. `#67` is the case in point: *"The live path raises NotImplementedError; M2 was closed as
   NOT RUN"*, after a documented training path had been described as working.

## What a CPU or AMD run may and may not support

| Situation | Permitted | Not permitted |
|---|---|---|
| CPU preflight only | establishing that the harness imports and the data pipeline materializes | any training claim |
| CPU smoke run | proving the retention paths ([12](12-ARTIFACT-RETENTION.md)) and validators execute | a capability or behaviour number |
| AMD primary runs | a full arm set, if preflight is `READY` with a non-null `rocm_version` and a stated basis | mixing AMD and NVIDIA rows in one comparison |
| NVIDIA primary runs | a full arm set | treating the H200/A100 distinction as irrelevant (`#34`, `#43`) |

The CPU row matters because it is how the design set is verified without a GPU at all: a smoke run on
`provider: cpu` with a tiny step budget exercises every retention path and every validator, which is the
cheapest possible answer to "does the scaffolding actually work" — the question Study 001 answered wrongly
twice (`#66`: a tokenizer gate hardcoded to `mock_compatible=True` so it always passed; `#67`: a training
path that raises).

## Engine changes

An inference-engine change (for example `vLLM` to `llama.cpp`) is **not** a hardware-agnosticism feature and
is not part of this study. Study 001 measured 21 decision flips between engines and attributed them to the
engine alone; the audit corrected that: *"Hardware also differs (H200 vs A100), and 3 of the 21 flips are
tokenizer-caused"* (`#34`), and the supporting claim was itself internally contradictory — *"The verdict
supports runtime choice: hardware is confounded"* (`#43`). Study 002 therefore fixes the engine for the
whole study and records it in the run fingerprint. If the engine must change, the entire comparison set is
re-run and the two sets are reported separately; they are never mixed.

## Recording the environment

Each run records: provider, device class, device count, driver, `cuda_version` **or** `rocm_version`
(whichever applies, with the other null), the engine and its version, dtype, quantization state, and the
frozen dependency set. The provenance record uses the existing schema fields (`run_id`, `status`,
`compute_provider`, `hardware`) and adds the requested/observed split from the preflight schema, so a reader
can see what was asked for and what was found.