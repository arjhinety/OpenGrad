# NVIDIA ModelOpt integration

NVIDIA ModelOpt (`nvidia-modelopt`) is an **optional** backend for the optimization
producer layer. It is not a core dependency, it is not installed by default, and nothing
in OpenGrad imports it unless a ModelOpt backend method that needs it is actually called.

## Optional dependency

ModelOpt is provided by the `optimization` extra:

```bash
pip install '.[optimization]'
```

Every ModelOpt symbol is resolved by a lazy `importlib.import_module` inside the method
that needs it. When it is absent, `optimize()` and `export_artifact()` raise a
`RuntimeError` that names the extra:

```
the modelopt optimization backend requires the optimization extra: pip install '.[optimization]'
```

Importing `opengrad.optimization`, importing any unrelated OpenGrad module, running
`opengrad train`, the dataset commands, readiness, and evaluation all continue to work
with ModelOpt absent. The deterministic `mock` backend needs neither ModelOpt nor a GPU.

## Capability discovery: discover, never assume

`capabilities()` returns a `CapabilityProbe` per technique, each with one of:

* `SUPPORTED` - established for this exact model.
* `UNSUPPORTED` - established as not working for this exact model.
* `PARTIAL` - established as working with a documented limitation.
* `UNKNOWN` - cannot be settled without running.

Each probe carries an evidence string and the exact backend revision. The lookup key is
the exact `(model_id, technique)` pair in an explicit, currently-empty table. There is
**no** model-family or size fallback: a fact about one `Qwen3.5` size never answers for
another, and a technique ModelOpt documents at the library level does not promote a
per-model answer. Consequently the probe for `Qwen/Qwen3.5-2B` is `UNKNOWN` for every
technique until a real run establishes otherwise.

Techniques covered: FP8 PTQ, NVFP4, other PTQ formats (INT8/INT4/AWQ/GPTQ), QAT, QAD,
distillation, pruning/sparsity, speculative-decoding training methods (EAGLE/DFlash-style,
where the installed revision exposes them), Hugging Face export, and vLLM-compatible
deployment.

The recorded matrix lives in
`reports/optimization/modelopt-capability-matrix.json` and is regenerated with:

```bash
python -m opengrad.optimization capability-matrix \
    --output reports/optimization/modelopt-capability-matrix.json
```

## Execution status

No ModelOpt execution has been performed in this repository. FP8 PTQ, NVFP4, other PTQ,
QAT/QAD, distillation, pruning/sparsity, and speculative-decoding training are **NOT
attempted**. The `_execute()` path is written against ModelOpt's documented surface but is
unverified: it has never been run, and the `optimization` extra has never been installed
here. Treat it as unproven until a real, reviewed run exists.

`registry/runtime_components.yaml` records `nvidia-modelopt` as `INTERFACE_ONLY` with
NVIDIA `NOT_TESTED` and AMD/CPU `UNKNOWN`.
