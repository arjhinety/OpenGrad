# Methodology

OpenGrad uses baseline → intervention → evaluation → diagnosis → next experiment. No post-training method is presumed superior. Accepted checkpoints require reproducible comparisons, variance/confidence, regression analysis, contamination review, and lineage explaining why checkpoint X was preferred over Y.

That is the standard, and the current promoted checkpoint does not meet all of it. M1-v2 checkpoint
30 is single-seed with no variance estimate or interval; its +0.0078 `call_f1` over M0 is 7 more
correct calls out of 453, within noise, and its promotion reflects a change of gate (v3 → v4) rather
than a measured improvement over M0. See [`reports/M1_DPO_EVALUATION.md`](../../reports/M1_DPO_EVALUATION.md).

Tool quality and inference efficiency are measured separately before any joint claim.

OpenGrad's initial research questions were motivated by deployment findings from the independently maintained [OpenWeights](https://github.com/alpharomercoma/openweights) project. OpenWeights observations are problem-discovery evidence, not OpenGrad results. Preserve that distinction with explicit labels such as `Observed in OpenWeights`, `Motivated by OpenWeights`, `OpenGrad hypothesis`, `OpenGrad planned experiment`, `OpenGrad reproduced`, and `OpenGrad result`; see [motivation and provenance](motivation.md).
