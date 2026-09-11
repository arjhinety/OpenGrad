# Hugging Face

**In use.** Hugging Face is the distribution home for large artifacts; GitHub remains the
canonical home for the code, schemas, manifests, audits and experiment definitions that produced
them.

Published, in two collections:

| Collection | Contents |
|---|---|
| [models](https://huggingface.co/collections/arrochi112/opengrad-models-6aa3c7ea9ae58be5adbb113e) | The M0-final checkpoint, the partial-v2 checkpoints, and the M1 DPO checkpoint |
| [datasets and evaluation records](https://huggingface.co/collections/arrochi112/opengrad-datasets-and-evaluation-records-6aa3c7eb4514d9bc27e5d160) | Canonical v2 (final), canonical v1, the partial-v2 snapshot, and the corpus-v1 evaluation record |

The tracked release definitions — dataset-card templates, licensing, and citations — live in
[`release/huggingface/`](../../release/huggingface/). Generated staging belongs under `.release/`
and is ignored by Git.

Uploading is a deliberate manual step; no training run publishes automatically. Every published
artifact's revision is recorded in [`reports/releases/`](../../reports/releases/).
