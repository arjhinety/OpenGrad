# Adding a data source to OpenGrad

This is the procedure for bringing a dataset into OpenGrad, whether for training or for evaluation. It exists
so that anyone reading a result can answer four questions without asking us:
- which bytes were used;
- under what terms;
- what else they overlap;
- which sources were considered and rejected.

The record format is `registry/dataset_record.schema.json` (JSON Schema 2020-12). Each field there states its
meaning, its allowed values and its counterpart in MLCommons
[Croissant 1.1](https://github.com/mlcommons/croissant/blob/main/docs/croissant-spec-1.1.md) and
[Croissant RAI 1.0](https://github.com/mlcommons/croissant/blob/main/docs/croissant-rai-spec.md). This page is
the order in which the fields get filled, and the evidence each one needs.

## The two files

| File | Holds | Read by |
|---|---|---|
| `registry/datasets.yaml` | Adopted sources and the corpora OpenGrad derives from them | the training firewall, readiness gates, `opengrad-validate` |
| `registry/source_screening.yaml` | Every candidate screened for a purpose, with a verdict per criterion, including the rejected ones | `opengrad-validate` only; never the firewall |

A source enters `datasets.yaml` only after it is adopted. The firewall
(`opengrad.data.materialize._training_split_allowlist`) decides what may enter SFT from `id`, `intended_stages`
and `allowed_splits` in that file. Keeping candidates out of it means a candidate can never become trainable
by accident.

## Step 1: Pin the bytes

1. **Pin an immutable revision.** Use the Hub commit, or the git commit for GitHub-hosted data. Never pin a
   branch or `main`. Record it as `source_revision` with `verified: true` and the `source` you read it from.
2. **List the files at that revision in `distribution`,** each with its sha256, size and format:
   - For a Hub dataset, read the digests from the Hub's own LFS record:
     `https://huggingface.co/api/datasets/<repo>/revision/<rev>?blobs=true`.
   - For a file the Hub does not store in LFS, or a file on GitHub, hash the bytes. Say in `verified_from` how
     you know they are the pinned bytes. For GitHub, the local file's `git hash-object` must equal the pinned
     tree's blob id.
3. **Put the revision in every `content_url`.** `opengrad-validate` rejects a URL that does not carry it, since
   that digest would pin nothing.

## Step 2: Record the terms

1. **Read the licence at the pinned revision** from the LICENSE file or the dataset card. Do not rely on a
   metadata tag. Record the SPDX identifier.
   - A local licence file needs `source_sha256` (`docs/PROVENANCE.md`).
   - If the code and the data carry different licences, record the data licence and say so in `notes`.
   - If the maintainers disclaim copyright in the content itself, as TriviaQA's do for its questions, say so
     in `notes`.
2. **Set `redistribution`** to one of `REDISTRIBUTION_WITH_ATTRIBUTION`, `PER_SOURCE`, `NOT_ASSESSED` or
   `NOT_PERMITTED`. Name the document that records the decision in `redistribution_basis`; it is required unless
   the status is `NOT_ASSESSED`.
3. **Fill the remaining terms:**
   - `upstream_access_mode`, from the Hub API's `gated` flag;
   - `downstream_access_requirement`;
   - `attribution_required`, `citation_required`, `citation_target` and `modifications_disclosed`;
   - `papers`, as ids in `docs/references/papers.yaml`. Add the paper there first.

## Step 3: Record what the data is

- **`role`.** `UPSTREAM_SOURCE` for data OpenGrad did not create, `DERIVED_CORPUS` for what it built. A derived
  corpus also needs `derived_from` (each upstream id and revision), `preprocessing_version` and a digest in
  `processed_dataset_hash.value`.
- **`lifecycle`.** `ACTIVE`, `HISTORICAL` (superseded but cited by published results) or `EXCLUDED` (recorded so
  the exclusion is explicit). An `EXCLUDED` record may not list an `sft` stage.
- **Counts.** `sample_count.published` is the upstream's own count and where it was checked.
  `retained_after_filtering` is what OpenGrad kept.
- **`responsible_use` (Croissant RAI).** Record only what a cited source states: how the data was collected,
  its intended uses, limitations, biases and personal information. List the sources in `evidence`. An empty
  field is better than a guessed one.

## Step 4: Record overlap and contamination

- **`benchmark_overlap_risk`.** `HIGH_KNOWN_BENCHMARK_OVERLAP` requires `overlap_benchmarks`.
- **`known_source_overlap`.** `AUDITED` or `INHERITED_FROM_SOURCES` requires `overlap_evidence`: the paths of the
  artifacts that record the audit. A shared upstream is overlap even when no text matches. When2Call's training
  data, for example, was generated from xLAM-60k.
- **`contamination_status`.** This comes from the contamination screen
  (`docs/evaluation/CONTAMINATION_ADJUDICATION.md`). Level 5 is never marked complete by hand.

## Step 5: Decide the firewall fields

- `intended_stages` takes only `sft`, `dpo`, `preference`, `evaluation` and `contaminated_experiment_only`. A
  `future_` prefix marks a stage that is planned but not yet executed.
- `allowed_splits` lists the only splits that may enter training.
- `forbidden_splits` lists every held-out, preference and evaluation split.
- A source used for evaluation must never list an evaluation split as allowed.

## Step 6: Validate

```bash
.venv/Scripts/opengrad-validate.exe
.venv/Scripts/python.exe -m pytest -q tests/registry/test_dataset_record_schema.py
```

A change to `intended_stages` or `allowed_splits` changes what may be trained on. The pinned allowlist in that
test fails on purpose. Update it in the same commit, and say why in the commit message.
