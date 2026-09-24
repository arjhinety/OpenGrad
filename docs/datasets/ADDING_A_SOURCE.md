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

## Step 0: Screen before you adopt

Candidates for a purpose (a training source, an evaluation population, a benchmark) are screened in
`registry/source_screening.yaml` before any is adopted. A screening records:
- **the need,** and the document that sizes it (`requirement_source`);
- **the criteria,** each a test a candidate passes or fails;
- **the search,** its date and method, the written report, and the artifacts behind its numbers;
- **every candidate considered,** with the fields below.

Each candidate carries:
- one verdict per criterion: PASS, PARTIAL, FAIL, UNKNOWN or NOT_APPLICABLE;
- a count of the items that fit the purpose, and the basis for that count;
- evidence, as https URLs or repository paths;
- a decision:
  - **SHORTLISTED**, which fails no criterion;
  - **EXCLUDED**, which names the failing criteria that decide it;
  - **WATCHLIST**, which names what would change the verdict.

The rejected candidates stay in the log. That is the point of it: a reader can see what was looked at and why
each exclusion happened, as in a PRISMA flow. The flow counts are generated, never typed:

```bash
.venv/Scripts/python.exe scripts/reporting/generate_source_views.py   # docs/datasets/SOURCE_SCREENING.md, SOURCE_REGISTRY.md
```

Adopting a candidate is the study owner's decision. Record it in the screening's `owner_decision`, then add the
source to `registry/datasets.yaml` with the steps below. The validator requires an adopted candidate to be both
shortlisted and registered.

A number in the screening report comes from a committed artifact, as in
`reports/source-screening/study-002-answer-heldout/`, where `scripts/audit_bfcl_answer_supply.py` regenerates
the sizing and overlap figures. A reader's sizing judgement is stored as its own file, labelled as not gold.

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

## Step 7: Export as Croissant

```bash
.venv/Scripts/opengrad.exe croissant <id>                 # one record as Croissant 1.1 JSON-LD
.venv/Scripts/opengrad.exe croissant --out DIR --strict   # every record; exit 1 if a required property is missing
```

- **Where the mapping comes from.** The export reads each field's Croissant or Croissant RAI counterpart from
  the schema's `x-croissant` annotations. It uses the standard Croissant context.
- **What it contains.** The export is metadata only: dataset-level fields, the pinned files as `cr:FileObject`s
  with their sha256, lineage as `prov:wasDerivedFrom`, and `responsible_use` as `rai:*`. It has no `RecordSet`,
  because the registry does not describe record fields.
- **What it leaves out.** A required property the registry cannot supply is reported, never invented. Today
  that is `distribution` for the derived corpora.
- **Validator result.** Checked with MLCommons' validator (`mlcroissant` 1.1.0), the export has no errors.
  - One warning is expected and kept: `version` holds the pinned commit, which is not a semantic version.
    Inventing one would describe the data less exactly.
  - The other is a missing `citeAs` where the source has no paper.
