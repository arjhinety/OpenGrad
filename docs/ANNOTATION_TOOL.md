# Annotation tool (`opengrad-annotate`)

A local, config-driven tool for human annotation: one item at a time, one keypress per label, every
change saved and recorded as it is made, and research output written as hash-pinned JSONL packages.
The first task is the frozen P-DET-v1 population
([`22-PDET-PROTOCOL.md`](research/study-002/22-PDET-PROTOCOL.md),
[`23-PDET-ANNOTATION-INSTRUMENT.md`](research/study-002/23-PDET-ANNOTATION-INSTRUMENT.md)); the engine is
generic, and a new task is a new YAML file, not new code.

**What it never does:** write to a source dataset, show a classifier prediction or model suggestion,
pre-fill a label, or let one pass see another. It produces no labels itself. A task may declare a
**model annotator** (see *Model annotators*). Its labels are then recorded and exported as model
judgments, in their own session, and never presented as human labels.

## Quick start (P-DET)

```bash
# once: Python extras and the UI build
uv pip install -e ".[annotation]"          # or: pip install -e ".[annotation]"
cd integrations/annotate-ui && npm install && npm run build && cd ../..

# before and after annotating (23 §1): the frozen population must still verify
python -m opengrad.verification.pdet --verify
opengrad-annotate check pdet-v1                  # read-only preflight of the task; creates nothing

# optional: every check `start` makes, stopping just before it creates the working state
opengrad-annotate start pdet-v1 --annotator <your-id> --session pass-a --dry-run

# annotate a pass -- run the same command again to resume exactly where you stopped
opengrad-annotate start pdet-v1 --annotator <your-id> --session pass-a
```

`start` (aliases `serve`, `resume`) opens <http://127.0.0.1:8765/>. It requires the task (`pdet-v1`), the
annotator id and the pass id, prints all three, and records the annotator and session on every change. A
second annotator runs the same command with their own id and `--session pass-b`. Read the
[annotator checklist](research/study-002/26-PDET-ANNOTATOR-CHECKLIST.md) first.

| Command | Does |
|---|---|
| `check TASK` | read-only preflight: config, source hash and item-count pins, ids, blinding, rubric, state location |
| `start TASK --annotator A --session S` | annotate one pass (start or resume); aliases `serve`, `resume` |
| `start … --dry-run` | every check `start` makes (source, rubric, ids, existing session, UI build, port); writes nothing |
| `adjudicate TASK --sessions pass-a pass-b --adjudicator C` | resolve disagreements between two finished passes |
| `adjudicate TASK --sessions pass-a --adjudicator C` | single-annotator re-read (22 §4) |
| `status TASK [--session S]` | progress per session; label counts only for the session named; creates nothing |
| `audit TASK` | re-verify every change-log hash chain in the working state |
| `export TASK --sessions ... [--out DIR]` | work-in-progress snapshot (never gold, never locks) |
| `freeze-gold TASK --sessions ... [--out DIR]` | final gold package; **locks** the sessions |
| `verify MANIFEST [--require-source]` | re-hash a package and re-check it against its manifest |
| `reference TASK --sessions pass-a model-a [--json]` | the composite reference's label counts (by source kind, and metric-eligible), recomputed from the store; creates nothing |
| `review-queue TASK --name N --sessions ... --seed S --out FILE [...]` | write a pinned review queue (see *Review queues*); never overwrites |

`TASK` is a task id with a config at `configs/annotation/<id>.yaml`, or a path to a config.
`python -m opengrad.annotation.cli` works the same way without the installed entry point.

| Where | P-DET |
|---|---|
| Working state (SQLite, git-ignored) | `.annotation/pdet-v1.sqlite3` — created by the first `start` |
| Work-in-progress snapshots | `reports/pdet/annotation/wip/` |
| Final gold package | `reports/pdet/annotation/` (manifest `pdet-v1.annotation-manifest.json`) |

## The annotation screen

- **Left, scrolling:** item number and id, metadata chips (P-DET component, source, ids), the preceding
  context (for P-DET: marked *not present in the source record*, because the frozen record keeps only the
  first user and last assistant turn), the user message, the assistant response **being annotated**
  (outlined), and the offered tools as readable parameter tables with a raw-schema toggle.
- **Right, fixed:** label buttons with their keys (`1` CALL, `2` DIRECT, `3` CLARIFY, `4` UNSUPPORTED,
  `5` UNKNOWN), each with a *definition* link into the rubric; the task's structured fields (P-DET:
  ambiguity status, optional rationale, boundary rule); an optional note; flag, skip, undo, previous,
  next; the saved state (label, revision, time, state hash); and this item's change history.
- **Drawers** (open over the context, so the buttons stay usable): *Items* (`/`) filters by
  unlabeled / completed / skipped / flagged / UNKNOWN and by source or component, jumps to an item by id
  or number, and shows the label counts — captioned as progress tracking, not a target. *Rubric* (`i`)
  renders the task's instruction files, read-only, with a section index; each label's *definition* link
  jumps to its section. A task can limit a file to named sections (`include_sections`) and list
  `instruction_forbidden_terms`; omitted text never reaches the browser, and the server refuses to start if
  a served excerpt contains a forbidden term. For P-DET the panel shows the checklist, 22 §1–§4 and 23 §2,
  §3, §5, §7, and omits the sections carrying sampling cues, classifier criteria, coverage counting and
  worked examples ([addendum A2.3](research/study-002/25-PDET-IMPLEMENTATION-ADDENDUM.md)).

Pressing a label key saves and moves to the next unlabeled item when the form is complete. For P-DET,
since [amendment 29](research/study-002/29-PDET-RATIONALE-OPTIONAL-AMENDMENT.md), no text is required:
CALL, DIRECT, CLARIFY and UNSUPPORTED save at once. UNKNOWN still needs an ambiguity status. Pressing `5`
without one holds the label and focuses that field, and choosing a status then saves it. The same happens
to a mode label while an ambiguity status is set. Typing in a text field never saves by itself; **Enter**
saves from it. Keys typed into a field stay in the field; **Ctrl+Enter** saves from anywhere and **Esc**
leaves a field. The progress header reads `183 / 581 completed · 31.5%`.

### Review queues and progress beside model judgments

A task may pin **review queues** (`review_queues` in the task config): fixed lists of items a pass may take
first, each a file pinned by SHA-256. `review-queue` builds one from the composite of the sessions named
(in priority order). The criteria are:

- `--flagged`: the source record is flagged uncertain;
- `--label L`: the composite label is L;
- `--sample L=N`: N model-sourced, metric-eligible items with label L, taken by lowest
  `sha256(seed:item_id)` and not already selected.

The queue is ordered by `sha256(seed:order:item_id)`, an order that ignores why each item is in it. The
file records each item's reasons, the seed and each source session's chain head. The same store and
arguments give the same bytes. The store refuses to open if a pinned file no longer matches its hash, or
names an item outside the population.

A pass sees only the queue's name, its item ids and its own progress through it. There is no reason, no
reference label, and nothing showing which session or model labeled an item. The *Order* selector switches
between a queue and every remaining item in frozen order. When the chosen queue has nothing left to label,
the screen says so and continues with all remaining items. Beside model sessions, the panel also counts:

- items this pass has labeled (**human-reviewed**);
- items resting only on a model judgment (**provisional, model only**);
- items with no label at all;
- items remaining for human review;
- this pass's flags.

These are counts only, and a model's label is never shown, before or after a submit. For P-DET the queue
is `priority-review` ([amendment 29](research/study-002/29-PDET-RATIONALE-OPTIONAL-AMENDMENT.md)); its file
names reference labels, so an annotator does not open it.

## Editing, provenance and freezing

Labels can be changed at any time **until the gold freeze**. Changing a saved label must be explicit
(the UI does this when you choose a new label on a labeled item) and may carry a reason. Every state
change — label, relabel, skip, flag, unflag, undo, adjudication; a note edit rides on the change it was
saved with — is appended to a **hash chain** (one per session, one per adjudication) in the same
transaction as the change:

| Recorded per change | |
|---|---|
| `recorded_at`, `actor_id`, `action`, `reason` | when, who, what, why |
| `before`, `after` | the full state before and after |
| `before_sha256`, `after_sha256` | canonical SHA-256 of each state |
| `prev_entry_sha256`, `entry_sha256` | the chain link and this entry's own hash |
| `reverts_entry_sha256` | for an undo, the entry it reverts |

Each annotation record carries its `revision`, `state_sha256` and `previous_state_sha256`. Undo never
deletes history; it appends an entry that names the one it reverts. `opengrad-annotate audit` re-derives
every hash and link and checks that each current record is the last state of its chain; export refuses to
run if it does not.

`freeze-gold` records a lock before it writes a byte (and releases it only if writing fails). From then
on every write to a frozen session — relabel, skip, flag, undo, adjudication — is refused (HTTP `423` in
the UI), and a package directory holding a gold freeze is never overwritten. Revising frozen labels means
a new task version.

## Passes, independence and adjudication

- A session is one pass by one annotator; it refuses to open under a different annotator id.
- The annotation server is bound to one session and has **no route** that returns another session's
  labels. Adjudication is a separate server mode and refuses to start until every compared pass is
  complete, so it cannot leak one pass into another in progress. `status` shows label distributions only
  for the session you name.
- **Disagreement** is defined by the task's `disagreement_keys` (P-DET, 23 §5.2: `gold_policy_label` or
  `ambiguity_status`). The adjudication screen shows both passes with differences highlighted, does not
  pre-fill a decision, and requires a rationale (P-DET also requires the deciding decision-tree step).
  Adjudication writes its own chained records; the passes are never modified.
- Adjudicating an item the passes agree on is allowed and flagged `ADJUDICATION_WITHOUT_DISAGREEMENT`. A
  pass edited after its item was adjudicated makes that adjudication **stale**, and freezing refuses
  until it is redone.
- **Single annotator** (22 §4): `adjudicate --sessions pass-a` opens the deterministic re-read queue —
  for P-DET every flagged item, every UNKNOWN, and every item citing a boundary rule. The freeze requires
  each to be reviewed, and the manifest records `design: single_annotator`,
  `inter_annotator_agreement_claimable: false` and the limitation. Two passes by the same annotator id are
  recorded as `repeated_pass_same_annotator`, never as independent.

## Freeze conditions

`freeze-gold` refuses — exit code 3, with every reason listed — unless:

1. every item is **labeled** in every named session (skipped counts as not labeled);
2. the design is allowed by the task (`freeze.allowed_designs`);
3. two passes: every disagreement is adjudicated (when `freeze.require_adjudication`, the default);
4. one pass: every item in the re-read queue is reviewed (when `freeze.single_annotator_review` is set);
5. no adjudication is stale, and the working state passes `audit`.

`export` writes the same files without the gold file into `<output.dir>/wip/`, with
`completion_state: INCOMPLETE` and the reasons, so work in progress can be inspected but never mistaken
for gold.

**Composite** (`--composite`, allowed only when the task lists `composite` in `freeze.allowed_designs`)
takes each item's label from the first listed session that labeled it, so the order of `--sessions` is
the priority, for example the human pass first, then a model session. It requires only that every item be
labeled in at least one listed session. Every gold record carries `label_source_session` and
`label_source_kind`. The manifest counts the `label_sources`, the items per session, and the overlapping
items where the sessions disagree. There is no adjudication and no single-annotator re-read.

## Model annotators

A task may declare language models as annotators (`model_annotators` in the task config): an id that
must start with `model.`, the model, a procedure file pinned by SHA-256, and the document that authorizes
it. Every `model.` id must be declared, and the tool refuses a model session when the procedure file no
longer matches its hash. Declarations are outside the task definition, so adding one never locks existing
labels.

| Command | Does |
|---|---|
| `model-batch TASK --session S --annotator model.X [--defer-to pass-a] [--size 50]` | renders the next items without a label in `S` or the deferred sessions: the rubric excerpts and each item exactly as the screen shows it, with no blinded column and no other session's labels. Writes `batch-NN.json` and `batch-NN.md` beside the working state. |
| `model-ingest TASK --batch batch-NN.json --answers answers.json` | checks every answer against the batch and the task's value rules, records nothing unless all of them pass, then records each in `S`. The change log names the batch, its content hash and the procedure hash. |

Packages say what the labels are:
- every session summary has `annotator_kind: human | model`;
- the manifest has `model_annotation` and the model declarations, and uses a statement that model labels
  are model judgments;
- a package with a model session is never `two_pass`, never claims independent annotators, and never
  claims inter-annotator agreement.

`verify` fails a package whose annotator kinds or gold `label_sources` do not follow from its records.
P-DET uses this under amendment
[`28-PDET-MODEL-LABEL-AMENDMENT.md`](research/study-002/28-PDET-MODEL-LABEL-AMENDMENT.md).

## Package format (P-DET)

Written under `reports/pdet/annotation/` — beside, never over, the frozen files, which the config lists as
`protected_paths`:

```
pdet-v1.annotation-manifest.json          manifest (+ .sha256 sidecar)
pdet-v1.annotations.<annotator>.<session>.jsonl   one row per annotated item, per pass
pdet-v1.disagreements.jsonl               both values and the differing keys
pdet-v1.adjudicated.jsonl                 both originals, the decision, rationale, adjudicator
pdet-v1.gold.jsonl                        gold labels with the full trail (freeze only)
audit/pdet-v1.history.<annotator>.<session>.jsonl  the pass's complete hash chain
audit/pdet-v1.adjudication-history.jsonl  the adjudication chain
```

Pass rows use the instrument's names (23 §2): `pdet_id`, `gold_policy_label`, `ambiguity_status`,
`annotator_rationale`, `boundary_rule_cited`, `annotator_id`, `annotation_version`, plus `session_id`,
`status`, `flagged`, `note`, `revision`, `created_at`, `timestamp`, `state_sha256`,
`previous_state_sha256`, `source_population_sha256` and `source_row_hash`. **One deliberate, naming-only
deviation from 23 §2:** pass files are named `…annotations.<annotator_id>.<session_id>.jsonl` rather than
`…annotations.<annotator_id>.jsonl`, and all files sit in `reports/pdet/annotation/`, so that a repeated
pass by the same annotator can never overwrite the first. Population, labels, procedure, pass isolation,
adjudication rules and P-DET semantics are unchanged; the record is
[`25-PDET-IMPLEMENTATION-ADDENDUM.md`](research/study-002/25-PDET-IMPLEMENTATION-ADDENDUM.md) (A1).

## P-DET decisions (recorded in the addendum, A2)

1. **`challenge_families` stays hidden** while labeling. The families are the cue predicates the challenge
   sample was drawn with; a name like `refusal_plain` beside an item would suggest a label. The field stays
   in the frozen population, and every record's `source_row_hash` links back to the full row.
2. **A rationale was required** for P-DET, because 23 §2 requires one. Since
   [amendment 29](research/study-002/29-PDET-RATIONALE-OPTIONAL-AMENDMENT.md) (`study_002_prereg_v3`) it
   is **optional**, an ergonomics change, not a taxonomy change. Rationales already written are kept, and a
   label without one is not treated as less certain. It is task configuration either way, and the generic
   note field stays optional.
3. **Pass files are session-qualified**, as above.
4. **The rubric panel shows excerpts** without sampling cues, classifier criteria, coverage counting or
   worked examples, and refuses to start if a forbidden term would be shown.
5. **Three items are `EXPOSED_WORKED_EXAMPLE`**
   ([`27-PDET-EXPOSED-WORKED-EXAMPLES.md`](research/study-002/27-PDET-EXPOSED-WORKED-EXAMPLES.md)): 23 §4
   shows illustrative labels for them. They are annotated normally, but excluded from classifier-validation
   metrics, agreement statistics and untouched-gold claims, leaving 578 of 581 items metric-eligible. The
   interface never shows the status.

Each gold row holds the gold `gold_policy_label` and `ambiguity_status`, its `gold_source` (`agreement`,
`adjudication`, `review` or `single_annotator`), every pass's value and state hash, and the adjudication
if any. All files are sorted by item id with sorted keys, so for the same stored state and destination
the bytes are identical.

The manifest records the task id and version, the task-definition hash, the source path, format, SHA-256
and item count, the design and whether agreement may be claimed, per-session counts (labeled, skipped,
flagged, unlabeled, label counts, UNKNOWN, relabels, undos, chain length and head), annotator and
adjudicator ids, disagreement counts, adjudication flags, completion state and reasons, gold counts and
sources, `model_assistance: false`, the SHA-256, size and record count of every output, the app version and
the export time. Its `metric_exclusions` section lists each exclusion group (status, reason, document,
what it excludes from, item ids) and the metric-eligible item count; sessions, disagreements and gold carry
metric-eligible counts beside the full ones, and every pass, disagreement, adjudication and gold record
carries its item's `metric_exclusions` (`[]` for most items).

`verify` recomputes all of it from the files: the sidecar, every output hash, size and count, label counts,
each chain's integrity, that every record's **content** equals its final chained state, that every gold
label follows from the records it cites, that every record carries exactly the exclusions the manifest
lists for its item and the metric-eligible counts add up, and that the source still hashes to the pinned
value. It finds
the repository — and so the source — from the package itself; if the source is unreachable it says so
(`source_unreachable`), and `--require-source` turns that into a failure. A package written outside the
repository (a scratch snapshot, say) cannot find the source by itself: pass `--root <repository>`.

## Writing a task

```yaml
task_id: my-task-v1              # required
version: "1"                     # required
annotation_schema_version: my-task-annotation-v1   # required; stamped on every record
task_type: single_label          # single_label | multi_label | binary | rating | free_text | pairwise | ranking
labels: [GOOD, BAD, UNSURE]      # shortcuts default to 1..9
unknown_labels: [UNSURE]
source:
  path: data/my.parquet          # read-only, relative to the repository root
  format: parquet                # jsonl | json | csv | parquet
  id_field: example_id           # omit to use a content hash as the id
  expected_sha256: <hash>        # optional: refuse to open if the bytes differ
fields:                          # what the annotator sees, in order
  context: messages              # dotted paths: messages.0.content, messages.-1.content
  user: {path: prompt, label: User}
  assistant: response            # "assistant" is outlined as the thing being annotated
  tools: tools                   # "tools" renders as schema tables
metadata: {split: split}         # chips above the item
filters: {split: split}          # navigator filters
blind_fields: [model_prediction] # never displayed, filtered on or sent to the browser
extra_fields:                    # structured inputs beside the label
  - {key: why, type: text, required: true, adjudication: false}   # adjudicators write their own rationale
  - {key: severity, type: select, options: [NONE, MINOR, MAJOR], default: NONE, required: true}
constraints:                     # label-conditional rules, checked on every save
  - {when_label_in: [BAD], field: severity, forbidden: [NONE], message: A BAD item needs a severity}
disagreement_keys: [label]
adjudication_fields: []
record_keys: {label: my_label}   # rename keys in exported records
output: {dir: reports/my-task}
freeze: {allowed_designs: [two_pass, single_annotator], require_adjudication: true}
metric_exclusions:               # optional: annotated normally, left out of the named metrics
  - status: EXPOSED_WORKED_EXAMPLE
    reason: Labeled as an example in the instructions
    document: docs/my-task-exclusions.md   # optional; must exist and name every id
    excluded_from: [agreement_statistics, model_validation_metrics]
    item_ids: [ex-17, ex-42]       # must be source items
definition_amendments:           # optional: declared relaxations after labels exist (see below)
  - document: docs/my-task-amendment-1.md   # must exist
    from_definition_sha256: <definition hash the labels were made under>
    to_definition_sha256: <the definition hash now>
review_queues:                   # optional: pinned review orders (see *Review queues*)
  - {name: priority-review, file: reports/my-task/review.json, sha256: <hash>}
```

Configs are validated strictly: a missing version, a shortcut to an unknown label, a display path into a
blinded field, `model_assistance: true`, or a constraint on a field the adjudicator does not fill are all
errors. Once any label exists, a change to the task *definition* (labels, fields, constraints, versions)
refuses to open rather than reinterpret existing labels; display-only changes are free. The one exception
is a **declared relaxation**. If `definition_amendments` leads from the stored definition to the new one,
and the change only turns required fields optional, the store opens: every existing label met the stricter
rule, so it meets the new one unchanged. It then appends one hash-chained entry to its `definition_history`,
holding both definitions, their hashes, the amendment document, the number of annotation records at that
moment and the time. `audit` re-verifies that chain, every export manifest carries it, and `verify` fails a
package whose history does not hash or does not end at its definition. Metric
exclusions are not part of the definition either: they change which labels a metric may use, not what a
label means, so an exposure found mid-annotation can be recorded without refusing the store. Every export
records the exclusions in force when it was written, and the interface never shows them.

Implemented task types: single-label, binary and pairwise (buttons), multi-label (toggles; also error
tagging), rating (scale buttons), free text, and ranking (reorder). Span selection is not implemented; the
value model (`value` as an object keyed by the task type's primary key) leaves room for it.

## Architecture

| Layer | Where | Notes |
|---|---|---|
| Task config | `src/opengrad/annotation/config.py` | YAML/JSON → validated `TaskConfig`; the definition hash |
| Import | `items.py` | JSONL / JSON / CSV / Parquet; one read, hashed and parsed from the same bytes |
| Values | `values.py` | server-side validation and canonical form of every value |
| Provenance | `provenance.py` | state hashes, chain entries, chain verification |
| Store | `store.py` | SQLite (WAL, `synchronous=FULL`); one transaction per change |
| Service | `service.py` | integrity on open, sessions, labeling, navigation, adjudication, audit |
| Export | `export.py` | snapshots, gold freeze and lock, manifest, `verify_package` |
| Review | `review.py` | composite reference counts; building and writing pinned review queues |
| Model batches | `model_batch.py` | rendering, checking and recording declared model annotators' batches |
| API | `server.py` | stdlib `http.server` on loopback; JSON routes; serves the UI build |
| CLI | `cli.py` | `opengrad-annotate` |
| UI | `integrations/annotate-ui/` | Next.js (App Router, TypeScript), static export; renders only |

The Python side has no dependencies beyond the standard library, PyYAML and (for Parquet) pyarrow. The UI
holds no rules: it mirrors validation only to say what is missing, and the server re-validates every
write. Working state lives in `.annotation/<task_id>.sqlite3` (git-ignored); a label is committed to disk
before the browser is told it was saved, so closing the tab, stopping the server or losing power loses
nothing already saved.

For UI development, run `opengrad-annotate start …` and, in `integrations/annotate-ui`, `npm run dev`
(<http://localhost:3000>, proxying `/api` to port 8765, or `ANNOTATE_API`).

## Integrity model and limitations

What the tool provides is **integrity evidence, not security**. It is designed to make accidental and
ordinary changes detectable, not to resist a determined adversary.

- **Hashes and `.sha256` sidecars detect accidental or ordinary changes** to an artifact: an edited label,
  a truncated file, a stale copy, a changed population. `verify` and `audit` recompute them.
- **Committing a package's manifest (or its `.sha256` digest) to Git gives an external historical anchor**
  relative to the package: a later rewrite of the package no longer matches the committed digest.
- **Out of scope:** an actor able to rewrite both the entire package — every chain entry, hash and sidecar
  recomputed — *and* the trusted Git history. Nothing in the tool can detect that.
- **Packages are not cryptographically signed.** A hash says the bytes are unchanged since it was
  computed; it does not say who produced them.
- **Annotator identities are declared identifiers, not authenticated identities.** `--annotator` is taken
  at face value and recorded; the tool cannot establish who typed it.
- **There is no application authentication.** The server listens on loopback only and refuses foreign
  `Host` headers and non-JSON writes, which keeps other web pages out; it does not tell users apart. Pass
  isolation is enforced by the server's routes, not by access control over the SQLite file or exported
  snapshots — keep an annotator away from other passes' files until their own pass is finished.
- **Agreement statistics** (raw agreement, Cohen's κ) are not computed by the application; the packages
  carry everything needed to compute them.
- **Span annotation is not implemented.**
- One server process serves one session or one adjudication; switch by restarting with other arguments.
- The application offers no deletion of sessions or labels: history is append-only. Disposable test work
  belongs in a separate `--state-db`, which can simply be deleted.
