# Provenance

> OpenGrad requires immutable provenance, not hashes for the sake of hashes.

A `verified: true` claim is an assertion that the evidence behind a value was
checked. On its own that assertion means nothing, because a claim is only as good
as the thing it resolves to. What makes it meaningful is that the evidence cannot
change while the record stays the same.

## The invariant

Every `verified: true` claim must resolve to an immutable evidence anchor. There
are two routes, and they are not interchangeable:

```text
verified claim
    |
    +-- mutable/local source
    |       -> content digest required (source_sha256)
    |
    +-- externally immutable source
            -> immutable revision/pin required
```

A repository path is mutable. `reports/foo.json` can hold different bytes
tomorrow while the record that points at it is unchanged, so a path alone pins
nothing and the claim must carry `source_sha256`:

```yaml
license:
  value: COMPOSITE_PER_SOURCE
  verified: true
  source: release/huggingface/toolpolicy-canonical-v2-final/source-licenses.md
  source_sha256: de9e6cbfe126d6368f509b9587132b1a2cf00b0808c3b7b814f12c3fb6c23ce0
```

An external source pinned to an exact commit or revision is already anchored and
is deliberately **not** asked for a redundant digest. Uniformity is not the
requirement; immutability is. Accepted immutable identifiers include a full Git
commit SHA, a Hugging Face model or dataset revision, an immutable release or tag
the upstream guarantees, a DOI with an explicit version, and any other
repository-supported frozen identifier.

What is not accepted is a moving target. `blob/main`, `tree/master`,
`releases/latest` and short abbreviated revisions all resolve to different bytes
at different times, so none of them pins anything.

The validator enforces the distinction rather than commenting on it. A rule that
demanded `source_sha256` everywhere would be as wrong as one that demanded
nothing: it would add meaningless hashes to sources that are already immutable
while proving nothing about the ones that are not.

## Derived artifacts

A derived artifact is held to the composition of the two routes:

```text
immutable source/input identity
            |
            v
     deterministic transform
            |
            v
       derived identity
```

The repository already names these, and this document uses the repository's
names rather than introducing new ones:

| Role | Existing field |
|---|---|
| input identity | `source_revision`, `exact_revision`, `parent_fingerprint`, `input_manifest_sha256` |
| derived identity | `processed_dataset_hash`, `checksum`, `derived_fingerprint` |

A derived record must record an immutable input identity and a derived identity,
and where it records two derived identities they must agree. A record that
records a derived digest but no input identity is incomplete.

Note what is *not* checked: where a record carries a digest alongside a pointer
to a manifest, the two are treated as complementary rather than contradictory.
Which of the two is authoritative is a research judgment, and the registry
records that judgment. The validator only rejects a genuine contradiction.

## Legacy exceptions

Artifacts published before this model are preserved as they were. They are not
rewritten to look as though they had always followed it, because the
distinction between *what was recorded at the time* and *what we now know is
correct* is part of the research record.

`provenance_version: legacy_single_digest_v1` is a **named allowlist**, checked
against `LEGACY_ARTIFACTS` in `src/opengrad/registry/provenance.py`. A record
carrying the marker without being in that map is a validation failure, an
unknown `provenance_version` is a validation failure, and the allowlist is
asserted to stay small. This is an exception that is intended to shrink, not an
escape hatch.

The currently named case is `OpenGrad-ToolPolicy-Canonical-v2-minus-xlam`, whose
interpretation-only card correction moved the derived fingerprint from `f8ba687e`
to `5fc73904` while its 118 parquet shards stayed byte-identical to the frozen
parent. Both fingerprints are preserved; see `reports/ERRATA.md`.

## Release identity and publication identity

OpenGrad has a release discipline and, historically, no publication discipline.
Every rule governed the payload -- hashed manifests, hashed shards, a validator
that fails closed, a fingerprint that survives a rebuild -- while the act of
publishing, which is when an artifact becomes citable, had no gate at all.

The two are now separate, and both are recorded:

* **Release identity** is the payload: the manifest and its shards. It is
  immutable, and a rebuild must reproduce it. This is what experiments pin.
* **Publication identity** is the set of files actually published at a Hub
  revision. It moves whenever anything published changes, including a licence
  file or a card.

A provenance correction is therefore a *publication event*, recorded in
`reports/releases/` as an append-only record naming the revision it superseded.
It is not a new version of the corpus. The 2026-09-14 correction to the v2-final
licence file is recorded that way, and the payload hashes it lists are unchanged
from the release before it.

Publication records carry publication dates, not sequence numbers, and several
describe the same repository on the same day, so filename order is not
chronology. The consistency check is order-independent: the set of revisions a
repository has been published at must resolve to exactly one current revision
once every explicitly superseded revision is removed.

A publication event should be atomic -- one commit, one recorded revision. Where
a correction was published as more than one Hub commit, the intermediate
revisions are recorded rather than tidied away.

## Content digests are over committed bytes

A `source_sha256` is the hash of the file **as committed**, not as it appears on
disk. This repository sets `core.autocrlf=true` and `.gitattributes` covers
`*.yaml`, `*.json` and `*.md` but not, for example, `*.jsonl`. A file checked out
on Windows therefore has CRLF where the committed blob has LF, and hashing the
working-tree bytes would make the digest a claim about the verifier's operating
system rather than about the artifact.

The rule is enforced, not assumed:

* the evidence path must be **tracked by Git**. An untracked path is not part of
  the research record, however stable it looks. This is the general form of the
  defect that let two licence claims point into `.release/`, a gitignored build
  directory, and look sound.
* the digest is computed over the committed blob (`git cat-file blob HEAD:<path>`).
* outside a work tree (unit-test fixtures) the working-tree bytes are used, and
  that fallback exists only so the rules can be tested without building a repo.

## A frozen identity must be reproducible or corroborated

A record asserting a `FULL_DATA_VALIDATED` derived identity must be able to show
what that identity is the identity *of*. Either:

* the record names an artifact, and the check **re-derives** the digest from its
  committed bytes; or
* the record declares `processed_dataset_hash.identity_artifact_unavailable`,
  names the artifact that is absent, and names a tracked record that
  **corroborates the same digest**.

A declaration with no corroboration is rejected, and a declaration whose
corroborating record disagrees is rejected with the disagreement stated
explicitly — because two records disagreeing about the identity of the same
published corpus is the case that matters most.

This rule exists because its absence made the check decorative.
`_local_manifest_for` previously returned nothing for all three canonical
corpora, so the strongest freeze assertion in the system was never executed and
the check reported PASS by doing nothing.

## Reconstructed events cannot read as contemporaneous

A publication made without a record may be reconstructed from immutable Git or
Hub history. That is worth doing, but it creates a specific hazard: an entry
written today about an event three days ago is indistinguishable from an entry
written at the time unless it says which it is, and a later reader asking "what
did we record then?" would be answered with something recorded afterwards.

So every reconstructed event carries three separate facts, and the distinction is
validated:

| Field | Meaning |
|---|---|
| `event_occurred_at` | when the publication actually happened |
| `entry_recorded_at` | when this entry was written |
| `reconstruction_evidence` | the immutable identifier it was reconstructed from |

`recorded_retroactively: true` requires all three, requires that the entry was
written strictly after the event, and requires immutable evidence.
`entry_recorded_at` must also match the date of the record it lives in. An entry
whose dates differ without declaring `recorded_retroactively` is a validation
failure: that is exactly the masquerade this prevents.

## A gate must prove it executed

A gate that returns success without examining anything is indistinguishable from
one that examined everything and found nothing wrong. Both happened here: freeze
validation skipped every canonical corpus because it looked for a manifest in a
field those records do not use, and `python -m opengrad.registry.validate`
imported its module, ran nothing, and exited 0.

Every gate therefore reports an execution census alongside its errors, in
`src/opengrad/verification/accounting.py`:

```text
discovered == checked + blocked + skipped
checked    == passed + failed
```

A counter set that does not add up is itself a failure, because counters that
disagree are a symptom of the same class of bug. So is a skip without a reason.
And a gate declares what its population is:

| Policy | Meaning |
|---|---|
| `REQUIRED_NONEMPTY` | The inputs exist by construction; discovering none means discovery is broken. |
| `CONDITIONALLY_REQUIRED` | Required only while a stated precondition holds. |
| `OPTIONAL` | May legitimately have no inputs; recorded as a skip with a reason. |

The distinction matters because making every empty population fail would be as
wrong as making every empty population pass. The freeze gate's requirement binds
*because this repository declares corpora*; a repository that declares none is
skipped with a reason rather than failed or silently passed.

`scripts/verify_publication.py` prints the census, and `--json` emits it
machine-readably with the verifier contract version. Exit 0 is only reachable
when every required population was actually examined.

### Verifier contract versions

The contract version is in `src/opengrad/verification/__init__.py` and reported by
the gate. Bump it whenever the set of checks, their discovery rules, or their
non-vacuity requirements change such a PASS means something different.

| Version | What a PASS meant |
|---|---|
| v1 | Assertions passed. It did not verify that they had run, so `opengrad.registry.validate` could exit 0 without validating and freeze validation could skip every canonical corpus. |
| v2 | Every required population was examined and passed, with an explicit census. |

A PASS recorded under v1 is not equivalent to one under v2, and no historical
record is rewritten to imply otherwise.

## The gate

`scripts/verify_publication.py` is the authoritative publication-readiness gate.
Run it before declaring a release, report or publication complete.

```bash
python scripts/verify_publication.py            # full verification
python scripts/verify_publication.py --offline  # registry only; see BLOCKED below
```

It returns:

| Status | Meaning | Exit |
|---|---|---|
| `PASS` | Every check ran and passed. | 0 |
| `FAIL` | A check ran and found a demonstrable defect. | 1 |
| `BLOCKED_NETWORK` | A required check could not reach the network. | 2 |
| `BLOCKED_OPTIONAL_DEPENDENCY` | A required check needs an optional dependency that is not installed. | 2 |

A blocked check is **never** reported as success. "We could not check" and "we
checked and it is fine" are different claims, and collapsing them is how a
provenance defect ships.

The same implementation backs the `network`-marked tests
(`pytest -m network`), which exist to catch regressions in this machinery. They
do not replace the command, and publication validity does not depend on anyone
remembering to run them.

## What the validator does not do

The registry is human-authored declarative research state. The validator verifies
that the judgment recorded in it is internally and evidentially well-formed; it
does not decide whether a result is scientifically sound, which of two recorded
values is authoritative, or whether a mixture is a good idea. Those are research
judgments, and they live in the registry.

## What this caught

Adding these rules surfaced prior provenance defects that no check had been able
to see. They are recorded in `reports/ERRATA.md` and corrected forward:

* `canonical_v2`'s licence claim pointed at a `.release/hf/` path that no longer
  existed, and its `verified_from` anchor pointed at an uncommitted build
  manifest. `.release/**` is build output under `.gitignore`, so neither could be
  reproduced from a fresh clone.
* `canonical_v1`'s licence claim pointed at a `.release/hf/` file that still
  existed on the machine that built it. The claim looked sound while being
  unreproducible from the repository, which is a worse failure mode than a
  missing file.
* The `qwen3.5-2b` licence claim was anchored to `blob/main/LICENSE`, a moving
  target.
* `CITATIONS.bib` for the v2-final corpus omitted the APIGen entry its own
  release manifest declares as required for xLAM.
