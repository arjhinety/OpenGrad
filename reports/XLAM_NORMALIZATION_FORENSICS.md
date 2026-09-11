# xLAM normalization forensics

**Status:** canonical normalization `IMPLEMENTED` and `MEASURED` over all 59,370 retained
records. **Trainability is recovered: 0 → 56,090 (94.5%)** under the supervision contract (§7).

**Date:** 2026-09-11
**Machine-readable evidence:** [`reports/data/xlam-normalization-forensics.json`](data/xlam-normalization-forensics.json)
**Code:** `src/opengrad/data/xlam_types.py`, adapter `xlam_function_calling_60k_v2`
**Reproduce:** `python scripts/audit_xlam_normalization.py --output reports/data/xlam-normalization-forensics.json`

---

## 1. Summary

| | Before | After |
|---|---:|---:|
| Records | 59,370 | 59,370 |
| Canonical records accepted | **33** (0.06%) | **57,342** (96.6%) |
| Records rejected at the schema layer | 59,337 | **2,028** (3.4%) |
| Tools | 166,781 | 166,781 |
| **Trainable records** | **0** | **56,090 (94.5%)** |
| Records with a tool-call target | 0 | 57,342 |

Both blockers are now resolved, and they were in different layers: the schema representation
(§2–§6) and the definition of what the corpus supervises (§7).

## 2. Why wrapping the parameter map is interpretation, not inference

xLAM's dataset card documents the source contract at the pinned revision
`26d14ebfe18b1f7b524bd39b404b50af5dc97866`:

> `parameters` (object): An object representing the parameters required by the tool.
> * Each parameter is represented as a key-value pair, **where the key is the parameter name**
>   and the value is an object with the following properties: `type` (string), `description`
>   (string), `required` (boolean).
>
> `answers` … `arguments` (object): … Each argument is represented as a key-value pair, **where
> the key is the parameter name** and the value is the corresponding value.

The retained derivative independently confirms it. For every record with a call, the call's
argument keys were compared against the corresponding tool's parameter-map keys:

```
call-argument key sets that are a subset of the parameter-map keys : 3,150
call-argument key sets that are not                            : 0
```

Argument keys are parameter names, so the map they index is a property-definition map. Wrapping
it as `{"type": "object", "properties": <map>}` is therefore a source-contract transformation.

**Upstream data access.** The upstream repository is access-gated; a bounded range read of the
pinned revision returns HTTP 401. No substitute revision was fetched. The card is public
documentation and was read at the pinned revision; the audited bytes are the Canonical-v1
release, which retains xLAM's `parameters` maps unmodified.

## 3. The ambiguity, and why no global heuristic is acceptable

A bare property map is structurally indistinguishable from a schema. The corpus contains tools
whose parameters are literally named after schema keywords:

| Parameter name | Tools |
|---|---:|
| `type` | 2,551 |
| `format` | 2,179 |
| `items` | 670 |
| `description` | 99 |
| `maximum` | 30 |
| `properties` | 19 |
| `pattern` | 16 |

A rule like "any bare dict is a schema" would silently reinterpret roughly 5,500 tools. Detection
is therefore exact: a map is already a schema only if `parameters["type"]` is the **string**
`"object"` *and* `parameters["properties"]` is a dict. A bare map cannot satisfy both even when
it has a parameter named `type`, because the value under that key is a descriptor object.

That this matters is not hypothetical. The previous path accepted **428 tools** by misreading
them:

```
raw  : {"type": {"default": "game", "description": "The type of giveaways…", "type": "str"}}
OLD  : {"type": "string"}
NEW  : {"type": "object", "properties": {"type": {"type": "string", …}}, "required": ["type"]}
```

The old result declares that the tool takes a *string*, discarding the parameter name,
description and default — while the gold call is `{"type": "game"}`. Those 428 "accepted" tools
were silently corrupted, not correct, which is why the pre-fix canonical yield of 33 records was
not a partial success.

## 4. Requiredness: the source does not support it, so none is asserted

The documented `required` boolean **does not occur anywhere in this revision**: 0 of 357,766
parameter descriptors carry a `required` key. The only optionality marker present is the
`, optional` suffix on the type annotation (113,420 occurrences).

The first implementation inferred requiredness from that marker — optional if marked, otherwise
required — and justified it on omission rates:

| Parameter class | Observed | Omitted from gold call | Omission rate |
|---|---:|---:|---:|
| says `optional`, no default | 1,092 | 290 | **26.6%** |
| says `optional`, has default | 20,654 | 10,526 | **51.0%** |
| no marker, no default | 11,950 | 93 | **0.8%** |
| no marker, has default | 31,029 | 1,260 | **4.1%** |

**The corpus rejected that inference.** With unmarked parameters marked required, 3,050 of
57,342 canonically valid records failed the training boundary with
`SEM_ARGUMENT_INVALID: ARG_REQUIRED` — on parameters xLAM's own gold call had simply not passed:

```
schema required : [genres, limit, network_ids, page, regions, …]
gold call       : {network_ids: "1,8", release_date_start: 20110101, sort_by: "release_date_desc"}
→ ARG_REQUIRED: arguments.genres
```

A canonical schema that rejects the corpus's own gold arguments is wrong, and the failure was
attributable to this adapter rather than to the source. Requiredness is therefore **not asserted
at all**. Three pieces of evidence agree that the source does not support it: its documented
mechanism is absent, the absence of `, optional` is not a positive statement of requiredness, and
its own calls omit unmarked parameters in 0.8–4.1% of observations.

The optional marker is preserved as recorded provenance (`xlam_optional_parameters`,
`xlam_unmarked_parameters`) instead of being converted into an obligation, and
`xlam_required_lists_emitted` is reported as `0` so the choice is visible in the artifact. This
is the least-assumptive reading the source permits, and it is the reading under which no xLAM
record is quarantined for a defect this adapter invented.

A `default` is preserved verbatim as a schema `default` but does **not** imply optionality:
`''` is the single most common default value (13,785 of 55,281) and 25.2% of defaults are
placeholder-like. Treating a placeholder as an optionality signal would be inference, not
interpretation. This ambiguity is recorded here rather than resolved silently.

## 5. Annotation grammar

Types are Python-flavoured strings, parsed by a small recursive-descent parser. `eval` is never
used. The complete observed vocabulary is 30 distinct strings over 357,766 parameters.

**Supported** — `str`/`string`/`text` → `string`; `int`/`integer`/`long` → `integer`;
`float`/`double`/`number` → `number`; `bool`/`boolean` → `boolean`; `dict`/`Dict`/`object` →
`object`; `list`/`List` → `array`; `List[T]` recursively to any depth including
`List[List[int]]`; bare `List` → untyped `array`; homogeneous `Tuple[T, …]` → fixed-length array
via `minItems`/`maxItems`.

**Quarantined with reason codes**, never approximated:

| Reason | Parameters | Records lost |
|---|---:|---:|
| `XLAM_TYPE_UNSUPPORTED_UNION` | 1,241 (`List[Union[int, float]]`) | 1,216 |
| `XLAM_TYPE_UNSUPPORTED_CALLABLE` | 584 (`Callable[[float], float]`) | 529 |
| `XLAM_TYPE_UNSUPPORTED_SET` | 566 (`set`) | 283 |

Total: 2,391 parameters, 2,028 records (3.4%). `set` is quarantined because JSON has no unique
collection and `uniqueItems` is not in the canonical keyword set; the union form
`Union[int, float]` is quarantined per the instruction that unions remain quarantined, though it
would be exactly representable as `number`.

## 6. Rule distribution and identity

| Rule | Applications |
|---|---:|
| `XLAM_SCALAR_ALIAS_NORMALIZED` | 338,800 |
| `XLAM_REQUIREDNESS_NOT_ASSERTED` | 340,713 |
| `XLAM_DEFAULT_PRESERVED` | 254,039 |
| `XLAM_PARAMETER_MAP_WRAPPED_AS_OBJECT` | 157,834 |
| `XLAM_OPTIONAL_SUFFIX_PRESERVED` | 111,718 |
| `XLAM_GENERIC_ANNOTATION_EXPANDED` | 21,084 |
| `XLAM_BARE_COLLECTION_NORMALIZED` | 3,408 |

Parameters: 340,713 total (111,718 carrying an `optional` marker, 228,995 unmarked); `required`
lists emitted: **0**.

Corpus fingerprint over the 57,342 accepted canonical records:
`4d4906568de073f3a6724dd1ae87ef32557876bfa3cc493e1fee10075e80d25e`.

(The earlier `b8e8bacb…` fingerprint in this report's history covered the same records before the
supervision contract added `metadata.supervision`, which changes the canonical hash of every
record. The current value is the one the audit emits.)
Duplicates: 0. Input shard digests are recorded per shard in the JSON artifact.

## 7. Trainability: 0 → 56,090, via the supervision contract

The remaining blocker was never a parser defect. xLAM's format is `query` + `tools` +
`answers`: a single turn naming the call to make, with no tool-result turn. OpenGrad's trajectory
policy required every call to be resolved, so it rejected the entire source — correctly for a
corpus claiming a complete trajectory, and incorrectly for one whose objective is next-call
prediction.

With the supervision contract in place (`reports/SUPERVISION_CONTRACT_REPORT.md`), xLAM declares
`CALL_PREDICTION` and the terminal call is the supervised target:

| | Before the contract | After |
|---|---:|---:|
| Records | 59,370 | 59,370 |
| Schema-valid | 57,342 | 57,342 |
| `SEM_UNRESOLVED_CALL` | 56,111 | **0** |
| `SEM_ARGUMENT_INVALID` | 1,231 | 1,231 |
| **Trainable** | **0** | **56,090 (94.5%)** |
| Tool-call targets | 0 | 57,342 |

Every accepted record is `CALL_PREDICTION` (57,342). The arithmetic closes exactly:

```text
57,342 schema-valid
 - 1,231 argument-invalid   (upstream placeholder defects, unchanged)
 -    21 target-truncated   (exceed the 2,048-token window)
 = 56,090 trainable
```

### What remains quarantined, and why

| Cause | Records | Attribution |
|---|---:|---|
| `XLAM_TYPE_UNSUPPORTED_UNION` | 1,216 | this adapter refuses `Union[int, float]` |
| `XLAM_TYPE_UNSUPPORTED_CALLABLE` | 529 | this adapter refuses `Callable[...]` |
| `XLAM_TYPE_UNSUPPORTED_SET` | 283 | this adapter refuses `set` (no JSON equivalent) |
| `SEM_ARGUMENT_INVALID` | 1,231 | **upstream data quality** |
| `TARGET_TRUNCATED` | 21 | window, not source |

2,028 records (3.4%) are unrepresentable at the schema layer and were never approximated; 1,231
(2.1%) are genuine upstream defects where the gold call's value contradicts the type the same
record declares — `dough` declared `object` and passed `"prepared_dough"`, `books` declared
`array` and passed `"<user-provided-books-data>"`. Those read exactly like the upstream project's
own disclosure of ~5% inaccurate arguments, and they are not recoverable without inventing data.

**No fabricated turn.** The 56,090 accepted records render exactly their messages: two turns,
`user` and `assistant`, with no `<|im_start|>tool` turn and no placeholder observation. The
contract changes which turns are *expected*, never which turns *exist*.

## 8. Reconciliation with the briefing's figures

The briefing cited 166,762 bare parameter maps plus 19 already-proper JSON Schema objects.
Measured from the retained derivative: **166,781 total tools**, which matches that sum exactly.

The split does not reproduce. Measured here: **166,781 bare parameter maps, 0 already-proper
objects**. The retained Canonical-v1 derivative contains no schema-shaped xLAM parameter block,
so the "19 proper" cannot be verified from available bytes; if those records exist, they are
either in the gated upstream revision or were dropped during v1 materialization. This is recorded
as an open discrepancy rather than reconciled by assumption.

## 9. Tests and reproduction

86 tests in `tests/data/test_xlam_normalization.py` cover every supported scalar, generic and
optional form; parameters literally named `type`/`format`/`items`/`properties`; empty, nested and
absent parameter maps; unsupported and malformed inputs; required/optional behaviour; idempotence
and determinism (including rule ordering); canonical validation after normalization; that the
generic validator still rejects an unwrapped bare map; and adapter-level provenance.

```
python scripts/audit_xlam_normalization.py --output reports/data/xlam-normalization-forensics.json
python -m pytest tests/data/test_xlam_normalization.py -q
```
