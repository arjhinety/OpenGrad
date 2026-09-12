# Tokenizer divergence — permanent regression fixtures

Six confirmatory prompts tokenize differently under the pinned HF tokenizer and stock llama.cpp.
That failure is recorded as `ENGINE_TOKENIZER_PARITY_FAILED` and is **not** being fixed. These
fixtures exist so the *characterisation* cannot drift unnoticed.

**These tests do not require the two tokenizers to agree.** They already disagree; asserting
agreement would quietly delete the finding. `test_fixtures_do_not_assert_tokenizer_agreement`
actively guards against a future edit that "fixes" the divergence by editing the fixtures.

## Root cause

| | letter-run branch of the pre-tokenizer regex |
|---|---|
| pinned checkpoint `tokenizer.json` | `[^\r\n\p{L}\p{N}]?` **`\p{L}+`** |
| stock llama.cpp `QWEN35` | `[^\r\n\p{L}\p{N}]?` **`[\p{L}\p{M}]+`** |

llama.cpp folds Unicode combining marks (`\p{M}`) into the letter run; the pinned tokenizer does
not. All six affected prompts are **Thai**, whose tone marks and vowel signs are non-spacing marks
(`Mn`), so llama.cpp consistently emits **fewer** tokens.

```
'ช่วยหาคุณแม่'
  pinned HF : ['ช', '่วยหาค', 'ุณแม', '่']   4 chunks
  llama.cpp : ['ช่วยหาคุณแม่']              1 chunk
```

Across all 1,277 confirmatory prompts the set predicted by this regex difference equals the
observed mismatch set **exactly** — no false positives, no false negatives. The declared
pre-tokenizer is `qwen35`, a recognised type, not a `default` fallback. The HF `NFC` normalizer
llama.cpp does not apply is inactive here: all 3,650 frozen prompts are already NFC.

## The six fixtures

| example | expected | HF tok | llama.cpp tok | Δ | decision (HF ids) | decision (llama.cpp tok) | behaviour |
|---|---|---:|---:|---:|---|---|---|
| `2bf3162c…` | CALL | 390 | 377 | −13 | `ANSWER` | `CALL` | **changes** |
| `8643445b…` | CLARIFY | 357 | 344 | −13 | `ANSWER` | `CALL` | **changes** |
| `b20b6ecc…` | UNSUPPORTED | 35 | 22 | −13 | `ANSWER` | `ANSWER` | inert |
| `c9cf98bc…` | CLARIFY | 439 | 413 | −26 | `CLARIFY` | `UNSUPPORTED` | **changes** |
| `e2425bb7…` | CALL | 439 | 419 | −20 | `ANSWER` | `ANSWER` | inert |
| `f9f7c30f…` | UNSUPPORTED | 51 | 31 | −20 | `ANSWER` | `ANSWER` | inert |

**3 of 6 change the evaluated policy decision** — verdict
`TOKENIZER_DIVERGENCE_BEHAVIORALLY_MATERIAL`. Not systematically favourable to either side: one
example is correct only under llama.cpp, one only under HF, one is wrong under both.

Downstream effect on the full 1,277-example metrics is under 0.003 absolute on every metric and
changes no gate verdict. **Material at the example level, negligible at the metric level** — both
are true and both are reported.

## Why the isolation experiment is valid

Comparing llama.cpp to vLLM would change engine *and* tokenizer together and could not attribute a
difference to either. So the engine was held constant: same llama.cpp `b10919`, same BF16 GGUF,
same greedy sampler, asked twice — once with prompt text, once with the pinned HF tokenizer's
**token ids submitted directly**.

That arm is only valid if the server evaluates exactly the ids supplied, so it was asserted, not
assumed:

| | |
|---|---|
| token ids submitted | 390 |
| `tokens_evaluated` reported | 390 |
| literal (no injection) | **true** |
| `/detokenize` round-trip reproduces the prompt | **true** |

The probe aborts rather than being interpreted if this fails. Given the earlier `<|im_start|>` BOS
collision in this project, this check is not optional.

**No special token is inserted, dropped, duplicated, or reinterpreted** on any fixture:
`<|im_start|>` counts, `<|im_end|>` counts, EOS counts and the first token (`248045`) are identical
on both sides. The divergence is confined to ordinary text tokens, and both streams reconverge
after the Thai span.

## What the tests lock

`tests/evaluation/test_tokenizer_divergence_regression.py` (9 tests):

- the fixture set is exactly these six, with the recorded parity and materiality verdicts
- **exactly three** change behaviour, and exactly which three
- prompt text still matches its recorded sha256
- **HF token ids recompute exactly** from the pinned tokenizer — a live check, not stored-vs-stored
- llama.cpp is shorter on every fixture, and by exactly the recorded delta
- no special token is inserted, dropped, or reinterpreted
- divergence is localized and reconverges
- the direct-token path was validated before interpretation
- the fixtures still represent a real disagreement

The llama.cpp arm cannot be recomputed locally (no llama.cpp on the host); its ids are held against
the recorded measurement and re-measured only when the Modal probe re-runs.

## If these tests fail

Do **not** edit the fixtures to make them pass. A failure means the characterisation changed — a
tokenizer revision, a llama.cpp bump, a template change — and the correct response is to re-run
`modal run scripts/modal/gguf_study.py --stage divergence-probe`, re-characterise deliberately, and
record a new fixture version.

Fixture file: `tests/fixtures/tokenizer_divergence_v1.json`
Generator: `scripts/build_tokenizer_regression_fixtures.py`
