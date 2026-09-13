## Change type

- [ ] infrastructure
- [ ] reproduction
- [ ] negative result
- [ ] dataset/benchmark audit
- [ ] model/runtime support

## Evidence

Describe tests and provenance. Do not include fabricated or unverified results.

## Guardrails

For a change that reports a result, or touches a card, the site or a status line, go through the
"Before publishing" checklist in [`docs/research/GUARDRAILS.md`](../docs/research/GUARDRAILS.md).
In particular:

- [ ] Every number is recomputed from committed JSON; one quantity has one value everywhere (G14)
- [ ] Every row states its partition and n and uses the same population (G9)
- [ ] Caveats, regressions and uncertainty travel with the numbers, on every surface (G10–G12)
- [ ] Each correction is applied to every copy: repository, site, cards, ledgers (G15)
