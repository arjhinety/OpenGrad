# OpenPapers workflow

OpenPapers is the first-level research server for OpenGrad ([repository](https://github.com/arjhinety/OpenPapers), release `v1.0.0`, commit `227fd2b76c86825d5faef39afe5195f6b0e362c0` on `master`). It is an Apache-2.0 MCP server for scholarly retrieval, paper ingestion, and reproducible research workflows; its release quality is gated by a five-level test program with recorded evidence ([test plan](https://github.com/arjhinety/OpenPapers/blob/master/docs/testing.md)).

It provides replaceable adapters for arXiv, Crossref, OpenAlex, Semantic Scholar, GitHub, and Hugging Face; preserves DOI/arXiv/provider identities; discovers citations and implementations; reads bounded paper and repository content without executing it; and records evidence, uncertainty, conflicts, and provider failures. It can run over stdio or Streamable HTTP and uses local SQLite or PostgreSQL/pgvector storage.

## First-level research server

OpenGrad deliberately does not pre-download a fixed paper corpus before training begins: which papers matter only becomes clear once dataset materialization, training, and evaluation are underway. OpenPapers is the first stop for every literature need during active research:

1. A question arises during research (a method claim, a training parameter, a benchmark definition).
2. OpenPapers resolves it on demand through bounded retrieval with preserved source identity, locator, and uncertainty.
3. The retrieved evidence is verified against the primary source before it enters OpenGrad records.

This just-in-time strategy keeps the OpenGrad repository small, avoids speculative bulk downloads that may never be used, and still satisfies provenance rules: every claim traces to a primary source with an explicit locator and uncertainty status. Identifier-shaped queries (arXiv IDs, DOIs, and their URL forms) are resolved natively by OpenPapers' identifier probing, which was evaluated at 1.0 resolution rate on the recorded live run.

## Research workflow

1. Search OpenPapers by title or identifier.
2. Resolve the candidate with a provider-native identifier (`arXiv`, DOI, or OpenAlex ID).
3. Retrieve paper metadata and, when needed, bounded paper sections or implementation files.
4. Record the source URL, revision/blob, locator, and evidence strength in `docs/references/papers.yaml` and an implementation note.
5. Check the canonical paper, project, or dataset page before treating a field as verified. OpenPapers improves provenance and discovery; it does not replace primary-source judgment.
6. Keep recommendations and implementation hypotheses separate from factual evidence.

A citation discovered through OpenPapers enters OpenGrad only after its source identity and relevant fields are recorded. Missing fields remain `null` with an explicit status and a `checked_sources` list.

## Safe local stdio connection

Clone and build OpenPapers separately using its own documented instructions (Node.js >=22.5). In an MCP client, point the server at the built entry point. The repository includes a non-secret conceptual example at `integrations/openpapers/mcp.example.json`:

```json
{
  "servers": {
    "openpapers": {
      "type": "stdio",
      "command": "node",
      "args": ["/path/to/openpapers/dist/mcp/server.js"]
    }
  }
}
```

Replace the path locally; do not commit absolute machine paths, tokens, or `.env` files. Client configuration keys vary, so follow the client and OpenPapers documentation for the final wrapper.

## Optional boundary

OpenGrad remains reproducible without OpenPapers: direct arXiv, official repository, Hugging Face, Crossref, OpenAlex, or conference sources may be used. OpenPapers is not a Python/runtime dependency, is not required by CI, and does not gate future training or evaluation runs.
