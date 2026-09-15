# annotate-ui

The browser UI for `opengrad-annotate` — Next.js (App Router, TypeScript), built as a static export that
the Python server hosts. It renders; it does not decide. Every rule — validation, isolation, freezes,
provenance — lives in `src/opengrad/annotation/`, and the server re-checks every write.

```bash
npm install
npm run build        # -> out/, served by `opengrad-annotate start` at http://127.0.0.1:8765/
npm run dev          # http://localhost:3000, proxies /api to ANNOTATE_API (default http://127.0.0.1:8765)
npm run typecheck
```

See [`docs/ANNOTATION_TOOL.md`](../../docs/ANNOTATION_TOOL.md).
