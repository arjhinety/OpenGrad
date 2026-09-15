import type { NextConfig } from "next";
import { PHASE_DEVELOPMENT_SERVER } from "next/constants";

// `next build` emits a static site into out/, which `opengrad-annotate serve` hosts beside its API, so
// annotation needs one process. `next dev` cannot host the API, so in development only it proxies /api
// to a running `opengrad-annotate serve` (rewrites are not available to a static export).
const api = process.env.ANNOTATE_API ?? "http://127.0.0.1:8765";

export default function config(phase: string): NextConfig {
  if (phase === PHASE_DEVELOPMENT_SERVER) {
    return {
      async rewrites() {
        return [{ source: "/api/:path*", destination: `${api}/api/:path*` }];
      },
    };
  }
  return { output: "export", images: { unoptimized: true } };
}
