"use client";

import { useEffect, useState } from "react";
import { AdjudicateView } from "@/components/AdjudicateView";
import { AnnotateView } from "@/components/AnnotateView";
import { api, ApiError } from "@/lib/api";
import type { AppState } from "@/lib/types";

// The server decides the mode (one pass, or adjudication) when it starts; the page only follows it.
export default function Page() {
  const [state, setState] = useState<AppState | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .state()
      .then((next) => {
        setState(next);
        document.title = `${next.task.title} · ${next.mode === "annotate" ? next.session?.session_id : "adjudication"}`;
      })
      .catch((reason: unknown) => setError(reason instanceof ApiError ? reason.message : String(reason)));
  }, []);

  if (error) {
    return (
      <div className="screen">
        <h1>Cannot load the annotation task</h1>
        <p>{error}</p>
        <pre>opengrad-annotate start pdet-v1 --annotator YOUR_ID --session pass-a</pre>
      </div>
    );
  }
  if (!state) return <div className="screen empty">Loading task…</div>;
  return state.mode === "annotate" ? <AnnotateView state={state} /> : <AdjudicateView state={state} />;
}
