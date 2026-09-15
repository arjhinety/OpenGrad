"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { AdjudicationView, AppState, InstructionDoc, QueueItem, Value } from "@/lib/types";
import { canonical, display, fieldsFor, formatTime, initialValue, problems, shortHash } from "@/lib/value";
import { ContextPane } from "./ContextPane";
import { ExtraFields, type FormHandle, PrimaryControl } from "./Form";
import { RubricPanel, type RubricTarget } from "./RubricPanel";
import { Header, useKeyboard } from "./Shell";

const SINGLE_CHOICE = new Set(["single_label", "binary", "pairwise", "rating"]);
type Notice = { kind: "good" | "bad" | "warn" | "info"; text: string } | null;
type QueueFilter = "undecided" | "decided" | "all";

function PassCard({ task, pass, other }: { task: AppState["task"]; pass: AdjudicationView["passes"][number]; other?: Value | null }) {
  const value = pass.annotation?.value ?? null;
  const differs = (key: string) => other !== undefined && JSON.stringify(value?.[key] ?? null) !== JSON.stringify(other?.[key] ?? null);
  const extra = [...task.extra_fields];
  return (
    <div className="pass">
      <div className="who">
        {pass.session_id} · {pass.annotator_id}
        {pass.annotation?.flagged ? <span className="flag"> · ⚑ flagged</span> : null}
      </div>
      <div className={`value${differs(task.primary_key) ? " diff" : ""}`}>{display(value?.[task.primary_key])}</div>
      <dl>
        {extra.map((field) => (
          <span key={field.key} style={{ display: "contents" }}>
            <dt>{field.label}</dt>
            <dd className={differs(field.key) ? "diff" : ""}>{display(value?.[field.key])}</dd>
          </span>
        ))}
        {pass.annotation?.note ? (
          <>
            <dt>Note</dt>
            <dd>{pass.annotation.note}</dd>
          </>
        ) : null}
        <dt>State</dt>
        <dd className="mono">{shortHash(pass.annotation?.state_sha256)} · rev {pass.annotation?.revision ?? "—"}</dd>
      </dl>
    </div>
  );
}

export function AdjudicateView({ state }: { state: AppState }) {
  const task = state.task;
  const info = state.adjudication!;
  const frozen = info.frozen;
  const fields = fieldsFor(task, true);

  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [filter, setFilter] = useState<QueueFilter>("undecided");
  const [view, setView] = useState<AdjudicationView | null>(null);
  const [value, setValue] = useState<Value>({});
  const [rationale, setRationale] = useState("");
  const [issues, setIssues] = useState<{ field: string; message: string }[]>([]);
  const [notice, setNotice] = useState<Notice>(null);
  const [busy, setBusy] = useState(false);
  const [drawer, setDrawer] = useState<"items" | "rubric" | null>(null);
  const [docs, setDocs] = useState<InstructionDoc[]>([]);
  const [target, setTarget] = useState<RubricTarget | null>(null);
  const form = useRef<FormHandle>(null);
  const rationaleRef = useRef<HTMLTextAreaElement>(null);

  const fail = (error: unknown) => setNotice({ kind: "bad", text: error instanceof ApiError ? error.message : String(error) });

  const refreshQueue = useCallback(async () => {
    const result = await api.queue();
    setQueue(result.items);
    return result.items;
  }, []);

  const load = useCallback(
    async (id: string, message: Notice = null) => {
      try {
        const next = await api.adjudicationItem(id);
        setView(next);
        // Never pre-fill from either pass: the adjudicator decides from the item and the rubric.
        setValue(initialValue(task, fields, next.adjudication?.value ?? null));
        setRationale(next.adjudication?.rationale ?? "");
        setIssues([]);
        setNotice(message);
        (document.activeElement as HTMLElement | null)?.blur?.();
        document.querySelector(".context")?.scrollTo({ top: 0 });
      } catch (error) {
        fail(error);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [task],
  );

  useEffect(() => {
    void (async () => {
      const items = await refreshQueue();
      const first = items.find((item) => !item.adjudicated) ?? items[0];
      if (first) await load(first.item_id);
    })().catch(fail);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (drawer === "rubric" && !docs.length) api.instructions().then((r) => setDocs(r.documents)).catch(fail);
  }, [drawer, docs.length]);

  const save = async (candidate: Value = value) => {
    if (!view || frozen || busy) return;
    const found = problems(task, fields, candidate);
    if (!rationale.trim()) found.push({ field: "__rationale", message: "An adjudication rationale is required" });
    if (found.length) {
      setIssues(found);
      const first = fields.find((field) => found.some((issue) => issue.field === field.key));
      if (first) form.current?.focus(first.key);
      else rationaleRef.current?.focus();
      return;
    }
    setBusy(true);
    try {
      await api.adjudicate(view.item_id, canonical(task, fields, candidate), rationale);
      const items = await refreshQueue();
      const position = items.findIndex((item) => item.item_id === view.item_id);
      const rotation = [...items.slice(position + 1), ...items.slice(0, Math.max(position, 0))];
      const next = rotation.find((item) => !item.adjudicated);
      if (next) await load(next.item_id, { kind: "good", text: "Adjudication saved" });
      else await load(view.item_id, { kind: "good", text: "Saved. Every item in the queue is decided." });
    } catch (error) {
      fail(error);
    } finally {
      setBusy(false);
    }
  };

  const choose = (next: Value) => {
    if (frozen) return;
    setValue(next);
    setIssues([]);
    if (SINGLE_CHOICE.has(task.task_type)) void save(next);
  };

  const step = (delta: number) => {
    if (!view || !queue.length) return;
    const position = queue.findIndex((item) => item.item_id === view.item_id);
    const next = queue[(position + delta + queue.length) % queue.length];
    if (next) void load(next.item_id);
  };

  useKeyboard({
    shortcuts: task.shortcuts,
    onLabel: (label) => view && choose({ ...value, [task.primary_key]: label }),
    onSave: () => void save(),
    onEscape: () => setDrawer(null),
    keys: {
      n: () => step(1),
      ArrowRight: () => step(1),
      p: () => step(-1),
      ArrowLeft: () => step(-1),
      i: () => setDrawer((current) => (current === "rubric" ? null : "rubric")),
      "/": () => setDrawer("items"),
      e: () => rationaleRef.current?.focus(),
    },
  });

  const decided = queue.filter((item) => item.adjudicated).length;
  const shown = queue.filter((item) => filter === "all" || (filter === "decided") === item.adjudicated);
  const problemKeys = new Set(issues.map((issue) => issue.field));
  const kind = info.kind === "adjudication" ? "Adjudication" : "Single-annotator review";
  const passes = view?.passes ?? [];

  return (
    <>
      <Header
        title={task.title}
        who={`${kind} · ${info.sessions.map((s) => `${s.session_id} (${s.annotator_id})`).join(" vs ")} · adjudicator ${info.adjudicator_id}`}
        drawer={drawer}
        onItems={() => setDrawer(drawer === "items" ? null : "items")}
        onRubric={() => setDrawer(drawer === "rubric" ? null : "rubric")}
        itemsLabel="Queue"
        extra={
          <span className="num" style={{ whiteSpace: "nowrap" }}>
            <b>{decided}</b> / {queue.length} decided
          </span>
        }
      />
      {frozen ? <div className="banner frozen">These sessions are frozen into a gold package. Adjudication is read-only.</div> : null}
      {!queue.length ? (
        <div className="screen">
          <h1>Nothing to decide</h1>
          <p>
            {info.kind === "adjudication"
              ? "The two passes agree on every item under the task's disagreement keys."
              : "No item matches the task's single-annotator re-read rule."}
          </p>
        </div>
      ) : (
        <div className="layout">
          <main className="context">{view ? <ContextPane task={task} item={view} /> : <div className="empty">Loading…</div>}</main>
          <aside className="panel" aria-label="Adjudication">
            <div className="section">
              <h3>{info.kind === "adjudication" ? (view?.disagreement ? "The passes disagree" : "The passes agree") : "Pass under review"}</h3>
              <div className="passes">
                {passes.map((pass, index) => (
                  <PassCard
                    key={pass.session_id}
                    task={task}
                    pass={pass}
                    other={passes.length === 2 ? passes[1 - index].annotation?.value ?? null : undefined}
                  />
                ))}
              </div>
              {view && queue.find((item) => item.item_id === view.item_id)?.reasons.length ? (
                <div className="notice info">
                  In the queue because: {queue.find((item) => item.item_id === view.item_id)?.reasons.join(", ")}
                </div>
              ) : null}
            </div>
            <div className="decide">
              <h2>Adjudicated label</h2>
              <PrimaryControl
                task={task}
                fields={fields}
                value={value}
                savedValue={view?.adjudication?.value}
                problemKeys={problemKeys}
                disabled={frozen || busy || !view}
                onChange={setValue}
                onChoose={choose}
                onSubmit={() => void save()}
                onDefinition={
                  task.instructions.length
                    ? (label) => {
                        setDrawer("rubric");
                        setTarget({ label, nonce: Date.now() });
                      }
                    : undefined
                }
              />
              {issues.length ? (
                <div className="notice bad" role="alert">
                  <ul>
                    {issues.map((issue) => (
                      <li key={issue.field + issue.message}>{issue.message}</li>
                    ))}
                  </ul>
                </div>
              ) : notice ? (
                <div className={`notice ${notice.kind}`}>{notice.text}</div>
              ) : null}
            </div>
            <div className="section">
              <ExtraFields
                ref={form}
                task={task}
                fields={fields}
                value={value}
                savedValue={view?.adjudication?.value}
                problemKeys={problemKeys}
                disabled={frozen || !view}
                onChange={setValue}
                onChoose={choose}
                onSubmit={() => void save()}
              />
              <div className={`form-row${problemKeys.has("__rationale") ? " problem" : ""}`}>
                <label htmlFor="rationale">Adjudication rationale</label>
                <textarea
                  id="rationale"
                  ref={rationaleRef}
                  rows={3}
                  disabled={frozen || !view}
                  value={rationale}
                  onChange={(event) => setRationale(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      void save();
                    }
                  }}
                  placeholder="Why this label, citing the rule that decided it"
                />
              </div>
              <button className="btn primary" style={{ width: "100%" }} disabled={frozen || busy || !view} onClick={() => void save()}>
                Save adjudication and next <kbd>Ctrl+Enter</kbd>
              </button>
              {view?.adjudication ? (
                <div className="saved-state">
                  <span>Decided: {display(view.adjudication.value[task.primary_key])}</span>
                  <span>by {view.adjudication.adjudicator_id}</span>
                  <span>rev {view.adjudication.revision}</span>
                  <span>{formatTime(view.adjudication.updated_at)}</span>
                  {view.adjudication.flag ? <span className="flag">{view.adjudication.flag}</span> : null}
                </div>
              ) : null}
              <div className="nav-row">
                <button className="btn" onClick={() => step(-1)}>
                  ← Previous <kbd>p</kbd>
                </button>
                <button className="btn" onClick={() => step(1)}>
                  Next <kbd>n</kbd> →
                </button>
              </div>
            </div>
          </aside>
        </div>
      )}
      {drawer === "items" ? (
        <aside className="drawer" aria-label="Queue">
          <header>
            <h2>Queue</h2>
            <button className="btn small" onClick={() => setDrawer(null)}>
              Close <kbd>Esc</kbd>
            </button>
          </header>
          <div className="controls">
            <div className="row">
              <select value={filter} onChange={(event) => setFilter(event.target.value as QueueFilter)} aria-label="Queue filter">
                <option value="undecided">undecided</option>
                <option value="decided">decided</option>
                <option value="all">all</option>
              </select>
              <span className="empty" style={{ fontSize: 12.5 }}>
                {shown.length} shown · {decided} of {queue.length} decided
              </span>
            </div>
          </div>
          <div className="scroll">
            <ul className="list">
              {shown.map((item) => (
                <li key={item.item_id}>
                  <button className={item.item_id === view?.item_id ? "current" : ""} onClick={() => void load(item.item_id)}>
                    <span className="n">#{item.index + 1}</span>
                    <span className={`dot ${item.adjudicated ? "labeled" : ""}`} />
                    <span className="id">{item.item_id}</span>
                    <span className="v">{item.values.map(display).join(" / ")}</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </aside>
      ) : null}
      {drawer === "rubric" ? <RubricPanel docs={docs} target={target} onClose={() => setDrawer(null)} /> : null}
    </>
  );
}
