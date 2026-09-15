"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type {
  AppState,
  InstructionDoc,
  ItemView,
  ListedItem,
  Progress,
  QueueProgress,
  ReferenceProgress,
  Value,
} from "@/lib/types";
import {
  canonical,
  display,
  fieldsFor,
  formatTime,
  initialValue,
  primaryChosen,
  problems,
  sameValue,
  shortHash,
} from "@/lib/value";
import { ContextPane } from "./ContextPane";
import { ExtraFields, type FormHandle, PrimaryControl } from "./Form";
import { History } from "./History";
import { Navigator, type NavigatorHandle, type StatusFilter } from "./Navigator";
import { RubricPanel, type RubricTarget } from "./RubricPanel";
import { Header, useKeyboard } from "./Shell";

type Drawer = "items" | "rubric" | null;
type Notice = { kind: "good" | "bad" | "warn" | "info"; text: string } | null;
interface Draft {
  value: Value;
  note: string;
  reason: string;
}

const SINGLE_CHOICE = new Set(["single_label", "binary", "pairwise", "rating"]);

export function AnnotateView({ state }: { state: AppState }) {
  const task = state.task;
  const session = state.session!;
  const frozen = session.frozen;
  const fields = fieldsFor(task, false);

  const [progress, setProgress] = useState<Progress | undefined>(state.progress);
  const [item, setItem] = useState<ItemView | null>(null);
  const [value, setValue] = useState<Value>({});
  const [note, setNote] = useState("");
  const [reason, setReason] = useState("");
  const [issues, setIssues] = useState<{ field: string; message: string }[]>([]);
  const [notice, setNotice] = useState<Notice>(null);
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);
  const [drawer, setDrawer] = useState<Drawer>(null);
  const [docs, setDocs] = useState<InstructionDoc[]>([]);
  const [target, setTarget] = useState<RubricTarget | null>(null);
  const [status, setStatus] = useState<StatusFilter>("all");
  const [filters, setFilters] = useState<Record<string, string>>({});
  const [listed, setListed] = useState<ListedItem[]>([]);
  const [queues, setQueues] = useState<QueueProgress[]>(state.queues ?? []);
  const [reference, setReference] = useState<ReferenceProgress | null>(state.reference ?? null);
  // Which order to take items in: "" is every remaining item in frozen order, otherwise a pinned review
  // queue. Remembered per browser only, as a convenience.
  const queueKey = `opengrad-annotate:${task.task_id}:${session.session_id}:queue`;
  const [queue, setQueue] = useState<string>(() => {
    try {
      const stored = window.localStorage.getItem(queueKey) ?? "";
      return (state.queues ?? []).some((entry) => entry.name === stored) ? stored : "";
    } catch {
      return "";
    }
  });

  const drafts = useRef(new Map<string, Draft>());
  // A chosen label the form held back (e.g. UNKNOWN before its ambiguity status is set).
  const held = useRef(false);
  const form = useRef<FormHandle>(null);
  const nav = useRef<NavigatorHandle>(null);
  const noteRef = useRef<HTMLTextAreaElement>(null);

  const saved = item?.annotation?.status === "labeled" ? item.annotation : null;
  const edited =
    saved !== null && (!sameValue(saved.value, canonical(task, fields, value)) || (saved.note ?? "") !== note.trim());

  const fail = (error: unknown) => {
    setNotice({ kind: "bad", text: error instanceof ApiError ? error.message : String(error) });
  };

  const refreshList = useCallback(async () => {
    const params: Record<string, string> = { status, ...filters, ...(queue ? { queue } : {}) };
    const result = await api.items(params);
    setListed(result.items);
    return result.items;
  }, [status, filters, queue]);

  const refreshCounts = useCallback(async () => {
    try {
      const next = await api.state();
      if (next.progress) setProgress(next.progress);
      setQueues(next.queues ?? []);
      setReference(next.reference ?? null);
    } catch {
      /* the counts refresh again after the next change */
    }
  }, []);

  const stash = () => {
    if (item) drafts.current.set(item.item_id, { value, note, reason });
  };

  const load = useCallback(
    async (id: string, message: Notice = null) => {
      try {
        const view = await api.item(id);
        setItem(view);
        const draft = drafts.current.get(id);
        setValue(draft?.value ?? initialValue(task, fields, view.annotation?.value));
        setNote(draft?.note ?? view.annotation?.note ?? "");
        setReason(draft?.reason ?? "");
        setIssues([]);
        held.current = false;
        setNotice(message);
        // A new item starts with shortcuts live: focus left in the previous item's text field would
        // otherwise swallow the next label key as typed text.
        (document.activeElement as HTMLElement | null)?.blur?.();
        window.history.replaceState(null, "", `#${encodeURIComponent(id)}`);
        document.querySelector(".context")?.scrollTo({ top: 0 });
        document.querySelector(".panel")?.scrollTo({ top: 0 });
      } catch (error) {
        fail(error);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [task],
  );

  const open = (id: string | null, message: Notice = null) => {
    if (!id) return;
    stash();
    void load(id, message);
  };

  // First load: the item in the URL, else where the session stopped.
  useEffect(() => {
    void (async () => {
      const fromHash = decodeURIComponent(window.location.hash.slice(1));
      if (fromHash) {
        try {
          await api.item(fromHash);
          await load(fromHash);
          return;
        } catch {
          /* fall through to the resume point */
        }
      }
      let next = await api.next(undefined, queue || undefined);
      if (!next.item_id && queue) next = await api.next();
      if (next.item_id) {
        await load(next.item_id);
        return;
      }
      setDone(true);
      const all = await api.items({ status: "all" });
      if (all.items.length) await load(all.items[0].item_id);
    })().catch(fail);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (drawer === "items") void refreshList().catch(fail);
  }, [drawer, refreshList]);

  useEffect(() => {
    if (drawer === "rubric" && !docs.length) {
      api
        .instructions()
        .then((result) => setDocs(result.documents))
        .catch(fail);
    }
  }, [drawer, docs.length]);

  const rememberQueue = (name: string) => {
    setQueue(name);
    try {
      window.localStorage.setItem(queueKey, name);
    } catch {
      /* a per-browser convenience; the order still works for this visit */
    }
  };

  /** Where to go after `after`: the next item in the chosen queue, else the server's next in frozen order. */
  const following = async (after: string, serverNext: string | null): Promise<{ id: string | null; switched: boolean }> => {
    if (!queue) return { id: serverNext, switched: false };
    const inQueue = await api.next(after, queue);
    if (inQueue.item_id && inQueue.item_id !== after) return { id: inQueue.item_id, switched: false };
    rememberQueue("");
    return { id: serverNext, switched: true };
  };

  const advance = async (after: string, serverNext: string | null) => {
    void refreshCounts();
    // With a narrowing filter open (e.g. flagged), stay inside that list; otherwise resume at the next
    // unlabeled item, which is what "automatically advance" means for a first pass.
    if (drawer === "items" && status !== "all" && status !== "unlabeled") {
      const list = await refreshList();
      const position = list.findIndex((entry) => entry.item_id === after);
      const nextInList = position >= 0 ? list[position + 1] : list[0];
      if (nextInList) return load(nextInList.item_id, { kind: "good", text: "Saved" });
    }
    if (drawer === "items") void refreshList();
    const { id, switched } = await following(after, serverNext);
    const message = switched ? `Saved. Nothing is left to label in ${queue}; continuing with all remaining items.` : "Saved";
    if (id && id !== after) {
      setDone(false);
      return load(id, { kind: "good", text: message });
    }
    setDone(true);
    return load(after, { kind: "good", text: "Saved. Every item in this session is labeled." });
  };

  const chooseQueue = (name: string) => {
    rememberQueue(name);
    void (async () => {
      const next = await api.next(undefined, name || undefined);
      if (next.item_id) open(next.item_id, { kind: "info", text: name ? `Taking items from ${name}` : "Taking every remaining item, in order" });
      else setNotice({ kind: "info", text: name ? `Nothing is left to label in ${name}.` : "Nothing is left to label." });
    })().catch(fail);
  };

  const save = async (candidate: Value = value) => {
    if (!item || frozen || busy) return;
    const found = problems(task, fields, candidate);
    if (found.length) {
      setIssues(found);
      const first = fields.find((field) => found.some((issue) => issue.field === field.key));
      if (first) form.current?.focus(first.key);
      return;
    }
    const clean = canonical(task, fields, candidate);
    if (saved && sameValue(saved.value, clean) && (saved.note ?? "") === note.trim()) {
      drafts.current.delete(item.item_id);
      const next = await api.next(item.item_id);
      return advance(item.item_id, next.item_id);
    }
    setBusy(true);
    try {
      const result = await api.annotate(item.item_id, {
        value: clean,
        note,
        replace: saved !== null,
        expected_revision: item.annotation?.revision ?? 0,
        reason: saved !== null ? reason : undefined,
      });
      drafts.current.delete(item.item_id);
      setProgress(result.progress);
      await advance(item.item_id, result.next_item_id);
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
    if (SINGLE_CHOICE.has(task.task_type)) {
      // Saved at once when the form is complete; otherwise held until the missing selection is made.
      held.current = problems(task, fields, next).length > 0;
      void save(next);
    }
  };

  const changeFields = (next: Value) => {
    const changed = fields.filter((field) => next[field.key] !== value[field.key]);
    setValue(next);
    setIssues([]);
    // A held label saves as soon as a selection completes it: UNKNOWN, then its ambiguity status.
    // Typing in a text field never saves by itself.
    if (
      held.current &&
      changed.length > 0 &&
      changed.every((field) => field.type === "select") &&
      primaryChosen(task, next) &&
      problems(task, fields, next).length === 0
    ) {
      held.current = false;
      void save(next);
    }
  };

  const skip = async () => {
    if (!item || frozen) return;
    try {
      const result = await api.skip(item.item_id, note);
      setProgress(result.progress);
      void refreshCounts();
      drafts.current.delete(item.item_id);
      const { id } = await following(item.item_id, result.next_item_id);
      if (id && id !== item.item_id) await load(id, { kind: "info", text: "Skipped" });
      else await load(item.item_id, { kind: "info", text: "Skipped; nothing else is waiting." });
    } catch (error) {
      fail(error);
    }
  };

  const toggleFlag = async () => {
    if (!item || frozen) return;
    try {
      const flagged = !(item.annotation?.flagged ?? false);
      const result = await api.flag(item.item_id, flagged, note);
      setProgress(result.progress);
      void refreshCounts();
      const view = await api.item(item.item_id);
      setItem(view);
      setNotice({ kind: "info", text: flagged ? "Flagged for review" : "Flag removed" });
    } catch (error) {
      fail(error);
    }
  };

  const undo = async () => {
    if (frozen) return;
    try {
      const result = await api.undo();
      setProgress(result.progress);
      void refreshCounts();
      drafts.current.delete(result.item_id);
      setDone(false);
      await load(result.item_id, { kind: "warn", text: "Undid the last change (recorded in the change log)" });
    } catch (error) {
      fail(error);
    }
  };

  const jump = async (text: string) => {
    const trimmed = text.trim();
    if (/^\d+$/.test(trimmed)) {
      const all = await api.items({ status: "all" });
      const match = all.items[Number(trimmed) - 1];
      if (match) return open(match.item_id);
    }
    open(trimmed);
  };

  const showDefinition = (label: string) => {
    setDrawer("rubric");
    setTarget({ label, nonce: Date.now() });
  };

  useKeyboard({
    shortcuts: task.shortcuts,
    onLabel: (label) => item && choose({ ...value, [task.primary_key]: label }),
    onSave: () => void save(),
    onEscape: () => setDrawer(null),
    keys: {
      n: () => open(item?.next_id ?? null),
      ArrowRight: () => open(item?.next_id ?? null),
      p: () => open(item?.prev_id ?? null),
      ArrowLeft: () => open(item?.prev_id ?? null),
      u: () => void undo(),
      s: () => void skip(),
      f: () => void toggleFlag(),
      i: () => setDrawer((current) => (current === "rubric" ? null : "rubric")),
      "/": () => {
        setDrawer("items");
        setTimeout(() => nav.current?.focusJump(), 0);
      },
      e: () => {
        const first = fields[0];
        if (first) form.current?.focus(first.key);
        else noteRef.current?.focus();
      },
    },
  });

  const problemKeys = new Set(issues.map((issue) => issue.field));

  return (
    <>
      <Header
        title={task.title}
        who={`${session.session_id} · ${session.annotator_id}`}
        progress={progress}
        drawer={drawer}
        onItems={() => setDrawer(drawer === "items" ? null : "items")}
        onRubric={() => setDrawer(drawer === "rubric" ? null : "rubric")}
        itemsLabel="Items"
      />
      {frozen ? (
        <div className="banner frozen">
          This session was frozen into a gold package at {formatTime(session.frozen_at)} ({session.frozen_manifest}). It is
          read-only.
        </div>
      ) : null}
      <div className="layout">
        <main className="context">{item ? <ContextPane task={task} item={item} /> : <div className="empty">Loading…</div>}</main>
        <aside className="panel" aria-label="Decision">
          {queues.length || reference ? (
            <div className="section review">
              {queues.length ? (
                <div className="form-row">
                  <label htmlFor="queue">Order</label>
                  <select
                    id="queue"
                    value={queue}
                    disabled={!item}
                    onChange={(event) => {
                      event.target.blur();
                      chooseQueue(event.target.value);
                    }}
                  >
                    <option value="">All remaining items · frozen order</option>
                    {queues.map((entry) => (
                      <option key={entry.name} value={entry.name}>
                        {entry.name} · {entry.completed} / {entry.total} labeled
                      </option>
                    ))}
                  </select>
                </div>
              ) : null}
              {reference ? (
                <>
                  <div className="tally" aria-label="Review progress">
                    <span>Human-reviewed</span>
                    <b>{reference.human_reviewed}</b>
                    <span>Provisional, model only</span>
                    <b>{reference.provisional_model_only}</b>
                    {reference.unlabeled ? (
                      <>
                        <span>Unlabeled</span>
                        <b>{reference.unlabeled}</b>
                      </>
                    ) : null}
                    <span>Remaining for human review</span>
                    <b>{reference.remaining_for_human_review}</b>
                    <span>Flagged by you</span>
                    <b>{reference.flagged}</b>
                  </div>
                  <div className="help">
                    Provisional items are not human-reviewed. A model&apos;s judgment is never shown here, before or
                    after you label.
                  </div>
                </>
              ) : null}
            </div>
          ) : null}
          <div className="decide">
            <h2>{saved ? "Label (saved — choosing again changes it)" : "Label"}</h2>
            <PrimaryControl
              task={task}
              fields={fields}
              value={value}
              savedValue={saved?.value}
              problemKeys={problemKeys}
              disabled={frozen || busy || !item}
              onChange={(next) => {
                setValue(next);
                setIssues([]);
              }}
              onChoose={choose}
              onSubmit={() => void save()}
              onDefinition={task.instructions.length ? showDefinition : undefined}
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
              <div className={`notice ${notice.kind}`} role="status">
                {notice.text}
              </div>
            ) : null}
          </div>

          <div className="section">
            <ExtraFields
              ref={form}
              task={task}
              fields={fields}
              value={value}
              savedValue={saved?.value}
              problemKeys={problemKeys}
              disabled={frozen || !item}
              onChange={changeFields}
              onChoose={choose}
              onSubmit={() => void save()}
            />
            <div className="form-row">
              <label htmlFor="note">
                Note <span className="opt">(optional)</span>
              </label>
              <textarea
                id="note"
                ref={noteRef}
                rows={2}
                disabled={frozen || !item}
                value={note}
                onChange={(event) => setNote(event.target.value)}
                placeholder="Anything worth keeping about this item"
              />
            </div>
            {saved && edited ? (
              <div className="form-row">
                <label htmlFor="reason">
                  Reason for the change <span className="opt">(optional, written to the change log)</span>
                </label>
                <input
                  id="reason"
                  disabled={frozen}
                  value={reason}
                  onChange={(event) => setReason(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") void save();
                  }}
                  placeholder={`Was ${display(saved.value?.[task.primary_key])}`}
                />
              </div>
            ) : null}
            <button className="btn primary" style={{ width: "100%" }} disabled={frozen || busy || !item} onClick={() => void save()}>
              {saved ? (edited ? "Save change and next" : "Keep and next") : "Save and next"} <kbd>Ctrl+Enter</kbd>
            </button>
            {done && !frozen ? (
              <div className="notice good">
                Every item in this session is labeled. Changes stay possible until the gold freeze and each one is
                recorded. Next: <code>opengrad-annotate export</code>, or adjudication once all passes are complete.
              </div>
            ) : null}
          </div>

          <div className="section">
            <div className="actions">
              <button className={`btn${item?.annotation?.flagged ? " on" : ""}`} disabled={frozen || !item} onClick={() => void toggleFlag()}>
                ⚑ {item?.annotation?.flagged ? "Flagged" : "Flag"} <kbd>f</kbd>
              </button>
              <button className="btn" disabled={frozen || !item} onClick={() => void skip()}>
                Skip <kbd>s</kbd>
              </button>
              <button className="btn" disabled={frozen} onClick={() => void undo()}>
                Undo <kbd>u</kbd>
              </button>
            </div>
            <div className="nav-row">
              <button className="btn" disabled={!item?.prev_id} onClick={() => open(item?.prev_id ?? null)}>
                ← Previous <kbd>p</kbd>
              </button>
              <button className="btn" disabled={!item?.next_id} onClick={() => open(item?.next_id ?? null)}>
                Next <kbd>n</kbd> →
              </button>
            </div>
            {item?.annotation ? (
              <div className="saved-state">
                <span>
                  {item.annotation.status === "labeled"
                    ? `Saved: ${display(item.annotation.value?.[task.primary_key])}`
                    : item.annotation.status === "skipped"
                      ? "Skipped"
                      : "Not labeled"}
                </span>
                <span>revision {item.annotation.revision}</span>
                <span>{formatTime(item.annotation.updated_at)}</span>
                <span className="mono" title={item.annotation.state_sha256}>
                  state {shortHash(item.annotation.state_sha256)}
                </span>
              </div>
            ) : (
              <div className="saved-state">Not labeled yet</div>
            )}
          </div>

          <div className="section">
            <h3>Change history</h3>
            <History changes={item?.history ?? []} />
          </div>

          <div className="section">
            <h3>Keys</h3>
            <div className="keys">
              <span>
                {task.shortcuts.map((shortcut) => (
                  <kbd key={shortcut.key} style={{ marginRight: 3 }}>
                    {shortcut.key}
                  </kbd>
                ))}
              </span>
              <span>choose a label: saves and moves on once the form is complete</span>
              <kbd>Enter</kbd>
              <span>save from a text field · Shift+Enter new line</span>
              <kbd>e</kbd>
              <span>edit the first field</span>
              <span>
                <kbd>n</kbd> <kbd>p</kbd>
              </span>
              <span>next / previous item</span>
              <span>
                <kbd>s</kbd> <kbd>f</kbd> <kbd>u</kbd>
              </span>
              <span>skip · flag · undo</span>
              <span>
                <kbd>i</kbd> <kbd>/</kbd>
              </span>
              <span>rubric · jump to item</span>
              <kbd>Esc</kbd>
              <span>leave a field / close a panel</span>
            </div>
          </div>
        </aside>
      </div>
      {drawer === "items" ? (
        <Navigator
          ref={nav}
          task={task}
          options={state.filter_options}
          progress={progress}
          status={status}
          filters={filters}
          items={listed}
          currentId={item?.item_id ?? null}
          queue={queue}
          onStatus={setStatus}
          onFilter={(name, filterValue) => setFilters((current) => ({ ...current, [name]: filterValue }))}
          onOpen={(id) => open(id)}
          onJump={(text) => void jump(text).catch(fail)}
          onClose={() => setDrawer(null)}
        />
      ) : null}
      {drawer === "rubric" ? <RubricPanel docs={docs} target={target} onClose={() => setDrawer(null)} /> : null}
    </>
  );
}
