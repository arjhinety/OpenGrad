"use client";

import { forwardRef, useImperativeHandle, useRef, useState } from "react";
import type { ListedItem, Progress, TaskPublic } from "@/lib/types";
import { display } from "@/lib/value";

export const STATUS_FILTERS = ["all", "unlabeled", "completed", "skipped", "flagged", "unknown"] as const;
export type StatusFilter = (typeof STATUS_FILTERS)[number];

export interface NavigatorHandle {
  focusJump(): void;
}

interface Props {
  task: TaskPublic;
  options: Record<string, string[]>;
  progress: Progress | undefined;
  status: StatusFilter;
  filters: Record<string, string>;
  items: ListedItem[];
  currentId: string | null;
  /** A pinned review queue the list follows, in its order; empty for every item in frozen order. */
  queue?: string;
  onStatus(status: StatusFilter): void;
  onFilter(name: string, value: string): void;
  onOpen(id: string): void;
  onJump(text: string): void;
  onClose(): void;
}

export const Navigator = forwardRef<NavigatorHandle, Props>(function Navigator(props, ref) {
  const { task, options, progress, status, filters, items, currentId, queue } = props;
  const [search, setSearch] = useState("");
  const jump = useRef<HTMLInputElement>(null);
  useImperativeHandle(ref, () => ({ focusJump: () => jump.current?.focus() }));
  const needle = search.trim().toLowerCase();
  const shown = needle ? items.filter((item) => item.item_id.toLowerCase().includes(needle)) : items;
  const hasUnknown = task.unknown_labels.length > 0;

  return (
    <aside className="drawer" aria-label="Items">
      <header>
        <h2>Items</h2>
        <button className="btn small" onClick={props.onClose}>
          Close <kbd>Esc</kbd>
        </button>
      </header>
      <div className="controls">
        <div className="row">
          <input
            ref={jump}
            value={search}
            placeholder="Jump to item id or number…"
            aria-label="Jump to item"
            onChange={(event) => setSearch(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && search.trim()) {
                const exact = shown.length === 1 ? shown[0].item_id : search.trim();
                props.onJump(exact);
              }
            }}
          />
        </div>
        <div className="row">
          <select value={status} onChange={(event) => props.onStatus(event.target.value as StatusFilter)} aria-label="Status">
            {STATUS_FILTERS.filter((name) => name !== "unknown" || hasUnknown).map((name) => (
              <option key={name} value={name}>
                {name === "unknown" ? task.unknown_labels.join("/") : name}
              </option>
            ))}
          </select>
          {task.filters.map((name) => (
            <select
              key={name}
              value={filters[name] ?? ""}
              onChange={(event) => props.onFilter(name, event.target.value)}
              aria-label={name}
            >
              <option value="">{name}: any</option>
              {(options[name] ?? []).map((value) => (
                <option key={value} value={value}>
                  {name}: {value}
                </option>
              ))}
            </select>
          ))}
          <span className="empty" style={{ fontSize: 12.5 }}>
            {shown.length} shown
          </span>
        </div>
        {queue ? (
          <div className="row empty" style={{ fontSize: 12.5 }}>
            The {queue} queue, in its order. Numbers are positions in the full population.
          </div>
        ) : null}
        {progress && Object.keys(progress.label_counts).length ? (
          <div className="counts">
            {Object.entries(progress.label_counts).map(([label, count]) => (
              <span key={label} style={{ display: "contents" }}>
                <span>{label}</span>
                <span>{count}</span>
              </span>
            ))}
            <span>remaining</span>
            <span>{progress.remaining}</span>
            <span className="caption">
              Counts track progress only. They are not a target — never choose a label to balance classes.
            </span>
          </div>
        ) : null}
      </div>
      <div className="scroll">
        <ul className="list">
          {shown.map((item) => (
            <li key={item.item_id}>
              <button className={item.item_id === currentId ? "current" : ""} onClick={() => props.onOpen(item.item_id)}>
                <span className="n">#{item.index + 1}</span>
                <span className={`dot ${item.status}`} title={item.status} />
                <span className="id">{item.item_id}</span>
                <span className="v">
                  {item.flagged ? <span className="flag" title="flagged">⚑ </span> : null}
                  {item.status === "labeled" ? display(item.value) : item.status === "skipped" ? "skipped" : ""}
                </span>
              </button>
            </li>
          ))}
        </ul>
        {!shown.length ? <div className="screen" style={{ margin: "32px auto" }}>No items match.</div> : null}
      </div>
    </aside>
  );
});
