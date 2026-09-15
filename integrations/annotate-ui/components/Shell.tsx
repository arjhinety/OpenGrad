"use client";

import { useEffect, useRef } from "react";
import type { Progress } from "@/lib/types";

export function Header({
  title,
  who,
  progress,
  drawer,
  onItems,
  onRubric,
  itemsLabel,
  extra,
}: {
  title: string;
  who: string;
  progress?: Progress;
  drawer: string | null;
  onItems(): void;
  onRubric(): void;
  itemsLabel: string;
  extra?: React.ReactNode;
}) {
  return (
    <header className="header">
      <button className={`btn small${drawer === "items" ? " on" : ""}`} onClick={onItems}>
        {itemsLabel} <kbd>/</kbd>
      </button>
      <span className="title">{title}</span>
      <span className="who">{who}</span>
      <span className="spacer" />
      {extra}
      {progress ? (
        <div className="progress" aria-label="Progress">
          <div className="bar" aria-hidden>
            <span style={{ width: `${progress.percent}%` }} />
          </div>
          <span className="num">
            <b>{progress.completed}</b> / {progress.total} completed · {progress.percent.toFixed(1)}%
          </span>
        </div>
      ) : null}
      <button className={`btn small${drawer === "rubric" ? " on" : ""}`} onClick={onRubric}>
        Rubric <kbd>i</kbd>
      </button>
    </header>
  );
}

interface KeyboardOptions {
  shortcuts: { key: string; label: string }[];
  onLabel(label: string): void;
  onSave(): void;
  onEscape(): void;
  keys: Record<string, () => void>;
}

function typingIn(target: EventTarget | null): boolean {
  const element = target as HTMLElement | null;
  return Boolean(element?.closest?.("input, textarea, select, [contenteditable='true']"));
}

/** Global shortcuts. Keys typed into a field belong to the field; Ctrl+Enter and Esc work everywhere. */
export function useKeyboard(options: KeyboardOptions) {
  const latest = useRef(options);
  latest.current = options;
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const current = latest.current;
      const typing = typingIn(event.target);
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        current.onSave();
        return;
      }
      if (event.key === "Escape") {
        if (typing) (event.target as HTMLElement).blur();
        else current.onEscape();
        return;
      }
      if (typing || event.ctrlKey || event.metaKey || event.altKey || event.repeat) return;
      const shortcut = current.shortcuts.find((item) => item.key === event.key || item.key.toLowerCase() === event.key.toLowerCase());
      if (shortcut) {
        event.preventDefault();
        current.onLabel(shortcut.label);
        return;
      }
      const action = current.keys[event.key];
      if (action) {
        event.preventDefault();
        action();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}
