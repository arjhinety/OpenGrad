"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { InstructionDoc } from "@/lib/types";

// The rubric is rendered from the task's instruction files, never restated in UI code: the files are
// the frozen protocol, and a copy here could silently drift from them.

interface Heading {
  level: number;
  text: string;
}

function headings(markdown: string): Heading[] {
  const found: Heading[] = [];
  let fenced = false;
  for (const line of markdown.split(/\r?\n/)) {
    if (/^\s*(```|~~~)/.test(line)) fenced = !fenced;
    if (fenced) continue;
    const match = /^(#{1,3})\s+(.*)$/.exec(line);
    if (match) found.push({ level: match[1].length, text: match[2].replace(/[`*_]/g, "").trim() });
  }
  return found;
}

function mentions(text: string, label: string): boolean {
  return new RegExp(`(^|[^A-Za-z0-9_])${label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}([^A-Za-z0-9_]|$)`).test(text);
}

export interface RubricTarget {
  label: string;
  nonce: number;
}

export function RubricPanel({
  docs,
  onClose,
  target,
}: {
  docs: InstructionDoc[];
  onClose(): void;
  target: RubricTarget | null;
}) {
  const [active, setActive] = useState(0);
  // A jump names its document; it is applied only once that document is the one rendered, so a heading
  // index is never resolved against the previous tab's DOM.
  const [request, setRequest] = useState<{ doc: number; index: number; nonce: number } | null>(null);
  const body = useRef<HTMLDivElement>(null);
  const toc = useMemo(() => docs.map((doc) => headings(doc.content)), [docs]);

  const scrollTo = (index: number) => {
    const nodes = body.current?.querySelectorAll("h1, h2, h3");
    const node = nodes?.[index] as HTMLElement | undefined;
    if (!node) return;
    node.scrollIntoView({ block: "start" });
    // Exactly one section is highlighted: the one just opened.
    body.current?.querySelectorAll(".flash").forEach((other) => other.classList.remove("flash"));
    void node.offsetWidth;
    node.classList.add("flash");
  };

  useEffect(() => {
    if (!target) return;
    for (let doc = 0; doc < docs.length; doc += 1) {
      const index = toc[doc].findIndex((heading) => mentions(heading.text, target.label));
      if (index >= 0) {
        setActive(doc);
        setRequest({ doc, index, nonce: target.nonce });
        return;
      }
    }
  }, [target, docs, toc]);

  useEffect(() => {
    body.current?.scrollTo({ top: 0 });
  }, [active]);

  useEffect(() => {
    if (!request || request.doc !== active) return;
    const frame = requestAnimationFrame(() => scrollTo(request.index));
    return () => cancelAnimationFrame(frame);
  }, [request, active]);

  const doc = docs[active];
  const byName = new Map(docs.map((item, index) => [item.path.split("/").pop() ?? item.path, index]));

  return (
    <aside className="drawer wide" aria-label="Rubric">
      <header>
        <h2>Rubric and instructions</h2>
        <button className="btn small" onClick={onClose}>
          Close <kbd>i</kbd>
        </button>
      </header>
      {docs.length > 1 ? (
        <nav className="rubric-tabs">
          {docs.map((item, index) => (
            <button key={item.path} className={index === active ? "on" : ""} onClick={() => setActive(index)}>
              {item.title}
            </button>
          ))}
        </nav>
      ) : null}
      {doc ? (
        <div className="rubric-body">
          <nav className="toc" aria-label="Sections">
            {toc[active].map((heading, index) => (
              <a
                key={index}
                href="#"
                className={`l${heading.level}`}
                title={heading.text}
                onClick={(event) => {
                  event.preventDefault();
                  scrollTo(index);
                }}
              >
                {heading.text}
              </a>
            ))}
          </nav>
          <div className="markdown" ref={body}>
            <div className="empty" style={{ fontSize: 12 }}>
              {doc.path}
            </div>
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                a({ href, children }) {
                  const file = (href ?? "").split("#")[0].split("/").pop() ?? "";
                  if (byName.has(file)) {
                    return (
                      <a
                        href="#"
                        onClick={(event) => {
                          event.preventDefault();
                          setActive(byName.get(file) ?? 0);
                        }}
                      >
                        {children}
                      </a>
                    );
                  }
                  if (href && /^https?:\/\//.test(href)) {
                    return (
                      <a href={href} target="_blank" rel="noopener noreferrer">
                        {children}
                      </a>
                    );
                  }
                  return <span title={href}>{children}</span>;
                },
              }}
            >
              {doc.content}
            </ReactMarkdown>
          </div>
        </div>
      ) : (
        <div className="screen">This task declares no instruction files.</div>
      )}
    </aside>
  );
}
