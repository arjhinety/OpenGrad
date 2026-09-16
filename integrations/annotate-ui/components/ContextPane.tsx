"use client";

import type { DisplayField, ItemCore, TaskPublic } from "@/lib/types";
import { ToolList } from "./ToolList";

function Text({ value, missing }: { value: unknown; missing?: string }) {
  if (value === null || value === undefined)
    return <div className="body empty">{missing || "Missing in the source record."}</div>;
  if (typeof value === "string") {
    // An empty or whitespace-only response is itself evidence (e.g. non-substantive), so say so plainly.
    if (!value.trim()) return <div className="body empty">(empty — {value.length} whitespace characters)</div>;
    return <div className="body">{value}</div>;
  }
  return <pre className="json">{JSON.stringify(value, null, 2)}</pre>;
}

function Conversation({ value }: { value: unknown }) {
  if (!Array.isArray(value) || !value.length) return <Text value={value} />;
  return (
    <div className="body">
      {value.map((turn, index) => {
        const message = (turn ?? {}) as { role?: unknown; content?: unknown };
        return (
          <div className="turn" key={index}>
            <div className="role">{String(message.role ?? "turn")}</div>
            <div>{typeof message.content === "string" ? message.content : JSON.stringify(message.content ?? turn)}</div>
          </div>
        );
      })}
    </div>
  );
}

// Structured tool calls of the turn being annotated: each call's name and its arguments, nothing else.
function Calls({ value }: { value: unknown[] }) {
  return (
    <div className="body">
      {value.map((raw, index) => {
        const call = (raw ?? {}) as { name?: unknown; arguments?: unknown };
        return (
          <div className="turn" key={index}>
            <div className="role">{String(call.name ?? "call")}</div>
            <pre className="json">{JSON.stringify(call.arguments ?? {}, null, 2)}</pre>
          </div>
        );
      })}
    </div>
  );
}

function Field({ field, value }: { field: DisplayField; value: unknown }) {
  let body: React.ReactNode;
  // A turn without calls shows no call section at all.
  if (field.render === "calls" && (!Array.isArray(value) || !value.length)) return null;
  if (!field.available) body = <div className="body empty">Not present in the source record.</div>;
  else if (field.render === "calls") body = <Calls value={value as unknown[]} />;
  else if (field.render === "tools") body = <ToolList tools={value} />;
  else if (field.render === "conversation") body = <Conversation value={value} />;
  else if (field.render === "json") body = <pre className="json">{JSON.stringify(value, null, 2)}</pre>;
  else body = <Text value={value} missing={field.missing_text} />;
  return (
    <section className={`field${field.emphasis ? " emphasis" : ""}`}>
      <div className="label">
        {field.label}
        {field.emphasis ? <span className="tag">being annotated</span> : null}
      </div>
      {body}
    </section>
  );
}

export function ContextPane({ task, item }: { task: TaskPublic; item: ItemCore }) {
  const candidates = new Set(task.candidates);
  const regular = task.display.filter((field) => !candidates.has(field.key));
  const compared = task.display.filter((field) => candidates.has(field.key));
  const copy = () => void navigator.clipboard?.writeText(item.item_id);
  return (
    <>
      <div className="item-head">
        <span className="index">
          Item {item.index + 1} <span style={{ color: "var(--muted)", fontWeight: 400 }}>of {item.total}</span>
        </span>
        <span className="chip id" title="Copy item id" onClick={copy}>
          {item.item_id}
        </span>
        {task.metadata.map((field) => {
          const value = item.metadata[field.key];
          if (value === null || value === undefined || value === "") return null;
          return (
            <span className="chip" key={field.key} title={`${field.label}: ${String(value)}`}>
              <b>{field.label}</b> {Array.isArray(value) ? value.join(", ") : String(value)}
            </span>
          );
        })}
      </div>
      {regular.map((field) => (
        <Field key={field.key} field={field} value={item.fields[field.key]} />
      ))}
      {compared.length ? (
        <div className="candidates">
          {compared.map((field, index) => (
            <Field
              key={field.key}
              field={{ ...field, label: `${String.fromCharCode(65 + index)} · ${field.label}` }}
              value={item.fields[field.key]}
            />
          ))}
        </div>
      ) : null}
    </>
  );
}
