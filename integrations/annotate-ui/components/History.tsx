import type { Change } from "@/lib/types";
import { display, formatTime, shortHash } from "@/lib/value";

const VERB: Record<string, string> = {
  label: "Labeled",
  relabel: "Changed label",
  skip: "Skipped",
  flag: "Flagged",
  unflag: "Unflagged",
  undo: "Undid",
};

/** This session's recorded changes to the item: when, by whom, from what, why, and the state hashes. */
export function History({ changes }: { changes: Change[] }) {
  if (!changes.length) return <div className="empty" style={{ fontSize: 12.5 }}>No changes recorded yet.</div>;
  return (
    <ol className="history">
      {[...changes].reverse().map((change) => (
        <li key={change.entry_sha256}>
          <div className="what">
            {VERB[change.action] ?? change.action}
            {change.action === "relabel" || change.action === "undo"
              ? ` ${display(change.from)} → ${display(change.to)}`
              : change.action === "label"
                ? ` ${display(change.to)}`
                : ""}
          </div>
          <div className="meta">
            {formatTime(change.recorded_at)} · {change.actor_id}
          </div>
          {change.reason ? <div>“{change.reason}”</div> : null}
          <div className="hash" title={`entry ${change.entry_sha256}`}>
            {shortHash(change.before_sha256)} → {shortHash(change.after_sha256)}
          </div>
        </li>
      ))}
    </ol>
  );
}
