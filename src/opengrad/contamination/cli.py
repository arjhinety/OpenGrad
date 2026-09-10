"""Contamination screening, Level-5 adjudication, and quarantine CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from opengrad.contamination.audit import (
    AUDIT_PATH,
    QUARANTINE_PATH,
    VERDICT_CONTAMINATED,
    VERDICT_INCIDENTAL,
    VERDICT_PENDING,
    AuditEvaluation,
    apply_verdict,
    benchmark_fingerprint,
    build_quarantine,
    default_reviewer,
    evaluate_audit,
    load_audit,
    load_quarantine,
    save_audit,
    save_quarantine,
    sync_audit,
    training_corpus_fingerprint,
)
from opengrad.contamination.heldout import OUTPUT as REPORT_PATH

_VERDICT_ALIASES = {
    "c": VERDICT_CONTAMINATED,
    "contaminated": VERDICT_CONTAMINATED,
    "i": VERDICT_INCIDENTAL,
    "incidental": VERDICT_INCIDENTAL,
    "incidental_overlap": VERDICT_INCIDENTAL,
    "s": VERDICT_PENDING,
    "skip": VERDICT_PENDING,
    "pending": VERDICT_PENDING,
    "p": VERDICT_PENDING,
}


def _print_summary(evaluation: AuditEvaluation, quarantine_count: int, audit_path: Path) -> None:
    print()
    print("Level-5 adjudication status")
    print("-" * 60)
    print(f"  scanner findings      : {evaluation.total}")
    print(f"  adjudicated           : {evaluation.resolved}")
    print(f"  contaminated          : {evaluation.contaminated}")
    print(f"  incidental overlap    : {evaluation.incidental}")
    print(f"  quarantined examples  : {quarantine_count}")
    print(f"  unresolved (pending)  : {evaluation.pending}")
    if evaluation.stale:
        print(f"  stale judgments       : {evaluation.stale}")
    print(f"  5_manual_audit        : {evaluation.level_5}")
    print(f"  contamination status  : {evaluation.effective_status}")
    if evaluation.problems:
        print("  problems:")
        for problem in evaluation.problems:
            print(f"    - {problem}")
    if evaluation.pending:
        print()
        print("  Remaining items block readiness. Run:")
        print("    opengrad-contamination adjudicate")
    if evaluation.contaminated and evaluation.pending == 0 and evaluation.problems:
        print()
        print("  Apply quarantine for contaminated verdicts, then rescan:")
        print("    opengrad-contamination quarantine --apply")
        print("    opengrad-contamination heldout-screen")
    print(f"  audit artifact        : {audit_path}")


def _show_item(index: int, total: int, item) -> None:
    print()
    print("=" * 74)
    print(f"  ITEM {index}/{total}   {item.record_id}")
    print("=" * 74)
    print()
    print("  HELD-OUT BENCHMARK ITEM")
    print(f"    prompt            : {item.heldout_text!r}")
    print(f"    expected decision : {item.expected_decision}")
    if item.tools:
        print(f"    tools available   : {', '.join(item.tools)}")
    if item.candidates:
        print("    candidate answers :")
        for name, text in sorted(item.candidates.items()):
            marker = "  <== GOLD" if name == item.expected_decision else ""
            print(f"      {name:18s} {str(text)[:130]}{marker}")
    if item.stale:
        print(f"    STALE             : {item.stale_reason}")
    print()

    print("  MATCHED TRAINING RECORDS")
    if not item.training_evidence:
        print("    (no training evidence captured)")
    for evidence in item.training_evidence:
        print(f"    - {evidence.get('record_id')}")
        print(f"        source          : {evidence.get('source')}")
        print(f"        behavior        : {evidence.get('behavior_decision')}")
        print(f"        prompt          : {evidence.get('prompt')!r}")
        if evidence.get("assistant"):
            print(f"        assistant       : {evidence.get('assistant')!r}")
        if evidence.get("tool_calls"):
            print(f"        tool_calls      : {json.dumps(evidence.get('tool_calls'))[:300]}")
    print()
    print("  VERDICT: [c] Contaminated   [i] Incidental overlap   [s] Skip/Pending   [q] Quit")
    print("  (Contaminated = exclude from B0 and all later held-out evaluations)")


def _run_interactive(artifact, audit_path: Path, quarantine_path: Path, root: Path, reviewer: str) -> int:
    unresolved = [item for item in artifact.items if not item.resolved()]
    if not unresolved:
        print("No unresolved findings.")
        return 0
    print(f"{len(unresolved)} unresolved finding(s). Reviewer: {reviewer}")

    for index, item in enumerate(unresolved, 1):
        _show_item(index, len(unresolved), item)
        while True:
            try:
                choice = input("  choice> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nStopped. Recorded verdicts are preserved.")
                save_audit(audit_path, artifact)
                return 0
            if choice in {"q", "quit", "exit"}:
                save_audit(audit_path, artifact)
                print("Stopped. Recorded verdicts are preserved.")
                return 0
            if choice in _VERDICT_ALIASES:
                verdict = _VERDICT_ALIASES[choice]
                break
            print("  Please enter c, i, s, or q.")
        try:
            reason = input("  reason (optional)> ").strip()
        except (EOFError, KeyboardInterrupt):
            reason = ""
        apply_verdict(artifact, item.record_id, verdict, reason=reason, reviewer=reviewer)
        save_audit(audit_path, artifact)
        print(f"  recorded: {verdict}")

    _report_after(artifact, audit_path, quarantine_path, root)
    return 0


def _report_after(artifact, audit_path: Path, quarantine_path: Path, root: Path) -> None:
    report = _load_report(root)
    quarantine = load_quarantine(quarantine_path)
    evaluation = evaluate_audit(
        report,
        artifact,
        quarantine,
        benchmark_fp=benchmark_fingerprint(root),
        training_fp=training_corpus_fingerprint(root),
    )
    _print_summary(evaluation, len(load_quarantine(quarantine_path).record_ids()), audit_path)
    print(f"  quarantine list       : {quarantine_path}")


def _load_report(root: Path) -> dict:
    path = root / REPORT_PATH
    if not path.is_file():
        raise SystemExit(
            f"scanner report not found: {path}\n"
            "Run `opengrad-contamination heldout-screen` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _adjudicate(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    report = _load_report(root)
    audit_path = root / AUDIT_PATH
    quarantine_path = root / QUARANTINE_PATH

    artifact = sync_audit(
        report,
        existing=load_audit(audit_path),
        benchmark_fp=report.get("benchmark_fingerprint"),
        training_fp=report.get("training_corpus_fingerprint"),
        quarantined_ids=load_quarantine(quarantine_path).record_ids(),
    )
    save_audit(audit_path, artifact)

    if args.status:
        quarantine = load_quarantine(quarantine_path)
        evaluation = evaluate_audit(
            report,
            artifact,
            quarantine,
            benchmark_fp=benchmark_fingerprint(root),
            training_fp=training_corpus_fingerprint(root),
        )
        _print_summary(evaluation, len(quarantine.record_ids()), audit_path)
        return 0

    if args.list:
        for item in artifact.items:
            print(f"{item.verdict:20s} {item.record_id}  {item.heldout_text[:60]!r}")
        return 0

    if args.id:
        if not args.verdict:
            raise SystemExit("--verdict is required with --id")
        verdict = _VERDICT_ALIASES.get(args.verdict.strip().lower())
        if verdict is None:
            raise SystemExit(
                "verdict must be one of: contaminated, incidental, pending "
                "(aliases: c, i, s)"
            )
        try:
            item = apply_verdict(
                artifact,
                args.id,
                verdict,
                reason=args.reason or "",
                reviewer=args.reviewer or default_reviewer(),
            )
        except KeyError as exc:
            raise SystemExit(f"unknown finding id: {args.id}") from exc
        save_audit(audit_path, artifact)
        print(f"recorded {item.verdict} for {item.record_id}")
        _report_after(artifact, audit_path, quarantine_path, root)
        return 0

    return _run_interactive(
        artifact, audit_path, quarantine_path, root, args.reviewer or default_reviewer()
    )


def _quarantine(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    audit_path = root / AUDIT_PATH
    quarantine_path = root / QUARANTINE_PATH
    artifact = load_audit(audit_path)
    if artifact is None:
        raise SystemExit(f"audit artifact not found: {audit_path}")

    if args.status:
        quarantine = load_quarantine(quarantine_path)
        print(f"quarantined examples: {len(quarantine.record_ids())}")
        for entry in quarantine.excluded:
            print(f"  {entry.get('record_id')}  {entry.get('reason', '')[:60]}")
        return 0

    if not args.apply:
        raise SystemExit("pass --apply to regenerate the quarantine list, or --status to view it")

    quarantine = build_quarantine(artifact)
    save_quarantine(quarantine_path, quarantine)
    print(f"quarantined {len(quarantine.excluded)} example(s) -> {quarantine_path}")
    for entry in quarantine.excluded:
        print(f"  {entry['record_id']}")
    print()
    print("Rescan so the generated report reflects the excluded examples:")
    print("  opengrad-contamination heldout-screen")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="opengrad-contamination",
        description="Contamination screening and Level-5 adjudication for frozen evaluation namespaces",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    screen = sub.add_parser(
        "heldout-screen",
        help="measure contamination levels 1-4 between the held-out set and the training corpus",
    )
    screen.add_argument("--root", type=Path, default=Path.cwd())
    screen.add_argument(
        "--release-dir",
        type=Path,
        default=None,
        help="canonical training release directory (default .release/hf/toolpolicy-canonical-v1)",
    )
    screen.add_argument("--max-df", type=int, default=1000, help="prune shingles above this training document frequency")
    screen.add_argument("--min-shingles", type=int, default=25, help="minimum held-out prompt length for containment flagging")
    screen.add_argument("--jaccard", type=float, default=0.5)
    screen.add_argument("--containment", type=float, default=0.6)
    screen.add_argument("--edit-similarity", type=float, default=0.8)
    screen.add_argument("--json", action="store_true", help="print the full report")

    adjudicate = sub.add_parser(
        "adjudicate",
        aliases=["audit"],
        help="record human Level-5 verdicts (interactive, or non-interactive with --id)",
    )
    adjudicate.add_argument("--root", type=Path, default=Path.cwd())
    adjudicate.add_argument("--id", help="finding id, e.g. when2call-mcq:<uuid> (non-interactive)")
    adjudicate.add_argument("--verdict", help="contaminated | incidental | pending")
    adjudicate.add_argument("--reason", help="short justification stored with the verdict")
    adjudicate.add_argument("--reviewer", help="reviewer identity (defaults to $OPENGRAD_REVIEWER or $USER)")
    adjudicate.add_argument("--status", action="store_true", help="print counts and exit")
    adjudicate.add_argument("--list", action="store_true", help="list findings and verdicts")

    quarantine = sub.add_parser(
        "quarantine",
        help="derive the held-out quarantine list from CONTAMINATED verdicts",
    )
    quarantine.add_argument("--root", type=Path, default=Path.cwd())
    quarantine.add_argument("--apply", action="store_true", help="regenerate the quarantine artifact")
    quarantine.add_argument("--status", action="store_true", help="show the current quarantine list")

    args = parser.parse_args()
    if args.command == "heldout-screen":
        from opengrad.contamination.heldout import screen_and_write

        report, path = screen_and_write(
            args.root.resolve(),
            release_dir=args.release_dir,
            max_df=args.max_df,
            min_shingles=args.min_shingles,
            jaccard_threshold=args.jaccard,
            containment_threshold=args.containment,
            edit_threshold=args.edit_similarity,
        )
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(f"status: {report['status']}")
            print(f"held-out records: {report['counts']['heldout_records']}")
            print(f"training records: {report['counts']['training_records']}")
            print(f"audit queue size: {report['audit_queue_size']}")
            print(f"scan fingerprint: {report['scan_fingerprint']}")
            print(f"report: {path}")
        return 0
    if args.command in {"adjudicate", "audit"}:
        return _adjudicate(args)
    if args.command == "quarantine":
        return _quarantine(args)
    return 1
