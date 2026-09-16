"""``opengrad-annotate``: check a task, start or resume a pass, adjudicate, report, export, freeze, verify.

    opengrad-annotate check  TASK                                          # read-only preflight
    opengrad-annotate start  TASK --annotator alice --session pass-a       # start or resume a pass
    opengrad-annotate start  TASK --annotator alice --session pass-a --dry-run   # checks only; writes nothing
    opengrad-annotate adjudicate TASK --sessions pass-a pass-b --adjudicator carol
    opengrad-annotate status TASK [--session pass-a]
    opengrad-annotate audit  TASK                                          # re-verify change logs
    opengrad-annotate export TASK --sessions pass-a pass-b [--out DIR]     # work-in-progress snapshot
    opengrad-annotate freeze-gold TASK --sessions pass-a pass-b            # final, locks the passes
    opengrad-annotate verify MANIFEST                                      # re-hash an exported package
    opengrad-annotate model-batch  TASK --session model-a --annotator model.X --defer-to pass-a
    opengrad-annotate model-ingest TASK --batch BATCH.json --answers ANSWERS.json
    opengrad-annotate reference TASK --sessions pass-a model-a             # composite counts, recomputed
    opengrad-annotate review-queue TASK --name NAME --sessions pass-a model-a --seed S --out FILE ...

TASK is a task id with a config at ``configs/annotation/<id>.yaml`` (e.g. ``pdet-v1``) or a path to a task
config. ``start`` (aliases ``serve``, ``resume``) is idempotent: running it again with the same session
resumes exactly where it stopped, because every label was committed to SQLite when it was made.
"""

from __future__ import annotations

import argparse
import json
import socket
import sqlite3
import sys
import webbrowser
from collections import Counter
from pathlib import Path
from typing import Any

from opengrad.annotation.config import TaskConfig, TaskConfigError, find_repo_root, load_task_config
from opengrad.annotation.export import (
    ExportError,
    IncompleteGoldError,
    default_out_dir,
    export_snapshot,
    freeze_gold,
    verify_package,
)
from opengrad.annotation.items import SourceError, load_source, project
from opengrad.annotation.model_batch import ingest_batch, parse_answers, prepare_batch, write_batch
from opengrad.annotation.review import build_review_queue, reference_summary, write_review_queue
from opengrad.annotation.server import ServerContext, create_server
from opengrad.annotation.service import (
    Workspace,
    WorkspaceError,
    check_identifier,
    load_instructions,
    load_review_queues,
    metric_exclusion_problems,
    metric_exclusion_summary,
    model_annotator_problem,
)
from opengrad.verification.accounting import PASS

UI_DIR = Path("integrations/annotate-ui/out")
TASK_DIR = Path("configs/annotation")


def resolve_config(value: str, root: str | None) -> Path:
    """A path to a task config, or a task id with a config at ``configs/annotation/<id>.yaml``."""
    candidate = Path(value)
    if candidate.is_file():
        return candidate
    base = Path(root) if root else find_repo_root(Path.cwd())
    named = base / TASK_DIR / f"{value}.yaml"
    if named.is_file():
        return named
    available = sorted(path.stem for path in (base / TASK_DIR).glob("*.yaml"))
    raise TaskConfigError(f"no task config {value!r}; pass a config path or one of {available}")


def _config(args: argparse.Namespace) -> TaskConfig:
    path = resolve_config(args.config, args.root)
    config = load_task_config(path, Path(args.root) if args.root else None)
    if not Path(args.config).is_file() and config.task_id != args.config:
        raise TaskConfigError(f"{path} declares task_id {config.task_id!r}, not {args.config!r}")
    return config


def _state_path(args: argparse.Namespace, config: TaskConfig) -> Path:
    return Path(args.state_db) if args.state_db else config.resolve(config.state_db)


def _open(args: argparse.Namespace) -> Workspace:
    config = _config(args)
    return Workspace.open(config, state_db=_state_path(args, config))


def _print_progress(progress: dict[str, Any], *, labels: bool) -> None:
    print(f"  {progress['completed']} / {progress['total']} completed   {progress['percent']}%")
    if labels:
        width = max([len(name) for name in progress["label_counts"]] + [9])
        for name, count in progress["label_counts"].items():
            print(f"    {name:<{width}}  {count:>5}")
        print(f"    {'remaining':<{width}}  {progress['remaining']:>5}")
    print(f"  skipped {progress['skipped']}   flagged {progress['flagged']}")


def _serve(workspace: Workspace, context: ServerContext, args: argparse.Namespace) -> int:
    ui_dir = Path(args.ui_dir) if args.ui_dir else workspace.config.root / UI_DIR
    context.ui_dir = ui_dir if (ui_dir / "index.html").is_file() else None
    server = create_server(context, host=args.host, port=args.port)
    url = f"http://{args.host}:{context.port}/"
    print(f"task      {workspace.task_id}  ({workspace.item_count} items)")
    print(f"source    {workspace.config.source.path}  sha256 {workspace.source_sha256}")
    print(f"state     {workspace.store.path}")
    if context.ui_dir is None:
        print("ui        not built -- run `npm install && npm run build` in integrations/annotate-ui,")
        print("          or `npm run dev` there and open http://localhost:3000")
    print(f"\nOpen {url}   (Ctrl+C stops; every label is already saved)")
    if not args.no_browser and context.ui_dir is not None:
        webbrowser.open(url)
    sys.stdout.flush()  # the banner names task, annotator and session; show it even when piped to a log
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
        workspace.close()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    if args.dry_run:
        return _dry_run_start(args)
    workspace = _open(args)
    try:
        session = workspace.open_session(args.session, args.annotator)
        state = workspace.session_state(args.session)
    except BaseException:
        workspace.close()
        raise
    print(f"session   {session['session_id']}  annotator {session['annotator_id']}")
    if state["frozen"]:
        print(f"          FROZEN at {state['frozen_at']} -- read-only")
    _print_progress(workspace.progress(args.session), labels=True)
    _print_reference_progress(workspace, args.session)
    return _serve(workspace, ServerContext(workspace, "annotate", session_id=args.session), args)


def _print_reference_progress(workspace: Workspace, session_id: str) -> None:
    reference = workspace.reference_progress(session_id)
    if reference is not None:
        print(
            f"  human-reviewed {reference['human_reviewed']}   provisional model-only "
            f"{reference['provisional_model_only']}   unlabeled {reference['unlabeled']}   "
            f"remaining for human review {reference['remaining_for_human_review']}"
        )
    for queue in workspace.queue_progress(session_id):
        print(f"  queue {queue['name']}: {queue['completed']} / {queue['total']} labeled in this session")


def _dry_run_start(args: argparse.Namespace) -> int:
    """Everything ``start`` checks before its first write, and none of its writes.

    It loads and re-hashes the source, renders the rubric, validates the ids, inspects any existing state
    read-only, and probes the port -- then stops where ``start`` would create the working state.
    """
    config = _config(args)
    problems: list[str] = []
    digest, items = load_source(config)  # raises on a hash or item-count mismatch
    load_instructions(config)  # raises on a missing section or a forbidden term
    problems.extend(metric_exclusion_problems(config, {item.item_id for item in items}))
    for value, what in ((args.annotator, "annotator id"), (args.session, "session id")):
        try:
            check_identifier(value, what)
        except WorkspaceError as exc:
            problems.append(str(exc))
    print(f"task      {config.task_id}  ({len(items)} items, source sha256 {digest}, pinned)")
    state = _state_path(args, config)
    if state.exists():
        with sqlite3.connect(f"file:{state.as_posix()}?mode=ro", uri=True) as conn:
            owner = conn.execute(
                "SELECT annotator_id FROM sessions WHERE task_id = ? AND session_id = ?",
                (config.task_id, args.session),
            ).fetchone()
            labeled = conn.execute(
                "SELECT COUNT(*) FROM annotations WHERE task_id = ? AND session_id = ? AND status = 'labeled'",
                (config.task_id, args.session),
            ).fetchone()[0]
        print(f"state     {state}  (exists)")
        if owner is None:
            print(f"session   {args.session} would be created for annotator {args.annotator}")
        elif owner[0] != args.annotator:
            problems.append(f"session {args.session!r} belongs to annotator {owner[0]!r}, not {args.annotator!r}")
        else:
            print(f"session   {args.session} exists for {args.annotator} ({labeled} labeled); start resumes it")
    else:
        print(f"state     {state}  (does not exist; start would create it)")
        print(f"session   {args.session} would be created for annotator {args.annotator}")
    ui_dir = Path(args.ui_dir) if args.ui_dir else config.root / UI_DIR
    if (ui_dir / "index.html").is_file():
        print(f"ui        {ui_dir}  (built)")
    else:
        problems.append(f"the UI is not built at {ui_dir}; run `npm install && npm run build` there")
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        problems.append("the annotation server only binds to loopback")
    else:
        family = socket.AF_INET6 if ":" in args.host else socket.AF_INET
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((args.host, args.port))
            except OSError:
                problems.append(f"{args.host}:{args.port} is in use; pass --port to choose another")
    print(f"url       http://{args.host}:{args.port}/")
    for problem in problems:
        print(f"PROBLEM   {problem}")
    if problems:
        print("DRY RUN   FAIL -- nothing was created")
        return 1
    print("DRY RUN   PASS -- nothing was created; run the same command without --dry-run to begin")
    return 0


def cmd_adjudicate(args: argparse.Namespace) -> int:
    workspace = _open(args)
    try:
        workspace.check_adjudication_sessions(args.sessions)
        queue = workspace.adjudication_queue(args.sessions)
    except BaseException:
        workspace.close()
        raise
    kind = "adjudication" if len(args.sessions) == 2 else "single-annotator review"
    done = sum(1 for item in queue if item["adjudicated"])
    print(f"{kind}   sessions {', '.join(args.sessions)}   adjudicator {args.adjudicator}")
    print(f"  queue {len(queue)}   decided {done}")
    context = ServerContext(
        workspace, "adjudicate", sessions=list(args.sessions), adjudicator_id=args.adjudicator
    )
    return _serve(workspace, context, args)


def _no_state(args: argparse.Namespace) -> Path | None:
    """The state path if it does not exist yet: reporting commands must not create it."""
    config = _config(args)
    state = _state_path(args, config)
    return None if state.exists() else state


def cmd_status(args: argparse.Namespace) -> int:
    missing = _no_state(args)
    if missing is not None:
        print(f"no working state yet at {missing}; start a pass with `opengrad-annotate start`")
        return 0
    workspace = _open(args)
    try:
        summary = workspace.summary()
        if args.json:
            if args.session:
                summary["progress"] = workspace.progress(args.session)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        print(f"task    {summary['task_id']} v{summary['version']}  ({summary['task_type']})")
        print(f"source  {summary['source_path']}  {summary['items']} items")
        print(f"        sha256 {summary['source_sha256']}  (matches the pinned hash)")
        print(f"state   {summary['state_db']}")
        if not summary["sessions"]:
            print("\nno sessions yet")
        for session in summary["sessions"]:
            frozen = "  FROZEN" if session["frozen"] else ""
            print(
                f"\nsession {session['session_id']}  ({session['annotator_id']}){frozen}"
            )
            # Label distributions are shown only for the session asked about, so checking status
            # during pass B does not reveal what pass A decided.
            _print_progress(
                workspace.progress(session["session_id"]),
                labels=session["session_id"] == args.session,
            )
        for freeze in summary["freezes"]:
            print(f"\nfrozen  {', '.join(freeze['sessions'])} at {freeze['frozen_at']} -> {freeze['manifest_path']}")
        return 0
    finally:
        workspace.close()


def cmd_check(args: argparse.Namespace) -> int:
    """Read-only preflight: config, pinned source, item ids, blinding, rubric, state. Writes nothing."""
    config = _config(args)
    problems: list[str] = []
    digest, items = load_source(config)  # raises on a hash or item-count mismatch
    documents = load_instructions(config)  # raises on a missing section or a forbidden term
    ids = [item.item_id for item in items]
    if len(set(ids)) != len(ids):
        problems.append("item ids are not unique")
    # A blinded column leaks when it reaches the browser as a key of the projected view (a display path that
    # returns a mapping holding it). The bare name inside prose or a tool description ("player" holds
    # "layer") carries no blinded value, so only key occurrences count.
    blinded_keys = [json.dumps(name) + ":" for name in config.blind_fields]
    exposed = [
        item.item_id
        for item in items
        if any(key in json.dumps(project(config, item.row)) for key in blinded_keys)
    ]
    if exposed:
        problems.append(f"{len(exposed)} items would expose a blinded field")
    print(f"task      {config.task_id} v{config.version}  ({config.task_type})  {config.config_path}")
    print(f"source    {config.source.path}  ({config.source.format}, read-only)")
    pin = "pinned, matches" if config.source.expected_sha256 == digest else "NOT PINNED"
    print(f"  sha256  {digest}  ({pin})")
    count = "pinned, matches" if config.source.expected_items == len(items) else "not pinned"
    print(f"  items   {len(items)}  ({count}); ids from {config.source.id_field or 'content hash'}, file order")
    for name, path in config.filters:
        counts = Counter(str(item.row.get(path)) for item in items)
        print(f"  {name:<7} " + ", ".join(f"{key} {value}" for key, value in sorted(counts.items())))
    print(f"labels    {', '.join(f'{key}={label}' for key, label in config.shortcuts)}")
    required = [field.key for field in config.extra_fields if field.required]
    print(f"required  {', '.join(required) or '(label only)'}; note optional")
    print(f"blinded   {', '.join(config.blind_fields) or '(none)'}")
    print(f"rubric    {len(documents)} documents, {len(config.instruction_forbidden_terms)} forbidden terms absent")
    for document in documents:
        print(f"  - {document['title']}")
    problems.extend(metric_exclusion_problems(config, set(ids)))
    exclusions = metric_exclusion_summary(config, len(items))
    for group in exclusions["groups"]:
        print(f"excluded  {group['status']}: {group['items']} items, annotated normally; excluded from")
        print(f"          {', '.join(group['excluded_from'])}  ({group['document'] or 'no document'})")
    print(f"metrics   {exclusions['metric_eligible_items']} of {len(items)} items metric-eligible")
    for declared in config.model_annotators:
        problem = model_annotator_problem(config, declared.annotator_id)
        print(f"model     {declared.annotator_id} = {declared.model}; procedure {declared.procedure}")
        print(f"          sha256 {declared.procedure_sha256}  ({'MISMATCH' if problem else 'pinned, matches'})")
        if problem:
            problems.append(problem)
    for amendment in config.definition_amendments:
        print(f"amended   definition {amendment.from_sha256[:12]}... -> {amendment.to_sha256[:12]}...")
        print(f"          {amendment.document}")
    try:
        queues = load_review_queues(config, set(ids), digest)
    except WorkspaceError as exc:
        problems.append(str(exc))
        queues = {}
    for name, order in queues.items():
        print(f"queue     {name}: {len(order)} items (file pinned, matches)")
    state = _state_path(args, config)
    if state.exists():
        with sqlite3.connect(f"file:{state.as_posix()}?mode=ro", uri=True) as conn:
            recorded = conn.execute(
                "SELECT source_sha256, definition_sha256 FROM tasks WHERE task_id = ?", (config.task_id,)
            ).fetchone()
            if recorded and recorded[0] != digest:
                problems.append("the working state was imported from different source bytes")
            if recorded and recorded[1] != config.definition_sha256():
                route = config.amendment_route(recorded[1])
                if route is None:
                    problems.append("the stored task definition differs and no declared amendment leads here")
                else:
                    print(f"definition  stored {recorded[1][:12]}...; the next open records {len(route)} amendment(s)")
            sessions = conn.execute(
                "SELECT s.session_id, s.annotator_id, COUNT(a.item_id) FROM sessions s "
                "LEFT JOIN annotations a ON a.task_id = s.task_id AND a.session_id = s.session_id "
                "AND a.status = 'labeled' WHERE s.task_id = ? GROUP BY s.session_id ORDER BY s.session_id",
                (config.task_id,),
            ).fetchall()
        print(f"state     {state}  (exists)")
        for session_id, annotator, labeled in sessions:
            print(f"  session {session_id} ({annotator}): {labeled} labeled")
    else:
        print(f"state     {state}  (not created yet; `start` creates it)")
    print(f"exports   snapshots {default_out_dir(config, gold=False)}")
    print(f"          gold      {default_out_dir(config, gold=True)}")
    for problem in problems:
        print(f"PROBLEM   {problem}")
    print("CHECK     " + ("PASS" if not problems else "FAIL"))
    return 0 if not problems else 1


def cmd_audit(args: argparse.Namespace) -> int:
    missing = _no_state(args)
    if missing is not None:
        print(json.dumps({"status": "PASS", "errors": [], "note": f"no working state at {missing}"}, indent=2))
        return 0
    workspace = _open(args)
    try:
        errors = workspace.audit()
    finally:
        workspace.close()
    print(json.dumps({"status": "PASS" if not errors else "FAIL", "errors": errors}, indent=2))
    return 0 if not errors else 1


def _export(args: argparse.Namespace, gold: bool) -> int:
    workspace = _open(args)
    try:
        out = Path(args.out) if args.out else None
        operation = freeze_gold if gold else export_snapshot
        manifest = operation(workspace, list(args.sessions), out, composite=args.composite)
    finally:
        workspace.close()
    print(f"{manifest['artifact_kind']}  {manifest['completion_state']}  -> {manifest['manifest_path']}")
    for reason in manifest["incomplete_reasons"]:
        print(f"  incomplete: {reason}")
    for name, output in manifest["outputs"].items():
        print(f"  {output['sha256'][:16]}  {output['records']:>5}  {name}")
    if manifest["gold"]:
        print(f"  gold labels: {manifest['gold']['label_counts']}")
        print(f"  gold label sources: {manifest['gold']['label_sources']}")
    if manifest["model_annotation"]:
        print("  model annotation: sessions " + ", ".join(
            s["session_id"] for s in manifest["sessions"] if s["annotator_kind"] == "model"
        ) + " hold model judgments, not human labels")
    if manifest["limitations"]:
        print(f"  limitation: {manifest['limitations'][0]}")
    return 0


def _batch_dir(args: argparse.Namespace, workspace: Workspace) -> Path:
    if args.out:
        return Path(args.out)
    return _state_path(args, workspace.config).parent / f"{workspace.task_id}.model-batches" / str(args.session)


def cmd_model_batch(args: argparse.Namespace) -> int:
    workspace = _open(args)
    try:
        directory = _batch_dir(args, workspace)
        # Only the batch files themselves count; answers and audit records sit beside them.
        prepared = [path for path in directory.glob("batch-*.json") if path.stem[len("batch-"):].isdigit()]
        batch_id = f"{len(prepared) + 1:02d}"
        batch = prepare_batch(
            workspace,
            args.session,
            args.annotator,
            size=args.size,
            defer_to=list(args.defer_to or []),
            batch_id=batch_id,
        )
        if not batch["item_ids"]:
            print(f"nothing left: every item is labeled in {args.session} or {args.defer_to}")
            return 0
        json_path, md_path = write_batch(batch, workspace.config, directory)
        remaining = workspace.progress(args.session)["remaining"] - len(batch["item_ids"])
    finally:
        workspace.close()
    numbers = [item["number"] for item in batch["items"]]
    print(f"batch     {batch_id}: {len(numbers)} items (#{numbers[0]}..#{numbers[-1]}), content sha256 {batch['content_sha256']}")
    print(f"model     {batch['annotator_id']} = {batch['model']}; procedure sha256 {batch['procedure_sha256']}")
    print(f"read      {md_path}")
    print(f"ingest    {json_path}")
    print(f"after it  {remaining} items still without a label in {args.session} (before deferring)")
    return 0


def cmd_model_ingest(args: argparse.Namespace) -> int:
    workspace = _open(args)
    try:
        batch = json.loads(Path(args.batch).read_text(encoding="utf-8"))
        answers = parse_answers(Path(args.answers).read_text(encoding="utf-8"))
        result = ingest_batch(workspace, batch, answers)
    finally:
        workspace.close()
    progress = result["progress"]
    print(f"recorded  batch {result['batch_id']}: {result['recorded']} labels in {result['session_id']} ({result['flagged']} flagged)")
    print(f"session   {progress['completed']} / {progress['total']} labeled")
    return 0


def cmd_reference(args: argparse.Namespace) -> int:
    missing = _no_state(args)
    if missing is not None:
        print(f"no working state yet at {missing}")
        return 0
    workspace = _open(args)
    try:
        summary = reference_summary(workspace, list(args.sessions))
    finally:
        workspace.close()
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    print(f"reference {summary['task_id']}: {summary['rule']}")
    for source in summary["priority"]:
        print(f"  {source['session_id']:<10} {source['annotator_id']}  ({source['annotator_kind']})")
    kinds = summary["items_by_kind"]
    print(
        f"  items     {summary['labeled_items']} of {summary['population_items']} labeled: "
        f"{kinds['human']} human, {kinds['model']} model; {summary['unlabeled_items']} unlabeled"
    )
    width = max(len(label) for label in summary["label_counts"])
    print(f"  {'label':<{width}}   total   human   model   metric-eligible")
    for label, count in summary["label_counts"].items():
        human = summary["label_counts_by_kind"]["human"][label]
        model = summary["label_counts_by_kind"]["model"][label]
        eligible = summary["metric_eligible_label_counts"][label]
        print(f"  {label:<{width}}   {count:>5}   {human:>5}   {model:>5}   {eligible:>5}")
    print(f"  flagged uncertain: {summary['flagged_by_kind']}")
    print("  frozen" if summary["frozen"] else "  not frozen: a working reference, not gold")
    return 0


def cmd_review_queue(args: argparse.Namespace) -> int:
    samples: dict[str, int] = {}
    for spec in args.sample or []:
        label, _, size = spec.partition("=")
        if not label or not size.isdigit():
            raise WorkspaceError(f"--sample takes LABEL=SIZE, not {spec!r}")
        samples[label] = int(size)
    workspace = _open(args)
    try:
        queue = build_review_queue(
            workspace,
            name=args.name,
            sessions=list(args.sessions),
            seed=args.seed,
            flagged=args.flagged,
            labels=list(args.label or []),
            samples=samples,
        )
    finally:
        workspace.close()
    out = Path(args.out)
    digest = write_review_queue(queue, out)
    # Counts per criterion only; which item met which criterion stays in the file.
    print(f"queue     {queue['name']}: {queue['counts']['items']} items -> {out}")
    print(f"  sha256  {digest}")
    for criterion, count in queue["counts"]["by_criterion"].items():
        print(f"  {criterion:<8} {count}")
    print("pin it in the task config:")
    print(f"  review_queues:\n    - name: {queue['name']}\n      file: {out.as_posix()}\n      sha256: {digest}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    root = Path(args.root) if args.root else None
    _, summary = verify_package(Path(args.manifest), root, require_source=args.require_source)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == PASS else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opengrad-annotate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def task_args(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "config", metavar="TASK", help="task id (e.g. pdet-v1) or a path to a task config"
        )
        command.add_argument("--root", help="repository root (default: found from the config)")
        command.add_argument("--state-db", help="override the SQLite working-state path")

    def server_args(command: argparse.ArgumentParser) -> None:
        command.add_argument("--host", default="127.0.0.1")
        command.add_argument("--port", type=int, default=8765)
        command.add_argument("--ui-dir", help="static UI build (default: integrations/annotate-ui/out)")
        command.add_argument("--no-browser", action="store_true", help="do not open a browser tab")

    check = sub.add_parser("check", help="read-only preflight of a task; creates nothing")
    task_args(check)
    check.set_defaults(func=cmd_check)

    serve = sub.add_parser("start", aliases=["serve", "resume"], help="annotate one pass (start or resume)")
    task_args(serve)
    server_args(serve)
    serve.add_argument("--annotator", required=True, help="your annotator id, recorded on every change")
    serve.add_argument("--session", required=True, help="pass id, e.g. pass-a; one pass per session")
    serve.add_argument(
        "--dry-run", action="store_true", help="run every check start makes, then stop; creates nothing"
    )
    serve.set_defaults(func=cmd_serve)

    adjudicate = sub.add_parser("adjudicate", help="compare two finished passes, or review one")
    task_args(adjudicate)
    server_args(adjudicate)
    adjudicate.add_argument("--sessions", nargs="+", required=True, help="one or two session ids")
    adjudicate.add_argument("--adjudicator", required=True, help="adjudicator id")
    adjudicate.set_defaults(func=cmd_adjudicate)

    status = sub.add_parser("status", help="progress per session")
    task_args(status)
    status.add_argument("--session", help="also show this session's label counts")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    audit = sub.add_parser("audit", help="re-verify every change-log hash chain in the store")
    task_args(audit)
    audit.set_defaults(func=cmd_audit)

    for name, gold, text in (
        ("export", False, "write a work-in-progress snapshot (never gold)"),
        ("freeze-gold", True, "write the final gold package and lock its sessions"),
    ):
        command = sub.add_parser(name, help=text)
        task_args(command)
        command.add_argument("--sessions", nargs="+", required=True)
        command.add_argument("--out", help="package directory (default from the task config)")
        command.add_argument(
            "--composite",
            action="store_true",
            help="each item's label from the first listed session that labeled it (list sessions in priority order)",
        )
        command.set_defaults(func=lambda a, gold=gold: _export(a, gold))

    batch = sub.add_parser("model-batch", help="prepare the next batch for a declared model annotator")
    task_args(batch)
    batch.add_argument("--session", required=True, help="the model's session, e.g. model-a")
    batch.add_argument("--annotator", required=True, help="a declared model annotator id (model.*)")
    batch.add_argument("--size", type=int, default=50)
    batch.add_argument("--defer-to", nargs="*", help="sessions whose labeled items the model skips")
    batch.add_argument("--out", help="batch directory (default: beside the working state)")
    batch.set_defaults(func=cmd_model_batch)

    ingest = sub.add_parser("model-ingest", help="validate a model's answers to a batch, then record them")
    task_args(ingest)
    ingest.add_argument("--batch", required=True, help="the batch JSON written by model-batch")
    ingest.add_argument("--answers", required=True, help="the model's answers (JSON array)")
    ingest.set_defaults(func=cmd_model_ingest)

    reference = sub.add_parser("reference", help="the composite reference's counts, recomputed from the store")
    task_args(reference)
    reference.add_argument("--sessions", nargs="+", required=True, help="sessions in priority order")
    reference.add_argument("--json", action="store_true")
    reference.set_defaults(func=cmd_reference)

    queue = sub.add_parser("review-queue", help="write a pinned review queue from the composite reference")
    task_args(queue)
    queue.add_argument("--name", required=True, help="neutral queue name shown to annotators")
    queue.add_argument("--sessions", nargs="+", required=True, help="sessions in priority order")
    queue.add_argument("--seed", required=True, help="seed for the sample draw and the presentation order")
    queue.add_argument("--flagged", action="store_true", help="items whose reference label is flagged")
    queue.add_argument("--label", action="append", help="items whose reference label is LABEL")
    queue.add_argument("--sample", action="append", help="LABEL=SIZE: a seeded sample of model-sourced items")
    queue.add_argument("--out", required=True, help="queue file to write (never overwritten)")
    queue.set_defaults(func=cmd_review_queue)

    verify = sub.add_parser("verify", help="re-hash an exported package against its manifest")
    verify.add_argument("manifest", help="path to the package manifest JSON")
    verify.add_argument("--root", help="repository root, to re-hash the source population")
    verify.add_argument(
        "--require-source",
        action="store_true",
        help="fail if the source population cannot be found and re-hashed",
    )
    verify.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except IncompleteGoldError as exc:
        print(str(exc), file=sys.stderr)
        print("A work-in-progress snapshot is still available: opengrad-annotate export ...", file=sys.stderr)
        return 3
    except (TaskConfigError, SourceError, WorkspaceError, ExportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
