"""Recording for every script and module entry point: one helper, one policy table.

A script's ``if __name__ == "__main__":`` block calls ``run_script(__file__, main)``. That records the run in the local
run-record folder, like ``opengrad train``, without changing what the script does. It is called only from the
``__main__`` block, never from ``main()``, so a test that imports a script and calls ``main()`` records nothing.

The policy for each entry point is fixed here, not inferred at run time:

* ``kind``: the run kind. An ``annotation`` log that uploads goes as a redacted copy.
* ``expected_exit_codes``: non-zero codes the script returns by design, each with a reason. Examples are a ``--check``
  generator reporting stale output, or a scan reporting findings. Such a run is recorded as ``PASS`` with its exit
  code kept, so it needs no disposition (owner's decision, 2026-11-13). Any other non-zero code, or a crash, is
  ``FAILED`` and needs one.
* held-out data: an entry point whose code reads held-out evaluation data (``HELD_OUT_MARKERS``) is listed in
  ``HELD_OUT_ENTRY_POINTS``. Its log is recorded locally but never uploaded (owner's decision, 2026-11-13).
  ``tests/runrecords/test_scripts_policy.py`` keeps that list, together with ``PINNED_CONSOLE``, equal to a scan
  of the code that follows imports: an entry point counts when its own code, or a module it imports, names a
  marker. The scan reads code only (``code_text``): a marker in a comment or docstring does not count.
* pinned console: an entry point whose console output a test pins to status (ids, counts, codes, fixed text) is
  in ``PINNED_CONSOLE`` and uploads although the scan reaches held-out data. The labelling runner is one (owner's
  decision, 2026-11-13: annotation logs go up as redacted copies).

Upload is an allowlist (owner's decision, 2026-11-13): a log uploads only when its entry point was reviewed and
approved (``PINNED_CONSOLE``, ``APPROVED_CONSOLE``); any other stays local and is listed until it is. The checks
below still apply to an approved entry point. The decision is made when a log is uploaded (``upload_decision``),
from the recorded command, so it also
covers a script launched through ``python -m opengrad.runrecords run``, a script not converted yet, and a run
whose arguments name held-out data. A command that cannot be traced to a repository entry point is kept local
and listed (owner's decision, 2026-11-13: fail closed).
"""

from __future__ import annotations

import ast
import io
import posixpath
import re
import subprocess
import sys
import tokenize
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from opengrad.runrecords.inputs import (
    HELD_OUT_MARKERS,
    argument_values,
    config_state,
    names_held_out,
    repo_relative,
)
from opengrad.runrecords.record import recorded_call


@dataclass(frozen=True)
class ScriptPolicy:
    kind: str = "script"
    #: Non-zero exit code -> why the script returns it by design.
    expected_exit_codes: Mapping[int, str] = field(default_factory=lambda: MappingProxyType({}))


#: Entry points whose policy differs from the default (kind ``script``, no expected non-zero codes), by repository path.
POLICY: Mapping[str, ScriptPolicy] = MappingProxyType(
    {
        "scripts/run_external_annotation.py": ScriptPolicy(kind="annotation"),
        "src/opengrad/annotation/cli.py": ScriptPolicy(kind="annotation"),
    }
)

#: Entry points that read held-out data, directly or through an import: their logs are never uploaded.
HELD_OUT_ENTRY_POINTS = frozenset(
    {
        "scripts/archive_external_model_labels.py",
        "scripts/assemble_m1_preference_dataset.py",
        "scripts/audit_canonical_v2_final_direct.py",
        "scripts/audit_corpus_direct_prevalence.py",
        "scripts/audit_first_reply_supply.py",
        "scripts/audit_normalization_v3_structure.py",
        "scripts/audit_pdet_coverage_v2_supply.py",
        "scripts/audit_punans_constructed_supply.py",
        "scripts/audit_punans_supply.py",
        "scripts/audit_punans_v2_supply.py",
        "scripts/build_engine_parity_report.py",
        "scripts/build_executorch_handoff.py",
        "scripts/build_findings_visual.py",
        "scripts/build_gguf_card.py",
        "scripts/build_gguf_handoff.py",
        "scripts/build_h200_benchmark_report.py",
        "scripts/build_m1_local_preference_pairs.py",
        "scripts/build_ptq_evaluation_report.py",
        "scripts/build_quantization_visual.py",
        "scripts/build_regression_analysis.py",
        "scripts/build_tokenizer_regression_fixtures.py",
        "scripts/close_ptq_phase.py",
        "scripts/compare_gemini_relabel.py",
        "scripts/evaluate_prose_classifier_dev.py",
        "scripts/evaluate_prose_classifier_v2_dev.py",
        "scripts/evaluate_sft_checkpoints.py",
        "scripts/freeze_eval_partition.py",
        "scripts/freeze_m0_final.py",
        "scripts/freeze_minus_xlam_ablation.py",
        "scripts/grade_openweights_parity.py",
        "scripts/prepare_imatrix_calibration.py",
        "scripts/prepare_quantization_inputs.py",
        "scripts/preserve_h200_state.py",
        "scripts/publish_card_corrections.py",
        "scripts/publish_executorch_artifact.py",
        "scripts/rebuild_eval_splits.py",
        "scripts/render_frozen_prompts.py",
        "scripts/reporting/generate_experiment_status.py",
        "scripts/run_gguf_ladder.py",
        "scripts/score_gguf_candidate.py",
        "scripts/score_quantized_artifact.py",
        "scripts/select_checkpoint.py",
        "scripts/update_quantization_ledger.py",
        "src/opengrad/verification/answer_strata_report.py",
        "src/opengrad/verification/flag_triage_population.py",
        "src/opengrad/verification/flag_triage_report.py",
        "src/opengrad/verification/pconf.py",
        "src/opengrad/verification/pdet.py",
        "src/opengrad/verification/pdet_coverage_reference.py",
        "src/opengrad/verification/punans.py",
        "src/opengrad/verification/punans_constructed.py",
        "src/opengrad/verification/punans_constructed_report.py",
        "src/opengrad/verification/punans_report.py",
        "src/opengrad/verification/punans_v2.py",
        "src/opengrad/verification/punans_v2_report.py",
        "src/opengrad/verification/study_002_gate.py",
        "src/opengrad/verification/study_002_readiness.py",
    }
)


#: Entry points the held-out scan reaches whose console output a test pins to status lines: their logs upload
#: (redacted when private details or an annotation run call for it). Path -> the test that pins the console.
#: The fifteen after the runner reach held-out data only through classifier_input's exclusion set; a read of every
#: object they print found no record text (owner's decision, 2026-11-13).
PINNED_CONSOLE: Mapping[str, str] = MappingProxyType(
    {
        "scripts/run_external_annotation.py": "tests/annotation/test_run_external_annotation.py",
        "scripts/audit_pdet_coverage_supply_v3.py": "tests/runrecords/test_console_pins.py",
        "scripts/audit_pdet_v1_representation.py": "tests/runrecords/test_console_pins.py",
        "src/opengrad/data/behaviour_labels.py": "tests/runrecords/test_console_pins.py",
        "src/opengrad/data/behaviour_labels_v3.py": "tests/runrecords/test_console_pins.py",
        "src/opengrad/data/canonical_v3.py": "tests/runrecords/test_console_pins.py",
        "src/opengrad/data/canonical_v3_balance.py": "tests/runrecords/test_console_pins.py",
        "src/opengrad/data/provenance_gate.py": "tests/runrecords/test_console_pins.py",
        "src/opengrad/verification/classifier_devcheck.py": "tests/runrecords/test_console_pins.py",
        "src/opengrad/verification/classifier_devset.py": "tests/runrecords/test_console_pins.py",
        "src/opengrad/verification/classifier_devset_v2.py": "tests/runrecords/test_console_pins.py",
        "src/opengrad/verification/flag_set.py": "tests/runrecords/test_console_pins.py",
        "src/opengrad/verification/pdet_coverage.py": "tests/runrecords/test_console_pins.py",
        "src/opengrad/verification/pdet_coverage_v2.py": "tests/runrecords/test_console_pins.py",
        "src/opengrad/verification/prose_classifier_oneshot.py": "tests/runrecords/test_console_pins.py",
        "src/opengrad/verification/prose_classifier_v2_oneshot.py": "tests/runrecords/test_console_pins.py",
    }
)


def code_text(source: str) -> str:
    """Python source with its comments and docstrings removed: the text the held-out scan reads."""
    docstrings: set[tuple[int, int]] = set()
    for node in ast.walk(ast.parse(source)):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                docstrings.add((value.lineno, value.col_offset))
    kept = [
        token.string
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type != tokenize.COMMENT and not (token.type == tokenize.STRING and token.start in docstrings)
    ]
    return " ".join(kept)


def repo_root(path: Path) -> Path | None:
    """The nearest directory above ``path`` that holds ``pyproject.toml``."""
    for parent in path.resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


_PYTHON = re.compile(r"(?i)^(?:python(?:\d+(?:\.\d+)*)?w?|pyw?)(?:\.exe)?$")
_LAUNCHER_SUFFIX = re.compile(r"(?i)(?:-script\.pyw?|\.exe|\.cmd|\.bat)$")
#: Short interpreter options that take no value and change nothing about what runs; a cluster of them (``-uB``) too.
_PLAIN_OPTIONS = frozenset("bBdEhiIOPqRsSuvVx")
#: The ``py`` launcher's version selector (``-3``, ``-3.11``, ``-3.11-64``).
_PY_VERSION = re.compile(r"^-\d+(?:\.\d+)?(?:-(?:32|64))?$")
#: Environment prefixes that make the interpreter run other code first.
_CODE_ENV = frozenset({"PYTHONSTARTUP", "PYTHONPATH", "PYTHONHOME", "PYTHONINSPECT", "PYTHONSAFEPATH"})


def console_commands(root: Path) -> dict[str, str]:
    """``[project.scripts]`` of the repository: console command -> ``module:function``."""
    try:
        data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return {str(k): str(v) for k, v in data.get("project", {}).get("scripts", {}).items()}


def _known(relative: str) -> str:
    """The table spelling of a repository path. Windows file names ignore case, so the tables are matched that way."""
    folded = relative.casefold()
    for table in (HELD_OUT_ENTRY_POINTS, PINNED_CONSOLE, POLICY):
        for key in table:
            if key.casefold() == folded:
                return key
    return relative


def _module_path(module: str, root: Path) -> str | None:
    base = "src/" + module.replace(".", "/")
    for candidate in (base + ".py", base + "/__main__.py"):
        if (root / candidate).is_file():
            return _known(candidate)
    return None


def repo_script(token: str, root: Path) -> str | None:
    """The repository path of a script argument (``./scripts/x.py``, ``scripts//x.py``, ``scripts/../scripts/x.py``,
    any case), or None when it is not a ``.py`` file under ``scripts/`` or ``src/``."""
    value = str(token).replace("\\", "/")
    if not value:
        return None
    relative = posixpath.normpath(value)
    if not relative.casefold().startswith(("scripts/", "src/")) or not relative.casefold().endswith(".py"):
        return None
    if not (root / relative).is_file():
        return None
    known = _known(relative)
    if known != relative:
        # On a case-sensitive file system a differently-cased name is another file: it gets no table entry.
        try:
            if not (root / relative).samefile(root / known):
                return relative
        except OSError:
            return relative
    return known


def _console_target(token: str, root: Path) -> str | None:
    """The ``module:function`` a console-command token runs: its bare name, or a launcher in the checkout's
    ``.venv``. A program of the same name elsewhere is not this repository's command."""
    value = posixpath.normpath(str(token).replace("\\", "/"))
    if "/" in value and posixpath.dirname(value).casefold() not in (".venv/scripts", ".venv/bin"):
        return None
    return console_commands(root).get(_LAUNCHER_SUFFIX.sub("", value.rsplit("/", 1)[-1]))


def console_target(command: Sequence[str], root: Path) -> str | None:
    """The ``module:function`` of the repository console command a recorded command runs, if it runs one."""
    tokens = [str(t).replace("\\", "/") for t in command]
    i = 0
    while i < len(tokens) and "=" in tokens[i] and not tokens[i].startswith("-"):
        i += 1
    if i + 1 < len(tokens) and _LAUNCHER_SUFFIX.sub("", tokens[i].rsplit("/", 1)[-1]).casefold() == "uv":
        i += 2 if tokens[i + 1] == "run" else len(tokens)
    return _console_target(tokens[i], root) if i < len(tokens) else None


def trace(command: Sequence[str], root: Path) -> tuple[str | None, int]:
    """The repository entry point a recorded command runs, as a repository path, or None when it cannot be traced.

    Understood: ``VAR=value`` prefixes (not ones that run other code, such as ``PYTHONSTARTUP``), ``uv run`` with no
    options, a Python interpreter with plain options (``-u``, ``-I``, ``-X utf8``, ``-W x``, ``--``), ``-m module``,
    a script under ``scripts/`` or ``src/``, and the repository's console commands. Anything else (a shell, pytest,
    an outside tool, ``-c``, an option this does not know) is None, and its log is kept local (owner's decision,
    2026-11-13: fail closed).
    """
    tokens = [str(t).replace("\\", "/") for t in command]
    i = 0
    while i < len(tokens) and "=" in tokens[i] and not tokens[i].startswith("-"):
        name, value = tokens[i].split("=", 1)
        if name.upper() in _CODE_ENV or value.casefold().endswith((".py", ".pyw")):
            return None, 0
        i += 1
    if i + 1 < len(tokens) and _LAUNCHER_SUFFIX.sub("", tokens[i].rsplit("/", 1)[-1]).casefold() == "uv":
        if tokens[i + 1] != "run":
            return None, 0
        i += 2
        if i < len(tokens) and tokens[i].startswith("-"):
            return None, 0  # a uv option may take a value, so what runs cannot be told apart from it
    if i >= len(tokens):
        return None, 0
    program = tokens[i].rsplit("/", 1)[-1]
    if _PYTHON.match(program):
        launcher = program.casefold().startswith("py") and not program.casefold().startswith("python")
        i += 1
        while i < len(tokens) and tokens[i].startswith("-") and tokens[i] != "-m":
            option = tokens[i]
            if option == "--":
                i += 1
                break
            plain = not option.startswith("--") and len(option) > 1 and set(option[1:]) <= _PLAIN_OPTIONS
            attached = option.startswith(("-X", "-W")) or bool(launcher and _PY_VERSION.match(option))
            if option in ("-X", "-W"):
                i += 2
            elif plain or attached:
                i += 1
            else:
                return None, 0  # -c, -mmodule, an option cluster ending in m or c, a long option, "-"
        if i >= len(tokens):
            return None, 0
        if tokens[i] == "-m":
            return (_module_path(tokens[i + 1], root), i + 2) if i + 1 < len(tokens) else (None, i)
        return repo_script(tokens[i], root), i + 1
    target = _console_target(tokens[i], root)
    if target is not None:
        return _module_path(target.split(":", 1)[0], root), i + 1
    return repo_script(tokens[i], root), i + 1


def command_entry(command: Sequence[str], root: Path) -> str | None:
    """The repository entry point a recorded command runs (``trace``), or None when it cannot be traced."""
    return trace(command, root)[0]


def subcommand(command: Sequence[str], root: Path) -> str | None:
    """The first argument after the entry point that is not an option (``train`` in ``opengrad train x``)."""
    entry, after = trace(command, root)
    if entry is None:
        return None
    return next((str(t) for t in list(command)[after:] if not str(t).startswith("-")), None)


#: The labelling runner: pinned by the owner's decision on annotation logs, whatever its arguments name.
LABELLING_RUNNER = "scripts/run_external_annotation.py"
#: The one file through which the other pinned entry points may reach held-out data: it loads the When2Call test
#: questions as a set to exclude matching training records, and never prints them.
EXCLUSION_LOOKUP = "src/opengrad/data/classifier_input.py"
#: Entry points that read held-out data in some subcommands only: path -> those subcommands. ``opengrad baseline``
#: runs the frozen baseline or a candidate checkpoint on the held-out set, with a default config that does not
#: appear in its command.
HELD_OUT_SUBCOMMANDS: Mapping[str, frozenset[str]] = MappingProxyType({"src/opengrad/cli.py": frozenset({"baseline"})})
#: Approved for upload (owner's decision, 2026-11-13: an allowlist). A log uploads only when its entry point was
#: reviewed and approved: the entry points in ``PINNED_CONSOLE``, and these subcommands of these console commands
#: (matched by ``module:function``: ``opengrad-data-audit`` shares ``opengrad.cli`` but is not approved). Any other
#: log stays local
#: and is listed as not approved, until its entry point's console output is reviewed and it is added here or to
#: ``PINNED_CONSOLE``. The checks before it still apply to an approved entry point.
APPROVED_CONSOLE: Mapping[str, frozenset[str]] = MappingProxyType(
    {"opengrad.cli:main": frozenset({"train", "evaluate"})}
)
#: The run-record command: a run it wraps is recorded as its own run, so one that wraps another cannot be traced.
RECORDER = frozenset({"src/opengrad/runrecords/__main__.py", "src/opengrad/runrecords/cli.py"})


def named_entry_points(command: Sequence[str], root: Path) -> set[str]:
    """Every repository entry point a recorded command names anywhere: a script path in any argument or in a string
    handed to a shell (``bash -c "python scripts/x.py"``), and a module after ``-m`` (or an option cluster ending in
    ``m``)."""
    pieces = [piece for token in command for piece in str(token).split()]
    found: set[str] = set()
    for i, piece in enumerate(pieces):
        for value in argument_values(piece):
            script = repo_script(repo_relative(value, root), root)
            if script is not None:
                found.add(script)
        module = None
        if piece.startswith("-") and not piece.startswith("--") and piece.endswith("m") and i + 1 < len(pieces):
            module = pieces[i + 1]
        elif piece.startswith("-m") and len(piece) > 2 and "=" not in piece:
            module = piece[2:]
        if module and (path := _module_path(module, root)) is not None:
            found.add(path)
    return found


def upload_decision(record: Mapping[str, Any], root: Path) -> tuple[str, str]:
    """Whether a recorded run's log may be uploaded, and the kind to upload it as.

    ``never`` (held-out data; listed as kept by policy), ``unrecognised`` (cannot be checked; kept local and listed,
    owner's decision 2026-11-13: fail closed), ``unapproved`` (not reviewed for upload yet; kept local and listed)
    or ``allowed``. An allowlist (owner's decision, 2026-11-13): only a reviewed entry point uploads. In order:

    1. ``never``: the record carries the never stamp, or the command names a held-out entry point anywhere (a script
       handed to another program or a shell, a module after ``-m``).
    2. ``unrecognised``: the command is the run-record command itself, wrapping another run.
    3. ``never``: a subcommand that reads held-out data (``HELD_OUT_SUBCOMMANDS``: ``opengrad baseline``).
    4. Pinned entry points (``PINNED_CONSOLE``): ``never`` when the scan of the entry point as it was on disk when
       the run started (``held_out_reach``) finds more than the pinned review allowed (nothing for the labelling
       runner, ``EXCLUSION_LOOKUP`` for the others). The labelling runner is then ``allowed``, when it started
       from reviewed code (step 8), whatever its arguments name (a P-UNANS session or config); the others go on
       to the argument checks.
    5. ``never``: the run's arguments named held-out data when it started (``held_out_inputs``), or the recorded
       arguments, or a ``configs/`` file they name, name a marker.
    6. ``never``: the entry point is in ``HELD_OUT_ENTRY_POINTS``, or, unpinned, its scan at start found any
       marker-naming file (a script not tracked yet, or with local edits).
    7. ``unrecognised``: a named ``configs/`` file is gone, or the command does not trace to an entry point.
    8. ``allowed`` only for an approved entry point: one in ``PINNED_CONSOLE``, or an approved subcommand of an
       approved console command (``APPROVED_CONSOLE``: ``opengrad train`` and ``opengrad evaluate``), that started
       from reviewed code (``_as_reviewed``: its commit on ``origin/master``'s history, scanned, nothing it loads
       and no ``configs/`` file modified, untracked or ignored). Anything else is ``unapproved``. An approved run given an existing file outside ``configs/``
       that the checks cannot read (``unchecked_inputs``) is ``unrecognised``.

    The kind comes from ``POLICY`` when the entry point has one: the labelling runner launched through ``run
    --kind script`` still uploads as a redacted annotation copy.
    """
    command = [str(t) for t in record.get("command") or []]
    entry = command_entry(command, root)
    kind = POLICY[entry].kind if entry is not None and entry in POLICY else str(record.get("kind"))
    reach = record.get("held_out_reach")
    if record.get("upload_policy") == "never" or named_entry_points(command, root) & HELD_OUT_ENTRY_POINTS:
        return "never", kind
    if entry in RECORDER:
        return "unrecognised", kind
    sub = subcommand(command, root)
    if entry is not None and sub in HELD_OUT_SUBCOMMANDS.get(entry, frozenset()):
        return "never", kind
    if entry in PINNED_CONSOLE:
        reviewed = set() if entry == LABELLING_RUNNER else {EXCLUSION_LOOKUP}
        if reach is not None and not set(reach) <= reviewed:
            return "never", kind
        if entry == LABELLING_RUNNER:
            return ("allowed" if _as_reviewed(record, root) else "unapproved"), kind
    if record.get("held_out_inputs") or any(names_held_out(token) for token in command):
        return "never", kind
    configs = [config_state(token, root) for token in command]
    if "held_out" in configs:
        return "never", kind
    if entry in HELD_OUT_ENTRY_POINTS or (entry not in PINNED_CONSOLE and reach):
        return "never", kind
    if "missing" in configs or entry is None:
        return "unrecognised", kind
    approved = entry in PINNED_CONSOLE or sub in APPROVED_CONSOLE.get(console_target(command, root) or "", frozenset())
    if not approved or not _as_reviewed(record, root):
        return "unapproved", kind
    if record.get("unchecked_inputs"):
        return "unrecognised", kind
    return "allowed", kind


#: The history that counts as reviewed: commits pushed to master, which the owner's rule (2026-10-03) has an
#: independent audit check before every push and CI checks after it. A commit only on a local branch, or on another
#: branch, has not been. The full ref name, so a local branch or tag called ``origin/master`` cannot stand in for it.
REVIEWED_REF = "refs/remotes/origin/master"


#: A full git object id (SHA-1 or SHA-256).
_OBJECT_ID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


def _reviewed_tip(root: Path) -> str | None:
    """The commit ``REVIEWED_REF`` points at; None when it does not exist. ``show-ref --verify`` does no name lookup,
    so a local branch or tag named like it never counts."""
    try:
        ref = subprocess.run(
            ["git", "--no-optional-locks", "show-ref", "--verify", "--hash", REVIEWED_REF],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    tip = ref.stdout.strip()
    return tip if ref.returncode == 0 and _OBJECT_ID.fullmatch(tip) else None


def on_reviewed_branch(sha: object, root: Path) -> bool:
    """True when ``sha`` is in ``REVIEWED_REF``'s history in this checkout (``git merge-base --is-ancestor``)."""
    if not isinstance(sha, str) or not _OBJECT_ID.fullmatch(sha):
        return False
    tip = _reviewed_tip(root)
    if tip is None:
        return False
    try:
        done = subprocess.run(
            ["git", "--no-optional-locks", "merge-base", "--is-ancestor", sha, tip],
            cwd=root,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0


def configs_pushed(blobs: object, root: Path) -> bool:
    """True when every consumed config, as the run read it, exists at that path in ``REVIEWED_REF``'s history: a
    run made with a tweaked config uploads once that exact config is pushed (UP-0040)."""
    if blobs is None:
        return False  # the consumed configs could not be determined
    if not isinstance(blobs, Mapping):
        return False
    if not blobs:
        return True
    tip = _reviewed_tip(root)
    if tip is None:
        return False
    for path, blob in blobs.items():
        if not (isinstance(path, str) and path.startswith("configs/") and isinstance(blob, str) and _OBJECT_ID.fullmatch(blob)):
            return False
        try:
            # Literal pathspecs: a recorded path is a file name, never a pattern. Path-limited history skips a
            # version that only a merge commit introduced, which fails closed.
            log = subprocess.run(
                ["git", "--no-optional-locks", "--literal-pathspecs", "log", tip, "--format=", "--raw", "--no-abbrev",
                 "--no-renames", "--", path],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        # ":100644 100644 <old blob> <new blob> M\t<path>": every version the path has had on that history.
        versions = {line.split()[3] for line in log.stdout.splitlines() if line.startswith(":") and len(line.split()) > 4}
        if log.returncode != 0 or blob not in versions:
            return False
    return True


def _as_reviewed(record: Mapping[str, Any], root: Path) -> bool:
    """The run started from reviewed code: its commit is on ``REVIEWED_REF``'s history, its start record has the scan
    (``held_out_reach``), no file the entry point loads, nor any ``configs/`` file it does not consume, was modified,
    untracked or ignored (``entry_modified`` empty), and each config it consumes, as it read it, exists at that path
    in ``REVIEWED_REF``'s history (``config_blobs``). A record from before these checks, or whose check failed, is
    not vouched for."""
    git = record.get("git")
    sha = git.get("sha") if isinstance(git, Mapping) else None
    return (
        record.get("held_out_reach") is not None
        and record.get("entry_modified") == []
        and on_reviewed_branch(sha, root)
        and configs_pushed(record.get("config_blobs", {}), root)
    )


def import_closure(entry: str, root: Path, tracked: set[str]) -> set[str]:
    """``entry`` and every repository file it loads, transitively: imports anywhere in the code (relative ones too),
    and ``.py`` files named in string constants (``spec_from_file_location``, ``runpy.run_path``)."""
    seen, stack = {entry}, [entry]
    while stack:
        path = stack.pop()
        for dep in _loads(path, root, tracked):
            if dep not in seen:
                seen.add(dep)
                stack.append(dep)
    return seen


def _module_file(module: str, base: str, tracked: set[str]) -> str | None:
    stem = (base + "/" if base else "") + module.replace(".", "/") if module else base
    for candidate in (stem + ".py", stem + "/__init__.py"):
        if candidate in tracked:
            return candidate
    return None


#: The policy files name every marker and every held-out path to define the policy; they read no data, so the scan
#: neither counts their markers nor follows the paths they list.
POLICY_FILES = frozenset(
    {
        "src/opengrad/runrecords/scripts.py",
        "src/opengrad/runrecords/record.py",
        "src/opengrad/runrecords/inputs.py",
    }
)


def _loads(path: str, root: Path, tracked: set[str]) -> set[str]:
    here = posixpath.dirname(path)
    if path in POLICY_FILES:
        return set()
    found: set[str] = set()

    def add(module: str, base: str) -> None:
        parts = module.split(".") if module else []
        for end in range(0 if base else 1, len(parts) + 1):
            hit = _module_file(".".join(parts[:end]), base, tracked)
            if hit:
                found.add(hit)

    for node in ast.walk(ast.parse((root / path).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.name, "src" if alias.name.split(".")[0] == "opengrad" else here)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = here
                for _ in range(node.level - 1):
                    base = posixpath.dirname(base)
            else:
                base = "src" if (node.module or "").split(".")[0] == "opengrad" else here
            module = node.module or ""
            add(module, base)
            for alias in node.names:
                add(f"{module}.{alias.name}" if module else alias.name, base)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.endswith(".py"):
            name = node.value.replace("\\", "/")
            for candidate in (posixpath.normpath(name), posixpath.normpath(posixpath.join(here, name))):
                if candidate in tracked:
                    found.add(candidate)
    return found


def marked_files(root: Path, paths: set[str]) -> set[str]:
    """The files among ``paths`` whose code names a held-out marker (the policy files do not count)."""
    return {
        path
        for path in paths
        if path not in POLICY_FILES
        and any(m in code_text((root / path).read_text(encoding="utf-8")) for m in HELD_OUT_MARKERS)
    }


def held_out_reach(root: Path, tracked: set[str], entry_points: Sequence[str]) -> dict[str, set[str]]:
    """Entry point -> the marker-naming files (itself or anything it loads) that reach held-out data.

    The policy files (``POLICY_FILES``) do not count.
    """
    cache: dict[str, bool] = {}

    def marked(path: str) -> bool:
        if path not in cache:
            cache[path] = path not in POLICY_FILES and any(
                m in code_text((root / path).read_text(encoding="utf-8")) for m in HELD_OUT_MARKERS
            )
        return cache[path]

    reach: dict[str, set[str]] = {}
    for entry in entry_points:
        hits = {path for path in import_closure(entry, root, tracked) if marked(path)}
        if hits:
            reach[entry] = hits
    return reach


def policy_for(relative: str) -> ScriptPolicy:
    return POLICY.get(relative, ScriptPolicy())


def _code(result: object) -> int:
    """The exit code ``raise SystemExit(result)`` would give: 0 for None, the int itself, else 1 with the value
    printed to standard error (as SystemExit prints a message), so converting an entry point changes nothing."""
    if result is None:
        return 0
    if isinstance(result, int):
        return int(result)
    print(result, file=sys.stderr)
    return 1


def run_script(file: str, main: Callable[[], object]) -> int:
    """Run ``main()`` as a recorded run under this entry point's policy, and return its exit code.

    Recording never blocks a run: outside a repository checkout, or if the record cannot start, ``main()`` runs
    unrecorded (``recorded_call`` warns in the second case).
    """
    path = Path(file).resolve()
    root = repo_root(path)
    if root is None:
        return _code(main())
    relative = path.relative_to(root).as_posix()
    policy = policy_for(relative)
    # The script's own repository path when it was run as a file (from any folder); a console command keeps its
    # own name, so two commands in one module stay apart in the record.
    first = sys.argv[0] if sys.argv else ""
    try:
        as_file = bool(first) and Path(first).resolve() == path
    except OSError:
        as_file = False
    argv0 = relative if as_file or not first else first
    return recorded_call(
        policy.kind,
        [argv0, *sys.argv[1:]],
        root,
        lambda: _code(main()),
        expected_exit_codes=dict(policy.expected_exit_codes),
        upload="never" if relative in HELD_OUT_ENTRY_POINTS else "allowed",
    )
