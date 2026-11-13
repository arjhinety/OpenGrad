"""Every entry point records its runs (run-record stage 3): ``opengrad.runrecords.scripts``."""

from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from opengrad.runrecords import cli as rr_cli
from opengrad.runrecords import record as rr
from opengrad.runrecords import scripts as rs

ROOT = Path(__file__).resolve().parents[2]
#: Not converted yet; later slices of stage 3 remove these. An entry may only leave this list.
PENDING = "not yet converted (run-record stage 3, a later slice)"
#: Entry points without the recording call, with the reason. The list may only shrink.
ALLOWLIST: dict[str, str] = {
    "scripts/agy_tool_gate.py": 'runs once per tool call inside an annotation run that is itself recorded',
    "scripts/archive_devset_model_labels.py": PENDING,
    "scripts/archive_external_model_labels.py": PENDING,
    "scripts/archive_pdet_model_batches.py": PENDING,
    "scripts/assemble_m1_preference_dataset.py": PENDING,
    "scripts/audit_bfcl_answer_supply.py": PENDING,
    "scripts/audit_campaign_final.py": PENDING,
    "scripts/audit_canonical_v2_final_direct.py": PENDING,
    "scripts/audit_corpus_direct_prevalence.py": PENDING,
    "scripts/audit_executorch_quantization.py": PENDING,
    "scripts/audit_first_reply_supply.py": PENDING,
    "scripts/audit_ladder_tokenizer_divergence.py": PENDING,
    "scripts/audit_normalization_v3_structure.py": PENDING,
    "scripts/audit_pdet_coverage_supply.py": PENDING,
    "scripts/audit_pdet_coverage_supply_v3.py": PENDING,
    "scripts/audit_pdet_coverage_v2_supply.py": PENDING,
    "scripts/audit_pdet_v1_representation.py": PENDING,
    "scripts/audit_punans_constructed_supply.py": PENDING,
    "scripts/audit_punans_supply.py": PENDING,
    "scripts/audit_punans_v2_supply.py": PENDING,
    "scripts/audit_sft_refusal_supervision.py": PENDING,
    "scripts/audit_toolace_call_final_shape.py": PENDING,
    "scripts/audit_xlam_normalization.py": PENDING,
    "scripts/build_capability_cost_ledger.py": PENDING,
    "scripts/build_capability_evidence.py": PENDING,
    "scripts/build_capability_report.py": PENDING,
    "scripts/build_checkpoint_ladder.py": PENDING,
    "scripts/build_engine_parity_report.py": PENDING,
    "scripts/build_executorch_handoff.py": PENDING,
    "scripts/build_final_verdict.py": PENDING,
    "scripts/build_final_yield_report.py": PENDING,
    "scripts/build_findings_visual.py": PENDING,
    "scripts/build_gguf_card.py": PENDING,
    "scripts/build_gguf_handoff.py": PENDING,
    "scripts/build_h200_benchmark_report.py": PENDING,
    "scripts/build_m1_local_preference_pairs.py": PENDING,
    "scripts/build_minus_xlam_release.py": PENDING,
    "scripts/build_ptq_evaluation_report.py": PENDING,
    "scripts/build_quantization_visual.py": PENDING,
    "scripts/build_regression_analysis.py": PENDING,
    "scripts/build_tokenizer_regression_fixtures.py": PENDING,
    "scripts/build_when2call_preference_pairs.py": PENDING,
    "scripts/build_xlam_from_derivative.py": PENDING,
    "scripts/characterize_refusal.py": PENDING,
    "scripts/close_ptq_phase.py": PENDING,
    "scripts/compare_gemini_relabel.py": PENDING,
    "scripts/derive_ablation_yield_report.py": PENDING,
    "scripts/evaluate_prose_classifier_dev.py": PENDING,
    "scripts/evaluate_prose_classifier_v2_dev.py": PENDING,
    "scripts/evaluate_sft_checkpoints.py": PENDING,
    "scripts/fetch_and_score_capability.py": PENDING,
    "scripts/freeze_eval_partition.py": PENDING,
    "scripts/freeze_m0_final.py": PENDING,
    "scripts/freeze_minus_xlam_ablation.py": PENDING,
    "scripts/freeze_minus_xlam_v2.py": PENDING,
    "scripts/grade_openweights_parity.py": PENDING,
    "scripts/measure_ablation_mass.py": PENDING,
    "scripts/prepare_capability_benchmarks.py": PENDING,
    "scripts/prepare_imatrix_calibration.py": PENDING,
    "scripts/prepare_quantization_inputs.py": PENDING,
    "scripts/preserve_h200_state.py": PENDING,
    "scripts/publish_card_corrections.py": PENDING,
    "scripts/publish_executorch_artifact.py": PENDING,
    "scripts/rebuild_eval_splits.py": PENDING,
    "scripts/render_frozen_prompts.py": PENDING,
    "scripts/repo/check_publication_hygiene.py": PENDING,
    "scripts/report_toolace_call_prediction.py": PENDING,
    "scripts/reporting/generate_experiment_status.py": PENDING,
    "scripts/reporting/generate_indexes.py": PENDING,
    "scripts/reporting/generate_source_views.py": PENDING,
    "scripts/reporting/generate_upstream_log.py": PENDING,
    "scripts/run_gguf_ladder.py": PENDING,
    "scripts/sanity_check_rendered_samples.py": PENDING,
    "scripts/score_gguf_candidate.py": PENDING,
    "scripts/score_gsm8k.py": PENDING,
    "scripts/score_ifeval.py": PENDING,
    "scripts/score_mmlu_pro.py": PENDING,
    "scripts/score_quantized_artifact.py": PENDING,
    "scripts/score_sentinel.py": PENDING,
    "scripts/select_checkpoint.py": PENDING,
    "scripts/smoke_gpu_dpo_opd.py": PENDING,
    "scripts/update_quantization_ledger.py": PENDING,
    "scripts/verify_publication.py": PENDING,
    "src/opengrad/data/behaviour_labels.py": PENDING,
    "src/opengrad/data/behaviour_labels_v3.py": PENDING,
    "src/opengrad/data/canonical_v3.py": PENDING,
    "src/opengrad/data/canonical_v3_balance.py": PENDING,
    "src/opengrad/data/normalization_v3.py": PENDING,
    "src/opengrad/data/provenance_gate.py": PENDING,
    "src/opengrad/optimization/__main__.py": PENDING,
    "src/opengrad/registry/validate.py": PENDING,
    "src/opengrad/results/cli.py": PENDING,
    "src/opengrad/runrecords/cli.py": 'the run-record command itself; recording it would record the recorder',
    "src/opengrad/training/exposure_planner.py": PENDING,
    "src/opengrad/training/preprocess.py": PENDING,
    "src/opengrad/verification/answer_strata.py": PENDING,
    "src/opengrad/verification/answer_strata_report.py": PENDING,
    "src/opengrad/verification/call_fidelity.py": PENDING,
    "src/opengrad/verification/classifier_devcheck.py": PENDING,
    "src/opengrad/verification/classifier_devset.py": PENDING,
    "src/opengrad/verification/classifier_devset_v2.py": PENDING,
    "src/opengrad/verification/flag_set.py": PENDING,
    "src/opengrad/verification/flag_triage_population.py": PENDING,
    "src/opengrad/verification/flag_triage_report.py": PENDING,
    "src/opengrad/verification/pconf.py": PENDING,
    "src/opengrad/verification/pdet.py": PENDING,
    "src/opengrad/verification/pdet_coverage.py": PENDING,
    "src/opengrad/verification/pdet_coverage_reference.py": PENDING,
    "src/opengrad/verification/pdet_coverage_v2.py": PENDING,
    "src/opengrad/verification/prose_classifier_oneshot.py": PENDING,
    "src/opengrad/verification/prose_classifier_v2_oneshot.py": PENDING,
    "src/opengrad/verification/punans.py": PENDING,
    "src/opengrad/verification/punans_constructed.py": PENDING,
    "src/opengrad/verification/punans_constructed_report.py": PENDING,
    "src/opengrad/verification/punans_report.py": PENDING,
    "src/opengrad/verification/punans_v2.py": PENDING,
    "src/opengrad/verification/punans_v2_report.py": PENDING,
    "src/opengrad/verification/study_002_gate.py": PENDING,
    "src/opengrad/verification/study_002_readiness.py": PENDING,
}
#: ``[project.scripts]`` commands that do not go through ``run_script``, with the reason.
CONSOLE_ALLOWLIST: dict[str, str] = {
    "opengrad": "records its own train and evaluate runs (opengrad.cli, recorded_call)",
    "opengrad-preflight": PENDING,
    "opengrad-validate": PENDING,
    "opengrad-contamination": PENDING,
    "opengrad-data-audit": PENDING,
    "opengrad-data": PENDING,
    "opengrad-benchmark": PENDING,
}
ALLOWLIST_CEILING = 119
CONSOLE_ALLOWLIST_CEILING = 7
RECORDING_CALLS = {"run_script", "recorded_main"}


def _tracked_python() -> list[str]:
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [p for p in out.split() if p.startswith(("scripts/", "src/"))]


def _main_block(tree: ast.Module) -> ast.If | None:
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and getattr(node.test.left, "id", "") == "__name__"
        ):
            return node
    return None


def _calls(node: ast.AST) -> set[str]:
    return {
        getattr(c.func, "id", getattr(c.func, "attr", ""))
        for c in ast.walk(node)
        if isinstance(c, ast.Call)
    }


def _entry_points() -> dict[str, ast.Module]:
    found = {}
    for path in _tracked_python():
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        if _main_block(tree) is not None:
            found[path] = tree
    return found


def test_every_entry_point_records_its_runs_or_is_allowlisted() -> None:
    entries = _entry_points()
    assert len(entries) >= 100, f"only {len(entries)} entry points found; the scan broke"
    missing = []
    for path, tree in entries.items():
        block = _main_block(tree)
        assert block is not None
        if not _calls(block) & RECORDING_CALLS and path not in ALLOWLIST:
            missing.append(path)
    assert missing == [], f"these entry points do not record their runs: {missing}"


def test_the_allowlist_only_shrinks() -> None:
    entries = _entry_points()
    stale = [
        path
        for path in ALLOWLIST
        if path not in entries or _calls(_main_block(entries[path]) or ast.Module([], [])) & RECORDING_CALLS
    ]
    assert stale == [], f"remove these from ALLOWLIST: {stale}"
    # Lower the ceiling as entry points are converted; raising it needs the owner's word.
    assert len(ALLOWLIST) <= ALLOWLIST_CEILING and len(CONSOLE_ALLOWLIST) <= CONSOLE_ALLOWLIST_CEILING


def test_a_recorded_main_wrapper_calls_run_script() -> None:
    for path, tree in _entry_points().items():
        wrappers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "recorded_main"]
        for wrapper in wrappers:
            assert "run_script" in _calls(wrapper), path


def test_every_console_command_records_or_is_allowlisted() -> None:
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    for name, target in scripts.items():
        if name in CONSOLE_ALLOWLIST:
            assert not target.endswith(":recorded_main"), f"{name} records now; remove it from CONSOLE_ALLOWLIST"
            continue
        assert target.endswith(":recorded_main"), f"{name} -> {target} does not record its runs"
    assert set(CONSOLE_ALLOWLIST) <= set(scripts), "CONSOLE_ALLOWLIST names a command that no longer exists"


def _held_out_reach() -> dict[str, set[str]]:
    """Entry point -> the marker-naming files it loads (``opengrad.runrecords.scripts.held_out_reach``)."""
    return rs.held_out_reach(ROOT, set(_tracked_python()), sorted(_entry_points()))


def test_the_held_out_list_equals_a_scan_of_the_code_and_its_imports() -> None:
    reach = _held_out_reach()
    assert len(reach) >= 60, f"only {len(reach)} entry points reach held-out data; the scan broke"
    assert set(rs.HELD_OUT_ENTRY_POINTS) == set(reach) - set(rs.PINNED_CONSOLE)


def test_a_pinned_console_entry_point_reaches_held_out_data_only_through_the_exclusion_lookup() -> None:
    # Owner's decision, 2026-11-13: an entry point that reaches held-out data only through classifier_input (which
    # loads the test questions as a set to exclude matching training records, and never prints them) may upload
    # once a test pins its console. One that names a marker itself, or imports P-UNANS / P-CONF code, may not.
    # The labelling runner reaches nothing; it is pinned by the owner's decision on annotation logs.
    reach = _held_out_reach()
    assert not set(rs.PINNED_CONSOLE) & set(rs.HELD_OUT_ENTRY_POINTS)
    for path, test in rs.PINNED_CONSOLE.items():
        assert (ROOT / test).is_file(), (path, test)
        assert reach.get(path, set()) <= {"src/opengrad/data/classifier_input.py"}, (path, reach.get(path))


def test_the_held_out_scan_reads_code_not_comments_or_docstrings() -> None:
    source = (
        '"""Module doc naming punans."""\n'
        "def f():\n"
        '    """Function doc naming confirmatory."""\n'
        "    return 1  # the punans-v2-trial session\n"
    )
    assert not any(m in rs.code_text(source) for m in rs.HELD_OUT_MARKERS)
    assert "punans" in rs.code_text(source + 'PATH = "reports/study-002/punans-v2"\n')


def test_the_labelling_runner_uploads_redacted_copies() -> None:
    # Owner's decision, 2026-11-13: its logs go up as redacted annotation copies, not kept local.
    assert "scripts/run_external_annotation.py" not in rs.HELD_OUT_ENTRY_POINTS
    assert rs.POLICY["scripts/run_external_annotation.py"].kind in rr_cli.ALWAYS_REDACTED_KINDS


def test_policies_are_well_formed() -> None:
    for path, policy in rs.POLICY.items():
        assert (ROOT / path).is_file(), path
        assert policy.kind in rr.KINDS, path
        for code, reason in policy.expected_exit_codes.items():
            assert isinstance(code, int) and code != 0 and reason.strip(), (path, code)


def _script(tmp_path: Path, relative: str) -> Path:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n', encoding="utf-8")
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def test_a_declared_exit_code_records_as_pass_and_keeps_the_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _script(tmp_path, "scripts/check_thing.py")
    monkeypatch.setattr(
        rs, "POLICY", {"scripts/check_thing.py": rs.ScriptPolicy(expected_exit_codes={1: "stale output reported"})}
    )
    monkeypatch.setattr(sys, "argv", ["check_thing.py", "--check"])
    assert rs.run_script(str(path), lambda: 1) == 1
    assert rs.run_script(str(path), lambda: 2) == 2
    by_code = {r["exit_code"]: r for r in rr.read_local(tmp_path)}
    assert by_code[1]["status"] == "PASS" and by_code[1]["expected_exit_codes"] == {"1": "stale output reported"}
    assert by_code[2]["status"] == "FAILED"


def test_a_held_out_entry_point_is_marked_never_uploaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _script(tmp_path, "scripts/score_thing.py")
    monkeypatch.setattr(rs, "HELD_OUT_ENTRY_POINTS", frozenset({"scripts/score_thing.py"}))
    monkeypatch.setattr(sys, "argv", ["score_thing.py"])
    assert rs.run_script(str(path), lambda: None) == 0
    (record,) = rr.read_local(tmp_path)
    assert record["upload_policy"] == "never" and record["status"] == "PASS" and record["kind"] == "script"


def test_outside_a_checkout_the_script_runs_unrecorded(tmp_path: Path) -> None:
    lone = tmp_path / "lone.py"
    lone.write_text("", encoding="utf-8")
    assert rs.run_script(str(lone), lambda: 3) == 3
    assert rr.read_local(tmp_path) == []


RUNNER = "scripts/run_external_annotation.py"


@pytest.mark.parametrize(
    ("command", "entry"),
    [
        (["python", "-u", RUNNER, "--session", "x"], RUNNER),
        ([".venv/Scripts/python.exe", "-X", "utf8", "-m", "opengrad.annotation.cli"], "src/opengrad/annotation/cli.py"),
        (["uv", "run", "python", "scripts\\audit_punans_supply.py"], "scripts/audit_punans_supply.py"),
        (["PYTHONHASHSEED=0", "python3.11", "./scripts/audit_punans_supply.py"], "scripts/audit_punans_supply.py"),
        ([".venv/Scripts/opengrad-annotate.exe", "label"], "src/opengrad/annotation/cli.py"),
        (["opengrad", "train", "configs/x.yaml"], "src/opengrad/cli.py"),
        (["src/opengrad/verification/punans.py", "--check"], "src/opengrad/verification/punans.py"),
        (["python", "-c", "print(1)"], None),
        (["bash", "-c", "python scripts/audit_punans_supply.py"], None),
        (["pytest", "tests"], None),
        (["python", "scripts/does_not_exist.py"], None),
        (["<absolute-path>"], None),
        ([], None),
    ],
)
def test_a_recorded_command_is_traced_to_its_entry_point(command: list[str], entry: str | None) -> None:
    assert rs.command_entry(command, ROOT) == entry


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        # Launched through `run --kind script`: still the held-out entry point, still never uploaded.
        ({"kind": "script", "command": ["python", "scripts/audit_punans_supply.py"]}, ("never", "script")),
        # The labelling runner on a P-UNANS session: its console is pinned, so it uploads as an annotation copy.
        ({"kind": "script", "command": ["python", RUNNER, "--session", "punans-v2-trial"]}, ("allowed", "annotation")),
        # An argument naming held-out data keeps an otherwise clean command local.
        ({"kind": "evaluation", "command": ["opengrad", "evaluate", "release/x/frozen_prompts_v1.jsonl"]}, ("never", "evaluation")),
        ({"kind": "evaluation", "command": ["opengrad", "evaluate", "RESULTS/Confirmatory/x"]}, ("never", "evaluation")),
        # Stamped at record time.
        ({"kind": "script", "command": ["opengrad", "train"], "upload_policy": "never"}, ("never", "script")),
        # Fail closed: a command that cannot be traced (owner's decision, 2026-11-13).
        ({"kind": "script", "command": ["bash", "run_all.sh"]}, ("unrecognised", "script")),
        ({"kind": "training", "command": ["opengrad", "train", "configs/benchmark_suites/smoke.yaml"]}, ("allowed", "training")),
        # A named config that is gone cannot be checked: fail closed.
        ({"kind": "training", "command": ["opengrad", "train", "configs/does_not_exist.yaml"]}, ("unrecognised", "training")),
        # Set when the run started, before redaction turned an outside path into <absolute-path>.
        ({"kind": "script", "command": ["python", "scripts/x.py", "<absolute-path>"], "held_out_inputs": True},
         ("never", "script")),
        # A path that is not normalised, or in another case, is still the held-out entry point.
        ({"kind": "script", "command": ["python", "scripts/../scripts/FREEZE_EVAL_PARTITION.py"]}, ("never", "script")),
        ({"kind": "script", "command": ["python", "scripts//freeze_eval_partition.py"]}, ("never", "script")),
        # A held-out script handed to a pinned one, or loaded by -c, or run first through PYTHONSTARTUP.
        ({"kind": "script", "command": ["python", "scripts/audit_pdet_v1_representation.py", "scripts/freeze_eval_partition.py"]},
         ("never", "script")),
        ({"kind": "script", "command": ["python", "-cimport runpy", "scripts/audit_pdet_v1_representation.py"]},
         ("unrecognised", "script")),
        ({"kind": "script", "command": ["PYTHONSTARTUP=x.py", "python", "scripts/audit_pdet_v1_representation.py"]},
         ("unrecognised", "script")),
        ({"kind": "script", "command": ["uv", "run", "--env-file", "scripts/audit_pdet_v1_representation.py", "python", "x"]},
         ("unrecognised", "script")),
        # The labelling runner on a P-UNANS task config: pinned, so allowed.
        ({"kind": "script", "command": ["python", RUNNER, "--task", "configs/annotation/punans-v2.yaml"]}, ("allowed", "annotation")),
        # Through the config it is given: this candidate config points at the confirmatory partition.
        ({"kind": "evaluation", "command": ["opengrad", "train", "configs/evaluation/candidates/m0_sft_canonical_v2_final/checkpoint-1800.yaml"]},
         ("never", "evaluation")),
        ({"kind": "script", "command": ["python", "scripts/train.py", "--config=configs\\evaluation\\candidates\\m0_sft_canonical_v2_final\\checkpoint-1800.yaml"]},
         ("never", "script")),
        # This experiment config says "confirmatory" only in its description (prose, not a path): it reaches nothing.
        ({"kind": "training", "command": ["opengrad", "train", "configs/experiments/m1_dpo_canonical_v2_final.yaml"]},
         ("allowed", "training")),
    ],
)
def test_the_upload_decision_is_made_from_the_recorded_command(record: dict, expected: tuple[str, str]) -> None:
    assert rs.upload_decision({**REVIEWED, **record}, ROOT) == expected



def test_opengrad_evaluate_cannot_reach_held_out_data_through_a_suite() -> None:
    # `opengrad evaluate TARGET --suite NAME` reads configs/benchmark_suites/NAME.yaml and the benchmark configs it
    # lists. Its record names only the suite, so upload_decision cannot see held-out data reached that way: no suite
    # or benchmark config may name it. If one must, give `opengrad evaluate` a policy first.
    named = [
        path.relative_to(ROOT).as_posix()
        for folder in ("configs/benchmark_suites", "configs/benchmarks")
        for path in sorted((ROOT / folder).rglob("*"))
        if path.is_file()
        and any(m in path.read_text(encoding="utf-8", errors="replace").lower() for m in rs.HELD_OUT_MARKERS)
    ]
    assert named == []


#: A start record from reviewed code: scanned, nothing loaded modified (inputs.py entry_scan), and a commit that
#: the fixture below treats as on origin/master's history (on_reviewed_branch has its own test).
REVIEWED = {"held_out_reach": [], "entry_modified": [], "git": {"sha": "0" * 40}}


@pytest.fixture(autouse=True)
def _reviewed_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rs, "on_reviewed_branch", lambda sha, root: sha == "0" * 40)


ABS = str(ROOT / "configs/evaluation/candidates/m0_sft_canonical_v2_final/checkpoint-1800.yaml")


@pytest.mark.parametrize(
    "command",
    [
        # Wrapped: the run-record command around a held-out module, or a shell around a held-out script.
        ["python", "-m", "opengrad.runrecords", "run", "--", "python", "-m", "opengrad.verification.pconf"],
        ["bash", "-c", "python scripts/select_checkpoint.py --x"],
        ["python", "-Bm", "opengrad.verification.pdet"],
        # The command that runs a model on the held-out set, with its default config.
        ["opengrad", "baseline"],
        [".venv/Scripts/opengrad.exe", "baseline", "--config", "configs/x.yaml"],
        # A pinned entry point given a held-out config: the console review assumed its usual inputs.
        ["python", "-m", "opengrad.verification.flag_set", "--config", "configs/annotation/punans-v2.yaml"],
    ],
)
def test_a_wrapped_or_held_out_run_is_never_uploaded(command: list[str]) -> None:
    assert rs.upload_decision({"kind": "script", "command": command}, ROOT)[0] == "never"


def test_the_run_record_command_wrapping_a_clean_run_cannot_be_traced() -> None:
    command = ["python", "-m", "opengrad.runrecords", "run", "--", "python", "-c", "print(1)"]
    assert rs.upload_decision({"kind": "script", "command": command}, ROOT)[0] == "unrecognised"


def test_an_absolute_config_path_inside_the_checkout_is_read(tmp_path: Path) -> None:
    for token in (ABS, f"--config={ABS}", ABS.replace("\\", "/"), ABS[0].lower() + ABS[1:]):
        assert rr.config_state(token, ROOT) == "held_out", token


def test_a_pinned_entry_point_whose_code_reaches_more_than_reviewed_is_never_uploaded() -> None:
    clean = {"kind": "script", "command": ["python", "scripts/audit_pdet_v1_representation.py"], **REVIEWED}
    reviewed = {**clean, "held_out_reach": ["src/opengrad/data/classifier_input.py"]}
    edited = {**clean, "held_out_reach": ["src/opengrad/data/classifier_input.py", "scripts/x.py"]}
    runner = {"kind": "script", "command": ["python", RUNNER], **REVIEWED, "held_out_reach": ["scripts/x.py"]}
    assert rs.upload_decision(reviewed, ROOT)[0] == "allowed"
    assert rs.upload_decision(edited, ROOT)[0] == "never"
    assert rs.upload_decision(runner, ROOT)[0] == "never"
    assert rs.upload_decision({**clean, "held_out_reach": ["<scan failed>"]}, ROOT)[0] == "never"



@pytest.mark.parametrize(
    ("command", "decision"),
    [
        (["opengrad", "train", "configs/benchmark_suites/smoke.yaml"], "allowed"),
        ([".venv/Scripts/opengrad.exe", "evaluate", "some-model", "--suite", "smoke"], "allowed"),
        # opengrad.cli has no __main__ block, so this runs nothing; approval is by console command.
        (["python", "-m", "opengrad.cli", "--json", "evaluate", "m"], "unapproved"),
        # Another console command sharing the module, and a program of the same name elsewhere.
        (["opengrad-data-audit", "--config", "train"], "unapproved"),
        (["tools/opengrad", "train", "configs/benchmark_suites/smoke.yaml"], "unrecognised"),
        # Not reviewed: another subcommand, an unlisted clean script, the annotation CLI.
        (["opengrad", "compare", "a", "b"], "unapproved"),
        (["python", "scripts/prepare_capability_benchmarks.py"], "unapproved"),
        ([".venv/Scripts/opengrad-annotate.exe", "status"], "unapproved"),
        # Reviewed: the labelling runner and one of the fifteen.
        (["python", RUNNER], "allowed"),
        (["python", "scripts/audit_pdet_v1_representation.py"], "allowed"),
    ],
)
def test_only_an_approved_entry_point_uploads(command: list[str], decision: str) -> None:
    # Owner's decision, 2026-11-13: an allowlist. Everything not reviewed stays local, listed.
    assert rs.upload_decision({"kind": "script", "command": command, **REVIEWED}, ROOT)[0] == decision


def test_every_approved_entry_point_exists() -> None:
    for path in rs.PINNED_CONSOLE:
        assert (ROOT / path).is_file(), path
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    for target in rs.APPROVED_CONSOLE:
        assert target in scripts.values(), target


def test_the_training_configs_keep_their_backup_and_the_reading_configs_do_not() -> None:
    # checkpoint_selection_manifest is exempt (only checked for existence); behavioral_manifest reads the set.
    for path in sorted((ROOT / "configs/experiments").glob("m0_*.yaml")):
        assert rr.config_state(path.relative_to(ROOT).as_posix(), ROOT) == "", path.name
    assert rr.config_state("configs/evaluation/tool_calling/qwen35_2b_baseline.yaml", ROOT) == "held_out"
    assert rr.config_state("configs/data/preferences/toolpolicy_dpo_v1.yaml", ROOT) == ""


@pytest.mark.parametrize(
    "start",
    [
        {},  # a record from before the run-start checks
        {"held_out_reach": [], "entry_modified": ["scripts/audit_pdet_v1_representation.py"]},  # a local edit
        {"held_out_reach": [], "entry_modified": ["configs/benchmark_suites/smoke.yaml"]},
        {"held_out_reach": [], "entry_modified": ["<git status failed>"]},
        {"held_out_reach": None, "entry_modified": None},
    ],
)
def test_an_approved_entry_point_uploads_only_from_the_reviewed_code(start: dict) -> None:
    for command in (["opengrad", "train", "configs/benchmark_suites/smoke.yaml"], ["python", RUNNER],
                    ["python", "scripts/audit_pdet_v1_representation.py"]):
        assert rs.upload_decision({"kind": "script", "command": command, **start}, ROOT)[0] == "unapproved"


def test_an_approved_run_given_a_file_outside_configs_is_not_uploaded() -> None:
    record = {"kind": "training", "command": ["opengrad", "train", "configs/benchmark_suites/smoke.yaml"],
              "unchecked_inputs": True}
    assert rs.upload_decision({**REVIEWED, **record}, ROOT)[0] == "unrecognised"


def test_a_run_from_a_commit_not_on_the_reviewed_branch_is_not_approved() -> None:
    record = {"kind": "training", "command": ["opengrad", "train", "configs/benchmark_suites/smoke.yaml"], **REVIEWED}
    assert rs.upload_decision(record, ROOT)[0] == "allowed"
    assert rs.upload_decision({**record, "git": {"sha": "1" * 40}}, ROOT)[0] == "unapproved"
    assert rs.upload_decision({**record, "git": {"sha": None}}, ROOT)[0] == "unapproved"


@pytest.mark.parametrize("launcher", [".venv/../tools/opengrad", ".venv/x/opengrad.cmd", ".venv/Scripts/../../evil/opengrad.exe"])
def test_only_the_checkouts_own_launcher_counts_as_the_console_command(launcher: str) -> None:
    record = {"kind": "training", "command": [launcher, "train", "configs/benchmark_suites/smoke.yaml"], **REVIEWED}
    assert rs.upload_decision(record, ROOT)[0] != "allowed"

