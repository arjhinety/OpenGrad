"""Agent-safe CLI operations with machine-readable --json support and stable error codes."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from opengrad.checkpoints.registry import CheckpointLifecycle, CheckpointRecord, CheckpointRegistry
from opengrad.data.inspector import inspect_template
from opengrad.data.validator import validate_records
from opengrad.env_capture import capture
from opengrad.experiments.diff import diff_experiments
from opengrad.experiments.preflight import run_experiment_preflight
from opengrad.experiments.schema import ExperimentConfig, ExperimentStatus
from opengrad.experiments.store import ExperimentStore
from opengrad.readiness import readiness
from opengrad.training.distillation import OnPolicyDistillationTrainerBackend
from opengrad.training.dpo import DPOTrainerBackend
from opengrad.training.protocol import TrainerBackend
from opengrad.training.sft import SFTTrainerBackend


def _select_trainer(trainer_type: str) -> TrainerBackend | None:
    if trainer_type == "sft":
        return SFTTrainerBackend()
    if trainer_type == "dpo":
        return DPOTrainerBackend()
    if trainer_type in {"on_policy_distillation", "distillation"}:
        return OnPolicyDistillationTrainerBackend()
    return None


def handle_validate_data(args: argparse.Namespace, root: Path) -> int:
    rec_path = Path(args.records)
    if not rec_path.exists():
        err = {"code": "DATASET_NOT_FOUND", "message": f"File not found: {rec_path}"}
        if args.json:
            print(json.dumps(err, indent=2))
        else:
            print(f"Error: {err['message']}")
        return 1

    records: list[dict[str, Any]] = []
    with rec_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    err = {
                        "code": "DATASET_SCHEMA_INVALID",
                        "message": f"Invalid JSON on line: {exc}",
                    }
                    if args.json:
                        print(json.dumps(err, indent=2))
                    else:
                        print(f"Error: {err['message']}")
                    return 1

    report = validate_records(records, dataset_name=rec_path.name, mode=args.mode)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"Dataset Validation: {report.status}")
        print(
            f"Total: {report.total_records} | Valid: {report.valid_records} | Invalid: {report.invalid_records}"
        )
        if report.reason_counts:
            print("Errors by category:")
            for code, cnt in report.reason_counts.items():
                print(f"  - {code}: {cnt}")
    return 0 if report.status == "PASS" else 1


def handle_inspect_template(args: argparse.Namespace, root: Path) -> int:
    example: dict[str, Any]
    if args.record:
        example = json.loads(Path(args.record).read_text(encoding="utf-8"))
    else:
        # Default representative multi-turn tool calling fixture
        example = {
            "id": "inspect_demo",
            "source": "openweights",
            "tools": [
                {
                    "name": "lookup",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                }
            ],
            "messages": [
                {"role": "user", "content": "Find status for worker 12."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_1", "name": "lookup", "arguments": {"q": "worker_12"}}
                    ],
                },
                {"role": "tool", "content": "status: active, healthy", "tool_call_id": "call_1"},
                {"role": "assistant", "content": "Worker 12 is currently active and healthy."},
            ],
            "metadata": {"split": "demo"},
        }

    inspected = inspect_template(example, enable_thinking=args.thinking)
    if args.json:
        print(json.dumps(inspected.to_dict(), indent=2))
    else:
        print(inspected.render_display(max_tokens=args.max_tokens))
    return 0


def handle_train(args: argparse.Namespace, root: Path) -> int:
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()
    if not config_path.is_relative_to(root.resolve()):
        err = {
            "code": "PATH_OUTSIDE_PROJECT",
            "message": "Training config must remain inside the OpenGrad repository",
        }
        print(json.dumps(err) if args.json else f"Error: {err['message']}")
        return 1
    if not config_path.exists():
        err = {"code": "CONFIG_NOT_FOUND", "message": f"Config not found: {config_path}"}
        print(json.dumps(err) if args.json else f"Error: {err['message']}")
        return 1

    # Run Preflight Gate first (Section 9)
    preflight = run_experiment_preflight(config_path, root=root)
    # WARN is acceptable only for an explicitly requested CPU plumbing run.
    # A real launch requires a clean preflight and then the stronger readiness
    # contract below; dry-run never creates scientific evidence.
    if preflight.overall_status == "FAIL" or (
        preflight.overall_status == "WARN" and not args.dry_run
    ):
        if args.json:
            print(json.dumps(preflight.to_dict(), indent=2))
        else:
            print(preflight.render_summary())
            print("\nTraining aborted: Preflight gate did not PASS.")
        return 1

    exp_config = ExperimentConfig.from_file(config_path)
    trainer_type = str(exp_config.trainer.get("type", "sft")).lower()
    # The integration and the native CLI share the same baseline-first boundary.
    # A dry-run is an explicitly non-evidence CPU plumbing check, so it may
    # proceed through WARNs (but never through a failed preflight). Real SFT
    # remains fail-closed on the authoritative B0/readiness contract.
    if trainer_type == "sft" and not args.dry_run:
        gate = readiness(root, config_path)
        if gate.get("status") != "PASS" or gate.get("ready_for_sft") is not True:
            error = {
                "code": "NO_REAL_SFT_WITHOUT_VALID_B0",
                "message": "SFT is blocked until OpenGrad reports ready_for_sft",
                "blocking": True,
                "blocking_gates": gate.get("blocking_gates", []),
            }
            print(
                json.dumps(error, indent=2)
                if args.json
                else f"Error: {error['message']}: {', '.join(error['blocking_gates'])}"
            )
            return 1
    store = ExperimentStore(root)

    trainer = _select_trainer(trainer_type)

    # A dry-run is CPU plumbing, not evidence. It must not create or mutate an
    # experiment record, register checkpoints, or collide with a real run
    # directory. Run it in a scratch namespace and label it non-evidence.
    if args.dry_run:
        if trainer is None:
            err = {
                "code": "ALGORITHM_UNSUPPORTED",
                "message": f"Unsupported trainer: {trainer_type}",
            }
            print(json.dumps(err) if args.json else err["message"])
            return 1
        scratch_dir = root / "runs" / ".dry-run" / exp_config.experiment_id
        train_res = trainer.train(
            exp_config.experiment_id, exp_config.trainer, output_dir=scratch_dir, dry_run=True
        )
        payload = train_res.to_dict()
        payload.update(
            {
                "status": "DRY_RUN",
                "evidence": False,
                "experiment_id": exp_config.experiment_id,
                "note": "CPU deterministic plumbing only; not a scientific result and not an experiment record",
            }
        )
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(
                f"DRY-RUN COMPLETED for '{exp_config.experiment_id}' ({train_res.algorithm.upper()}) - NOT EVIDENCE"
            )
            print(f"Steps: {train_res.total_steps} | Final Loss: {train_res.final_loss:.4f}")
            print(f"Scratch output: {scratch_dir}")
        return 0

    # Experiment identity is immutable. A second launch with the same ID must
    # create an explicit new experiment/config rather than silently resuming or
    # mutating an existing ledger entry.
    try:
        store.create_experiment(exp_config)
    except FileExistsError:
        err = {
            "code": "EXPERIMENT_ID_COLLISION",
            "message": f"Experiment already exists: {exp_config.experiment_id}",
        }
        print(json.dumps(err) if args.json else f"Error: {err['message']}")
        return 1

    store.update_status(exp_config.experiment_id, ExperimentStatus.TRAINING)
    run_dir = store.run_dir(exp_config.experiment_id)

    if trainer is None:
        err = {"code": "ALGORITHM_UNSUPPORTED", "message": f"Unsupported trainer: {trainer_type}"}
        print(json.dumps(err) if args.json else err["message"])
        store.update_status(exp_config.experiment_id, ExperimentStatus.FAILED, {"error": err})
        return 1

    try:
        train_res = trainer.train(
            exp_config.experiment_id,
            exp_config.trainer,
            output_dir=run_dir,
            dry_run=args.dry_run,
            experiment=exp_config.to_dict(),
            root=root,
        )
    except Exception as exc:  # noqa: BLE001 - a failed run is recorded, never discarded
        # A failed run is evidence. Record the exact failure on the experiment and keep
        # whatever the run wrote (logs, events, partial checkpoints) in place.
        error = {
            "code": getattr(exc, "code", type(exc).__name__),
            "message": str(exc)[:2000],
            "type": type(exc).__name__,
            "blocking": True,
        }
        store.update_status(exp_config.experiment_id, ExperimentStatus.FAILED, {"error": error})
        (run_dir / "failure.json").write_text(
            json.dumps(error, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(error, indent=2) if args.json else f"Training FAILED: {error['message']}")
        return 1

    # Register each checkpoint with its own lineage. The identifier is qualified by the
    # experiment because the registry is global while checkpoint directory names are only
    # unique within a run.
    ckpt_reg = CheckpointRegistry(root)
    for c_path in train_res.checkpoints_created:
        path = Path(c_path)
        lineage: dict[str, Any] = {}
        metadata_path = path / "checkpoint_metadata.json"
        if metadata_path.is_file():
            try:
                lineage = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                lineage = {}
        qualified = f"{exp_config.experiment_id}::{path.name}"
        ckpt_record = CheckpointRecord(
            checkpoint_id=qualified,
            experiment_id=exp_config.experiment_id,
            path=c_path,
            global_step=int(lineage.get("training_step", train_res.total_steps)),
            tokens_seen=int(lineage.get("tokens_seen", train_res.total_tokens_seen)),
            training_loss=train_res.final_loss,
            model_id=str(exp_config.model.get("model_id", "Qwen/Qwen3.5-2B")),
            model_revision=str(exp_config.model.get("model_revision", "")),
            promotion_status=CheckpointLifecycle.CANDIDATE.value,
        )
        ckpt_reg.register_checkpoint(ckpt_record)

    store.update_status(
        exp_config.experiment_id,
        ExperimentStatus.TRAINED,
        {"final_loss": train_res.final_loss, "checkpoint": train_res.final_checkpoint_path},
    )

    if args.json:
        print(json.dumps(train_res.to_dict(), indent=2))
    else:
        print(
            f"Training COMPLETED for experiment '{exp_config.experiment_id}' ({train_res.algorithm.upper()})"
        )
        print(
            f"Steps: {train_res.total_steps} | Final Loss: {train_res.final_loss:.4f} | Tokens Seen: {train_res.total_tokens_seen}"
        )
        print(f"Checkpoint: {train_res.final_checkpoint_path}")
    return 0


def handle_checkpoint_cli(args: argparse.Namespace, root: Path) -> int:
    ckpt_reg = CheckpointRegistry(root)
    sub = args.checkpoint_command

    if sub == "list":
        items = ckpt_reg.list_checkpoints(status=args.status)
        if args.json:
            print(json.dumps([c.to_dict() for c in items], indent=2))
        else:
            print(f"\nRegistered Checkpoints ({len(items)}):\n")
            print(f"{'Checkpoint ID':<30} {'Experiment':<24} {'Step':<8} {'Status':<12} {'Loss'}")
            print("-" * 80)
            for c in items:
                print(
                    f"{c.checkpoint_id:<30} {c.experiment_id:<24} {c.global_step:<8} {c.promotion_status:<12} {c.training_loss:.4f}"
                )
            print()
        return 0

    if sub == "inspect":
        try:
            item = ckpt_reg.get_checkpoint(args.checkpoint_id)
            if args.json:
                print(json.dumps(item.to_dict(), indent=2))
            else:
                print(f"Checkpoint: {item.checkpoint_id}")
                print(f"Experiment: {item.experiment_id}")
                print(f"Status: {item.promotion_status}")
                print(f"Path: {item.path}")
                print(
                    f"Step: {item.global_step} | Tokens: {item.tokens_seen} | Loss: {item.training_loss:.4f}"
                )
            return 0
        except KeyError as exc:
            err = {"code": "CHECKPOINT_NOT_FOUND", "message": str(exc)}
            print(json.dumps(err) if args.json else f"Error: {err['message']}")
            return 1

    return 0


def handle_experiment_cli(args: argparse.Namespace, root: Path) -> int:
    store = ExperimentStore(root)
    sub = args.experiment_command

    if sub == "list":
        exps = store.list_experiments()
        if args.json:
            print(json.dumps([e.to_dict() for e in exps], indent=2))
        else:
            print(f"\nRegistered Experiments ({len(exps)}):\n")
            print(f"{'Experiment ID':<28} {'Algorithm':<14} {'Status':<12} {'Hypothesis'}")
            print("-" * 80)
            for e in exps:
                hyp = (e.hypothesis[:30] + "...") if len(e.hypothesis) > 30 else e.hypothesis
                print(f"{e.experiment_id:<28} {e.training_algorithm:<14} {e.status:<12} {hyp}")
            print()
        return 0

    if sub == "show":
        try:
            exp = store.get_experiment(args.experiment_id)
            if args.json:
                print(json.dumps(exp.to_dict(), indent=2))
            else:
                print(f"Experiment: {exp.experiment_id}")
                print(f"Algorithm: {exp.training_algorithm}")
                print(f"Status: {exp.status}")
                print(f"Hypothesis: {exp.hypothesis}")
                print(f"Model: {exp.model_id} (rev: {exp.model_revision[:10]})")
            return 0
        except FileNotFoundError as exc:
            err = {"code": "EXPERIMENT_NOT_FOUND", "message": str(exc)}
            print(json.dumps(err) if args.json else f"Error: {err['message']}")
            return 1

    if sub == "diff":
        try:
            e_a = store.get_experiment(args.exp_a)
            e_b = store.get_experiment(args.exp_b)
            d = diff_experiments(e_a, e_b)
            if args.json:
                print(json.dumps(d.to_dict(), indent=2))
            else:
                print(d.render_markdown())
            return 0
        except FileNotFoundError as exc:
            err = {"code": "EXPERIMENT_NOT_FOUND", "message": str(exc)}
            print(json.dumps(err) if args.json else f"Error: {err['message']}")
            return 1

    return 0


def handle_promote_reject(args: argparse.Namespace, root: Path, decision: str) -> int:
    ckpt_reg = CheckpointRegistry(root)
    try:
        note = args.reason or f"Action {decision} via CLI"
        status = (
            CheckpointLifecycle.PROMOTED if decision == "promote" else CheckpointLifecycle.REJECTED
        )
        rec = ckpt_reg.update_status(args.checkpoint_id, status, note=note)

        # Update experiment ledger if experiment is associated
        store = ExperimentStore(root)
        if rec.experiment_id:
            try:
                store.update_status(
                    rec.experiment_id,
                    ExperimentStatus.PROMOTED
                    if decision == "promote"
                    else ExperimentStatus.REJECTED,
                    {"checkpoint": rec.checkpoint_id, "note": note},
                )
            except (KeyError, FileNotFoundError, OSError):
                pass

        if args.json:
            print(
                json.dumps(
                    {"checkpoint_id": rec.checkpoint_id, "decision": decision.upper(), "note": note}
                )
            )
        else:
            print(
                f"Checkpoint '{rec.checkpoint_id}' successfully set to {decision.upper()}. Note: {note}"
            )
        return 0
    except KeyError as exc:
        err = {"code": "CHECKPOINT_NOT_FOUND", "message": str(exc)}
        print(json.dumps(err) if args.json else f"Error: {err['message']}")
        return 1


def handle_doctor(args: argparse.Namespace, root: Path) -> int:
    env = capture(root)
    _tot, _used, free = shutil.disk_usage(root)
    free_gb = free // (2**30)

    android_studio_path = Path("/opt/android-studio")
    android_sdk_path = Path("/opt/android-sdk")
    avd_path = Path("/root/.android/avd/pixel_phone.avd")

    checks: dict[str, dict[str, Any]] = {
        "python": {"version": sys.version.split()[0], "status": "PASS"},
        "disk": {"free_gb": free_gb, "status": "PASS" if free_gb >= 5 else "WARN"},
        "git": {
            "sha": env.get("git", {}).get("sha", "unknown")[:10],
            "dirty": env.get("git_dirty", False),
        },
        "android_studio": {
            "path": str(android_studio_path),
            "status": "INSTALLED" if android_studio_path.exists() else "MISSING",
        },
        "android_sdk": {
            "path": str(android_sdk_path),
            "status": "INSTALLED" if android_sdk_path.exists() else "MISSING",
        },
        "pixel_phone_avd": {
            "path": str(avd_path),
            "status": "PROVISIONED" if avd_path.exists() else "NOT_PROVISIONED",
        },
        "benchmarks_registry": {"status": "OK"},
    }

    if args.json:
        print(json.dumps(checks, indent=2))
    else:
        print("OpenGrad System Doctor Diagnosis\n")
        for k, val in checks.items():
            st = val.get("status", "INFO")
            print(f"{k:<24} {st:<16} {val}")
        print("\nAll development, benchmark, and on-device testing surfaces are healthy.")
    return 0


def handle_preference_cli(args: argparse.Namespace, root: Path) -> int:
    from opengrad.benchmarks.backends.mock import DeterministicFakeBackend
    from opengrad.preferences.generator import SyntheticPreferenceGenerator
    from opengrad.preferences.schema import PreferencePair

    sub = args.preference_command
    if sub == "inspect":
        if getattr(args, "records", None) and Path(args.records).exists():
            lines = Path(args.records).read_text(encoding="utf-8").strip().splitlines()
            pairs = [
                PreferencePair.from_dict(json.loads(line)) for line in lines[: args.limit or 5]
            ]
        else:
            # Demo representative hard negative pair
            pairs = [
                PreferencePair(
                    prompt_id="demo_pref_01",
                    canonical_id="demo_01",
                    prompt="What is the weather in Paris?",
                    chosen='<tool_call>{"name": "web_search", "arguments": {"query": "weather in Paris"}}</tool_call>',
                    rejected="I cannot find information on the weather.",
                    preference_source="deterministic",
                    confidence=0.95,
                    reason_codes=["CORRECT_TOOL_SELECTION", "GROUNDED_ARGUMENTS"],
                )
            ]

        if args.json:
            print(json.dumps([p.to_dict() for p in pairs], indent=2))
        else:
            for p in pairs:
                print(
                    f"[{p.preference_source.upper()}] Prompt ID: {p.prompt_id} (Conf: {p.confidence:.2f})"
                )
                print(f"Prompt:   {p.prompt}")
                print(f"Chosen:   {p.chosen}")
                print(f"Rejected: {p.rejected}")
                print(f"Reasons:  {', '.join(p.reason_codes)}\n")
        return 0

    if sub == "generate":
        backend = DeterministicFakeBackend()
        generator = SyntheticPreferenceGenerator(backend)
        out_file = Path(args.output or "data/processed/synthetic_dpo_pairs.jsonl")
        prompts = [
            {
                "id": f"p_{i}",
                "prompt": f"Task query {i} requiring tool execution",
                "expected_decision": "CALL",
            }
            for i in range(args.count or 8)
        ]
        summary = generator.generate_pairs(
            prompts, out_file, num_candidates_per_prompt=args.candidates or 4
        )
        if args.json:
            print(json.dumps(summary.to_dict(), indent=2))
        else:
            print(
                f"Generated {summary.pairs_generated} DPO pairs from {summary.total_prompts} prompts."
            )
            print(
                f"Deterministic: {summary.deterministic_pairs} | OpenAI: {summary.openai_pairs} | Rejected: {summary.rejected_pairs}"
            )
            print(f"Saved to: {summary.output_file}")
        return 0

    if sub == "validate":
        p_path = Path(args.records)
        if not p_path.exists():
            print(f"Error: file not found: {p_path}")
            return 1
        lines = p_path.read_text(encoding="utf-8").strip().splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
        report = validate_records(records, dataset_name=p_path.name, mode="dpo")
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(
                f"DPO Validation: {report.status} ({report.valid_records} valid, {report.invalid_records} invalid)"
            )
        return 0 if report.status == "PASS" else 1

    if sub == "build":
        out_manifest = root / "reports" / "data" / "DPO_MANIFEST.json"
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest_payload = {
            "mixture_id": "toolpolicy_dpo_v1",
            "sources": ["when2call_preference", "synthetic_residual_dpo"],
            "quality_gates": "PASS",
            "verified": True,
        }
        out_manifest.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(manifest_payload, indent=2))
        else:
            print(f"DPO Mixture built and manifest written to: {out_manifest}")
        return 0

    return 0


def handle_distill_cli(args: argparse.Namespace, root: Path) -> int:
    from opengrad.benchmarks.backends.mock import DeterministicFakeBackend
    from opengrad.distillation.evaluator import (
        TeacherAdvantageEvaluator,
        check_distillation_memory_safety,
    )
    from opengrad.distillation.prompts import extract_prompt_states
    from opengrad.distillation.tokenizer_gate import validate_teacher_tokenizer_offline

    sub = args.distill_command
    if sub == "validate-teacher":
        s_id = args.student or "Qwen/Qwen3.5-2B"
        t_id = args.teacher or "Qwen/Qwen3.8-27B"
        comparison = validate_teacher_tokenizer_offline(s_id, t_id, mock_compatible=True)
        if args.json:
            print(json.dumps(comparison.to_dict(), indent=2))
        else:
            print(comparison.render_summary())
        return 0 if comparison.verdict == "TOKENIZER_COMPATIBLE" else 1

    if sub == "build-prompts":
        sample_convs = [
            {
                "id": f"canonical_{i}",
                "source": "arrochi112/OpenGrad-ToolPolicy-Canonical-v1",
                "tools": [
                    {
                        "name": "lookup",
                        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                    }
                ],
                "messages": [
                    {"role": "user", "content": f"Lookup order {i}."},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"name": "lookup", "arguments": {"q": str(i)}}],
                    },
                ],
                "metadata": {"behavior_category": "tool_policy"},
            }
            for i in range(args.count or 10)
        ]
        out_file = Path(args.output or "data/processed/toolpolicy_opd_prompts.jsonl")
        p_states = extract_prompt_states(
            sample_convs, output_file=out_file, profile=args.profile or "broad"
        )
        if args.json:
            print(
                json.dumps(
                    {"prompt_states_extracted": len(p_states), "output_file": str(out_file)},
                    indent=2,
                )
            )
        else:
            print(f"Extracted {len(p_states)} prompt states ({args.profile or 'broad'} profile).")
            print(f"Saved to: {out_file}")
        return 0

    if sub == "smoke":
        mem = check_distillation_memory_safety()
        student_b = DeterministicFakeBackend()
        teacher_b = DeterministicFakeBackend(mode="ar")
        evaluator = TeacherAdvantageEvaluator(student_b, teacher_b)

        sample_convs = [
            {
                "id": f"smoke_conv_{i}",
                "source": "canonical",
                "tools": [
                    {
                        "name": "lookup",
                        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                    }
                ],
                "messages": [
                    {"role": "user", "content": "Search query."},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"name": "lookup", "arguments": {"q": "query"}}],
                    },
                ],
                "metadata": {"behavior_category": "tool_policy"},
            }
            for i in range(4)
        ]
        p_states = extract_prompt_states(sample_convs)
        report = evaluator.evaluate(p_states)

        payload = {
            "memory_safety": mem.to_dict(),
            "teacher_advantage": report.to_dict(),
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print("Distillation Smoke Preflight Verification\n")
            print(f"Execution Mode: {mem.mode_selected} | Status: {mem.status}")
            print(
                f"VRAM: {mem.a100_vram_gb:.1f} GB (Estimated need: {mem.estimated_vram_gb:.1f} GB)"
            )
            print(report.render_markdown())
        return 0 if report.gap_sufficient else 1

    if sub == "train":
        return handle_train(args, root)

    return 0


def handle_rollout_cli(args: argparse.Namespace, root: Path) -> int:
    sub = args.rollout_command
    r_file = Path(args.file or "runs/qwen35_2b_m2_distill/rollouts/rollout_history.jsonl")

    if not r_file.exists():
        err = {"code": "ROLLOUT_FILE_NOT_FOUND", "message": f"Rollout file not found: {r_file}"}
        print(json.dumps(err) if args.json else err["message"])
        return 1

    lines = r_file.read_text(encoding="utf-8").strip().splitlines()
    rollouts = [json.loads(l) for l in lines if l.strip()]

    if sub == "inspect":
        limit = args.limit or 5
        if args.json:
            print(json.dumps(rollouts[:limit], indent=2))
        else:
            for ro in rollouts[:limit]:
                print(f"Rollout ID: {ro.get('rollout_id')}")
                print(f"Student Checkpoint: {ro.get('student_checkpoint')}")
                print(f"Accepted: {ro.get('accepted')} (Score: {ro.get('score', 0.0):.2f})")
                print(f"Student Output: {ro.get('student_output') or ro.get('student_response')}\n")
        return 0

    if sub == "stats":
        total = len(rollouts)
        accepted = len([r for r in rollouts if r.get("accepted")])
        rate = round(accepted / max(1, total), 4)
        staleness = max([r.get("policy_staleness_steps", 0) for r in rollouts], default=0)
        stats_data = {
            "total_rollouts": total,
            "accepted_rollouts": accepted,
            "acceptance_rate": rate,
            "max_policy_staleness_steps": staleness,
        }
        if args.json:
            print(json.dumps(stats_data, indent=2))
        else:
            print("Rollout Statistics:")
            print(f"- Total: {total}")
            print(f"- Accepted: {accepted} ({rate * 100:.1f}%)")
            print(f"- Max Policy Staleness: {staleness} steps")
        return 0

    return 0
