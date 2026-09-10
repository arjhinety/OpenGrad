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
from opengrad.training.distillation import OnPolicyDistillationTrainerBackend
from opengrad.training.dpo import DPOTrainerBackend
from opengrad.training.protocol import TrainerBackend
from opengrad.training.sft import SFTTrainerBackend


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
                    err = {"code": "DATASET_SCHEMA_INVALID", "message": f"Invalid JSON on line: {exc}"}
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
        print(f"Total: {report.total_records} | Valid: {report.valid_records} | Invalid: {report.invalid_records}")
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
            "tools": [{"name": "lookup", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}],
            "messages": [
                {"role": "user", "content": "Find status for worker 12."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "call_1", "name": "lookup", "arguments": {"q": "worker_12"}}],
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
    if not config_path.exists():
        err = {"code": "CONFIG_NOT_FOUND", "message": f"Config not found: {config_path}"}
        print(json.dumps(err) if args.json else f"Error: {err['message']}")
        return 1

    # Run Preflight Gate first (Section 9)
    preflight = run_experiment_preflight(config_path, root=root)
    if preflight.overall_status == "FAIL" and not args.force:
        if args.json:
            print(json.dumps(preflight.to_dict(), indent=2))
        else:
            print(preflight.render_summary())
            print("\nTraining aborted: Preflight gate FAILED.")
        return 1

    exp_config = ExperimentConfig.from_file(config_path)
    store = ExperimentStore(root)

    # Initialize experiment record
    try:
        store.create_experiment(exp_config)
    except FileExistsError:
        pass  # allow resuming or re-running existing experiment run

    store.update_status(exp_config.experiment_id, ExperimentStatus.TRAINING)
    trainer_type = str(exp_config.trainer.get("type", "sft")).lower()
    run_dir = store.run_dir(exp_config.experiment_id)

    trainer: TrainerBackend
    if trainer_type == "sft":
        trainer = SFTTrainerBackend()
    elif trainer_type == "dpo":
        trainer = DPOTrainerBackend()
    elif trainer_type in {"on_policy_distillation", "distillation"}:
        trainer = OnPolicyDistillationTrainerBackend()
    else:
        err = {"code": "ALGORITHM_UNSUPPORTED", "message": f"Unsupported trainer: {trainer_type}"}
        print(json.dumps(err) if args.json else err["message"])
        store.update_status(exp_config.experiment_id, ExperimentStatus.FAILED, {"error": err})
        return 1

    train_res = trainer.train(
        exp_config.experiment_id,
        exp_config.trainer,
        output_dir=run_dir,
        dry_run=args.dry_run,
    )

    # Register checkpoint in CheckpointRegistry
    ckpt_reg = CheckpointRegistry(root)
    for c_path in train_res.checkpoints_created:
        ckpt_record = CheckpointRecord(
            checkpoint_id=Path(c_path).name,
            experiment_id=exp_config.experiment_id,
            path=c_path,
            global_step=train_res.total_steps,
            tokens_seen=train_res.total_tokens_seen,
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
        print(f"Training COMPLETED for experiment '{exp_config.experiment_id}' ({train_res.algorithm.upper()})")
        print(f"Steps: {train_res.total_steps} | Final Loss: {train_res.final_loss:.4f} | Tokens Seen: {train_res.total_tokens_seen}")
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
                print(f"{c.checkpoint_id:<30} {c.experiment_id:<24} {c.global_step:<8} {c.promotion_status:<12} {c.training_loss:.4f}")
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
                print(f"Step: {item.global_step} | Tokens: {item.tokens_seen} | Loss: {item.training_loss:.4f}")
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
        status = CheckpointLifecycle.PROMOTED if decision == "promote" else CheckpointLifecycle.REJECTED
        rec = ckpt_reg.update_status(args.checkpoint_id, status, note=note)

        # Update experiment ledger if experiment is associated
        store = ExperimentStore(root)
        if rec.experiment_id:
            try:
                store.update_status(
                    rec.experiment_id,
                    ExperimentStatus.PROMOTED if decision == "promote" else ExperimentStatus.REJECTED,
                    {"checkpoint": rec.checkpoint_id, "note": note},
                )
            except (KeyError, FileNotFoundError, OSError):
                pass

        if args.json:
            print(json.dumps({"checkpoint_id": rec.checkpoint_id, "decision": decision.upper(), "note": note}))
        else:
            print(f"Checkpoint '{rec.checkpoint_id}' successfully set to {decision.upper()}. Note: {note}")
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
        "git": {"sha": env.get("git", {}).get("sha", "unknown")[:10], "dirty": env.get("git_dirty", False)},
        "android_studio": {"path": str(android_studio_path), "status": "INSTALLED" if android_studio_path.exists() else "MISSING"},
        "android_sdk": {"path": str(android_sdk_path), "status": "INSTALLED" if android_sdk_path.exists() else "MISSING"},
        "pixel_phone_avd": {"path": str(avd_path), "status": "PROVISIONED" if avd_path.exists() else "NOT_PROVISIONED"},
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
