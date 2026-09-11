"""Command-line surface for the optimization producer layer.

Two commands, both of which run without a GPU, without ModelOpt, and without touching
any optimization weights:

* ``capability-matrix`` - emit the per-model capability matrix (discovery only).
* ``optimize`` - run a recipe through a selected backend; with ``--backend mock`` this is
  deterministic CPU plumbing, never evidence.

The module never imports an optimization backend at import time; ``yaml`` is imported
only inside the config loader so the command works with the base install.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from opengrad.optimization import select_optimization_backend
from opengrad.optimization.protocol import OptimizationRecipe, SourceCheckpoint


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dev extra provides yaml
        raise RuntimeError("reading an optimization config requires the research extra") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("optimization config must be a YAML object")
    return data


def load_smoke_config(path: Path) -> tuple[SourceCheckpoint, OptimizationRecipe]:
    """Load a pinned source checkpoint and recipe from a smoke config."""
    data = _load_yaml(path)
    source = SourceCheckpoint.from_dict(dict(data["source"]))
    recipe = OptimizationRecipe.from_dict(dict(data["recipe"]))
    return source, recipe


def _capability_matrix(args: argparse.Namespace) -> int:
    backend = select_optimization_backend(args.backend)
    if backend is None:
        print(f"Error: unknown optimization backend: {args.backend}", file=sys.stderr)
        return 1
    matrix = backend.capability_matrix(model_id=args.model, revision=args.revision)
    payload = matrix.to_dict()
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _optimize(args: argparse.Namespace) -> int:
    backend = select_optimization_backend(args.backend)
    if backend is None:
        print(f"Error: unknown optimization backend: {args.backend}", file=sys.stderr)
        return 1
    source, recipe = load_smoke_config(Path(args.config))
    output_dir = Path(args.output_dir)
    try:
        result = backend.optimize(
            source, recipe, output_dir, dry_run=args.dry_run, runtime=args.runtime
        )
    except (RuntimeError, ValueError) as exc:
        error = {"code": type(exc).__name__, "message": str(exc)[:2000], "blocking": True}
        print(json.dumps(error, indent=2) if args.json else f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opengrad.optimization")
    sub = parser.add_subparsers(dest="command")

    cap = sub.add_parser("capability-matrix", help="emit the per-model capability matrix")
    cap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    cap.add_argument("--revision", default="15852e8c16360a2fea060d615a32b45270f8a8fc")
    cap.add_argument("--backend", default="modelopt", choices=["modelopt", "mock"])
    cap.add_argument("--output", help="write JSON here instead of stdout")
    cap.set_defaults(func=_capability_matrix)

    opt = sub.add_parser("optimize", help="run an optimization recipe through a backend")
    opt.add_argument("--config", required=True, help="optimization smoke config YAML")
    opt.add_argument("--backend", default="mock", choices=["modelopt", "mock"])
    opt.add_argument("--output-dir", default="runs/.dry-run/optimization")
    opt.add_argument("--runtime", default=None, help="declared deployment runtime")
    opt.add_argument("--dry-run", action="store_true", help="plan only; no weights touched")
    opt.add_argument("--json", action="store_true")
    opt.set_defaults(func=_optimize)

    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
