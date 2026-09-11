import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def capture(root: Path) -> dict[str, Any]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        sha = None
    try:
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        dirty = None
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "python": sys.version.split()[0],
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": None,
        "gpu": None,
        "cuda": None,
        "rocm": None,
        "git_sha": sha,
        "git_dirty": dirty,
        "gpu_probe": "not performed by Phase 0.5",
    }


def tracked_tree_provenance(root: Path) -> dict[str, Any]:
    """Commit and *tracked-tree* cleanliness, for deciding whether something is evidence.

    Untracked files are deliberately ignored. Every run creates untracked outputs -- its own
    predictions, metrics, and record -- so asking `git status` as a whole would report every
    successful run as dirty and make "produced from a clean checkout" unsatisfiable. What
    matters is whether the tracked code and configs matched a known commit.

    Fails closed: when git cannot be queried the result is dirty, so an unreadable repository
    can never be presented as clean provenance.
    """
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": True}
    return {"sha": sha or None, "dirty": bool(porcelain.strip())}


def git_identity(environment: dict[str, Any] | None) -> tuple[str, bool]:
    """The commit an artifact was produced from, and whether the tree was dirty.

    This module produces two shapes -- `capture()` returns a flat `git_sha`/`git_dirty`, and
    `tracked_tree_provenance()` a `{"sha", "dirty"}` -- and three call sites read a nested
    `environment["git"]["sha"]` that neither produces. So every experiment record, benchmark
    record and doctor report named its commit as `"unknown"`, which defeats the point of
    recording it: an artifact that cannot name the code that produced it is not reproducible
    evidence. Reading both real shapes here means a future caller cannot repeat the mistake.
    """
    env = environment or {}
    sha = env.get("git_sha") or env.get("sha") or "unknown"
    dirty = env.get("git_dirty")
    if dirty is None:
        dirty = env.get("dirty")
    return str(sha), bool(dirty)


def write_capture(root: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(capture(root), indent=2) + "\n")
    return destination
