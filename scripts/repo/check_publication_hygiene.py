#!/usr/bin/env python3
"""Scan tracked files for public-repository hygiene problems: secrets, local paths, leftover notes.

Three kinds of tracked file are read:

* **text** (docs, code, configs): every category below;
* **data** (``.jsonl``, ``.csv``, ``.bib``): secrets and home-directory paths only -- the other
  categories are ordinary vocabulary inside training and evaluation text;
* **tarballs** (``.tar.gz``): each member, as data. An earlier version skipped both data and
  tarballs, so it never saw the one live-format token the repository carries (`reports/ERRATA.md` §22).

Findings a reviewer has accepted -- the token that came with the upstream When2Call data, the author's
paths inside hash-pinned audit trails, genuine false positives -- are listed in
``publication_hygiene_allowlist.yaml`` with a reason and an **exact count**. A new finding in an
allowlisted file changes the count and fails, and an entry that no longer matches fails too, so the
allowlist cannot silently cover more than it names.

    python scripts/repo/check_publication_hygiene.py
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
import tarfile
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = Path(__file__).resolve().with_name("publication_hygiene_allowlist.yaml")

SECRET = re.compile(
    r"(?:sk-[A-Za-z0-9]{20,}"
    r"|OPENAI_API_KEY\s*=\s*['\"][A-Za-z0-9_\-]+['\"]"
    r"|ghp_[A-Za-z0-9]{36}"
    r"|gh[ousr]_[A-Za-z0-9]{36}"
    r"|github_pat_[A-Za-z0-9_]{22,}"
    r"|\bhf_[A-Za-z0-9]{30,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----)"
)
HOME_PATH = re.compile(
    r"(?:[A-Za-z]:(?:\\\\|\\|/)+Users(?:\\\\|\\|/)|/Users/[^/\s\"']+/|/home/[^/\s\"']+/)"
)

PATTERNS: dict[str, re.Pattern[str]] = {
    # "system prompt" is ordinary vocabulary in a tool-use repository and is not listed.
    "assistant_reference": re.compile(
        r"\b(chatgpt|copilot|developer message|conversation context)\b", re.IGNORECASE
    ),
    "user_reference": re.compile(
        r"\b(as (the )?user|user (requested|asked|wants|specified|provided))\b", re.IGNORECASE
    ),
    "prompt_reference": re.compile(
        r"\b(based on the prompt|original prompt|prompt requirement|according to the prompt)\b",
        re.IGNORECASE,
    ),
    "agent_scratch_language": re.compile(
        r"\b(agent note|agent should|agent instructions|next steps for agent|status for agent)\b",
        re.IGNORECASE,
    ),
    "private_path": re.compile(
        r"(?:/home/|/Users/|C:\\\\Users\\\\|C:/Users/|/mnt/data/|/workspace/|/tmp/)", re.IGNORECASE
    ),
    "tracking_url": re.compile(r"[?&]utm_[^\s)]+", re.IGNORECASE),
    "api_secret": SECRET,
    "placeholder": re.compile(
        r"\b(?:YOUR_(?:NAME|USERNAME|EMAIL)|INSERT_HERE|REPLACE_ME|CHANGE_ME|LOREM IPSUM)\b",
        re.IGNORECASE,
    ),
}
DATA_PATTERNS: dict[str, re.Pattern[str]] = {"api_secret": SECRET, "private_path": HOME_PATH}

TEXT_SUFFIXES = {
    ".md", ".mdx", ".txt", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx", ".sh", ".bash",
    ".yaml", ".yml", ".json", ".toml", ".cff",
}  # fmt: skip
DATA_SUFFIXES = {".jsonl", ".csv", ".bib"}


def tracked_files(root: Path) -> list[str]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return [name for name in output.decode().split("\0") if name]


def _lines(
    text: str, patterns: dict[str, re.Pattern[str]], where: str
) -> list[dict[str, str | int]]:
    findings: list[dict[str, str | int]] = []
    for number, line in enumerate(text.splitlines(), 1):
        for category, pattern in patterns.items():
            for _ in pattern.finditer(line):
                findings.append(
                    {
                        "category": category,
                        "file": where,
                        "line": number,
                        "text": line.strip()[:200],
                    }
                )
    return findings


def scan(root: Path = ROOT) -> list[dict[str, str | int]]:
    findings: list[dict[str, str | int]] = []
    # The scanner and its allowlist quote the patterns they describe.
    own = {Path(__file__).resolve(), ALLOWLIST.resolve()}
    for name in tracked_files(root):
        path = root / name
        if path.resolve() in own or not path.is_file():
            continue
        lowered = name.lower()
        if lowered.endswith(".tar.gz"):
            with tarfile.open(path) as archive:
                for member in archive.getmembers():
                    handle = archive.extractfile(member) if member.isfile() else None
                    if handle is not None:
                        text = handle.read().decode("utf-8", "replace")
                        findings += _lines(text, DATA_PATTERNS, f"{name}!{member.name}")
            continue
        suffix = Path(lowered).suffix
        if suffix not in TEXT_SUFFIXES and suffix not in DATA_SUFFIXES:
            continue
        try:
            text = path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        findings += _lines(text, DATA_PATTERNS if suffix in DATA_SUFFIXES else PATTERNS, name)
    return findings


def load_allowlist(path: Path = ALLOWLIST) -> list[dict]:
    entries = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    for entry in entries:
        missing = {"path", "category", "count", "reason"} - set(entry)
        if missing:
            raise ValueError(f"allowlist entry {entry!r} lacks {sorted(missing)}")
    return entries


def evaluate(
    findings: list[dict[str, str | int]], allowlist: list[dict]
) -> tuple[list[str], list[dict]]:
    """Split findings into problems (strings) and accepted ones. Every allowlist entry must match exactly."""
    problems: list[str] = []
    accepted: list[dict] = []
    remaining = list(findings)
    for entry in allowlist:
        matched = [
            f
            for f in remaining
            if f["category"] == entry["category"]
            and fnmatch.fnmatchcase(str(f["file"]), entry["path"])
        ]
        if len(matched) != int(entry["count"]):
            problems.append(
                f"allowlist: {entry['category']} {entry['path']}: expected exactly {entry['count']} "
                f"finding(s), found {len(matched)}"
            )
        accepted += matched
        remaining = [f for f in remaining if f not in matched]
    problems += [f"{f['category']}: {f['file']}:{f['line']}: {f['text']}" for f in remaining]
    return problems, accepted


def main() -> int:
    findings = scan(ROOT)
    problems, accepted = evaluate(findings, load_allowlist())
    for problem in problems:
        print(problem)
    if problems:
        return 1
    counts = Counter(str(f["category"]) for f in accepted)
    print(
        f"publication hygiene: PASS ({len(accepted)} allowlisted finding(s): {dict(sorted(counts.items()))})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
