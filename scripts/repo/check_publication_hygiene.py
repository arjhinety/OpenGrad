#!/usr/bin/env python3
"""Scan tracked text files for common public-repository hygiene problems."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

PATTERNS: dict[str, re.Pattern[str]] = {
    "assistant_reference": re.compile(
        r"\b(chatgpt|copilot|system prompt|developer message|conversation context)\b", re.IGNORECASE
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
        r"(?:/home/|/Users/|C:\\\\Users\\\\|/mnt/data/|/workspace/|/tmp/)", re.IGNORECASE
    ),
    "tracking_url": re.compile(r"[?&]utm_[^\s)]+", re.IGNORECASE),
    "placeholder": re.compile(
        r"\b(?:YOUR_(?:NAME|USERNAME|EMAIL)|INSERT_HERE|REPLACE_ME|CHANGE_ME|LOREM IPSUM)\b",
        re.IGNORECASE,
    ),
}

TEXT_SUFFIXES = {
    ".md",
    ".mdx",
    ".txt",
    ".rst",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".sh",
    ".bash",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".cff",
}


def tracked_files(root: Path) -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return [
        root / name
        for name in output.decode().split("\0")
        if name and Path(name).suffix.lower() in TEXT_SUFFIXES
    ]


def scan(root: Path) -> list[dict[str, str | int]]:
    findings: list[dict[str, str | int]] = []
    scanner_path = Path(__file__).resolve()
    for path in tracked_files(root):
        if path.resolve() == scanner_path:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(lines, 1):
            for category, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append(
                        {
                            "category": category,
                            "file": str(path.relative_to(root)),
                            "line": number,
                            "text": line.strip(),
                        }
                    )
    return findings


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    findings = scan(root)
    if findings:
        for item in findings:
            print(f"{item['category']}: {item['file']}:{item['line']}: {item['text']}")
        return 1
    print("publication hygiene: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
