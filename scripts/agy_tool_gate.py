"""The PreToolUse hook that confines an agy labelling run to reading its own input (docs/UPSTREAM_ISSUES.md UP-0012).

On this machine agy runs every tool without asking (``toolPermission: always-proceed``), and ``--sandbox`` restricts
terminal commands only. ``scripts/run_external_annotation.py`` therefore writes ``.agents/hooks.json`` into each
isolated run directory, so agy calls this script before every tool call. It allows one thing, reading the run's own
input file (``view_file`` with ``AbsolutePath``, or ``read_file`` with ``Path``, the name newer agy versions use;
UP-0039), and denies everything else: web search, URL reads, file browsing, MCP tools, commands. A hook that fails
to start blocked the tool call in the three forms measured on 2026-11-13 (UP-0039); that is not shown for every way
a hook can fail.

agy runs the hook with the working directory set to the folder holding ``hooks.json`` (``<run dir>/.agents``), the
tool call as JSON on standard input, and reads the decision as JSON from standard output (agy's hooks guide,
``PreToolUse``). Each decision is appended to ``gate-log.jsonl`` beside ``hooks.json`` with the tool name only, never
its arguments, so the runner can record what was attempted.

    python scripts/agy_tool_gate.py input.md < tool-call.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

#: The file-reading tools agy has used, with the argument that holds the path: ``view_file`` up to 2026-11,
#: ``read_file`` after (UP-0039).
READ_TOOLS = {"view_file": "AbsolutePath", "read_file": "Path"}
DENY_REASON = "Only reading the input file in the working directory is permitted for this task."


def decide(payload: Any, run_dir: Path, input_name: str) -> dict[str, str]:
    """Allow reading ``run_dir / input_name`` (``READ_TOOLS``); deny anything else, including what does not parse."""
    call = payload.get("toolCall") if isinstance(payload, dict) else None
    if isinstance(call, dict) and call.get("name") in READ_TOOLS:
        args = call.get("args")
        target = args.get(READ_TOOLS[call["name"]]) if isinstance(args, dict) else None
        if isinstance(target, str):
            try:
                if Path(target).resolve() == (run_dir / input_name).resolve():
                    return {"decision": "allow"}
            except OSError:
                pass
    return {"decision": "deny", "reason": DENY_REASON}


def main(argv: list[str]) -> int:
    input_name = argv[1] if len(argv) > 1 else "input.md"
    here = Path.cwd()
    try:
        payload: Any = json.loads(sys.stdin.read())
    except ValueError:
        payload = None
    result = decide(payload, here.parent, input_name)
    call = payload.get("toolCall") if isinstance(payload, dict) else None
    tool = str(call.get("name", "?")) if isinstance(call, dict) else "?"
    with (here / "gate-log.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps({"tool": tool, "decision": result["decision"]}) + "\n")
    sys.stdout.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
