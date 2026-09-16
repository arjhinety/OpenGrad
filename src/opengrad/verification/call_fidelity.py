"""Structural CALL evidence for P-DET-COVERAGE-v1 layer A: call conservation and argument fidelity.

30 §4 says layer A is validated exhaustively, as a code check. Comparing routing with message structure
cannot fail (routing *is* that structure), so this module checks something that can: every call the raw
upstream row expresses in its own syntax, re-read here by parsers that share **no code** with
``opengrad.data.adapters``, must be the structured call normalization-v3 holds, with the same name and
the same arguments, in the same order.

It covers every accepted normalization-v3 row of each layer A source, split by trajectory-gate status, so
records the input contract rejects (for example ToolACE's ``MISSING_TOOL_RESULT``) are checked too. Raw
rows the normalization itself rejected or dropped as duplicates are accounted for from the disposition
ledger: how many raw calls they held, by disposition and reason.

Counts and record ids only. No record text is written or printed: the study owner is the blind annotator.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from opengrad.data.normalization_v3 import (
    OUTPUT_DIR,
    file_sha256,
    iter_raw,
    iter_rows,
    load_source_manifest,
    source_specs,
)

FIDELITY_VERSION = "call-fidelity-v1"
SOURCES = ("glaive", "toolace", "xlam")
MISMATCH_ID_LIMIT = 25

EXACT = "exact"
COUNT_MISMATCH = "count_mismatch"
NAME_MISMATCH = "name_mismatch"
ARGUMENT_MISMATCH = "argument_mismatch"
RAW_UNPARSEABLE = "raw_unparseable_independently"
OUTCOMES = (EXACT, COUNT_MISMATCH, NAME_MISMATCH, ARGUMENT_MISMATCH, RAW_UNPARSEABLE)

TRAJECTORY_VALID = "trajectory_valid"
TRAJECTORY_ISSUE = "trajectory_issue"

Call = tuple[str, str]  # (name, canonical JSON of the arguments)


class RawCallError(ValueError):
    """The raw text holds call syntax that this independent reader cannot parse."""


def canonical_arguments(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


# ── independent raw readers ──────────────────────────────────────────────────────────────────────────

_GLAIVE_MARKER = "<functioncall>"
_GLAIVE_TURN_END = re.compile(
    r"<\|endoftext\|>|\n\s*\n\s*\n\s*(?:USER|ASSISTANT|FUNCTION RESPONSE)\s*:"
)
_JSON_STRING = r'"((?:[^"\\]|\\.)*)"'
_GLAIVE_NAME = re.compile(r'"name"\s*:\s*' + _JSON_STRING)
_GLAIVE_ARGUMENTS = re.compile(r'"arguments"\s*:\s*')


def glaive_raw_calls(chat: str) -> list[Call]:
    """Calls in a Glaive ``chat``: ``<functioncall> {"name": …, "arguments": '<json>'}``, in order."""
    calls: list[Call] = []
    decoder = json.JSONDecoder()
    for match in re.finditer(re.escape(_GLAIVE_MARKER), chat):
        rest = chat[match.end() :]
        end = _GLAIVE_TURN_END.search(rest)
        segment = (rest[: end.start()] if end else rest).strip()
        name = _GLAIVE_NAME.search(segment)
        arguments = _GLAIVE_ARGUMENTS.search(segment)
        if not name:
            raise RawCallError("glaive: no name")
        call_name = json.loads(f'"{name.group(1)}"')
        if not arguments:
            calls.append((call_name, canonical_arguments({})))
            continue
        tail = segment[arguments.end() :]
        if tail.startswith("'"):
            closing = tail.rstrip().rstrip("}").rstrip().rfind("'")
            if closing <= 0:
                raise RawCallError("glaive: unterminated argument string")
            text = tail[1:closing]
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RawCallError("glaive: argument string is not JSON") from exc
        else:
            try:
                value, _ = decoder.raw_decode(tail)
            except json.JSONDecodeError as exc:
                raise RawCallError("glaive: argument object is not JSON") from exc
        calls.append((call_name, canonical_arguments(value)))
    return calls


def _dotted(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    raise RawCallError("toolace: call target is not a name")


_JSON_LITERAL_NAMES = {"true": True, "false": False, "null": None}


def _mask_call_names(body: str) -> tuple[str, dict[str, str]]:
    """Replace call names and keyword-argument names with Python identifiers, so they parse.

    ToolACE call names may contain spaces, dots and hyphens (``Get Weather(...)``), and argument names
    may be Python keywords or contain hyphens (``from=``, ``page-size=``); Python's parser rejects all of
    them. A call name is the text between ``[`` or a top-level ``,`` and the next ``(``; an argument name
    is the text between a call's ``(`` or an argument-level ``,`` and the next ``=``. Both outside quotes.
    Returns the masked text and ``placeholder -> original name``.
    """
    names: dict[str, str] = {}
    out: list[str] = []
    depth = 0
    quote: str | None = None
    escaped = False
    capture: tuple[str, int] | None = None  # ("call" | "arg", start index)
    for index, char in enumerate(body):
        if quote:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if capture is not None:
            kind, start = capture
            if kind == "call" and char == "(":
                placeholder = f"__call{len(names)}"
                names[placeholder] = body[start:index].strip()
                out.append(placeholder + "(")
                capture = ("arg", index + 1)
                depth += 1
                continue
            if kind == "arg" and char == "=" and body[index + 1 : index + 2] != "=":
                text = body[start:index].strip()
                placeholder = f"__arg{len(names)}"
                names[placeholder] = text
                out.append(placeholder + "=")
                capture = None
                continue
            if kind == "call" and char in "[]),\"'{}":
                raise RawCallError("toolace: malformed call name")
            if kind == "arg" and char in "[](){},\"'":
                # Not a keyword argument name (positional value or empty call): emit what was held back.
                out.append(body[start:index])
                capture = None
            else:
                continue
        if char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
            if char == "[" and depth == 1:
                out.append(char)
                capture = ("call", index + 1)
                continue
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 1:
            out.append(char)
            capture = ("call", index + 1)
            continue
        elif char == "," and depth == 2:
            out.append(char)
            capture = ("arg", index + 1)
            continue
        out.append(char)
    if capture is not None:
        out.append(body[capture[1] :])
    return "".join(out), names


def _literal(node: ast.expr) -> Any:
    if isinstance(node, ast.Name) and node.id in _JSON_LITERAL_NAMES:
        return _JSON_LITERAL_NAMES[node.id]
    if isinstance(node, ast.List):
        return [_literal(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {
            _literal(key) if key is not None else None: _literal(value)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    try:
        return ast.literal_eval(node)
    except ValueError as exc:
        raise RawCallError("toolace: non-literal argument") from exc


def toolace_turn_calls(text: str) -> list[Call]:
    """Calls in one ToolACE bracket turn ``[f(a=1), g(b="x")]``, read with Python's own parser.

    Call names are masked first (they may contain spaces). Arguments must be literals, with JSON's
    ``true``/``false``/``null`` accepted; nothing is evaluated. Trailing prose after the closing bracket
    is ignored by trying each closing bracket from the right.
    """
    body = text.lstrip()
    for end in [index for index, char in enumerate(body) if char == "]"][::-1]:
        try:
            masked, names = _mask_call_names(body[: end + 1])
            tree = ast.parse(masked, mode="eval")
        except (SyntaxError, RawCallError):
            continue
        if not isinstance(tree.body, ast.List) or not all(
            isinstance(item, ast.Call) for item in tree.body.elts
        ):
            continue
        calls: list[Call] = []
        for item in tree.body.elts:
            assert isinstance(item, ast.Call)
            if item.args:
                raise RawCallError("toolace: positional arguments")
            arguments = {
                names.get(str(keyword.arg), str(keyword.arg)): _literal(keyword.value)
                for keyword in item.keywords
            }
            target = _dotted(item.func)
            calls.append((names.get(target, target), canonical_arguments(arguments)))
        return calls
    raise RawCallError("toolace: bracket turn does not parse as a list of calls")


PRIMARY = "python_parser"
GRAMMAR_FALLBACK = "format_grammar"


def _top_level_split(text: str, *, after_open: bool) -> list[str]:
    """Split at commas outside quotes and brackets. With ``after_open``, a comma counts only once the
    current segment has opened a parenthesis, so a call name may itself contain commas."""
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    escaped = False
    start = 0
    opened = False
    for index, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
            opened = opened or char == "("
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0 and (opened or not after_open):
            parts.append(text[start:index].strip())
            start = index + 1
            opened = False
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _argument_opening(segment: str) -> int:
    """Index of the ``(`` whose ``)`` ends the segment: names may contain their own parentheses."""
    depth = 0
    quote: str | None = None
    escaped = False
    openings: list[int] = []
    last_closed = -1
    for index, char in enumerate(segment):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "\"'" and openings:
            quote = char
        elif char == "(":
            openings.append(index)
            depth += 1
        elif char == ")" and openings:
            last_closed = openings.pop()
            depth -= 1
    if depth != 0 or last_closed <= 0 or not segment.endswith(")"):
        raise RawCallError("toolace: segment is not name(arguments)")
    return last_closed


def _format_value(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RawCallError("toolace: value is neither a Python nor a JSON literal") from exc


def toolace_turn_calls_by_format(text: str) -> list[Call]:
    """Second reader, from ToolACE's documented format rather than Python's grammar.

    ``[Name(key=value, ...), ...]``: calls separated by top-level commas, a name that may contain spaces,
    commas or its own parentheses (the argument list is the parenthesis that closes the call), and values
    that are Python or JSON literals. It shares that reading of the format with the adapter, so rows it
    reads are weaker evidence than :func:`toolace_turn_calls` and are counted separately.
    """
    body = text.strip()
    if not body.startswith("["):
        raise RawCallError("toolace: no opening bracket")
    closing = body.rfind("]")
    if closing <= 0:
        raise RawCallError("toolace: no closing bracket")
    calls: list[Call] = []
    for segment in _top_level_split(body[1:closing].strip(), after_open=True):
        opening = _argument_opening(segment)
        name = segment[:opening].strip()
        arguments: dict[str, Any] = {}
        for part in _top_level_split(segment[opening + 1 : -1], after_open=False):
            key, equals, value = part.partition("=")
            if not equals or not key.strip():
                raise RawCallError("toolace: argument without a name")
            arguments[key.strip()] = _format_value(value.strip())
        calls.append((name, canonical_arguments(arguments)))
    if not calls:
        raise RawCallError("toolace: no call in bracket turn")
    return calls


def toolace_raw_calls(
    conversations: Iterable[Mapping[str, Any]], methods: list[str] | None = None
) -> list[Call]:
    """Calls in every assistant bracket turn. The method used per turn is appended to ``methods``."""
    calls: list[Call] = []
    for turn in conversations or []:
        value = turn.get("value")
        if (
            turn.get("from") in {"assistant", "gpt"}
            and isinstance(value, str)
            and value.lstrip().startswith("[")
            and "(" in value
        ):
            try:
                calls.extend(toolace_turn_calls(value))
                method = PRIMARY
            except RawCallError:
                calls.extend(toolace_turn_calls_by_format(value))
                method = GRAMMAR_FALLBACK
            if methods is not None:
                methods.append(method)
    return calls


def xlam_raw_calls(answers: Any) -> list[Call]:
    value = json.loads(answers) if isinstance(answers, str) else answers
    value = [value] if isinstance(value, dict) else (value or [])
    calls: list[Call] = []
    for item in value:
        if isinstance(item, dict) and item.get("name"):
            calls.append((str(item["name"]), canonical_arguments(item.get("arguments", {}))))
    return calls


def raw_calls(source: str, raw: Mapping[str, Any], methods: list[str] | None = None) -> list[Call]:
    if source == "glaive":
        return glaive_raw_calls(str(raw.get("chat") or ""))
    if source == "toolace":
        return toolace_raw_calls(raw.get("conversations") or [], methods)
    if source == "xlam":
        return xlam_raw_calls(raw.get("answers"))
    raise ValueError(f"no independent call reader for {source}")


def structured_calls(record: Mapping[str, Any]) -> list[Call]:
    return [
        (str(call["name"]), canonical_arguments(call.get("arguments", {})))
        for message in record["messages"]
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
    ]


def compare(raw: list[Call], structured: list[Call]) -> str:
    if len(raw) != len(structured):
        return COUNT_MISMATCH
    if any(a[0] != b[0] for a, b in zip(raw, structured, strict=True)):
        return NAME_MISMATCH
    if any(a[1] != b[1] for a, b in zip(raw, structured, strict=True)):
        return ARGUMENT_MISMATCH
    return EXACT


# ── measurement ──────────────────────────────────────────────────────────────────────────────────────


def _empty_cell() -> dict[str, Any]:
    return {
        "rows": 0,
        "rows_with_calls": 0,
        "raw_calls": 0,
        "structured_calls": 0,
        "rows_read_by_format_grammar": 0,
        **{outcome: 0 for outcome in OUTCOMES},
    }


def measure_rows(
    source: str,
    rows: Iterable[Mapping[str, Any]],
    raw_by_index: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Compare each accepted row with its raw row. Pure: rows and raw rows are passed in."""
    cells: dict[str, dict[str, Any]] = defaultdict(_empty_cell)
    mismatch_ids: dict[str, list[str]] = defaultdict(list)
    for record in rows:
        metadata = record["metadata"]
        status = (
            TRAJECTORY_ISSUE
            if metadata["structure"].get("trajectory_issue_codes")
            else TRAJECTORY_VALID
        )
        cell = cells[status]
        cell["rows"] += 1
        structured = structured_calls(record)
        raw_row = raw_by_index[int(metadata["source"]["raw_row_index"])]
        methods: list[str] = []
        try:
            raw = raw_calls(source, raw_row, methods)
        except RawCallError:
            outcome, raw = RAW_UNPARSEABLE, []
        else:
            outcome = compare(raw, structured)
        if not raw and not structured and outcome == EXACT:
            continue
        cell["rows_with_calls"] += 1
        cell["raw_calls"] += len(raw)
        cell["structured_calls"] += len(structured)
        cell[outcome] += 1
        cell["rows_read_by_format_grammar"] += GRAMMAR_FALLBACK in methods
        if outcome != EXACT and len(mismatch_ids[outcome]) < MISMATCH_ID_LIMIT:
            mismatch_ids[outcome].append(str(record["id"]))
    return dict(sorted(cells.items())), dict(sorted(mismatch_ids.items()))


def measure_ledger(
    source: str,
    ledger: Iterable[Mapping[str, Any]],
    raw_by_index: Mapping[int, Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    """Raw calls held by rows normalization-v3 did not accept, by disposition and reason."""
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for entry in ledger:
        key = f"{entry['disposition']}:{entry.get('reason') or '-'}"
        try:
            count = len(raw_calls(source, raw_by_index[int(entry["raw_row_index"])]))
        except RawCallError:
            table[key]["raw_unparseable_independently"] += 1
            continue
        table[key]["rows"] += 1
        table[key]["rows_with_calls"] += bool(count)
        table[key]["raw_calls"] += count
    return {key: dict(sorted(counter.items())) for key, counter in sorted(table.items())}


def measure(root: Path) -> dict[str, Any]:
    """Call conservation and argument fidelity for every layer A source of the recorded artifact."""
    specs = {spec.name: spec for spec in source_specs(load_source_manifest(root), root)}
    artifact = root / OUTPUT_DIR
    report: dict[str, Any] = {"version": FIDELITY_VERSION, "sources": {}}
    for source in SOURCES:
        spec = specs[source]
        if file_sha256(spec.raw_path) != spec.raw_sha256:
            raise ValueError(f"{source}: raw artifact bytes are not the recorded ones")
        raw_by_index = dict(enumerate(iter_raw(spec.raw_path)))
        cells, mismatch_ids = measure_rows(source, iter_rows(artifact, source), raw_by_index)
        ledger_path = artifact / source / "dispositions.jsonl"
        ledger = [
            json.loads(line)
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        report["sources"][source] = {
            "accepted_rows": cells,
            "not_accepted_rows": measure_ledger(source, ledger, raw_by_index),
            "mismatch_record_ids_first": mismatch_ids,
            "mismatched_rows": sum(
                cell[outcome] for cell in cells.values() for outcome in MISMATCHES
            ),
            "unverified_rows": sum(cell[RAW_UNPARSEABLE] for cell in cells.values()),
        }
    report["status"] = fidelity_status(report["sources"])
    return report


#: Outcomes that are evidence of an extraction error.
MISMATCHES = (COUNT_MISMATCH, NAME_MISMATCH, ARGUMENT_MISMATCH)


def fidelity_status(sources: Mapping[str, Mapping[str, Any]]) -> str:
    """``FAIL`` on any mismatch; ``INCOMPLETE`` when some rows could not be read independently (they are
    unverified, never counted as passing); ``PASS`` only when every call row was compared and matched."""
    if any(item["mismatched_rows"] for item in sources.values()):
        return "FAIL"
    if any(item["unverified_rows"] for item in sources.values()):
        return "INCOMPLETE"
    return "PASS"


if __name__ == "__main__":  # pragma: no cover - measurement entry point, counts only
    print(json.dumps(measure(Path.cwd()), indent=2, sort_keys=True))
