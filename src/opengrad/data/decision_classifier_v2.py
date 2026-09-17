"""The Study 002 prose decision classifier, ``prose-decision-classifier-v2`` (under development, 37).

It started as a copy of the frozen ``prose-decision-classifier-v1`` (``decision_classifier.py``, tag
``prose-decision-classifier-v1``), which stays unchanged. It reads the first reply of a record through the input
contract ``prose-decision-input-v2`` (the same four features as v1's single exchange)
(:class:`opengrad.data.classifier_input.ClassifierFeatures`: the user message, the assistant response, the
offered tools, and the structured-call flag) and says what the response does:

* ``CALL`` -- only for a machine-readable textual call to an offered tool (30 §4: a structured call never
  reaches this classifier);
* ``DIRECT`` -- answers or performs the request;
* ``CLARIFY`` -- withholds the answer and asks for something required;
* ``UNSUPPORTED`` -- declines, or explains that it cannot act;
* ``ABSTAIN`` -- the rules cannot decide (the annotators' UNKNOWN).

The rules follow the annotation decision tree of 22 §3, in its order, with the discriminations of 22 §2.
They are deterministic and read nothing else: no source, id, stratum, label or file. Each decision carries
the tree step that decided it and the text that matched, so any label can be checked by hand.

How the rules are developed is fixed in docs/research/study-002/37-PROSE-DECISION-CLASSIFIER-V2-DEVELOPMENT-PLAN.md:
against model-labelled development sets only, never against a validation population. Agreement with those
development labels is not accuracy.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from opengrad.data import versions
from opengrad.data.classifier_input import ClassifierFeatures

CLASSIFIER_VERSION = versions.DECISION_CLASSIFIER_V2_VERSION

CALL, DIRECT, CLARIFY, UNSUPPORTED, ABSTAIN = "CALL", "DIRECT", "CLARIFY", "UNSUPPORTED", "ABSTAIN"
LABELS = (CALL, DIRECT, CLARIFY, UNSUPPORTED, ABSTAIN)

# Tree steps (22 §3), recorded on every decision.
STEP_CALL = "1_textual_call"
STEP_CALL_UNVERIFIABLE = "1_call_payload_unverifiable"
STEP_NON_SUBSTANTIVE = "2_non_substantive"
STEP_DECLINE = "3_decline"
STEP_DECLINE_WITH_CONTENT = "3_decline_with_delivered_content"
STEP_DECLINE_FOR_MISSING_INPUT = "3_decline_for_missing_user_input"
STEP_REQUEST = "4_required_request"
STEP_REQUEST_WITH_CONTENT = "4_optional_request_after_content"
STEP_EXTERNAL_SERVICE = "5_external_service_instead"
STEP_NARRATED_CALL = "6_narrated_call"
STEP_DELIVERED = "7_delivered"

#: Words of prose that is left after the declines, requests, offers and courtesies are removed, from which
#: a response counts as having delivered substantive content (22 §1.2: delivery, not completeness).
CONTENT_WORDS_AFTER_DECLINE = 40
CONTENT_WORDS_BEFORE_REQUEST = 12
CONTENT_WORDS_AFTER_REQUEST = 50
#: A clause after "but"/"however" in a decline that is itself content ("I can't know X, but at latitude X
#: the answer is Y") delivers an answer when it has at least this many words (22 §1.4 exclusion).
CONTENT_WORDS_IN_BUT_CLAUSE = 6
#: A narrated call is only read as one in a short response; a long one is an explanation.
NARRATION_MAX_WORDS = 60


@dataclass(frozen=True)
class Decision:
    label: str
    step: str
    evidence: tuple[str, ...] = ()
    classifier_version: str = CLASSIFIER_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "step": self.step,
            "evidence": list(self.evidence),
            "classifier_version": self.classifier_version,
        }


# ── text helpers ─────────────────────────────────────────────────────────────────────────────────────

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
_CODE_FENCE = re.compile(r"```.*?(?:```|$)", re.DOTALL)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


def _clean(text: str) -> str:
    return text.translate(_APOSTROPHES)


def _words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", text))


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_END.split(text) if part and part.strip()]


def _any(patterns: Iterable[re.Pattern[str]], text: str) -> re.Match[str] | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match
    return None


def _rx(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


def _tool_names(tools: Iterable[Mapping[str, Any]]) -> list[str]:
    names = []
    for tool in tools:
        spec = tool.get("function", tool) if isinstance(tool, Mapping) else {}
        name = spec.get("name") if isinstance(spec, Mapping) else None
        if name:
            names.append(str(name))
    return names


def _norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().strip("\"'`")).casefold()


# ── step 1: machine-readable call payloads (22 §1.1) ────────────────────────────────────────────────

_TAGGED_CALL = re.compile(r"<functioncall>|<tool_call>|<toolcall>", re.IGNORECASE)
_JSON_NAME = re.compile(r'\{\s*"name"\s*:\s*"([^"]+)"')
_JSON_ARGUMENTS = re.compile(r'"(arguments|parameters)"\s*:', re.IGNORECASE)
# ToolACE's native form: [Name(key=value, ...), Other(...)] -- names may hold spaces, dots or slashes.
_BRACKET_CALL = re.compile(
    r"\[\s*((?:[A-Za-z_][\w .\-/]*?\s*\((?:[^()]|\([^()]*\))*\)\s*,?\s*)+)\]"
)
_BRACKET_CALL_ITEM = re.compile(r"([A-Za-z_][\w .\-/]*?)\s*\(((?:[^()]|\([^()]*\))*)\)")
_FUNC_NAME_KWARG = re.compile(r"""\bname\s*=\s*["']([^"']+)["']""")
_QUOTED = re.compile(r'"[^"]*"|\'[^\']*\'')
# A call-block response whose whole text is a bracketed list of calls in any delimiter style, e.g.
# {Name:<k--"v">}, (Name-{k--v}), [[Name]|{k|v}], <Name:[k==v]>. The head is an unquoted name followed by an
# argument opener.
_BLOCK_HEAD = re.compile(
    r"(?:^|[\[{(<,|]\s*)([A-Za-z_][\w .\-/]*[\w)])\s*[\]})>]?\s*(?:=>|->|--|-|\||:|=)?\s*[\[{(<]"
)
_PROSE_WORDS = re.compile(
    r"\b(the|you|your|please|sorry|cannot|can't|i'm|function|functions|here|this|that|would|could)\b",
    re.IGNORECASE,
)


def _json_calls(text: str) -> list[str] | None:
    """Names of JSON call objects, when the response carries a call-shaped JSON payload."""
    if not (_TAGGED_CALL.search(text) or _JSON_ARGUMENTS.search(text)):
        return None
    names = _JSON_NAME.findall(text)
    if not names:
        for chunk in re.findall(r"\{.*\}", text, re.DOTALL):
            try:
                value = json.loads(chunk)
            except ValueError:
                continue
            if isinstance(value, Mapping) and value.get("name"):
                names.append(str(value["name"]))
    return names or None


def _bracket_calls(text: str) -> list[str] | None:
    match = _BRACKET_CALL.search(text)
    if not match:
        return None
    names = []
    for head, args in _BRACKET_CALL_ITEM.findall(match.group(1)):
        if args.strip() and "=" not in args:
            return None  # "[see figure (a)]" is not a call
        kwarg = _FUNC_NAME_KWARG.search(args) if head.strip() == "func" else None
        names.append(kwarg.group(1) if kwarg else head)
    return names or None


def _block_calls(text: str) -> list[str] | None:
    stripped = text.strip()
    if len(stripped) < 6 or stripped[0] not in "[{(<" or stripped[-1] not in "]})>":
        return None
    unquoted = _QUOTED.sub("", stripped)
    if _PROSE_WORDS.search(unquoted):
        return None
    heads = _BLOCK_HEAD.findall(unquoted)
    return heads or None


def _call_payload(text: str) -> tuple[str, list[str]] | None:
    for kind, finder in (("json", _json_calls), ("bracket", _bracket_calls), ("block", _block_calls)):
        names = finder(text)
        if names:
            return kind, names
    return None


# ── step 2: non-substantive (22 §1.2 exclusion, §3 step 2) ──────────────────────────────────────────

_MARKER_ONLY = re.compile(r'^[\s"\'`\[\]().…*_-]*(no need to ask)?[\s"\'`\[\]().…*_-]*$', re.IGNORECASE)


# ── steps 3 and 5: declines, inability and external services (22 §1.4) ─────────────────────────────

# Hedges about the world are not inability to act (22 §2 rows 3 and 7).
_EPISTEMIC = _rx(
    r"\b(can't|cannot|can not) (be (sure|certain)|say for (sure|certain)|guarantee|know for (sure|certain))\b",
    r"\bi'?m not (sure|certain)\b",
    r"\bi don't know (for sure|exactly)\b",
    # v2 round 1: an exclamation, not inability.
    r"\b(can't|cannot) believe\b",
)
_DECLINE = _rx(
    r"\b(i|we)\s*(?:'m| am)?\s*(?:really |truly )?(sorry|afraid|apologi[sz]e)\b[^.!?]{0,80}"
    r"\b(can't|cannot|can not|unable|not able|don't have|do not have|won't be able|not possible)\b",
    r"\bapologies\b[^.!?]{0,80}\b(can't|cannot|unable|not able|don't have|do not have)\b",
    r"\b(i|we)(?:'m| am|'re| are)? (?:currently |really |just |simply )?(can't|cannot|can not|unable to|not able to|"
    r"won't be able to)\b",
    r"\b(i'm|i am) (designed|built|programmed|here) to\b[^.!?]{0,80}\bnot (to )?",
    r"\b(i|we) (don't|do not) have (the |any )?(capability|capabilities|ability|access|real-?time|functions?|tools?)\b",
    r"\bas an ai\b[^.!?]{0,60}\b(can't|cannot|unable|don't|do not)\b",
    r"\b(unable|not able) to (assist|help|perform|provide|fulfil|fulfill|complete|do|book|order|access|retrieve|"
    r"fetch|browse|search|generate|find|check|process|make|place|call|use|get|answer)\b",
    r"\bnot possible to\b|\bit is not possible\b|\bimpossible to\b",
    r"\b(cannot|can't|could not|couldn't) (be )?(fulfill|fulfil|fulfilled|achieved|addressed|obtained|retrieved)\b",
    # v2 round 2: having no information about the subject, stated as the reply.
    r"\b(i|we) (don't|do not) have (any )?(specific |further |additional |more |enough |sufficient )?"
    r"(information|data|details|knowledge) (about|on|regarding|of)\b",
)
# The missing thing is a capability, tool or scope (22 §2 rows 2 and 6).
_CAPABILITY = _rx(
    # v2 round 5: "none of the required arguments are given" is missing input, not a missing tool.
    r"\b(none|neither) of (the|these|those|them)\b"
    r"(?! (required |necessary |needed |mandatory )?(arguments?|parameters?|details|information|fields?|inputs?|values?)\b)"
    r"[^.!?]{0,40}\b(functions?|tools?|apis?)?\b",
    r"\bno (available |suitable |relevant |specific |such |appropriate |defined |matching |corresponding )?"
    r"(functions?|tools?|apis?)\b",
    r"\b(does|do|did) not apply\b|\b(doesn't|don't) apply\b|\bunrelated to\b",
    r"\b(functions?|tools?|apis?) (list )?(is|are) empty\b|\bempty (functions?|tools?) list\b",
    r"\bnot covered by\b|\bwould be (necessary|needed|required) to fulfil",
    # v2 round 5: "perform" and "handle" added.
    r"\b(functions?|tools?|apis?|capabilities)\b[^.!?]{0,120}\b(do|does|did) not (directly |currently )?(support|include|"
    r"cover|provide|have|match|allow|offer|pertain|relate|address|fit|perform|handle)\b",
    r"\b(functions?|tools?|apis?|capabilities)\b[^.!?]{0,120}\b(don't|doesn't) (directly |currently )?(support|include|cover|provide|have|"
    r"match|allow|offer|pertain|relate|address|fit|perform|handle)\b",
    r"\b(not|isn't|aren't) (applicable|suitable|designed|capable|relevant|meant|related)\b",
    # v2 round 1: the request is outside what the listed functions do.
    r"\b(not|isn't|aren't) supported by\b|\b(do|does) not align with\b",
    r"\bcannot be (accomplished|done|performed|completed|carried out) with\b",
    r"\b(are|is) (only )?(meant|designed|tailored|intended|specific(ally)?( designed)?|limited|focused) (for|to|on)\b",
    r"\b(outside|beyond) (of )?(the |my )?(scope|capabilit)",
    r"\bfalls? outside\b",
    # v2 round 2: only a negated or limited ability ("lawyers have the ability to see" is not inability).
    r"(\bnot|n't|\bno|\bcannot|\blacks?|\blacking|\bwithout|\bunable|\bbeyond|\boutside|\blimited)\b[^.!?]{0,60}"
    r"\b(capability|capabilities|ability) to\b",
    r"\bcapabilities are\b|\bmy (current )?(capabilities|functions?|abilities)\b",
    r"\b(real-?time|physical|external) (data|tasks?|information|access|actions?)\b",
    r"\b(can't|cannot|unable to|not able to) (directly )?(call|use|invoke)\b[^.!?]{0,40}\b(functions?|tools?|apis?)\b",
    r"\bno function (in|from|provided|available|that|can)\b",
    # v2 round 2: the tools are lacking, or cannot do the task.
    r"\b(lacks?|lacking|without) (the |any |a |an )?(necessary |required |appropriate |relevant |suitable |needed )?"
    r"(functions?|tools?|apis?)\b",
    # v2 round 3: "the functions cannot be called" for want of arguments is not a capability gap.
    r"\b(functions?|tools?|apis?)\b[^.!?]{0,40}\b(cannot|can't|can not|could not|couldn't|(is|are) unable to) "
    # v2 round 4: "cannot be used to achieve the purpose" stays a capability gap; only uncallable is excluded.
    r"(?!be (called|invoked|executed|run|made)\b)\w+",
    r"\bcannot be (processed|handled|answered|addressed|fulfilled|resolved) (using|with|by) (the )?(given|provided|"
    r"available|listed|offered) (functions?|tools?|apis?)\b",
    r"\b(none|neither) of which (is|are) (relevant|applicable|suitable|useful|related)\b",
    # Asking whether a capability exists: the missing thing is a tool, not user input (22 §1.3 boundary).
    r"\bdo you have (a|an|any|another) [\w\- ]{0,30}(tool|function|api|integration|plugin)\b",
    r"\bonly (allow|support|provide|cover|retrieve|pertain|relate|focus|convert|generate|search|give|return|deal|work|"
    r"handle)s?\b",
    r"\bdon't have (access|the ability)\b|\bdo not have (access|the ability)\b",
)
# A decline whose stated reason is missing user input (22 §1.4 exclusion to CLARIFY).
_MISSING_INPUT_REASON = _rx(
    r"\bwithout (knowing|the|your|more|additional|further|a|an)\b",
    r"\bunless you (provide|specify|tell|give)\b",
    r"\b(you|the (query|question|request)) (did not|didn't|has not|hasn't|have not|haven't) (provide|specif|mention|give|include)",
)
_EXTERNAL_SERVICE = _rx(
    r"\b(i )?(would |'d )?(recommend|suggest)(ing)? (checking|using|visiting|contacting|consulting|you (check|use|visit|contact|consult))\b",
    r"\byou (may|might|could|should) (want to )?(check|visit|contact|consult|refer to)\b",
    r"\b(check|use|visit|try) (a|an|the) (reliable |official |specialized |specialised |dedicated )?(online |weather |news )?"
    r"(source|website|site|service|generator|database|app|platform)\b",
    r"\byou would need (access to|to use|to consult)\b",
    # v2 round 2: a referral phrased as a need.
    r"\byou (may|might|could|would|will) (need|want|have) to (consult|check|visit|contact|refer to|look up|use|search)\b",
    r"\b(please )?(consult|contact|refer to|check with) (a|an|the|your)\b",
    r"\bonline (generator|tool|service|source|resource)s?\b",
)
_OFFER = _rx(
    r"\b(i|we) (can|could|would be happy to|'d be happy to|am happy to|'m happy to) (still )?(help|assist)\b",
    r"\bif you(?: would|'d)? (like|need|want|prefer|wish)\b|\bif you have\b",
    r"\bfeel free\b",
    r"\bis there (anything|something|a specific|any)\b",
    r"\b(would|do) you (like|want) (me )?to\b",
    r"\blet me know\b",
)

# ── step 4: requests to the user (22 §1.3) and courtesy questions (22 §1.2) ─────────────────────────

_COURTESY = _rx(
    r"\bis there anything else\b",
    r"\banything else (i|you)\b",
    r"\b(does|did|do) (that|this|these) help\b",
    r"\bhope (this|that|it) helps\b",
    r"\blet me know if\b",
    r"\bfeel free to\b",
    r"\bhow (can|may) i (assist|help) you( today)?\b",
    r"\bdo you have any (other|more|further)? ?questions\b",
    r"\bif you have any (other|more|further)? ?questions\b",
    r"\b(want|would you like) me to (expand|elaborate|explain|go into)\b",
    r"^(hello|hi|hey)[!,.]",
)
_REQUEST = _rx(
    r"\b(could|can|would|will) you (please |kindly )?(provide|specify|tell|share|give|confirm|clarify|let me know|"
    r"send|enter|supply|indicate|clarify|list|describe|paste|upload)\b",
    r"\bplease (provide|specify|share|tell|give|confirm|clarify|let me know|send|enter|supply|indicate|include)\b",
    r"^[-*\d.\s]*(provide|specify|supply|enter|share) (the|your|a|an|me|us)\b",
    r"\b(i|i'll|i will|we|we'll) (would )?(first )?need (to know |you to (provide|specify|tell|give) )?"
    r"(the|your|a|an|some|more|additional|further|to know|which|what|details|information|specific|certain|particular)\b",
    r"\byou('ll| will)? (need|have) to (provide|specify|give|supply|tell)\b",
    r"\b(required|necessary|mandatory)\b[^.!?]{0,60}\b(parameters?|arguments?|information|details|fields?|inputs?)\b"
    r"[^.!?]{0,40}\b(missing|not (been )?(provided|specified|given)|lack)",
    r"\b(lacks?|missing|is missing|are missing)\b[^.!?]{0,60}\b(required|necessary)?\b[^.!?]{0,20}"
    r"\b(parameters?|arguments?|information|details|fields?|inputs?|values?|id|identifier)\b",
    r"\b(parameters?|arguments?)\b[^.!?]{0,40}\b(missing|not (been )?(provided|specified|given))\b",
    r"\b(does not|doesn't|did not|didn't) (provide|include|specify)\b[^.!?]{0,40}\b(sufficient|enough|required|"
    r"necessary|the|a|an|any)\b",
    # A statement of what the user left out asks for it (22 §2 row 6).
    r"\b(you|the (query|question|request|user)) (have|has|did|does) not (provided?|specif(y|ied)|given?|"
    r"mention(ed)?|include[d]?|supplied|supply)\b",
    r"\b(you|the (query|question|request|user)) (haven't|hasn't|didn't|doesn't) (provided?|specif(y|ied)|given?|"
    r"mention(ed)?|include[d]?|supplied|supply)\b",
    r"\bto proceed\b[^.!?]*\b(is|are) (needed|required)\b|\b(is|are) (needed|required) to proceed\b",
    r"\bfor which\b[^.!?]*\?",
    r"\b(needs?|requires?|need to know) (more|additional|further|some|a few) (information|details|input|context)\b",
    r"\b(is|are) missing\s*:|\b(missing|need|require) the following\b",
    # v2 round 1: statements that information is still needed.
    r"\b(additional|more|further|this|that|the following) (information|details|input|data) (is|are) (lacking|missing|"
    r"needed|required)\b",
    r"\b(shall|should|may) i (go ahead|proceed|continue|call|use|run)\b",
    r"\bdo you want me to (go ahead|proceed|continue|call|use|run)\b",
    r"^(what|which|who|where|when|how many|how much|do you|would you prefer|should i)\b[^.!?]*\?",
    # v2 round 1: polite and measure questions that withhold the answer.
    r"^(may|can|could) i (have|know|get|ask( for)?)\b[^.!?]*\?",
    r"^how (long|big|large|old|often|far|soon)\b[^.!?]*\?",
    # v2 round 3: asking which of several things the user means.
    r"^(are|were) you (interested in|looking for|asking about|referring to)\b[^.!?]*\?",
)
_INVOCATION_TALK = _rx(
    r"\b(i('ll| will| would| could| can| am going to|'m going to)|let me) (use|call|run|invoke|check)\b[^.!?]{0,60}"
    r"\b(functions?|tools?|apis?)\b",
    r"\busing the [\w .\-/]{1,60} (function|tool|api)\b",
    r"\bthe (right|best|appropriate|correct|relevant) (tool|function|api)\b",
)
_INVOCATION_VERB = re.compile(r"\b(use|call|invoke|run|using|calling|invoking)\b", re.IGNORECASE)
# Sentences about the offered tools, their parameters, or the request itself: talk about the task, not content.
_ABOUT_FUNCTIONS = re.compile(
    r"\b(functions?|tools?|apis?)\b[^.!?]{0,80}\b(required|requires|parameters?|arguments?|provided|given|available|"
    r"listed|call|called|calling|can be used)\b"
    r"|\b(required|provided|given|available|listed|following)\s+(functions?|tools?|apis?)\b"
    r"|\b(functions?|tools?|apis?)\s+[\"'`]|[\"'`][\w .\-/]+[\"'`]\s+(functions?|tools?|apis?)\b"
    r"|\b(functions?|tools?|apis?)\b[^.!?]{0,80}\b(can|could|will|would) (help|be used|provide|retrieve|find|get|"
    r"search|fetch|collect|add|update|generate|check)\b"
    r"|\b(parameters?|arguments?)\b"
    # v2 round 3: naming which offered function does the job ("The function that retrieves X is ...").
    r"|\b(functions?|tools?|apis?) (that|which) (retrieves?|returns?|gets?|fetches?|provides?|lists?|finds?|searches)\b"
    # v2 round 4: what the assistant will do once the requested input arrives.
    r"|^once (i have|i receive|i get|you provide|you've provided|you have provided)\b"
    r"|\b(the|your|this) (given |user's |original )?(query|question|request)\b"
    r"|\b(helpful|useful|necessary|needed) to (have|know)\b|\bfollowing (information|details)\b"
    r"|\b(missing|lacks?|ambiguous|once provided|not explicitly|not given)\b"
    r"|`[^`\s]+`",
    re.IGNORECASE,
)


def _strip(sentences: list[str], *groups: tuple[re.Pattern[str], ...]) -> list[str]:
    return [s for s in sentences if not any(_any(group, s) for group in groups)]


_LIST_MARKER = re.compile(r"^\s*(\d+[.)]|[-*•])\s*$")
_LIST_ITEM = re.compile(r"^\s*([-*•]|\d+[.)])\s+")


def _task_list_items(sentences: list[str]) -> set[int]:
    """v2 round 4: indexes of list items introduced by a sentence ending in ":" that talks about the task, the
    tools or a decline ("…the given functions do not support: 1. X 2. Y"). The items restate what was asked or
    what is missing; they are part of that sentence, not delivered content."""
    items: set[int] = set()
    for i, sentence in enumerate(sentences):
        intro = sentence.rstrip().endswith(":") and (
            _ABOUT_FUNCTIONS.search(sentence) or _any(_DECLINE, sentence) or _any(_CAPABILITY, sentence)
        )
        if not intro:
            continue
        j, after_marker = i + 1, False
        while j < len(sentences):
            current = sentences[j]
            if _LIST_MARKER.match(current):
                after_marker = True
            elif _LIST_ITEM.match(current) or after_marker:
                after_marker = False
            else:
                break
            items.add(j)
            j += 1
    return items


def _without(sentences: list[str], drop: set[int], start: int = 0) -> list[str]:
    return [s for i, s in enumerate(sentences, start) if i not in drop]


# v2 round 4: a label introducing rewritten text ("Corrected sentence: I can't find it."): what follows is content.
_REWRITE_LABEL = re.compile(
    r"^\W*(corrected|revised|rewritten|edited|fixed|improved|rephrased|paraphrased|translated|condensed|shortened)"
    r"( \w+)?( (sentence|version|text|paragraph|passage)s?)?\s*:",
    re.IGNORECASE,
)
# v2 round 4: a worked result ("gamma(3) = 2", "≈ 1.79") delivers an answer, whatever functions it names.
_WORKED_RESULT = re.compile(r"[\d)]\s*[=≈]\s*-?\d")

def _content_words(sentences: list[str]) -> int:
    return sum(_words(s) for s in sentences if not _ABOUT_FUNCTIONS.search(s))


_ACKNOWLEDGEMENT = _rx(
    r"^(sure|certainly|of course|absolutely|okay|ok|great|got it|i understand|i see|understood)\b",
    r"^to (assist|help|provide|proceed|give|find|get|check|fetch|retrieve)\b[^.!?]*$",
)
_DELIVERY = re.compile(r"^(here's|here is|here are)\b", re.IGNORECASE)
# v2 round 1: a sentence that only announces what the assistant is about to do ("Let me check that for you."), the
# usual first reply before a call in a continuing conversation. It delivers nothing (22 §3 step 2).
_ANNOUNCEMENT = re.compile(
    r"^(?:(?:sure|certainly|of course|absolutely|okay|ok|great|alright|no problem)[,!.]?\s*)?"
    r"(let me|let's|let us|i'll|i will|i am going to|i'm going to)\s+(?!know\b|explain\b|help\b|tell\b|share\b|"
    r"describe\b|walk\b|break\b|give\b|provide\b|list\b|show\b|summari[sz]e\b|outline\b)\w+\b[^:]{0,80}[.!]?$",
    re.IGNORECASE,
)
_HELP_OFFER = re.compile(
    r"^(i can|i'd be happy to|i would be happy to|happy to) (definitely |certainly )?help( you)?( with (that|this))?[.!]?$",
    re.IGNORECASE,
)
ANNOUNCEMENT_MAX_WORDS = 30
# v2 round 5: a reply made only of plans for later ("I will start by analyzing ... Then, I will ... I will keep you
# updated") delivers nothing yet, however long the plan.
_PLAN = re.compile(
    r"^(?:(?:then|first|next|after that|finally|afterwards),?\s+)?(i will|i'll|i am going to|i'm going to|let me)\s+"
    r"(?!know\b|explain\b|help\b|tell\b|share\b|describe\b|walk\b|break\b|give\b|provide\b|list\b|show\b|"
    r"summari[sz]e\b|outline\b)\w+",
    re.IGNORECASE,
)
_THANKS = re.compile(r"^(thank(s| you)|great|sure|certainly|of course|okay|ok|absolutely)\b[^.?]{0,40}[.!]?$", re.IGNORECASE)
PLAN_MAX_WORDS = 80
_TOOL_LIMIT_TAIL = re.compile(
    r"^(they|it|this|these|those)( functions?| tools?)? (do|does|did|can|could) ?(not|n't) (directly |currently )?"
    r"(support|include|cover|provide|have|offer|retrieve|handle|return|fetch|find|get|allow|perform)\b",
    re.IGNORECASE,
)
_LEAD_IN = re.compile(r"^(i can tell you (that )?|i can say (that )?|here is what i know:?\s*)", re.IGNORECASE)


def _but_clause_content(sentence: str) -> str | None:
    """The clause after "but"/"however" in a decline, or a sentence opening with one after it, when it is
    itself content rather than a decline, an offer, a request or advice to go elsewhere."""
    parts = re.split(r"\b(?:but|however|that said|nevertheless)\b,?", sentence, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) < 2:
        # v2 round 3: a concessive disclaimer ("Though I can't give medical advice, it's a good idea to ...").
        concessive = re.match(r"\s*(?:though|although|while|even though)\b[^,]*,\s*(.+)$", sentence, re.IGNORECASE)
        if not concessive:
            return None
        parts = [sentence[: concessive.start(1)], concessive.group(1)]
    tail = _LEAD_IN.sub("", parts[1].strip())
    if not tail or _any(_DECLINE, tail) or _any(_CAPABILITY, tail) or _any(_OFFER, tail):
        return None
    if _any(_EXTERNAL_SERVICE, tail) or _any(_REQUEST, tail) or _ABOUT_FUNCTIONS.search(tail):
        return None
    # v2 round 1: "..., but they do not retrieve X" restates what the tools cannot do.
    if _TOOL_LIMIT_TAIL.search(tail):
        return None
    return tail if _words(tail) >= CONTENT_WORDS_IN_BUT_CLAUSE else None


# ── the classifier ───────────────────────────────────────────────────────────────────────────────────


def classify(features: ClassifierFeatures) -> Decision:
    """The decision one assistant response represents, by the tree of 22 §3."""
    if features.structured_call_present:
        raise ValueError("a record with a structured call is CALL by structure (layer A), never classified here")
    raw = _clean(features.assistant_response)
    offered = {_norm_name(name) for name in _tool_names(features.tools)}

    # 1. A machine-readable call payload. CALL only when every call names an offered tool; a payload naming
    #    an unoffered tool, or offered with no tools at all, cannot be confirmed (22 §1.1, §1.5).
    payload = _call_payload(raw)
    if payload:
        kind, names = payload
        # A block's later heads can be argument names ({customerData|[...]}), so a block needs only its first
        # head offered; JSON and bracket calls name every call explicitly and all must be offered.
        named = [_norm_name(name) for name in (names[:1] if kind == "block" else names)]
        if offered and all(name in offered for name in named):
            return Decision(CALL, STEP_CALL, (kind, *names))
        return Decision(ABSTAIN, STEP_CALL_UNVERIFIABLE, (kind, *names))

    prose = _CODE_FENCE.sub(" ", raw)
    # Backticks around a lone identifier are not delivered code.
    has_code = any(len(block.strip("`").strip()) >= 40 for block in _CODE_FENCE.findall(raw))
    # 2. Empty, a marker, or nothing but punctuation.
    if not raw.strip() or (_MARKER_ONLY.match(raw) and not has_code) or _words(raw) == 0:
        return Decision(ABSTAIN, STEP_NON_SUBSTANTIVE)

    sentences = _sentences(prose)
    # v2 round 4: the rewritten text after "Corrected sentence:" is content, even when it reads like a decline.
    label_at = next((i for i, s in enumerate(sentences) if _REWRITE_LABEL.match(s)), None)
    rewritten = set(range(label_at, len(sentences))) if label_at is not None else set()
    epistemic_free = [s for i, s in enumerate(sentences) if not _any(_EPISTEMIC, s) and i not in rewritten]
    listed = _task_list_items(sentences)

    # 3. A decline or statement of inability.
    decline = next((m for s in epistemic_free if (m := _any(_DECLINE, s) or _any(_CAPABILITY, s))), None)
    if decline:
        capability = next((m for s in sentences if (m := _any(_CAPABILITY, s))), None)
        declining = [s for s in sentences if _any(_DECLINE, s) or _any(_CAPABILITY, s)]
        contrasting = [s for s in sentences if re.match(r"(however|but|that said|nevertheless|still)\b", s, re.IGNORECASE)]
        for sentence in declining + contrasting:
            tail = _but_clause_content(sentence)
            if tail:
                return Decision(DIRECT, STEP_DECLINE_WITH_CONTENT, (decline.group(0), tail[:80]))
        rest = _strip(_without(sentences, listed), _DECLINE, _CAPABILITY, _OFFER, _EXTERNAL_SERVICE, _COURTESY, _REQUEST)
        # "Here's the answer: ..." after a note that the tools don't fit delivers the answer (22 §1.4 exclusion).
        # v2 round 2: "Here is why I cannot proceed:" introduces the decline's reasons, not an answer.
        delivery = next(
            (i for i, s in enumerate(sentences) if _DELIVERY.match(s) and not _any(_OFFER, s) and not _any(_DECLINE, s)),
            None,
        )
        if delivery is not None:
            delivered = " ".join(sentences[delivery:])
            if _words(delivered) >= CONTENT_WORDS_IN_BUT_CLAUSE and not _any(_EXTERNAL_SERVICE, delivered):
                return Decision(DIRECT, STEP_DECLINE_WITH_CONTENT, (decline.group(0), sentences[delivery][:80]))
        # v2 round 2: with no capability gap stated, what lies between the first and last request (a list of the
        # missing fields) belongs to the request, as in step 4, not to delivered content.
        asks = [i for i, s in enumerate(sentences) if _any(_REQUEST, s)]
        if capability is None and asks:
            rest = _strip(
                _without(sentences[: asks[0]], listed) + _without(sentences[asks[-1] + 1 :], listed, asks[-1] + 1),
                _DECLINE, _CAPABILITY, _OFFER, _EXTERNAL_SERVICE, _COURTESY, _REQUEST,
            )
        if has_code or _content_words(rest) >= CONTENT_WORDS_AFTER_DECLINE:
            return Decision(DIRECT, STEP_DECLINE_WITH_CONTENT, (decline.group(0),))
        if capability is None:
            reason = next((m for s in declining if (m := _any(_MISSING_INPUT_REASON, s))), None)
            request = next((m for s in sentences if (m := _any(_REQUEST, s))), None)
            if reason or request:
                return Decision(
                    CLARIFY, STEP_DECLINE_FOR_MISSING_INPUT, (decline.group(0), (reason or request).group(0))
                )
        return Decision(UNSUPPORTED, STEP_DECLINE, (decline.group(0),) + ((capability.group(0),) if capability else ()))

    # 4. A request to the user: CLARIFY when the answer is withheld, DIRECT when content was delivered and the
    #    question is optional (22 §2 rows 1, 4 and 8).
    non_courtesy = _strip(sentences, _COURTESY)
    request = next((m for s in non_courtesy if (m := _any(_REQUEST, s))), None)
    if request:
        # Content counts before the first request and after the last one: what lies between (a list of the
        # missing fields) belongs to the request itself.
        asks = [i for i, s in enumerate(non_courtesy) if _any(_REQUEST, s)]
        # An answer delivered before the question makes the question optional; text after the question usually
        # explains what is asked (formats, defaults, options), so only a long passage there counts as delivery
        # (an opening rhetorical question followed by the answer).
        listed_here = _task_list_items(non_courtesy)
        before = _strip(_without(non_courtesy[: asks[0]], listed_here), _REQUEST, _OFFER, _ACKNOWLEDGEMENT)
        after = _strip(_without(non_courtesy[asks[-1] + 1 :], listed_here, asks[-1] + 1), _REQUEST, _OFFER, _ACKNOWLEDGEMENT)
        if (
            has_code
            or _content_words(before) >= CONTENT_WORDS_BEFORE_REQUEST
            or _content_words(after) >= CONTENT_WORDS_AFTER_REQUEST
        ):
            return Decision(DIRECT, STEP_REQUEST_WITH_CONTENT, (request.group(0),))
        return Decision(CLARIFY, STEP_REQUEST, (request.group(0),))

    # 5. Advice to use an outside service instead of doing the task.
    external = next((m for s in sentences if (m := _any(_EXTERNAL_SERVICE, s))), None)
    if external and not has_code and _content_words(_strip(sentences, _EXTERNAL_SERVICE, _OFFER, _COURTESY)) < CONTENT_WORDS_AFTER_DECLINE:
        return Decision(UNSUPPORTED, STEP_EXTERNAL_SERVICE, (external.group(0),))

    # 6. A call that is only described or proposed, in a short response. A reply made only of acknowledgements and an
    #    announcement of the next action delivers nothing either (v2 round 1).
    if (
        not has_code
        and _words(prose) <= ANNOUNCEMENT_MAX_WORDS
        and any(_ANNOUNCEMENT.match(s.strip()) for s in sentences)
        and all(_ANNOUNCEMENT.match(s.strip()) or _any(_ACKNOWLEDGEMENT, s) or _HELP_OFFER.match(s.strip()) for s in sentences)
    ):
        announced = next(s for s in sentences if _ANNOUNCEMENT.match(s.strip()))
        return Decision(ABSTAIN, STEP_NARRATED_CALL, (announced[:80],))
    if (
        not has_code
        and _words(prose) <= PLAN_MAX_WORDS
        and not _WORKED_RESULT.search(prose)
        and any(_PLAN.match(s.strip()) for s in sentences)
        and all(_PLAN.match(s.strip()) or _THANKS.match(s.strip()) or _HELP_OFFER.match(s.strip()) for s in sentences)
    ):
        planned = next(s for s in sentences if _PLAN.match(s.strip()))
        return Decision(ABSTAIN, STEP_NARRATED_CALL, (planned[:80],))
    narrated = _any(_INVOCATION_TALK, prose)
    evidence = narrated.group(0) if narrated else None
    if evidence is None and _INVOCATION_VERB.search(prose):
        low = prose.casefold()
        evidence = next((name for name in sorted(offered) if len(name) >= 4 and name in low), None)
    if evidence and not has_code and _words(prose) <= NARRATION_MAX_WORDS and not _WORKED_RESULT.search(prose):
        return Decision(ABSTAIN, STEP_NARRATED_CALL, (evidence,))

    # 7. Delivered content, including an ordinary conversational reply.
    return Decision(DIRECT, STEP_DELIVERED)
