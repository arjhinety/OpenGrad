"""Shared accounting for the general-capability benchmarks: refusal, answer rate, failure class.

The whole point of this module is the distinction between *the model got it wrong* and *the model
declined to try*. A benchmark that reports only accuracy cannot tell those apart, and the question
under investigation is exactly which one is happening.

`accuracy_given_answer` is the metric that separates them:

    accuracy            = correct / all_examples
    answer_rate         = attempted / all_examples
    accuracy_given_answer = correct / attempted

A checkpoint whose `accuracy` falls while `accuracy_given_answer` holds has not lost the ability;
it has stopped using it. A checkpoint whose `accuracy_given_answer` falls has.

REFUSAL DETECTION IS HEURISTIC and is labelled as such everywhere it is reported. It never feeds
an official benchmark metric -- IFEval strict/loose accuracy, GSM8K accuracy and MMLU-Pro accuracy
are computed exactly as their benchmarks specify, with refusals counted as ordinary failures. The
refusal flag is an additional diagnostic column, and every flagged example keeps its raw text and
the pattern that matched so a human can audit the call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Refusal patterns are deliberately high-precision rather than high-recall: a false positive would
# manufacture the very finding this study is testing for. Each is anchored near the start of the
# response, because a genuine refusal leads with the refusal -- a hedge in the middle of a real
# answer ("I can't be certain, but the answer is 16") is NOT a refusal and must not be flagged.
REFUSAL_WINDOW_CHARS = 400

REFUSAL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("apology_inability",
     (r"\b(?:apologies|i\s+apologi[sz]e|i'?m\s+sorry|i\s+am\s+sorry|sorry)\b[^.]{0,90}?"
      r"\b(?:unable|cannot|can'?t|not\s+able|don'?t\s+have\s+the\s+abilit)")),
    ("leading_unable",
     r"^\s*(?:i'?m|i\s+am)\s+(?:unable|not\s+able)\s+to\b"),
    ("leading_cannot",
     r"^\s*i\s+(?:cannot|can'?t)\s+(?:help|assist|do|perform|answer|provide|complete|comply|solve|calculate)\b"),
    ("as_an_ai",
     r"\bas\s+an\s+ai\b[^.]{0,80}?\b(?:cannot|can'?t|unable|do\s+not\s+have)"),
    ("no_capability",
     r"\bi\s+(?:do\s+not|don'?t)\s+have\s+the\s+(?:abilit|capabilit|capacity)"),
    ("decline_to",
     r"^\s*i\s+(?:must\s+)?(?:decline|refuse)\b"),
    ("unable_to_perform",
     r"\b(?:unable|not\s+able)\s+to\s+(?:perform|carry\s+out|execute|do)\s+(?:calculations?|computations?|math|arithmetic)\b"),
)

_COMPILED = tuple((name, re.compile(pat, re.IGNORECASE | re.MULTILINE)) for name, pat in REFUSAL_PATTERNS)

FAILURE_CATEGORIES = (
    "REFUSAL",
    "WRONG_CONTENT",
    "FORMAT_VIOLATION",
    "LENGTH_VIOLATION",
    "MISSING_REQUIRED_ELEMENT",
    "EXTRA_FORBIDDEN_CONTENT",
    "PARSE_FAILURE",
    "OTHER",
)

# IFEval instruction ids map to a failure category deterministically: the instruction that failed
# *is* the description of what went wrong. This mapping is exhaustive over the 25 registry entries
# and is asserted against the live registry in the tests, so an upstream addition cannot silently
# fall through to OTHER.
IFEVAL_ID_TO_CATEGORY: dict[str, str] = {
    "change_case:capital_word_frequency": "FORMAT_VIOLATION",
    "change_case:english_capital": "FORMAT_VIOLATION",
    "change_case:english_lowercase": "FORMAT_VIOLATION",
    "combination:repeat_prompt": "MISSING_REQUIRED_ELEMENT",
    "combination:two_responses": "FORMAT_VIOLATION",
    "detectable_content:number_placeholders": "MISSING_REQUIRED_ELEMENT",
    "detectable_content:postscript": "MISSING_REQUIRED_ELEMENT",
    "detectable_format:constrained_response": "FORMAT_VIOLATION",
    "detectable_format:json_format": "FORMAT_VIOLATION",
    "detectable_format:multiple_sections": "FORMAT_VIOLATION",
    "detectable_format:number_bullet_lists": "FORMAT_VIOLATION",
    "detectable_format:number_highlighted_sections": "FORMAT_VIOLATION",
    "detectable_format:title": "MISSING_REQUIRED_ELEMENT",
    "keywords:existence": "MISSING_REQUIRED_ELEMENT",
    "keywords:forbidden_words": "EXTRA_FORBIDDEN_CONTENT",
    "keywords:frequency": "MISSING_REQUIRED_ELEMENT",
    "keywords:letter_frequency": "MISSING_REQUIRED_ELEMENT",
    "language:response_language": "WRONG_CONTENT",
    "length_constraints:nth_paragraph_first_word": "MISSING_REQUIRED_ELEMENT",
    "length_constraints:number_paragraphs": "LENGTH_VIOLATION",
    "length_constraints:number_sentences": "LENGTH_VIOLATION",
    "length_constraints:number_words": "LENGTH_VIOLATION",
    "punctuation:no_comma": "EXTRA_FORBIDDEN_CONTENT",
    "startend:end_checker": "MISSING_REQUIRED_ELEMENT",
    "startend:quotation": "FORMAT_VIOLATION",
}


@dataclass(frozen=True)
class RefusalVerdict:
    """Why a response was or was not called a refusal. Always carries its own evidence."""

    is_refusal: bool
    pattern: str | None = None
    matched_text: str | None = None
    method: str = "HEURISTIC_REGEX_v1"


def detect_refusal(text: str | None) -> RefusalVerdict:
    """Flag a response as a refusal. High precision by construction; see module docstring."""
    if not text or not text.strip():
        return RefusalVerdict(False)
    window = text[:REFUSAL_WINDOW_CHARS]
    for name, rx in _COMPILED:
        m = rx.search(window)
        if m:
            return RefusalVerdict(True, pattern=name, matched_text=m.group(0)[:160])
    return RefusalVerdict(False)


@dataclass
class AnswerAccounting:
    """Per-benchmark tally that keeps 'wrong' and 'did not try' separate.

    Every example lands in exactly one of the four buckets, and `total` is asserted to equal their
    sum -- a silently dropped example would otherwise inflate every rate in the report.
    """

    total: int = 0
    correct: int = 0
    incorrect_attempted: int = 0
    refusals: int = 0
    parse_failures: int = 0
    refusal_patterns: dict[str, int] = field(default_factory=dict)

    def record(self, *, correct: bool, attempted: bool, refusal: RefusalVerdict) -> str:
        """Classify one example and return its bucket name."""
        self.total += 1
        if refusal.is_refusal and not attempted:
            self.refusals += 1
            self.refusal_patterns[refusal.pattern] = self.refusal_patterns.get(refusal.pattern, 0) + 1
            return "REFUSAL"
        if not attempted:
            self.parse_failures += 1
            return "PARSE_FAILURE"
        if correct:
            self.correct += 1
            return "CORRECT"
        self.incorrect_attempted += 1
        return "INCORRECT"

    @property
    def attempted(self) -> int:
        return self.correct + self.incorrect_attempted

    def summary(self) -> dict:
        if self.total != self.correct + self.incorrect_attempted + self.refusals + self.parse_failures:
            raise AssertionError("answer accounting does not reconcile; an example was dropped")
        t = self.total or 1
        return {
            "total": self.total,
            "correct": self.correct,
            "incorrect_attempted": self.incorrect_attempted,
            "refusals": self.refusals,
            "parse_failures": self.parse_failures,
            "attempted": self.attempted,
            "accuracy": self.correct / t,
            "answer_rate": self.attempted / t,
            "refusal_rate": self.refusals / t,
            "parse_failure_rate": self.parse_failures / t,
            # None, not 0.0: a model that never attempted anything has no conditional accuracy, and
            # reporting 0.0 would read as "it tried and got everything wrong".
            "accuracy_given_answer": (self.correct / self.attempted) if self.attempted else None,
            "refusal_detection_method": "HEURISTIC_REGEX_v1",
            "refusal_patterns": dict(sorted(self.refusal_patterns.items())),
        }


def classify_ifeval_failure(
    failed_instruction_ids: list[str],
    response: str | None,
    refusal: RefusalVerdict,
) -> tuple[str, str]:
    """Return (category, basis) for one failed IFEval example.

    A refusal is reported as REFUSAL even though an instruction also failed, because "declined the
    task" is the more informative description of what happened. The basis string records which rule
    fired so DETERMINISTIC and HEURISTIC calls are distinguishable downstream.
    """
    if response is None or not response.strip():
        return "PARSE_FAILURE", "DETERMINISTIC:empty_response"
    if refusal.is_refusal:
        return "REFUSAL", f"HEURISTIC:{refusal.pattern}"
    if not failed_instruction_ids:
        return "OTHER", "DETERMINISTIC:no_failed_instruction"
    categories = [IFEVAL_ID_TO_CATEGORY.get(i, "OTHER") for i in failed_instruction_ids]
    # With several failures, report the first in the example's own instruction order rather than a
    # majority vote: order is the benchmark's, a vote would be ours.
    return categories[0], f"DETERMINISTIC:{failed_instruction_ids[0]}"
