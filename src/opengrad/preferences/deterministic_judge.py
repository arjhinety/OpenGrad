"""Deterministic preference judge evaluating tool actions and hard negative selection."""

from __future__ import annotations

from opengrad.formatting.parser import parse_qwen_native_output
from opengrad.preferences.schema import PreferenceCandidate, PreferencePair


class DeterministicJudge:
    """Evaluates candidates using deterministic schema, syntax, and behavioral decision criteria."""

    def score_candidate(
        self,
        candidate_text: str,
        expected_decision: str = "CALL",
        expected_tool: str | None = None,
        available_tools: list[str] | None = None,
    ) -> tuple[float, list[str]]:
        """Score a single candidate response. Returns (score, reason_codes)."""
        reasons: list[str] = []
        score = 0.0

        parsed = parse_qwen_native_output(candidate_text)
        if parsed.status != "RAW_VALID":
            reasons.append("FORMAT_ERROR")
            return -1.0, reasons

        reasons.append("SYNTAX_COMPLIANT")
        score += 1.0

        avail = set(available_tools or [])

        if expected_decision == "CALL":
            if parsed.decision != "CALL":
                reasons.append("MISSED_TOOL")
                score -= 2.0
            else:
                reasons.append("CORRECT_CALL_DECISION")
                score += 2.0
                called_tools = [c.name for c in parsed.calls]
                for ct in called_tools:
                    if avail and ct not in avail:
                        reasons.append("HALLUCINATED_TOOL")
                        score -= 3.0
                    elif expected_tool and ct == expected_tool:
                        reasons.append("CORRECT_TOOL_SELECTION")
                        score += 3.0
                    else:
                        reasons.append("WRONG_TOOL")
                        score -= 1.0

                # Check argument objects
                for c in parsed.calls:
                    if c.arguments and isinstance(c.arguments, dict):
                        score += 1.0
                        reasons.append("GROUNDED_ARGUMENTS")
                    elif not isinstance(c.arguments, dict):
                        score -= 2.0
                        reasons.append("MALFORMED_ARGUMENTS")

        elif expected_decision in {"ANSWER", "DIRECT"}:
            if parsed.decision == "CALL":
                reasons.append("UNNECESSARY_TOOL")
                score -= 3.0
            else:
                reasons.append("CORRECT_NO_CALL_DECISION")
                score += 3.0

        elif expected_decision == "CLARIFY":
            if parsed.decision == "CLARIFY" or "clarif" in candidate_text.lower():
                reasons.append("CORRECT_CLARIFICATION")
                score += 3.0
            else:
                reasons.append("FAILED_CLARIFICATION")
                score -= 2.0

        return score, reasons

    def judge_pair(
        self,
        cand_a: PreferenceCandidate,
        cand_b: PreferenceCandidate,
        prompt: str,
        prompt_id: str,
        expected_decision: str = "CALL",
        expected_tool: str | None = None,
        available_tools: list[str] | None = None,
    ) -> PreferencePair | None:
        """Form a deterministic preference pair if clear separation exists."""
        score_a, reasons_a = self.score_candidate(
            cand_a.response_text, expected_decision, expected_tool, available_tools
        )
        score_b, reasons_b = self.score_candidate(
            cand_b.response_text, expected_decision, expected_tool, available_tools
        )

        margin = score_a - score_b
        # Require clear score margin (at least 1.5 delta)
        if abs(margin) < 1.5:
            return None  # Ambiguous or tie: defer to OpenAI judge if configured

        if margin > 0:
            chosen = cand_a.response_text
            rejected = cand_b.response_text
            codes = reasons_a
        else:
            chosen = cand_b.response_text
            rejected = cand_a.response_text
            codes = reasons_b

        return PreferencePair(
            prompt_id=prompt_id,
            canonical_id=prompt_id,
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            preference_source="deterministic",
            confidence=0.95,
            reason_codes=codes,
        )
