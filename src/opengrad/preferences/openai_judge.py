"""OpenAI DPO preference judge with budget safety, caching, and structured outputs."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opengrad.preferences.schema import PreferenceCandidate, PreferencePair


@dataclass
class JudgeBudget:
    max_requests: int = 500
    max_cost_usd: float = 10.0
    requests_made: int = 0
    estimated_cost_usd: float = 0.0

    def is_exceeded(self) -> bool:
        return (
            self.requests_made >= self.max_requests or self.estimated_cost_usd >= self.max_cost_usd
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_requests": self.max_requests,
            "max_cost_usd": self.max_cost_usd,
            "requests_made": self.requests_made,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
            "exceeded": self.is_exceeded(),
        }


class OpenAIJudge:
    """Evaluates ambiguous semantic preference pairs using OpenAI with budget protection and caching."""

    def __init__(
        self,
        model: str | None = None,
        budget: JudgeBudget | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.model = model or os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4o-mini")
        self.budget = budget or JudgeBudget()
        self.cache_dir = (cache_dir or Path.cwd() / ".cache" / "openai_judge") / self.model.replace(
            "/", "_"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_api_key(self) -> str | None:
        # Secret safety: read only from environment variable (Section 11)
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        return key if key else None

    def _cache_key(self, prompt: str, cand_a: str, cand_b: str) -> str:
        combined = f"{self.model}###{prompt}###{cand_a}###{cand_b}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def judge_pair(
        self,
        cand_a: PreferenceCandidate,
        cand_b: PreferenceCandidate,
        prompt: str,
        prompt_id: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> PreferencePair | None:
        if self.budget.is_exceeded():
            # Hard stop on budget limit (Section 13)
            return None

        key = self._cache_key(prompt, cand_a.response_text, cand_b.response_text)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
                return self._parse_verdict(cached_data, cand_a, cand_b, prompt, prompt_id)
            except (json.JSONDecodeError, KeyError):
                pass

        api_key = self._get_api_key()
        if not api_key:
            # Offline mock adjudication for CPU/CI testing
            return self._mock_adjudicate(cand_a, cand_b, prompt, prompt_id)

        # Call OpenAI Chat Completions API with structured output
        system_rubric = (
            "You are an expert AI evaluator for tool-calling policy in small language models.\n"
            "Compare Candidate A and Candidate B. Determine which response is strictly superior in terms of:\n"
            "1. Correct decision (calling vs. answering vs. clarifying)\n"
            "2. Schema validity and argument correctness\n"
            "3. Absence of hallucinated parameters or syntax leakage\n"
            "Respond strictly in JSON with keys: preferred_candidate ('A' or 'B'), decision ('PREFERENCE', 'TIE', or 'SKIP'), "
            "confidence (0.0 to 1.0), reason_codes (list of short strings)."
        )

        user_content = (
            f"Prompt: {prompt}\n\n"
            f"Candidate A:\n{cand_a.response_text}\n\n"
            f"Candidate B:\n{cand_b.response_text}\n"
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_rubric},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result_data = json.loads(resp.read().decode("utf-8"))
                self.budget.requests_made += 1
                self.budget.estimated_cost_usd += 0.002  # Approximate cost accounting

                choice = result_data["choices"][0]["message"]["content"]
                parsed_verdict = json.loads(choice)
                cache_file.write_text(json.dumps(parsed_verdict, indent=2) + "\n", encoding="utf-8")
                return self._parse_verdict(parsed_verdict, cand_a, cand_b, prompt, prompt_id)

        except (urllib.error.URLError, json.JSONDecodeError, KeyError):
            return None

    def _parse_verdict(
        self,
        verdict: dict[str, Any],
        cand_a: PreferenceCandidate,
        cand_b: PreferenceCandidate,
        prompt: str,
        prompt_id: str,
    ) -> PreferencePair | None:
        dec = verdict.get("decision", "SKIP")
        if dec != "PREFERENCE":
            return None

        pref = verdict.get("preferred_candidate", "A").strip().upper()
        conf = float(verdict.get("confidence", 0.9))
        reasons = list(verdict.get("reason_codes") or ["OPENAI_ADJUDICATED"])

        if pref == "A":
            chosen = cand_a.response_text
            rejected = cand_b.response_text
        else:
            chosen = cand_b.response_text
            rejected = cand_a.response_text

        return PreferencePair(
            prompt_id=prompt_id,
            canonical_id=prompt_id,
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            preference_source="openai_adjudicated",
            confidence=conf,
            reason_codes=reasons,
        )

    def _mock_adjudicate(
        self,
        cand_a: PreferenceCandidate,
        cand_b: PreferenceCandidate,
        prompt: str,
        prompt_id: str,
    ) -> PreferencePair | None:
        """Deterministic mock when no API key is supplied."""
        # Candidate with tool call wins on tool prompt, otherwise shorter clean answer wins
        if "<tool_call>" in cand_a.response_text and "<tool_call>" not in cand_b.response_text:
            return PreferencePair(
                prompt_id=prompt_id,
                canonical_id=prompt_id,
                prompt=prompt,
                chosen=cand_a.response_text,
                rejected=cand_b.response_text,
                preference_source="openai_adjudicated_mock",
                confidence=0.88,
                reason_codes=["MOCK_JUDGE_SELECTION"],
            )
        return None
