"""Lock the characterised tokenizer divergence in place.

These tests do **not** require the two tokenizers to agree. They already disagree on six
confirmatory prompts, that failure is recorded as `ENGINE_TOKENIZER_PARITY_FAILED`, and asserting
agreement here would quietly delete the finding.

What is locked is the *characterisation*: which prompts diverge, by how many tokens, where the
streams part and rejoin, that no special token is inserted or dropped, and which three of the six
actually change the policy decision. If any of that drifts — a tokenizer update, a llama.cpp bump,
a template change — these fail and a human has to re-characterise deliberately.

The HF side is genuinely recomputed from the pinned tokenizer, so this is a live check on one arm
rather than a comparison of stored numbers to themselves. The llama.cpp arm cannot run locally, so
its ids are compared against the recorded measurement and re-measured only when the Modal probe is
re-run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/tokenizer_divergence_v1.json"
TOKENIZER_DIR = ROOT / ".workspace/quantization/source/m1-v2/dpo-checkpoint-30"

# From the frozen characterisation. Hardcoded on purpose: if the fixture file itself is regenerated
# with different content, these constants are what notices.
EXPECTED_FIXTURES = 6
EXPECTED_BEHAVIOUR_CHANGING = 3
BEHAVIOUR_CHANGING_IDS = {
    "2bf3162c-8dbd-4305-8dbc-b765998783fa",
    "8643445b-e72d-4d15-85bf-df55a1f1811f",
    "c9cf98bc-59d6-4cf2-b62c-0473ae094cee",
}


@pytest.fixture(scope="module")
def payload():
    if not FIXTURES.is_file():
        pytest.skip("tokenizer divergence fixtures not built")
    return json.loads(FIXTURES.read_text(encoding="utf-8"))


def test_fixture_set_is_the_characterised_six(payload):
    assert payload["summary"]["examples"] == EXPECTED_FIXTURES
    assert payload["summary"]["parity_verdict"] == "ENGINE_TOKENIZER_PARITY_FAILED"
    assert payload["summary"]["materiality_verdict"] == "TOKENIZER_DIVERGENCE_BEHAVIORALLY_MATERIAL"
    assert len(payload["fixtures"]) == EXPECTED_FIXTURES


def test_exactly_three_examples_change_behaviour(payload):
    """The materiality split is the finding. Drift in either direction must be caught."""
    changing = {
        f["example_id"] for f in payload["fixtures"] if f["divergence"]["changes_behaviour"]
    }
    assert changing == BEHAVIOUR_CHANGING_IDS
    assert payload["summary"]["behaviour_changing"] == EXPECTED_BEHAVIOUR_CHANGING
    assert payload["summary"]["behaviour_preserving"] == EXPECTED_FIXTURES - EXPECTED_BEHAVIOUR_CHANGING


def test_prompt_text_matches_its_recorded_hash(payload):
    _items = list(payload["fixtures"])
    assert _items, "nothing to check: an empty collection would pass this test vacuously"
    for f in _items:
        digest = hashlib.sha256(f["prompt_text"].encode("utf-8")).hexdigest()
        assert digest == f["prompt_sha256"], f"{f['example_id']} prompt text drifted"


def test_hf_token_ids_recompute_exactly(payload):
    """Live recomputation of the HF arm against the pinned tokenizer."""
    if not (TOKENIZER_DIR / "tokenizer.json").is_file():
        pytest.skip("pinned tokenizer not materialised locally")
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    assert tok.add_bos_token is False, "pinned tokenizer must not prepend BOS"

    for f in payload["fixtures"]:
        ids = tok(f["prompt_text"], add_special_tokens=False)["input_ids"]
        assert ids == f["hf"]["token_ids"], (
            f"{f['example_id']}: HF tokenization changed "
            f"({len(ids)} tokens now vs {f['hf']['token_count']} recorded)"
        )


def test_llamacpp_side_is_shorter_on_every_fixture(payload):
    """The direction of the divergence is part of the characterisation, not incidental.

    llama.cpp folds combining marks into the letter run, producing coarser chunks, so it must emit
    FEWER tokens on every affected prompt. A fixture where it emitted more would mean a different
    mechanism is at work and the recorded cause no longer explains it.
    """
    _items = list(payload["fixtures"])
    assert _items, "nothing to check: an empty collection would pass this test vacuously"
    for f in _items:
        assert f["divergence"]["token_count_delta"] < 0, f["example_id"]
        assert f["llamacpp"]["token_count"] < f["hf"]["token_count"]
        assert (
            f["llamacpp"]["token_count"] - f["hf"]["token_count"]
            == f["divergence"]["token_count_delta"]
        )


def test_no_special_token_is_inserted_dropped_or_reinterpreted(payload):
    """The earlier <|im_start|> BOS collision is why this is asserted rather than assumed."""
    _items = list(payload["fixtures"])
    assert _items, "nothing to check: an empty collection would pass this test vacuously"
    for f in _items:
        hf, lc = f["hf"]["special_census"], f["llamacpp"]["special_census"]
        assert hf["im_start"] == lc["im_start"], f["example_id"]
        assert hf["im_end"] == lc["im_end"], f["example_id"]
        assert hf["eos"] == lc["eos"], f["example_id"]
        assert hf["first"] == lc["first"] == 248045, f["example_id"]
        assert f["divergence"]["special_tokens_identical"] is True


def test_divergence_is_localized_and_reconverges(payload):
    """Both streams rejoin after the Thai span; an unbounded divergence would be a different bug."""
    _items = list(payload["fixtures"])
    assert _items, "nothing to check: an empty collection would pass this test vacuously"
    for f in _items:
        d = f["divergence"]
        assert d["reconverged_suffix_tokens"] > 0, f["example_id"]
        assert d["first_divergence_index"] is not None
        assert d["divergent_span_hf"] > d["divergent_span_llamacpp"] >= 0


def test_direct_token_path_was_validated_before_interpretation(payload):
    """The isolation experiment is void if llama-server did not evaluate the ids literally."""
    v = payload["provenance"]["direct_token_path_validation"]
    assert v["literal"] is True
    assert v["tokens_evaluated"] == v["submitted_token_count"]
    assert v["detokenize_roundtrip_matches_prompt"] is True


def test_fixtures_do_not_assert_tokenizer_agreement(payload):
    """Guard against a future edit turning this suite into an agreement requirement.

    If someone 'fixes' the divergence by making the fixtures agree, the recorded failure would
    silently disappear. Every fixture must still represent a real disagreement.
    """
    _items = list(payload["fixtures"])
    assert _items, "nothing to check: an empty collection would pass this test vacuously"
    for f in _items:
        assert f["hf"]["token_ids"] != f["llamacpp"]["token_ids"], (
            f"{f['example_id']} no longer diverges; the parity failure must not be erased by "
            "editing fixtures — re-characterise deliberately instead"
        )
        assert f["divergence"]["known"] is True
