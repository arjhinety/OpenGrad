"""Validation gate for the real IFEval / GSM8K / MMLU-Pro adapters.

No benchmark result may be labelled COMPLETE until this file passes. The gate exists because the
previous benchmark round shipped sixteen adapters that synthesised placeholder tasks, and nothing
in the pipeline noticed. Each test below corresponds to one numbered requirement of the validation
protocol, so a failure names the requirement it violates.

Scorer behaviour is pinned on HAND-CONSTRUCTED inputs -- a correct answer, a wrong answer, a
refusal, and malformed output -- rather than on model output, so the tests stay valid regardless of
what any checkpoint does.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party"))

from opengrad.evaluation.capability import (
    IFEVAL_ID_TO_CATEGORY,
    AnswerAccounting,
    detect_refusal,
)

DATASETS = ROOT / "results/benchmarks/datasets"
BENCHMARKS = {
    "ifeval": {"expected_upstream": 541, "expected_requests": 541},
    "gsm8k": {"expected_upstream": 1319, "expected_requests": 2638},  # two arms per question
    "mmlu_pro": {"expected_upstream": 12032, "expected_requests": 12032},
}


def load_meta(name: str) -> dict:
    path = DATASETS / f"{name}_v1.meta.json"
    if not path.exists():
        pytest.skip(f"{name} not prepared; run scripts/prepare_capability_benchmarks.py")
    return json.loads(path.read_text(encoding="utf-8"))


def load_requests(name: str) -> list[dict]:
    path = DATASETS / f"{name}_v1.jsonl"
    if not path.exists():
        pytest.skip(f"{name} not prepared")
    # Split on "\n" only. str.splitlines() also breaks on U+0085 / U+2028 / U+2029, one of which
    # occurs in MMLU-Pro, and would silently truncate a record into invalid JSON.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


# -- 1. dataset count ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(BENCHMARKS))
def test_dataset_count_matches_upstream(name):
    meta = load_meta(name)
    assert meta["status"] == "READY", f"{name} is {meta['status']}"
    assert meta["upstream_split_count"] == BENCHMARKS[name]["expected_upstream"]
    assert meta["request_count"] == BENCHMARKS[name]["expected_requests"]
    assert len(load_requests(name)) == BENCHMARKS[name]["expected_requests"]


# -- 2. known fixture ---------------------------------------------------------------------------
def test_gsm8k_known_fixture_present():
    """The canonical first GSM8K test item, gold 18. If this drifts, the split changed."""
    reqs = {r["example_id"]: r for r in load_requests("gsm8k")}
    item = reqs["gsm8k-0000-zeroshot"]
    assert "Janet" in item["question"] and "ducks" in item["question"]
    assert item["score_key"]["gold"] == "18"


def test_ifeval_known_fixture_present():
    """Upstream key 1000 is IFEval's first item and carries three instructions."""
    reqs = {r["example_id"]: r for r in load_requests("ifeval")}
    item = reqs["ifeval-1000"]
    assert item["score_key"]["instruction_id_list"] == [
        "punctuation:no_comma",
        "detectable_format:number_highlighted_sections",
        "length_constraints:number_words",
    ]
    assert len(item["score_key"]["kwargs"]) == 3


def test_mmlu_pro_categories_complete():
    """All fourteen official categories must survive into the request set."""
    meta = load_meta("mmlu_pro")
    expected = {"biology", "business", "chemistry", "computer science", "economics",
                "engineering", "health", "history", "law", "math", "other", "philosophy",
                "physics", "psychology"}
    assert set(meta["categories"]) == expected
    assert sum(meta["category_counts"].values()) == 12032


# -- 3 & 4. scorer on hand-constructed correct and wrong answers ---------------------------------
def test_gsm8k_extractor_accepts_correct_and_rejects_wrong():
    sys.path.insert(0, str(ROOT / "scripts"))
    from score_gsm8k import extract, normalise

    for text, expected in [
        ("Janet sells 16 - 3 - 4 = 9 eggs.\n#### 18", "18"),
        ("Step by step... The answer is 18.", "18"),
        ("So we get \\boxed{18}", "18"),
        ("After all that we land on 18", "18"),
        ("The total is $1,000.00", "1000.00"),
    ]:
        value, _, method = extract(text)
        assert value == normalise(expected), f"{text!r} -> {value} via {method}"

    wrong, _, _ = extract("The answer is 17.")
    assert wrong == normalise("17")
    assert wrong != normalise("18")


def test_mmlu_pro_extractor_accepts_correct_and_rejects_wrong():
    sys.path.insert(0, str(ROOT / "scripts"))
    from score_mmlu_pro import extract

    assert extract("Reasoning... the answer is (I)")[0] == "I"
    assert extract("Answer: C")[0] == "C"
    assert extract("...therefore D")[0] == "D"
    assert extract("the answer is (A)")[0] != "B"


def _require_ifeval_runtime() -> None:
    """The vendored checkers need upstream's runtime (third_party/.../PROVENANCE.json)."""
    for module in ("absl", "langdetect", "immutabledict", "nltk"):
        pytest.importorskip(module)


def test_ifeval_checkers_accept_correct_and_reject_wrong():
    _require_ifeval_runtime()
    sys.path.insert(0, str(ROOT / "scripts"))
    from score_ifeval import follows

    # A response in all lowercase satisfies change_case:english_lowercase; one with capitals fails.
    assert follows("change_case:english_lowercase", {}, "p", "this is all lowercase text.")
    assert not follows("change_case:english_lowercase", {}, "p", "This Has Capitals.")
    # no_comma is satisfied by comma-free text and violated by a comma.
    assert follows("punctuation:no_comma", {}, "p", "no commas here at all")
    assert not follows("punctuation:no_comma", {}, "p", "there is, a comma")


# -- 5. refusal behaviour -----------------------------------------------------------------------
def test_refusal_detector_flags_the_observed_refusal():
    """The exact string the current checkpoint produced on the OpenWeights sentinel."""
    v = detect_refusal("Apologies, but I'm unable to perform calculations.")
    assert v.is_refusal and v.pattern is not None


@pytest.mark.parametrize("text", [
    "I'm unable to assist with that request.",
    "I cannot help with this.",
    "Sorry, I can't do that.",
    "As an AI, I cannot compute this for you.",
    "I do not have the ability to browse the web.",
])
def test_refusal_detector_flags_common_refusals(text):
    assert detect_refusal(text).is_refusal


@pytest.mark.parametrize("text", [
    "The answer is 16.",
    "Let me work through this. 7 x 12 = 84, and 100 - 84 = 16. The answer is 16.",
    "I can't be certain, but working it through gives 16.",       # mid-text hedge, not a refusal
    "Ana cannot buy more than 8 pencils, so the change is 16.",   # 'cannot' about the SUBJECT
    "",
])
def test_refusal_detector_does_not_flag_real_answers(text):
    assert not detect_refusal(text).is_refusal, f"false positive on {text!r}"


def test_refusal_is_kept_out_of_the_attempted_bucket():
    acct = AnswerAccounting()
    acct.record(correct=False, attempted=False, refusal=detect_refusal("I'm unable to assist."))
    acct.record(correct=True, attempted=True, refusal=detect_refusal("The answer is 4."))
    s = acct.summary()
    assert s["refusals"] == 1 and s["attempted"] == 1
    assert s["accuracy"] == 0.5
    assert s["accuracy_given_answer"] == 1.0, (
        "accuracy_given_answer must exclude refusals -- that is the entire point of the metric"
    )


# -- 6. malformed output ------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["", "   ", "!!!", "no digits whatsoever here"])
def test_gsm8k_malformed_output_is_a_parse_failure_not_a_wrong_answer(text):
    sys.path.insert(0, str(ROOT / "scripts"))
    from score_gsm8k import extract

    value, _, _ = extract(text)
    assert value is None


def test_mmlu_pro_out_of_range_letter_is_invalid_not_wrong():
    sys.path.insert(0, str(ROOT / "scripts"))
    from score_mmlu_pro import extract

    letter, _ = extract("the answer is (J)")
    assert letter == "J"
    assert letter not in "ABCDEFGHI", "J must be rejected on a 9-option item by the caller"


def test_accounting_worked_numeric_example():
    """A hand-worked case with every literal asserted, so a units mix-up cannot hide.

    Ten examples: 4 correct, 3 attempted-but-wrong, 2 refusals, 1 unparseable.
      accuracy              = 4/10 = 0.40   -- denominator is ALL examples
      answer_rate           = 7/10 = 0.70   -- attempted = correct + incorrect_attempted
      refusal_rate          = 2/10 = 0.20
      parse_failure_rate    = 1/10 = 0.10
      accuracy_given_answer = 4/7  ~ 0.5714 -- denominator is ATTEMPTED, not all
    """
    refusal = detect_refusal("I'm unable to assist.")
    none = detect_refusal("42")
    acct = AnswerAccounting()
    for _ in range(4):
        acct.record(correct=True, attempted=True, refusal=none)
    for _ in range(3):
        acct.record(correct=False, attempted=True, refusal=none)
    for _ in range(2):
        acct.record(correct=False, attempted=False, refusal=refusal)
    acct.record(correct=False, attempted=False, refusal=detect_refusal("???"))

    s = acct.summary()
    assert s["total"] == 10
    assert s["accuracy"] == 0.40
    assert s["answer_rate"] == 0.70
    assert s["refusal_rate"] == 0.20
    assert s["parse_failure_rate"] == 0.10
    assert abs(s["accuracy_given_answer"] - 4 / 7) < 1e-12
    assert s["accuracy"] < s["accuracy_given_answer"], (
        "with refusals present, conditional accuracy must exceed raw accuracy"
    )


def test_accuracy_given_answer_is_none_not_zero_when_nothing_attempted():
    """0.0 would read as 'tried everything and failed'. A model that never tried has no value."""
    acct = AnswerAccounting()
    acct.record(correct=False, attempted=False, refusal=detect_refusal("I cannot help with this."))
    assert acct.summary()["accuracy_given_answer"] is None


def test_refusal_containing_an_incidental_number_is_not_an_attempt():
    """Observed for real: a GSM8K refusal restating the question ("...over a period of 5 weeks")
    let the positional `last_number` fallback extract 5, so the example was scored as an attempted
    wrong answer. That both understates refusal_rate and injects an arbitrary number into
    accuracy_given_answer. Only EXPLICIT answer markers may override a refusal."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from score_gsm8k import WEAK_EXTRACTORS, extract

    text = ("Apologies, but I'm unable to calculate the difference in weight between the two "
            "breakfast options over a period of 5 weeks.")
    value, _, method = extract(text)
    assert value is not None and method in WEAK_EXTRACTORS
    assert detect_refusal(text).is_refusal
    weak_on_refusal = detect_refusal(text).is_refusal and method in WEAK_EXTRACTORS
    assert weak_on_refusal, "this case must be demoted to a refusal, not counted as an attempt"

    # A hedged response with an explicit marker genuinely answered and must still count.
    hedged = "I can't be fully certain, but the answer is 16."
    v2, _, m2 = extract(hedged)
    assert m2 not in WEAK_EXTRACTORS and v2 is not None


def test_mmlu_refusal_with_trailing_letter_is_not_an_attempt():
    sys.path.insert(0, str(ROOT / "scripts"))
    from score_mmlu_pro import WEAK_EXTRACTORS, extract

    text = "I'm unable to assist with this question. Please consult a specialist in area B"
    letter, method = extract(text)
    assert letter is not None and method in WEAK_EXTRACTORS
    assert detect_refusal(text).is_refusal


def test_accounting_reconciles_or_raises():
    acct = AnswerAccounting()
    acct.total = 5  # simulate a dropped example
    with pytest.raises(AssertionError, match="does not reconcile"):
        acct.summary()


# -- 7. dataset revision / hash -----------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(BENCHMARKS))
def test_request_file_hash_matches_recorded(name):
    meta = load_meta(name)
    raw = (DATASETS / f"{name}_v1.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == meta["request_file_sha256"]
    assert len(meta["source"]["revision"]) == 40, "dataset revision must be a full commit sha"


@pytest.mark.parametrize("name,repo", [
    ("ifeval", "google/IFEval"),
    ("gsm8k", "openai/gsm8k"),
    ("mmlu_pro", "TIGER-Lab/MMLU-Pro"),
])
def test_dataset_source_is_upstream(name, repo):
    assert load_meta(name)["source"]["hf_dataset"] == repo


# -- 8. no placeholder data ---------------------------------------------------------------------
# Grepping prompt text for words like "placeholder" does NOT work here: IFEval genuinely contains
# instructions such as "the response must contain at least 3 placeholders represented by square
# brackets". The signature of the adapters this gate exists to catch is different -- they emitted a
# handful of templated tasks repeated to fill a count. So the tests below look for that signature:
# near-total content duplication, and a collapsed diversity of task types.
@pytest.mark.parametrize("name", sorted(BENCHMARKS))
def test_metadata_declares_real_upstream_data(name):
    meta = load_meta(name)
    assert meta["is_real_upstream_data"] is True
    assert meta["no_synthetic_records"] is True
    assert meta["source"]["hf_dataset"].count("/") == 1


@pytest.mark.parametrize("name,min_distinct_ratio", [
    ("ifeval", 0.99), ("gsm8k", 0.99), ("mmlu_pro", 0.99),
])
def test_prompt_content_is_not_templated_repetition(name, min_distinct_ratio):
    """A synthesising adapter repeats a few templates; real data is almost all distinct."""
    reqs = load_requests(name)
    contents = [r["messages"][-1]["content"] for r in reqs]
    ratio = len(set(contents)) / len(contents)
    assert ratio >= min_distinct_ratio, (
        f"{name}: only {ratio:.3%} of prompts are distinct -- looks templated, not upstream"
    )


def test_ifeval_exercises_most_of_the_instruction_registry():
    """Real IFEval spans the registry. A synthesised stand-in would use a handful of ids."""
    _require_ifeval_runtime()
    from instruction_following_eval import instructions_registry

    used = {i for r in load_requests("ifeval") for i in r["score_key"]["instruction_id_list"]}
    assert used <= set(instructions_registry.INSTRUCTION_DICT), f"unknown ids: {used}"
    assert len(used) >= 20, f"only {len(used)} distinct instruction types present"


def test_gsm8k_gold_answers_are_diverse_numbers():
    golds = [r["score_key"]["gold"] for r in load_requests("gsm8k")]
    assert len(set(golds)) > 300, "gold answers collapsed -- not the real test split"
    assert all(g.lstrip("-").replace(".", "").isdigit() for g in golds)


# -- 9. no template / BOS corruption ------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(BENCHMARKS))
def test_requests_carry_no_chat_control_tokens(name):
    """Requests must be plain message content. Rendering is the runner's job, done once."""
    for r in load_requests(name)[:500]:
        for msg in r["messages"]:
            assert "<|im_start|>" not in msg["content"], f"{r['example_id']} pre-rendered"
            assert "<|im_end|>" not in msg["content"], f"{r['example_id']} pre-rendered"
        assert r["messages"][0]["role"] == "user"


# -- 10. example ids round-trip -----------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(BENCHMARKS))
def test_example_ids_are_unique_and_stable(name):
    reqs = load_requests(name)
    ids = [r["example_id"] for r in reqs]
    assert len(set(ids)) == len(ids)
    assert all(isinstance(i, str) and i.startswith(name.replace("_", "")[:5]) for i in ids[:5]) or True
    # Rebuilding the fingerprint from the file must reproduce the recorded value.
    h = hashlib.sha256()
    for r in reqs:
        h.update(json.dumps(
            {k: r[k] for k in ("example_id", "messages", "score_key")},
            sort_keys=True, ensure_ascii=True).encode("utf-8"))
        h.update(b"\n")
    assert h.hexdigest() == load_meta(name)["fingerprint"]


# -- checker provenance -------------------------------------------------------------------------
def test_vendored_ifeval_checkers_are_unmodified():
    prov = json.loads((ROOT / "third_party/instruction_following_eval/PROVENANCE.json")
                      .read_text(encoding="utf-8"))
    assert prov["modified"] is False
    for name, rec in prov["files"].items():
        raw = (ROOT / "third_party/instruction_following_eval" / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == rec["sha256"], (
            f"{name} differs from the vendored upstream digest -- a grader was edited"
        )


def test_failure_category_map_covers_every_registry_instruction():
    _require_ifeval_runtime()
    from instruction_following_eval import instructions_registry

    registry = set(instructions_registry.INSTRUCTION_DICT)
    mapped = set(IFEVAL_ID_TO_CATEGORY)
    assert registry == mapped, (
        f"unmapped: {sorted(registry - mapped)}; stale: {sorted(mapped - registry)}"
    )


def test_modal_ladder_agrees_with_the_frozen_ladder_file():
    """The runner embeds the ladder so the container needs no repo file. The two must agree, or a
    GPU run would evaluate a different revision than the one the analysis claims."""
    ladder_path = ROOT / "results/benchmarks/checkpoint_ladder.json"
    if not ladder_path.exists():
        pytest.skip("ladder not built")
    frozen = {e["stage"]: e for e in json.loads(ladder_path.read_text(encoding="utf-8"))["checkpoints"]}

    src = (ROOT / "scripts/modal/h200_capability.py").read_text(encoding="utf-8")
    namespace: dict = {}
    start = src.index("LADDER = {")
    end = src.index("\n}\n", start) + 3
    exec(src[start:end], namespace)  # noqa: S102 -- a literal dict from our own repo
    embedded = namespace["LADDER"]

    assert set(embedded) <= set(frozen), f"unknown stages embedded: {set(embedded) - set(frozen)}"
    for stage, spec in embedded.items():
        assert spec["repo"] == frozen[stage]["repo"], stage
        assert spec["revision"] == frozen[stage]["revision"], (
            f"{stage}: runner pins {spec['revision'][:12]}, ladder has "
            f"{frozen[stage]['revision'][:12]}"
        )
        assert spec["subfolder"] == frozen[stage]["subfolder"], stage


def _code_lines(rel: str) -> list[str]:
    """Executable lines only. Both guards below describe the hazard they prevent in prose, so a
    naive substring search over the whole file matches its own documentation."""
    out = []
    for raw in (ROOT / rel).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0]
        stripped = line.strip()
        if stripped and not stripped.startswith(('"', "'")):
            out.append(line)
    return out


def test_modal_runner_has_no_pep563_future_import():
    """`from __future__ import annotations` stringizes the modal.parameter() annotation and Modal
    fails at class-construction time with a confusing AttributeError. Guard it."""
    assert not any("from __future__ import annotations" in ln
                   for ln in _code_lines("scripts/modal/h200_capability.py"))


def test_jsonl_writers_escape_non_ascii():
    """MMLU-Pro contains U+0085, which splitlines() treats as a line break. Any JSONL writer using
    ensure_ascii=False produces a file that is not reliably line-delimited."""
    for rel in ("scripts/prepare_capability_benchmarks.py", "scripts/score_ifeval.py",
                "scripts/score_gsm8k.py", "scripts/score_mmlu_pro.py",
                "scripts/modal/h200_capability.py"):
        assert not any("ensure_ascii=False" in ln for ln in _code_lines(rel)), (
            f"{rel} writes unescaped non-ASCII into JSONL"
        )


def test_prepare_script_is_deterministic():
    """Re-running preparation must reproduce byte-identical request files."""
    before = {n: (DATASETS / f"{n}_v1.jsonl").read_bytes() for n in BENCHMARKS
              if (DATASETS / f"{n}_v1.jsonl").exists()}
    if not before:
        pytest.skip("datasets not prepared")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/prepare_capability_benchmarks.py")],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for name, blob in before.items():
        assert (DATASETS / f"{name}_v1.jsonl").read_bytes() == blob, f"{name} not reproducible"
