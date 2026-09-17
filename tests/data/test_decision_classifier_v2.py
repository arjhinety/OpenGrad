"""prose-decision-classifier-v2 (37) on the rubric's own boundary cases, the same ones v1 is held to, (22 §1-§2) and on its contract.

The example responses are written from the rubric's definitions and examples, never taken from P-DET-v1 or
P-DET-COVERAGE-v1.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from opengrad.data import decision_classifier_v2 as dc
from opengrad.data.classifier_input import ClassifierFeatures

WEATHER = {"name": "get_weather", "description": "Current weather for a city.", "parameters": {"type": "object"}}
RATES = {"name": "get_exchange_rate", "description": "Exchange rate between currencies.", "parameters": {"type": "object"}}


def decide(response: str, tools: list[dict] | None = None, user: str = "Help me, please.") -> dc.Decision:
    return dc.classify(ClassifierFeatures(user, response, tuple(tools or ()), False))


@pytest.mark.parametrize(
    ("response", "tools", "label"),
    [
        # 22 §1.1 CALL: a machine-readable payload naming an offered tool.
        ('{"name": "get_weather", "arguments": {"city": "Paris"}}', [WEATHER], dc.CALL),
        ('<functioncall> {"name": "get_weather", "arguments": {"city": "Paris"}}', [WEATHER], dc.CALL),
        ('[get_weather(city="Paris")]', [WEATHER], dc.CALL),
        # ... naming a tool that is not offered, or with no tools at all: unverifiable.
        ('{"name": "book_flight", "arguments": {"to": "Paris"}}', [WEATHER], dc.ABSTAIN),
        ("[get_weather(city=\"Paris\")]", [], dc.ABSTAIN),
        # Prose about a call is not a call.
        ("I would use get_weather here.", [WEATHER], dc.ABSTAIN),
        ("The right tool for this is get_weather.", [WEATHER], dc.ABSTAIN),
        # Ceding control to the user is a request (22 §1.1 boundary).
        ("I can look that up with the weather tool. Shall I go ahead and call it?", [WEATHER], dc.CLARIFY),
        # 22 §1.2 DIRECT, with a courtesy question after a complete answer.
        ("Paris is the capital of France. It has been the capital since 987. Does that help?", [], dc.DIRECT),
        # A JSON answer that is data, not a call.
        ('{"name": "Alice", "age": 30}', [], dc.DIRECT),
        # 22 §1.2 non-substantive.
        ("...", [], dc.ABSTAIN),
        ("", [], dc.ABSTAIN),
        # 22 §1.3 CLARIFY, with and without a question mark.
        ("Which city would you like the weather for?", [WEATHER], dc.CLARIFY),
        ("Please tell me the city you are asking about.", [WEATHER], dc.CLARIFY),
        # Refusal-shaped wording that only asks for missing information (22 §1.4 exclusion).
        ("I'm sorry, I can't check that without knowing the city. Could you tell me which one?", [WEATHER], dc.CLARIFY),
        # A capability question is UNSUPPORTED, not CLARIFY (22 §1.3 boundary).
        ("Do you have a tool that can read VINs?", [WEATHER], dc.UNSUPPORTED),
        # 22 §1.4 UNSUPPORTED.
        ("I can't retrieve today's rate.", [], dc.UNSUPPORTED),
        ("I can't perform calculations.", [], dc.UNSUPPORTED),
        ("I'm sorry, but I don't have the capability to book flights.", [WEATHER], dc.UNSUPPORTED),
        ("You may want to check a reliable currency website for that.", [], dc.UNSUPPORTED),
        # ... but an epistemic hedge inside a delivered answer is DIRECT (22 §2 rows 3 and 7).
        ("I can't know your exact location, but at latitude 40 north the sun sets around 7:45 pm in June.", [], dc.DIRECT),
        ("I'm not certain of today's rate. Last week one euro was worth about 1.08 dollars.", [RATES], dc.DIRECT),
        # An offer after a decline delivers nothing (22 §1.4).
        ("I'm sorry, I can't book flights. However, I can tell you about Lisbon if you'd like.", [WEATHER], dc.UNSUPPORTED),
        ("The only tool gives exchange rates, so it does not apply to finding hotels.", [RATES], dc.UNSUPPORTED),
        ("The listed tool is unrelated to train timetables.", [RATES], dc.UNSUPPORTED),
        # Saying what the user left out asks for it (22 §2 row 6).
        ("You have not provided the city. Please specify it.", [WEATHER], dc.CLARIFY),
        # A mismatch note followed by the answer itself is DIRECT (22 §1.4 exclusion).
        ("None of the tools fit this request. Here's the answer anyway: 12 times 12 is 144 and 9 plus 6 is 15.", [RATES], dc.DIRECT),
    ],
)
def test_rubric_boundary_cases(response: str, tools: list[dict], label: str) -> None:
    decision = decide(response, tools)
    assert decision.label == label, decision


def test_a_long_answer_with_an_opening_rhetorical_question_is_direct() -> None:
    response = (
        "Why does the loop never stop? The condition compares the counter with a value that the body never "
        "changes, so it stays true forever. Move the increment inside the loop body, after the print statement, "
        "and the counter will reach the limit after ten iterations. You can also rewrite it as a for loop over "
        "range(10), which makes the bound explicit and removes the manual increment entirely, and it is the more "
        "idiomatic form in Python for a fixed number of repetitions."
    )
    assert decide(response).label == dc.DIRECT


def test_a_list_of_missing_fields_is_part_of_the_request() -> None:
    response = (
        "The scheduling function needs more information. Your request is missing:\n"
        "- the attendee email addresses for every participant in the meeting\n"
        "- the meeting title that should appear in the calendar invitation\n"
        "- the time zone in which the start and end times are expressed\n"
        "- the location or video link where the meeting will take place\n"
        "- whether reminders should be sent and how long before the start\n"
        "Please provide these details."
    )
    assert decide(response).label == dc.CLARIFY


def test_every_decision_names_its_step_and_version() -> None:
    decision = decide("I can't retrieve today's rate.")
    assert decision.step == dc.STEP_DECLINE and decision.evidence
    assert decision.as_dict()["classifier_version"] == "prose-decision-classifier-v2"


def test_a_structured_call_is_never_classified_here() -> None:
    with pytest.raises(ValueError, match="layer A"):
        dc.classify(ClassifierFeatures("u", "a", (), True))


def test_classification_is_deterministic() -> None:
    features = ClassifierFeatures("u", "I'm sorry, I can't do that.", (WEATHER,), False)
    assert dc.classify(features) == dc.classify(features)


def test_the_classifier_module_reads_no_file_and_names_no_validation_population() -> None:
    """33 §5, 37 §4: the classifier never opens a validation population. Checked statically: no file access, no
    path to a population or reference, and no import beyond the input contract and versions."""
    source = Path(dc.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }
    assert imported <= {"__future__", "json", "re", "collections.abc", "dataclasses", "typing",
                        "opengrad.data", "opengrad.data.classifier_input"}
    called = {node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
              for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert not called & {"open", "read_text", "read_bytes", "load", "loads_file", "Path", "glob", "iterdir"}
    for forbidden in ("reports/", "pdet", "pdet-coverage", "reference", ".jsonl", "gold"):
        strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        assert not any(forbidden in s.casefold() for s in strings if not s.startswith(("The ", "It ", "How ", "Agree"))), forbidden


def test_the_classifier_opens_no_file_while_classifying(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    import io

    def refuse(*_args, **_kwargs):
        raise AssertionError("the classifier opened a file")

    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(io, "open", refuse)
    monkeypatch.setattr(Path, "open", refuse)
    decide("I can't retrieve today's rate.")
    decide('{"name": "get_weather", "arguments": {}}', [WEATHER])
    decide("Which city would you like the weather for?", [WEATHER])


# ── v2 round 1 (37 §4): first replies of continuing conversations and tool-limit wording ──────────────


@pytest.mark.parametrize(
    "response",
    [
        "Sure, let me check the weather for you.",
        "Of course, I can help with that. Let me look up the exchange rate.",
        "Certainly! I'll get that information now.",
    ],
)
def test_a_reply_that_only_announces_the_next_action_abstains(response: str) -> None:
    decision = decide(response, [WEATHER, RATES])
    assert decision.label == dc.ABSTAIN and decision.step == dc.STEP_NARRATED_CALL


def test_announcing_an_explanation_that_follows_is_not_an_empty_announcement() -> None:
    response = "Sure, let me explain. Rain forms when water vapour condenses into droplets heavy enough to fall."
    assert decide(response).label == dc.DIRECT


@pytest.mark.parametrize(
    "response",
    ["May I have the city you want the forecast for?", "How long would you like the password to be?"],
)
def test_a_polite_or_measure_question_that_withholds_the_answer_is_clarify(response: str) -> None:
    assert decide(response, [WEATHER]).label == dc.CLARIFY


def test_a_tool_limit_after_but_is_not_delivered_content() -> None:
    response = (
        "The available tools are designed for weather lookups, but they do not provide exchange rates for any "
        "currency pair."
    )
    assert decide(response, [WEATHER]).label == dc.UNSUPPORTED


def test_an_exclamation_of_disbelief_is_not_a_decline() -> None:
    assert decide("Wow, I can't believe it!").label == dc.DIRECT


# ── v2 round 2 (37 §7): capability wording, affirmative abilities, referrals and request lists ────────


def test_an_affirmative_ability_is_not_a_decline() -> None:
    response = "A teacher is like a gardener: gardeners have the ability to see what each plant needs to grow."
    assert decide(response).label == dc.DIRECT


@pytest.mark.parametrize(
    "response",
    [
        "The question lacks the functions required to translate documents. Without the appropriate functions, "
        "I am unable to do the translation.",
        "The given functions cannot retrieve exchange rates. They pertain to weather lookups, neither of which are "
        "relevant, so the question lacks the appropriate parameters and the necessary functions.",
        "I do not have any specific information about Jane Roe's research area.",
    ],
)
def test_a_lack_of_tools_or_of_information_about_the_subject_is_unsupported(response: str) -> None:
    assert decide(response, [WEATHER]).label == dc.UNSUPPORTED


def test_a_referral_phrased_as_a_need_is_not_delivered_content() -> None:
    response = (
        "The given question asks for detailed information about the history of every medieval castle in Europe and "
        "their architects. Since there are no functions provided for historical data, it is not possible to make "
        "any function calls. To finish your research, you may need to consult history books, archives or databases "
        "about medieval architecture."
    )
    assert decide(response).label == dc.UNSUPPORTED


def test_the_reasons_for_not_proceeding_and_a_list_of_missing_ids_belong_to_the_request() -> None:
    response = (
        "To answer your query, I'll need specific identifiers for each city you mentioned. The functions require "
        "the city_id parameter. Here is why I cannot proceed:\n1. **Forecast**: Requires the city_id of the city.\n"
        "2. **Alerts**: Requires the city_id of the city.\nYou need to provide the city_id values for Paris and Rome "
        "in order to call these functions.\nExample of missing information:\n- city_id for Paris (e.g., 2988507)\n"
        "- city_id for Rome\nOnce the identifiers are provided, I can proceed to collect the forecast and alerts for "
        "these cities."
    )
    assert decide(response, [WEATHER]).label == dc.CLARIFY


# ── v2 round 3 (37 §7): concessive disclaimers, uncallable functions and which-one questions ─────────


def test_lacking_a_function_is_a_capability_gap() -> None:
    response = "The provided question lacks a function related to booking hotels. Therefore, it cannot be processed using the given functions."
    assert decide(response, [WEATHER]).label == dc.UNSUPPORTED


def test_a_concessive_disclaimer_before_an_answer_is_direct() -> None:
    response = (
        "Though I can't give legal advice, it's usually wise to keep copies of every signed contract and read the "
        "termination clause before you sign."
    )
    assert decide(response).label == dc.DIRECT


def test_functions_that_cannot_be_called_for_want_of_arguments_is_not_a_capability_gap() -> None:
    response = (
        "The query does not provide the city_id required by the API. Without this identifier, the functions cannot "
        "be called."
    )
    assert decide(response, [WEATHER]).label == dc.CLARIFY


def test_naming_the_function_that_does_the_job_is_not_delivered_content() -> None:
    response = (
        'The function that retrieves a list of daily weather forecasts for a city is "Forecast". The required arguments are: 1. "days" '
        '(integer) Your query names the city but does not provide the "days" argument which is required.'
    )
    assert decide(response, [WEATHER]).label == dc.CLARIFY


def test_asking_which_thing_the_user_means_is_clarify() -> None:
    response = "I'm sorry but your request is too broad. Are you interested in a particular city, region, country or the general weather trends worldwide? Could you please specify?"
    assert decide(response, [WEATHER]).label == dc.CLARIFY
