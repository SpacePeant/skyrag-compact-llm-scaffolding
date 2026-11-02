"""Arm 3 under V5: the flat-map S2 payload.

V5 exists to answer one question: does the schema cost retrieval *because it is
a schema*, or *because of what V1's schema asks the model to write*? That only
works if V5 differs from V1 in payload shape and in nothing else, so most of
what is pinned here is sameness rather than behaviour.

CB2 is the reason this file exists before the run rather than after it: the
`spec_plus_exemplar` path shipped with zero coverage and could have produced
three hours of rows labelled as a treatment they were not.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from skyrag import selection
from skyrag.schemas import (
    EXEMPLARS,
    STAGE_SELECTION,
    STAGE_SELECTION_V5,
    keyword_selection_schema,
    keyword_selection_schema_v5,
    parse_and_validate,
    render,
    validate,
)

PLATFORMS = ["coursera", "edx"]
CAP = 40


# --- the schema contract is preserved ---------------------------------------


def test_v5_keeps_every_contract_element_v1_has():
    """Payload shape is the only thing V5 gives up.

    If V5 also dropped the platform enum or the closed top level it would be a
    weaker contract as well as a cheaper one, and a retrieval recovery could be
    attributed to either.
    """
    v1 = keyword_selection_schema(PLATFORMS, CAP)
    v5 = keyword_selection_schema_v5(PLATFORMS, CAP)

    assert v5["type"] == v1["type"] == "object"
    assert v5["required"] == v1["required"] == ["platform", "keywords"]
    assert v5["additionalProperties"] is False
    assert v5["properties"]["platform"] == v1["properties"]["platform"]


def test_v5_enforces_the_same_keyword_cap():
    """maxProperties is the flat-map counterpart of V1's maxItems."""
    v5 = keyword_selection_schema_v5(PLATFORMS, CAP)
    assert v5["properties"]["keywords"]["maxProperties"] == CAP
    assert v5["properties"]["keywords"]["minProperties"] == 1

    over = {"platform": "coursera", "keywords": {f"kw{i}": 0.5 for i in range(CAP + 1)}}
    assert validate(over, v5), "a reply over the cap must not validate"

    at_cap = {"platform": "coursera", "keywords": {f"kw{i}": 0.5 for i in range(CAP)}}
    assert validate(at_cap, v5) == []


def test_v5_still_constrains_the_score_range():
    """The score bound survives the move from field value to map value."""
    v5 = keyword_selection_schema_v5(PLATFORMS, CAP)
    assert validate({"platform": "coursera", "keywords": {"a": 1.5}}, v5)
    assert validate({"platform": "coursera", "keywords": {"a": -0.1}}, v5)
    assert validate({"platform": "coursera", "keywords": {"a": "high"}}, v5)
    assert validate({"platform": "coursera", "keywords": {"a": 0.5}}, v5) == []


def test_v5_still_rejects_an_unknown_platform():
    v5 = keyword_selection_schema_v5(PLATFORMS, CAP)
    assert validate({"platform": "udemy", "keywords": {"a": 0.5}}, v5)


def test_v5_retains_the_copy_exactly_instruction_somewhere_in_the_schema():
    """V1 carries it on `keyword`; V5 has no such property, so it moves.

    Losing it would confound payload shape with the presence of the fidelity
    instruction -- the confound that made the V4 comparison uninterpretable.
    """
    v5 = keyword_selection_schema_v5(PLATFORMS, CAP)
    description = v5["properties"]["keywords"]["description"].lower()
    assert "exactly" in description
    assert "pluralise" in description


# --- the validator extension is additive ------------------------------------


def test_existing_schemas_validate_exactly_as_before():
    """minProperties/maxProperties and dict-valued additionalProperties are new.

    They must not perturb V1, which uses neither. A silent change here would
    move condition A's violation count without any run having been repeated.
    """
    v1 = keyword_selection_schema(PLATFORMS, CAP)
    good = {
        "platform": "coursera",
        "keywords": [{"keyword": "marine biology", "score": 0.9, "copied_verbatim": True}],
    }
    assert validate(good, v1) == []

    bad = {"platform": "coursera", "keywords": [{"keyword": "x", "score": 0.9}]}
    assert validate(bad, v1), "a missing required field must still fail"


def test_additional_properties_false_still_rejects_extras():
    v5 = keyword_selection_schema_v5(PLATFORMS, CAP)
    assert validate({"platform": "coursera", "keywords": {"a": 0.5}, "extra": 1}, v5)


# --- the exemplar ------------------------------------------------------------


def test_v5_exemplar_is_flat_and_validates_against_v5():
    """An exemplar in V1's shape beside a V5 schema teaches a violation."""
    example = EXEMPLARS[STAGE_SELECTION_V5]
    assert isinstance(example["keywords"], dict)
    assert validate(example, keyword_selection_schema_v5(PLATFORMS, CAP)) == []


def test_v1_and_v5_exemplars_carry_the_same_subject_matter():
    """Same distance from the ten benchmark domains, so neither leaks more."""
    v1_terms = {e["keyword"] for e in EXEMPLARS[STAGE_SELECTION]["keywords"]}
    assert v1_terms == set(EXEMPLARS[STAGE_SELECTION_V5]["keywords"])


def test_v5_exemplar_names_no_benchmark_domain():
    banned = (
        "data science", "machine learning", "python", "business", "health",
        "law", "language", "education", "arts", "communication", "science",
        "mathematics", "engineering", "medicine",
    )
    text = str(EXEMPLARS[STAGE_SELECTION_V5]).lower()
    for term in banned:
        assert term not in text, f"V5 exemplar mentions {term!r}, a benchmark domain term"


# --- the prompt --------------------------------------------------------------


def _render(template: str, schema: dict) -> str:
    return template.format(
        platform="coursera",
        vocabulary="marine biology, oceanography",
        query="learn about reefs",
        max_keywords=CAP,
        schema=render(schema),
    )


def test_v5_prompt_differs_from_v1_only_by_the_forced_edits():
    """Every V1 rule except the copied_verbatim one must survive verbatim."""
    v5_text = selection.PROMPT_SCHEMA_V5
    for rule in (
        "- Choose ONLY from the list below. Never invent a keyword.",
        "- Score each from 0.0 to 1.0 by how relevant it is.",
        "- Return AT MOST {max_keywords} keywords, the most relevant ones.",
        "- Copy each keyword EXACTLY as written below.",
    ):
        assert rule in v5_text
        assert rule in selection.PROMPT_SCHEMA_V1

    assert "copied_verbatim" not in v5_text
    assert "copied_verbatim" in selection.PROMPT_SCHEMA_V1


def test_v5_prompt_example_is_a_flat_map():
    text = _render(selection.PROMPT_SCHEMA_V5, keyword_selection_schema_v5(PLATFORMS, CAP))
    assert '"keywords": {"machine learning": 0.95' in text


# --- parsing -----------------------------------------------------------------


def test_a_flat_reply_parses_and_validates():
    reply = '{"platform": "coursera", "keywords": {"Marine Biology": 0.9, "oceanography": 0.7}}'
    result = parse_and_validate(reply, keyword_selection_schema_v5(PLATFORMS, CAP), "keyword_selection")
    assert result.ok, result.violations
    assert result.value["keywords"] == {"Marine Biology": 0.9, "oceanography": 0.7}


def test_a_v1_shaped_reply_does_not_validate_against_v5():
    """The two payloads must be mutually exclusive.

    If a V1-shaped reply satisfied V5, a mis-dispatched run would be scored as
    the treatment it was not -- CB2's failure mode in a new place.
    """
    reply = '{"platform": "coursera", "keywords": [{"keyword": "a", "score": 0.5, "copied_verbatim": true}]}'
    assert not parse_and_validate(
        reply, keyword_selection_schema_v5(PLATFORMS, CAP), "keyword_selection"
    ).ok


# --- budget parity -----------------------------------------------------------


def test_v5_is_budgeted_at_the_arm_2_rate_not_the_schema_rate():
    """A V5 answer is Arm 2-shaped, so Arm 2's rate is its structural minimum.

    Budgeting V5 at the schema rate would hand it room V1 never had, turning a
    payload-shape comparison into a budget comparison.
    """
    assert selection._TOKENS_PER_KEYWORD == 32
    assert selection._TOKENS_PER_KEYWORD_SCHEMA == 64
    arm2_budget = selection._TOKEN_OVERHEAD + CAP * selection._TOKENS_PER_KEYWORD
    assert arm2_budget == 1408


# --- dispatch ----------------------------------------------------------------


# --- end to end through select_keywords --------------------------------------
#
# Every test above passed against a build in which the flat branch's assignment
# sat inside `if not key: continue` and was therefore unreachable: V5 returned
# `empty` for every reply. Schema-level and prompt-level tests cannot see that,
# because the defect is in the wiring between them. Hence these.


@dataclass
class _Reply:
    text: str
    truncated: bool = False
    prompt_tokens: int = 100
    completion_tokens: int = 50


class _StubBackend:
    def __init__(self, text: str) -> None:
        self.text = text
        self.prompt: str | None = None
        self.params = None

    def generate(self, prompt, model=None, params=None):
        self.prompt = prompt
        self.params = params
        return _Reply(self.text)


VOCAB = {"marine biology": 10, "oceanography": 8, "reef ecology": 5}
V5_REPLY = (
    '{"platform":"coursera","keywords":'
    '{"Marine Biology":0.9,"oceanography":0.7,"quantum python":0.5}}'
)


def _select(reply: str, version: str):
    backend = _StubBackend(reply)
    result = selection.select_keywords(
        backend, "reefs", "coursera", VOCAB,
        model="m", schema_enabled=True,
        platforms=["coursera", "edx"], sweep_version=version, max_keywords=CAP,
    )
    return result, backend


def test_v5_reply_reaches_retrieval():
    result, _ = _select(V5_REPLY, "v5")
    assert result.outcome == "ok"
    assert result.keywords == {"marine biology": 0.9, "oceanography": 0.7}


def test_v5_still_classifies_invention():
    """The whole point of the arm survives the payload change."""
    result, _ = _select(V5_REPLY, "v5")
    assert result.hallucinated == ["quantum python"]


def test_v5_lowercases_keys_the_way_arm_2_does():
    """`Marine Biology` must reach `classify` as `marine biology`.

    A V5 keyword normalising differently from an Arm 2 one would move M2.1 by
    casing rather than by capability.
    """
    result, _ = _select(V5_REPLY, "v5")
    assert "marine biology" in result.keywords


def test_v5_requests_the_arm_2_budget_and_v2_still_requests_the_schema_budget():
    _, v5_backend = _select(V5_REPLY, "v5")
    v2_reply = (
        '{"platform":"coursera","keywords":'
        '[{"keyword":"marine biology","score":0.9,"copied_verbatim":true}]}'
    )
    _, v2_backend = _select(v2_reply, "v2")
    assert v5_backend.params.max_tokens == 1408
    assert v2_backend.params.max_tokens == 2688


@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4", "v5"])
def test_arm_2_prompt_is_unchanged_by_every_version(version):
    """The control cannot drift under a version change. Extends the V4 test."""
    assert selection.PROMPT.startswith("You are an expert curriculum designer.")
    assert "copied_verbatim" not in selection.PROMPT
    assert "Schema:" not in selection.PROMPT
