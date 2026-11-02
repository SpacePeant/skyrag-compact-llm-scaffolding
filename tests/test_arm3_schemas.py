"""Tests for B1 -- the Arm 3 skill-schema layer.

What these pin, in the order the risks matter:

  1. The validator says yes to conforming values and no to each way of not
     conforming. Every schema violation exported by a live run is one of these
     verdicts, so a validator that is wrong is a results table that is wrong.
  2. **A schema violation never salvages keywords.** This is the sharpest edge
     in the whole build: Arm 2's parser maps a dict's top-level keys to
     keywords, so feeding it a schema-shaped reply would enter "platform" and
     "keywords" into retrieval as invented keywords. The harness would be
     manufacturing the failure it is measuring.
  3. **`copied_verbatim` never filters.** It is an instrument. A keyword the
     model admits it recalled must reach `classify` exactly like any other, or
     Arm 3 deletes the failure mode it exists to observe.
  4. **`cited_ids` still finds course IDs in a JSON path.** Grounding, M3.2 and
     `cited_count` all read that regex. If it broke, nothing would raise -- the
     grounding score would simply go quiet, which is the failure mode B15 was
     about.
  5. **Arm 2 is untouched.** `schema_enabled=False` must produce the prompts
     and the parse path it produced before this module existed, or the control
     arm moved when the treatment arm was added.
  6. B17: the treatment flags route the pipeline, and an explicit flag beats
     the arm label it disagrees with.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from skyrag import schemas
from skyrag.config import Config
from skyrag.data.schema import Course
from skyrag.embedding import EmbeddingStore
from skyrag.llm.echo import EchoBackend
from skyrag.pipelines.skyrag import run_skyrag
from skyrag.prompts import EXPAND_QUERY, STEP_BACK
from skyrag.selection import (
    OUTCOME_OK,
    OUTCOME_SCHEMA_VIOLATION,
    PROMPT,
    select_keywords,
)


class _MockEmbedder:
    def encode_one(self, text: str) -> np.ndarray:
        v = np.ones(384, dtype=np.float32)
        return v / np.linalg.norm(v)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.encode_one(t) for t in texts])


@pytest.fixture
def stores():
    course = Course(
        id_="COU1",
        platform="coursera",
        title="Intro to Python",
        keywords=["python", "programming"],
        description="Learn Python.",
        url="https://coursera.org/python",
    )
    matrix = _MockEmbedder().encode([course.title])
    return {"coursera": EmbeddingStore(courses=[course], matrix=matrix)}


def _s2():
    return schemas.keyword_selection_schema(["coursera", "edx"], 40)


def _valid_selection(**overrides):
    entry = {"keyword": "python", "score": 0.9, "copied_verbatim": True}
    entry.update(overrides)
    return {"platform": "coursera", "keywords": [entry]}


# ---------------------------------------------------------------------------
# 1. The validator
# ---------------------------------------------------------------------------

class TestValidator:
    def test_conforming_value_has_no_violations(self):
        assert schemas.validate(_valid_selection(), _s2()) == []

    def test_missing_required_property_is_named(self):
        bad = _valid_selection()
        del bad["keywords"][0]["copied_verbatim"]
        errors = schemas.validate(bad, _s2())
        assert len(errors) == 1
        assert "copied_verbatim" in errors[0]
        # The path, not just the property: a violation the reader cannot locate
        # in the reply is one they cannot check.
        assert "$.keywords[0]" in errors[0]

    def test_additional_property_is_rejected(self):
        bad = _valid_selection()
        bad["keywords"][0]["confidence"] = "high"
        errors = schemas.validate(bad, _s2())
        assert any("confidence" in e for e in errors)

    def test_platform_outside_the_enum_is_rejected(self):
        errors = schemas.validate(_valid_selection() | {"platform": "udemy"}, _s2())
        assert any("udemy" in e for e in errors)

    def test_exceeding_max_items_is_rejected(self):
        """The cap overrun becomes a violation the harness reads.

        Stage 1 recorded `selection_overrun = 1` on the 8B, so this is a real
        behaviour, not a hypothetical. Under Arm 2 it surfaced only as a
        truncation to be inferred from a finish_reason.
        """
        schema = schemas.keyword_selection_schema(["coursera"], 2)
        bad = {
            "platform": "coursera",
            "keywords": [
                {"keyword": f"k{i}", "score": 0.5, "copied_verbatim": True}
                for i in range(3)
            ],
        }
        errors = schemas.validate(bad, schema)
        assert any("maximum is 2" in e for e in errors)

    def test_a_boolean_is_not_an_integer(self):
        """`isinstance(True, int)` is True in Python, and must not be here.

        A schema that accepted `true` for `step` would admit a malformed
        learning path as conforming, and `step` is the only structured signal
        O1's ordering criterion has.
        """
        bad = {"path": [{"step": True, "course_id": "COU1", "rationale": "x"}]}
        errors = schemas.validate(bad, schemas.LEARNING_PATH_SCHEMA)
        assert any("expected integer" in e for e in errors)

    def test_an_integer_is_a_valid_number(self):
        assert schemas.validate(_valid_selection(score=1), _s2()) == []

    def test_max_length_is_enforced_on_rationale(self):
        bad = {"path": [{"step": 1, "course_id": "C", "rationale": "x" * 301}]}
        errors = schemas.validate(bad, schemas.LEARNING_PATH_SCHEMA)
        assert any("maximum is 300" in e for e in errors)

    def test_type_mismatch_does_not_cascade(self):
        """One wrong type reports one violation, not a pile of derived ones.

        Recursing into a value of the wrong type buries the single real cause
        under complaints about its contents, and these strings are what a
        reader uses to work out what the model actually did.
        """
        errors = schemas.validate({"path": "step 1: COU1"}, schemas.LEARNING_PATH_SCHEMA)
        assert len(errors) == 1


class TestParseAndValidate:
    def test_extracts_an_object_from_surrounding_prose(self):
        text = f"Sure! Here you go:\n{json.dumps(_valid_selection())}\nHope that helps."
        assert schemas.parse_and_validate(text, _s2(), "keyword_selection").ok

    def test_an_apostrophe_in_the_preamble_does_not_lose_the_object(self):
        """The `_balanced_spans` contraction defect, re-pinned through this path.

        `Here's` used to open a string literal that never closed, so the
        scanner consumed the reply and returned nothing -- charging a harness
        parsing bug to the model, preferentially to whichever model is
        chattier.
        """
        text = f"Here's the JSON: {json.dumps(_valid_selection())}"
        assert schemas.parse_and_validate(text, _s2(), "keyword_selection").ok

    def test_prose_with_no_json_is_a_violation_not_an_exception(self):
        result = schemas.parse_and_validate("I recommend Python.", _s2(), "sel")
        assert not result.ok
        assert result.violations
        # The reply survives on the failure path. A violation the harness
        # cannot show is a number nobody can check.
        assert result.raw == "I recommend Python."

    def test_empty_reply_is_a_violation(self):
        assert not schemas.parse_and_validate("", _s2(), "sel").ok

    def test_violations_carry_the_stage_name(self):
        result = schemas.parse_and_validate("nope", _s2(), "keyword_selection")
        assert all(v.startswith("keyword_selection:") for v in result.violations)


# ---------------------------------------------------------------------------
# 2. Selection under a schema
# ---------------------------------------------------------------------------

class TestSchemaSelection:
    def _vocab(self):
        return {"python": 5, "programming": 3}

    def test_a_conforming_reply_selects_and_records_the_claim(self):
        backend = EchoBackend(responses=[json.dumps(_valid_selection())])
        sel = select_keywords(
            backend, "learn python", "coursera", self._vocab(),
            schema_enabled=True, platforms=["coursera", "edx"],
        )
        assert sel.outcome == OUTCOME_OK
        assert sel.keywords == {"python": 0.9}
        assert sel.copied_verbatim == {"python": True}
        assert sel.schema_violations == []

    def test_the_score_survives_into_retrieval_ordering(self):
        """S2 carries a score because Arm 2's ordering depends on one.

        `ranked()` orders the keywords `keyword_buckets` consumes, and
        first-keyword-wins decides every retrieved course's `via_keyword`. A
        schema without a score would push that onto the alphabetical
        tie-break, changing what is retrieved for reasons unrelated to the
        treatment.
        """
        reply = {
            "platform": "coursera",
            "keywords": [
                {"keyword": "programming", "score": 0.4, "copied_verbatim": True},
                {"keyword": "python", "score": 0.95, "copied_verbatim": True},
            ],
        }
        backend = EchoBackend(responses=[json.dumps(reply)])
        sel = select_keywords(
            backend, "q", "coursera", self._vocab(), schema_enabled=True,
        )
        assert sel.ranked() == ["python", "programming"]

    def test_a_violating_reply_is_recorded_not_salvaged(self):
        """The sharpest edge in B1.

        Arm 2's `_coerce` maps a dict's top-level keys to keywords, so falling
        back to it here would enter `platform` and `keywords` into retrieval as
        two invented keywords -- and they would then score as model
        hallucinations. The harness would be manufacturing the failure it is
        measuring, in the arm whose whole purpose is to measure it.
        """
        reply = {"platform": "udemy", "keywords": []}
        backend = EchoBackend(responses=[json.dumps(reply)])
        sel = select_keywords(
            backend, "q", "coursera", self._vocab(),
            schema_enabled=True, platforms=["coursera", "edx"],
        )
        assert sel.outcome == OUTCOME_SCHEMA_VIOLATION
        assert sel.keywords == {}
        assert sel.hallucinated == []
        assert "platform" not in sel.keywords
        assert "keywords" not in sel.keywords
        assert sel.schema_violations

    def test_a_recalled_keyword_is_never_filtered(self):
        """`copied_verbatim` is an instrument, never a control.

        A model admitting it wrote a keyword from memory must be treated
        exactly like one claiming it copied. Dropping these would remove the
        invented keywords from M2.1 -- moving the headline metric toward the
        hypothesis by deleting its counter-evidence, the same error as
        repairing near-misses.
        """
        reply = {
            "platform": "coursera",
            "keywords": [
                {"keyword": "python", "score": 0.9, "copied_verbatim": False},
                {"keyword": "quantum alchemy", "score": 0.8, "copied_verbatim": False},
            ],
        }
        backend = EchoBackend(responses=[json.dumps(reply)])
        sel = select_keywords(
            backend, "q", "coursera", self._vocab(), schema_enabled=True,
        )
        assert sel.keywords == {"python": 0.9}          # real, kept
        assert sel.hallucinated == ["quantum alchemy"]  # invented, counted
        assert sel.copied_verbatim["quantum alchemy"] is False

    def test_the_claim_is_recorded_for_invented_keywords_too(self):
        """The interesting half of the split.

        Restricting the claim to validated keywords would answer "does the
        model know when it copied a *real* keyword", which is the easy
        question. The one the field exists for is what it claims about the
        ones it invented.
        """
        reply = {
            "platform": "coursera",
            "keywords": [
                {"keyword": "quantum alchemy", "score": 0.8, "copied_verbatim": True},
            ],
        }
        backend = EchoBackend(responses=[json.dumps(reply)])
        sel = select_keywords(
            backend, "q", "coursera", self._vocab(), schema_enabled=True,
        )
        assert sel.hallucinated == ["quantum alchemy"]
        assert sel.copied_verbatim == {"quantum alchemy": True}

    def test_the_schema_reaches_the_prompt_and_arm_2s_does_not(self):
        backend = EchoBackend(responses=[json.dumps(_valid_selection())])
        select_keywords(
            backend, "q", "coursera", self._vocab(),
            schema_enabled=True, platforms=["coursera", "edx"],
        )
        prompt = backend.calls[-1]["prompt"]
        assert "copied_verbatim" in prompt
        assert '"enum"' in prompt

    def test_arm2_selection_is_unchanged(self):
        """The control arm must not move when the treatment arm is added."""
        backend = EchoBackend(responses=['{"python": 0.9}'])
        sel = select_keywords(backend, "q", "coursera", self._vocab())
        assert sel.outcome == OUTCOME_OK
        assert sel.keywords == {"python": 0.9}
        assert sel.copied_verbatim == {}
        assert sel.schema_violations == []
        assert backend.calls[-1]["prompt"] == PROMPT.format(
            platform="coursera",
            vocabulary="python, programming",
            query="q",
            max_keywords=40,
        )

    def test_the_schema_budget_is_larger_than_arm_2s(self):
        """A schema entry is structurally longer than a `"key": score` pair.

        Sizing Arm 3's budget from Arm 2's per-keyword figure would reinstate
        the 2026-08-17 truncation defect in Arm 3 only -- and truncation is a
        *harness* outcome, so the arm would lose cells to a budget error while
        reporting nothing against the model.
        """
        plain = EchoBackend(responses=['{"python": 0.9}'])
        select_keywords(plain, "q", "coursera", self._vocab())
        schema = EchoBackend(responses=[json.dumps(_valid_selection())])
        select_keywords(schema, "q", "coursera", self._vocab(), schema_enabled=True)
        assert (
            schema.calls[-1]["params"].max_tokens
            > plain.calls[-1]["params"].max_tokens
        )


# ---------------------------------------------------------------------------
# 3. The pipeline end to end
# ---------------------------------------------------------------------------

def _schema_run_responses():
    return [
        json.dumps({
            "expansions": ["learn python", "python basics"],
            "expanded_query": "An introductory Python programming course.",
        }),
        json.dumps({
            "step_back": "programming fundamentals",
            "principles": ["Start simple", "Build up", "Practise", "Review"],
        }),
        json.dumps(_valid_selection()),
        json.dumps({
            "path": [{"step": 1, "course_id": "COU1", "rationale": "Starts at zero."}]
        }),
    ]


class TestPipeline:
    def _run(self, stores, responses, **kwargs):
        return run_skyrag(
            query="Learn Python",
            stores=stores,
            backend=EchoBackend(responses=responses),
            config=Config.load(),
            embedder=_MockEmbedder(),
            **kwargs,
        )

    def test_a_clean_arm3_run_reports_no_violations(self, stores):
        res = self._run(stores, _schema_run_responses(), arm="3")
        assert res.status == "ok"
        assert res.schema_violations == 0
        assert res.schema_violation_stages == ""
        assert res.keywords_claimed_copied == 1
        assert res.keywords_claimed_recalled == 0

    def test_cited_ids_still_reads_a_json_path(self, stores):
        """Grounding hangs off a regex over the path text.

        `cited_ids` finds course IDs by pattern, and every grounding metric,
        `cited_count` and M3.2 read it. If a JSON path broke it, nothing would
        raise -- the grounding score would go quiet and the row would look like
        a model that cited nothing.
        """
        res = self._run(stores, _schema_run_responses(), arm="3")
        assert res.cited_ids() == ["COU1"]
        assert res.hallucinated_ids() == []

    def test_the_path_text_is_the_model_s_own(self, stores):
        """S3 validates the path; it does not rewrite it.

        O1 scores this text. Substituting a re-rendered version would put the
        harness's formatting between the model and its judged score.
        """
        responses = _schema_run_responses()
        res = self._run(stores, responses, arm="3")
        assert res.learning_path == responses[-1]

    def test_violations_are_counted_per_stage_and_the_run_completes(self, stores):
        """Arm 2's replies through Arm 3: what an ignoring model looks like."""
        responses = [
            "Expanded: learn Python",
            "Step back: programming basics",
            "python\nprogramming",
            "A good Python course is COU1.",
        ]
        res = self._run(stores, responses, arm="3")
        assert res.schema_violations > 0
        stages = res.schema_violation_stages.split(",")
        assert "query_expansion" in stages
        assert "step_back" in stages
        assert "keyword_selection" in stages

    def test_a_rejected_expansion_falls_back_to_the_raw_reply(self, stores):
        """A violation must not change what gets *retrieved*.

        Dropping the expansion on a schema failure would make every downstream
        metric for that cell a measurement of the harness's reaction rather
        than of the model.
        """
        responses = _schema_run_responses()
        responses[0] = "Expanded: an introductory Python course."
        res = self._run(stores, responses, arm="3")
        assert res.expanded_query == "Expanded: an introductory Python course."
        assert res.schema_violations == 1

    def test_arm2_runs_the_unschemad_prompts(self, stores):
        backend = EchoBackend(responses=[
            "Expanded: learn Python",
            "Step back: programming basics",
            '{"python": 0.9}',
            "A good Python course is COU1.",
        ])
        res = run_skyrag(
            query="Learn Python",
            stores=stores,
            backend=backend,
            config=Config.load(),
            embedder=_MockEmbedder(),
            arm="2",
        )
        assert res.schema_violations == 0
        assert res.keywords_claimed_copied == 0
        assert backend.calls[0]["prompt"] == EXPAND_QUERY.format(query="Learn Python")
        assert backend.calls[1]["prompt"] == STEP_BACK.format(query="Learn Python")
        assert "copied_verbatim" not in backend.calls[2]["prompt"]


class TestFlagRouting:
    """B17 -- the treatment flags route the pipeline; the label only names it."""

    def _run(self, stores, **kwargs):
        return run_skyrag(
            query="Learn Python",
            stores=stores,
            backend=EchoBackend(responses=_schema_run_responses()),
            config=Config.load(),
            embedder=_MockEmbedder(),
            **kwargs,
        )

    def test_schema_enabled_activates_without_an_arm_label(self, stores):
        res = self._run(stores, schema_enabled=True)
        assert res.keywords_claimed_copied == 1

    def test_an_explicit_false_beats_the_arm_label(self, stores):
        """The flag is the instruction; the label is the name for it.

        Two encodings of one axis that can disagree surface as a mislabelled
        results table rather than as an exception, which is why one of them has
        to win explicitly.
        """
        res = self._run(stores, arm="3", schema_enabled=False)
        assert res.schema_violations == 0
        assert res.keywords_claimed_copied == 0

    def test_arm3_label_alone_infers_the_schema(self, stores):
        """With no flag passed, the legacy path infers the schema from the label."""
        res = self._run(stores, arm="3")
        assert res.keywords_claimed_copied == 1
