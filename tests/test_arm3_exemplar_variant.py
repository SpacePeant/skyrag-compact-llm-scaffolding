"""A2 -- Arm 3 condition B: the `spec_plus_exemplar` prompt variant.

Why this module exists, and why nearly every assertion below is about *prompt
text* rather than about a return value:

The 2026-08-31 condition-A run (`spec_only`, 50 queries, 8B) measured a
well-powered null -- invalid keyword rate 0.676 -> 0.659, p = 0.80 -- and its
own log rules that **no Arm 3 claim may be published on condition A alone**. A
bare schema specification with no worked example tests "can an 8B read a spec",
not "does structured output help". Condition B is therefore the deciding run
for the Arm 3 leg of the thesis.

That run costs ~3 hours of paid inference, and **its artifacts cannot audit
themselves**. The export stamps `schema_prompt_variant` per row, but the prompt
actually sent is never written to disk -- only `raw_response` is. So if the
flag failed to reach `schemas.render()`, the result would be a directory of
rows labelled `spec_plus_exemplar` that were in fact condition A: the treatment
compared against itself, with nothing in the output able to reveal it. A silent
null on top of a silent null.

Until this module, nothing exercised the path at all (`grep -ri exemplar
tests/` returned nothing) -- and it is the exact path the `bcaff6a` merge
conflict deleted and a hand-resolution restored on 2026-09-01.

One thing the sweep driver already gets right, and which these tests therefore
do not need to re-check: `run_multi_model_eval` derives the `schema_exemplar`
argument it passes to the pipeline from `eff_schema_variant`, the same value it
stamps on the row. Label and behaviour share one source and cannot drift apart
downstream of that point. What was untested is everything upstream of it --
whether the flag reaches the prompt at all, at every stage.
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
from skyrag.selection import PROMPT, select_keywords

EXEMPLAR_MARKER = "Example of a valid response"


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


def _vocab():
    return {"python": 5, "programming": 3}


def _valid_selection():
    return {
        "platform": "coursera",
        "keywords": [{"keyword": "python", "score": 0.9, "copied_verbatim": True}],
    }


def _schema_run_responses():
    """One schema-conforming reply per stage, in pipeline order."""
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


def _select(backend, **kwargs):
    return select_keywords(backend, "q", "coursera", _vocab(), **kwargs)


# ---------------------------------------------------------------------------
# 1. The flag reaches the selection prompt
# ---------------------------------------------------------------------------

class TestTheFlagReachesThePrompt:
    def test_the_exemplar_is_appended_when_asked_for(self):
        backend = EchoBackend(responses=[json.dumps(_valid_selection())])
        _select(backend, schema_enabled=True, schema_exemplar=True,
                platforms=["coursera", "edx"])
        prompt = backend.calls[-1]["prompt"]
        assert EXEMPLAR_MARKER in prompt
        assert "marine biology" in prompt

    def test_spec_only_is_the_default_and_carries_no_exemplar(self):
        """Condition A must keep meaning what it meant when it was run.

        If the exemplar ever became the default, the completed 50-query
        condition-A dataset would stop being reproducible by the command that
        produced it -- and the A/B contrast it anchors would collapse without
        raising anything.
        """
        backend = EchoBackend(responses=[json.dumps(_valid_selection())])
        _select(backend, schema_enabled=True, platforms=["coursera", "edx"])
        prompt = backend.calls[-1]["prompt"]
        assert EXEMPLAR_MARKER not in prompt
        assert "marine biology" not in prompt

    def test_the_variant_only_adds_and_never_rewrites(self):
        """The exemplar is an addition, so the contrast isolates one change.

        A condition B that also reworded the instructions would confound the
        worked example with whatever else moved, and the A/B difference would
        no longer be attributable to the exemplar.
        """
        plain = EchoBackend(responses=[json.dumps(_valid_selection())])
        _select(plain, schema_enabled=True, platforms=["coursera", "edx"])
        exemplar = EchoBackend(responses=[json.dumps(_valid_selection())])
        _select(exemplar, schema_enabled=True, schema_exemplar=True,
                platforms=["coursera", "edx"])

        spec_only = plain.calls[-1]["prompt"]
        spec_plus = exemplar.calls[-1]["prompt"]

        # The schema is rendered *inside* a surrounding instruction template,
        # so the example lands mid-prompt rather than at the end. Deleting
        # exactly the block `render()` adds must recover the condition-A prompt
        # character for character -- anything else means a second change rode
        # along with the exemplar.
        selection_schema = schemas.keyword_selection_schema(["coursera", "edx"], 40)
        base = schemas.render(selection_schema)
        added = schemas.render(
            selection_schema, exemplar_stage=schemas.STAGE_SELECTION
        )[len(base):]
        assert added.strip(), "render() appended nothing for the exemplar variant"
        assert spec_plus.replace(added, "") == spec_only

    def test_the_token_budget_is_unchanged_by_the_variant(self):
        """The exemplar lengthens the prompt, not the answer.

        Arm 3's completion budget is what decides whether an over-long reply is
        blamed on the model (`reply_overran_cap`) or on us (`truncated`). If
        the exemplar moved that budget, condition B's overrun rate would not be
        comparable with condition A's -- and overrun is implicated in 23 of
        condition A's 24 mute cells, so it is the number most likely to carry
        the finding.
        """
        plain = EchoBackend(responses=[json.dumps(_valid_selection())])
        _select(plain, schema_enabled=True, platforms=["coursera", "edx"])
        exemplar = EchoBackend(responses=[json.dumps(_valid_selection())])
        _select(exemplar, schema_enabled=True, schema_exemplar=True,
                platforms=["coursera", "edx"])
        assert (
            exemplar.calls[-1]["params"].max_tokens
            == plain.calls[-1]["params"].max_tokens
        )


# ---------------------------------------------------------------------------
# 2. The control arm does not move
# ---------------------------------------------------------------------------

class TestArm2IsUntouched:
    def test_the_flag_is_inert_without_a_schema(self):
        """`schema_exemplar` is meaningless unless `schema_enabled` is set.

        The sweep driver hands the argument to every arm it runs, so an
        unguarded implementation would leak a JSON example into Arm 2's plain
        prompt -- partly schema-priming the zero-shot baseline that the whole
        comparison is measured against.
        """
        backend = EchoBackend(responses=['{"python": 0.9}'])
        _select(backend, schema_exemplar=True)
        assert backend.calls[-1]["prompt"] == PROMPT.format(
            platform="coursera",
            vocabulary="python, programming",
            query="q",
            max_keywords=40,
        )


# ---------------------------------------------------------------------------
# 3. The example we teach is itself sound
# ---------------------------------------------------------------------------

# Built here rather than imported: `schemas` exposes no stage -> schema
# registry, and the selection schema is constructed per call because its
# `platform` enum depends on which platforms are loaded.
def _schema_for(stage: str) -> dict:
    return {
        schemas.STAGE_EXPANSION: schemas.QUERY_EXPANSION_SCHEMA,
        schemas.STAGE_STEP_BACK: schemas.STEP_BACK_SCHEMA,
        schemas.STAGE_SELECTION: schemas.keyword_selection_schema(
            ["coursera", "edx"], 40
        ),
        # V5 selects the same stage under a flat-map payload, so it carries its
        # own exemplar and must be checked against its own schema. Validating it
        # against V1's would pass a shape the V5 run would then reject.
        schemas.STAGE_SELECTION_V5: schemas.keyword_selection_schema_v5(
            ["coursera", "edx"], 40
        ),
        schemas.STAGE_PATH: schemas.LEARNING_PATH_SCHEMA,
    }[stage]


class TestTheExemplarsThemselves:
    def test_every_stage_has_an_exemplar(self):
        """A missing entry degrades silently: `render` returns the spec alone.

        That stage would then run condition A inside a run stamped condition B,
        and the row would still read `spec_plus_exemplar`.
        """
        assert set(schemas.EXEMPLARS) == {
            schemas.STAGE_EXPANSION,
            schemas.STAGE_STEP_BACK,
            schemas.STAGE_SELECTION,
            schemas.STAGE_SELECTION_V5,
            schemas.STAGE_PATH,
        }

    def test_every_exemplar_conforms_to_its_own_schema(self):
        """An invalid example teaches the model to violate the schema.

        This fails quietly rather than loudly: the run completes, the violation
        count rises, and Arm 3 records as model failure something the harness
        induced.
        """
        for stage, example in schemas.EXEMPLARS.items():
            result = schemas.parse_and_validate(
                json.dumps(example), _schema_for(stage), stage
            )
            assert result.ok, (
                f"{stage} exemplar violates its own schema: {result.violations}"
            )

    def test_no_exemplar_leaks_a_benchmark_domain(self):
        """The example shows shape, never content.

        Its subject matter is drawn from outside the ten benchmark domains on
        purpose. An exemplar naming a scored domain would hand the model a
        usable keyword for free -- turning a formatting aid into contamination
        and inflating Arm 3 over Arm 2 for a reason that has nothing to do with
        schemas.
        """
        text = json.dumps(schemas.EXEMPLARS).lower()
        for term in (
            "python", "programming", "machine learning", "data science",
            "cybersecurity", "web development", "statistics",
        ):
            assert term not in text, (
                f"exemplar text mentions {term!r}, a benchmark domain term"
            )


# ---------------------------------------------------------------------------
# 4. The flag reaches every stage of a real run
# ---------------------------------------------------------------------------

# One phrase unique to each stage's exemplar. Matching on the rendered example
# text is the only check that distinguishes "the flag was passed" from "the
# flag was passed and used".
_STAGE_MARKERS = {
    schemas.STAGE_EXPANSION: "beginner music theory for songwriting",
    schemas.STAGE_STEP_BACK: "foundational music literacy",
    schemas.STAGE_SELECTION: "marine biology",
    schemas.STAGE_PATH: "Establishes the vocabulary the later steps assume.",
}


class TestTheWholeRun:
    def test_every_stage_of_an_arm3_run_carries_its_exemplar(self, stores):
        """The wiring check the paid condition-B run depends on.

        Four separate call sites thread this flag -- expansion, step-back and
        path in `pipelines/skyrag.py`, selection in `selection.py` -- and each
        passes its own `exemplar_stage`. A miss at any one of them produces a
        *partial* condition B, which the export would still stamp as a whole
        one, and which no artifact could distinguish from the real thing.
        """
        backend = EchoBackend(responses=_schema_run_responses())
        run_skyrag(
            query="Learn Python",
            stores=stores,
            backend=backend,
            config=Config.load(),
            embedder=_MockEmbedder(),
            arm="3",
            schema_exemplar=True,
        )
        prompts = [c["prompt"] for c in backend.calls]
        for stage, marker in _STAGE_MARKERS.items():
            assert any(marker in p for p in prompts), (
                f"no prompt carried the {stage} exemplar -- that stage is unwired, "
                "so a run stamped spec_plus_exemplar would be part condition A"
            )

    def test_an_arm3_run_without_the_flag_carries_no_exemplar_anywhere(self, stores):
        backend = EchoBackend(responses=_schema_run_responses())
        run_skyrag(
            query="Learn Python",
            stores=stores,
            backend=backend,
            config=Config.load(),
            embedder=_MockEmbedder(),
            arm="3",
        )
        joined = "\n".join(c["prompt"] for c in backend.calls)
        assert EXEMPLAR_MARKER not in joined
        for marker in _STAGE_MARKERS.values():
            assert marker not in joined
