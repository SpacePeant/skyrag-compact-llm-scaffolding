"""The V4 compact keyword-selection scaffolding, on the schema arms.

Why this module exists:

`keyword_selection_schema_v4` and `PROMPT_SCHEMA_V4` are reachable from exactly
one place -- `selection.py`, under `schema_enabled and sweep_version == "v4"` --
and as of 2026-09-02 that branch **had never executed against a live model**:
Arm 3 has never been run on V4. It also had no test of any kind.

So a V4 Arm 3 sweep would be the first execution of this code path anywhere,
and -- exactly as with the exemplar variant -- the prompt sent is never
persisted, only `raw_response`. A branch that silently failed to swap the
prompt would produce rows stamped `sweep_version = v4` that were really V1,
with nothing in the artifact able to show it.

What is deliberately *not* asserted here: that V4 is better. That is what the
run measures. These tests only establish that V4 is what actually gets sent.
"""

from __future__ import annotations

import json

import pytest

from skyrag import schemas
from skyrag.llm.echo import EchoBackend
from skyrag.selection import PROMPT, PROMPT_SCHEMA_V4, select_keywords


def _vocab():
    return {"python": 5, "programming": 3}


def _valid_selection():
    return {
        "platform": "coursera",
        "keywords": [{"keyword": "python", "score": 0.9, "copied_verbatim": True}],
    }


def _select(backend, **kwargs):
    return select_keywords(backend, "q", "coursera", _vocab(), **kwargs)


def _prompt_for(sweep_version, **kwargs):
    backend = EchoBackend(responses=[json.dumps(_valid_selection())])
    _select(
        backend,
        schema_enabled=True,
        platforms=["coursera", "edx"],
        sweep_version=sweep_version,
        **kwargs,
    )
    return backend.calls[-1]["prompt"]


# A phrase unique to each template. The V1 line is the instruction V4 drops,
# which is the substantive difference between them and the one that matters
# most to the thesis -- see TestWhatV4Removes.
V1_MARKER = "Copy each keyword EXACTLY as written"
V4_MARKER = "Choose relevant keywords for the query from the list below."


class TestTheVersionReachesThePrompt:
    def test_v4_sends_the_compact_template(self):
        prompt = _prompt_for("v4")
        assert V4_MARKER in prompt
        assert V1_MARKER not in prompt

    def test_v2_sends_the_original_template(self):
        prompt = _prompt_for("v2")
        assert V1_MARKER in prompt
        assert V4_MARKER not in prompt

    @pytest.mark.parametrize("version", ["v1", "v2", "v3"])
    def test_only_v4_differs(self, version):
        """v1, v2 and v3 must be byte-identical on the schema arms.

        `selection.py` branches on `"v4"` and `"v5"` alone. This pins that,
        because it is what makes the condition-A run (stamped `v2`) a valid
        comparator whichever of v1, v2 or v3 a later run names.
        """
        assert _prompt_for(version) == _prompt_for("v2")

    def test_the_default_is_not_v4(self):
        """A caller that forgets the argument must get the original prompt.

        `select_keywords` is called from the pipeline, from tests and from
        scripts. A default of v4 would silently re-treat every schema arm.
        """
        backend = EchoBackend(responses=[json.dumps(_valid_selection())])
        _select(backend, schema_enabled=True, platforms=["coursera", "edx"])
        assert V1_MARKER in backend.calls[-1]["prompt"]


class TestArm2IsUntouched:
    @pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4"])
    def test_no_version_changes_the_unschemad_prompt(self, version):
        """Arm 2 is the control, and `sweep_version` must not reach it.

        Every V4 prompt swap in the pipeline sits behind `if schema_enabled`.
        If one ever escaped that guard, the baseline would move underneath the
        comparison and the Arm 2 rows already on disk would stop being
        reproducible -- silently, since the prompt is not exported.
        """
        backend = EchoBackend(responses=['{"python": 0.9}'])
        _select(backend, sweep_version=version)
        assert backend.calls[-1]["prompt"] == PROMPT.format(
            platform="coursera",
            vocabulary="python, programming",
            query="q",
            max_keywords=40,
        )


class TestTheV4Schema:
    def test_v4_strips_field_descriptions(self):
        """The compaction is the treatment, so it has to be real.

        V1 carries a long `description` on every field; V4 drops them. If they
        survived, a "compact" run would send nearly the same token count and
        the comparison would measure nothing.
        """
        v1 = schemas.keyword_selection_schema(["coursera", "edx"], 40)
        v4 = schemas.keyword_selection_schema_v4(["coursera", "edx"], 40)
        assert "description" in json.dumps(v1)
        assert "description" not in json.dumps(v4)
        assert len(json.dumps(v4)) < len(json.dumps(v1))

    def test_v4_keeps_the_fields_the_metrics_read(self):
        """Compaction may shorten the prose, never drop a measured field.

        `score` orders retrieval (first-keyword-wins decides every course's
        `via_keyword`), and `copied_verbatim` is the instrument M2.1 is split
        by. Losing either would change what is retrieved, or delete a failure
        mode, for reasons unrelated to the treatment.
        """
        v4 = schemas.keyword_selection_schema_v4(["coursera", "edx"], 40)
        item = v4["properties"]["keywords"]["items"]
        assert set(item["required"]) == {"keyword", "score", "copied_verbatim"}
        assert item["additionalProperties"] is False
        assert v4["properties"]["platform"]["enum"] == ["coursera", "edx"]

    def test_v4_honours_the_configured_cap(self):
        """The live path passes the real cap through.

        `KEYWORD_SELECTION_SCHEMA_V4` is a module-level constant built at 20,
        but nothing reads it -- `selection.py` calls the function with
        `config.retrieval.max_keywords`. Pinned because a V4 run that silently
        capped at 20 would confound compaction with the cap, which is the
        separate CB5 question.
        """
        assert schemas.keyword_selection_schema_v4(["coursera"], 40)[
            "properties"]["keywords"]["maxItems"] == 40
        assert schemas.keyword_selection_schema_v4(["coursera"], 80)[
            "properties"]["keywords"]["maxItems"] == 80

    def test_a_v4_reply_validates_against_the_v4_schema(self):
        v4 = schemas.keyword_selection_schema_v4(["coursera", "edx"], 40)
        result = schemas.parse_and_validate(
            json.dumps(_valid_selection()), v4, schemas.STAGE_SELECTION
        )
        assert result.ok, result.violations


class TestWhatV4Removes:
    def test_v4_drops_the_verbatim_copy_instruction(self):
        """Recorded as a test because it is a research decision, not a detail.

        V1 tells the model: "Copy each keyword EXACTLY as written below. Do not
        correct, pluralise, expand or tidy the characters." That instruction
        targets exactly the deficit the thesis measures -- inventing and
        mis-transcribing catalogue vocabulary. V4 does not carry it.

        So a V4 arm that retrieves more may have done so by relaxing the
        constraint under study. This test does not object to that; it makes the
        removal impossible to overlook when the numbers are read.
        """
        v1_prompt = _prompt_for("v2")
        v4_prompt = _prompt_for("v4")
        assert V1_MARKER in v1_prompt
        assert V1_MARKER not in v4_prompt
        assert "copied_verbatim" not in PROMPT_SCHEMA_V4
        # The compact prompt still forbids invention, which is the one
        # instruction both variants must share for the arms to be comparable.
        assert "Never invent" in v4_prompt

    def test_v4_is_substantially_shorter(self):
        v1_prompt = _prompt_for("v2")
        v4_prompt = _prompt_for("v4")
        assert len(v4_prompt) < len(v1_prompt)


class TestPipelineLevelVersionEquivalence:
    """v2 and v3 are the same treatment on Arm 3, at every stage.

    Verified empirically 2026-09-02 rather than argued from the code: no stage
    of the Arm 3 pipeline branches on v2 versus v3.

    This matters for two reasons. The condition-A run is stamped `v2`, so a
    reader comparing stamps against a `v3` run would reasonably assume the
    treatment moved with them. And a three-way v2/v3/v4
    comparison on Arm 3 is not a real experiment -- two of its cells are the
    same cell. Pinned here so that if a future version *does* reach the
    schema-only path, this fails loudly instead of quietly splitting one arm
    into two labels.
    """

    def _prompts(self, stores, version):
        import json as _json

        import numpy as _np

        from skyrag.config import Config
        from skyrag.llm.echo import EchoBackend
        from skyrag.pipelines.skyrag import run_skyrag

        class _Emb:
            def encode_one(self, text):
                v = _np.ones(384, dtype=_np.float32)
                return v / _np.linalg.norm(v)

            def encode(self, texts):
                return _np.vstack([self.encode_one(t) for t in texts])

        responses = [
            _json.dumps({"expansions": ["a", "b"], "expanded_query": "An intro course."}),
            _json.dumps({"step_back": "fundamentals", "principles": ["a", "b", "c", "d"]}),
            _json.dumps(_valid_selection()),
            _json.dumps({"path": [{"step": 1, "course_id": "COU1", "rationale": "Start."}]}),
        ]
        backend = EchoBackend(responses=responses)
        run_skyrag(
            query="Learn Python",
            stores=stores,
            backend=backend,
            config=Config.load(),
            embedder=_Emb(),
            arm="3",
            sweep_version=version,
        )
        return [c["prompt"] for c in backend.calls]

    @pytest.fixture
    def stores(self):
        import numpy as _np

        from skyrag.data.schema import Course
        from skyrag.embedding import EmbeddingStore

        class _Emb:
            def encode(self, texts):
                v = _np.ones((len(texts), 384), dtype=_np.float32)
                return v / _np.linalg.norm(v, axis=1, keepdims=True)

        course = Course(
            id_="COU1",
            platform="coursera",
            title="Intro to Python",
            keywords=["python", "programming"],
            description="Learn Python.",
            url="https://coursera.org/python",
        )
        return {
            "coursera": EmbeddingStore(
                courses=[course], matrix=_Emb().encode([course.title])
            )
        }

    def test_v2_and_v3_are_identical_at_every_stage(self, stores):
        assert self._prompts(stores, "v2") == self._prompts(stores, "v3")

    def test_v4_differs_at_every_stage(self, stores):
        """All four stages must actually change, or "compact" is partial.

        A V4 run where only some stages compacted would be a mixture the export
        still stamps as one version.
        """
        v2 = self._prompts(stores, "v2")
        v4 = self._prompts(stores, "v4")
        assert len(v2) == len(v4) == 4
        for i, (a, b) in enumerate(zip(v2, v4)):
            assert a != b, f"stage {i} did not change under v4"
            assert len(b) < len(a), f"stage {i} did not get shorter under v4"
