"""`expansion_mode`: the Stage 1 rewrite, set apart from the rest of the arm.

Arm 3's schema switches every stage at once, and the rewrite's `expanded_query`
is what keyword selection reads. The 5090 2x2 (2026-10-01) put the schema arm's
invention reduction in that rewrite, not in the selection schema, so the
harness needs to run the rewrite on its own -- and to run Arm 3's rewrite
*wording* without its schema, because the two prompts differ in both.

Coverage:
  1. Defaults: no override reproduces each arm's own rewrite exactly.
  2. Overrides: each mode sends its prompt and changes nothing after Stage 1.
  3. Refusals: an unknown mode, and `schema_wording` outside the V4 family,
     raise -- in the pipeline and in the sweep before the first call.
  4. Export: the mode and the rewrite text reach the row and survive the CSV.
"""

from __future__ import annotations

import csv
import io
import json

import numpy as np
import pytest

from skyrag.config import Config
from skyrag.data.schema import Course
from skyrag.embedding import EmbeddingStore
from skyrag.llm.echo import EchoBackend
from skyrag.pipelines.eval import (
    BenchmarkMetric,
    export_benchmark_csv,
    run_multi_model_eval,
)
from skyrag.pipelines.skyrag import (
    EXPANSION_PLAIN,
    EXPANSION_SCHEMA,
    EXPANSION_SCHEMA_WORDING,
    run_skyrag,
)
from skyrag.prompts import (
    EXPAND_QUERY,
    EXPAND_QUERY_SCHEMA_WORDING,
    STEP_BACK,
)


class _MockEmbedder:
    def encode_one(self, text: str) -> np.ndarray:
        v = np.ones(384, dtype=np.float32)
        return v / np.linalg.norm(v)

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
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


SCHEMA_EXPANSION = json.dumps({
    "expansions": ["learn python", "python basics"],
    "expanded_query": "An introductory Python programming course.",
})


def _run(stores, first_reply, **kwargs):
    backend = EchoBackend(responses=[
        first_reply,
        "Step back: programming basics",
        "python\nprogramming",
        "A good Python course is COU1.",
    ])
    res = run_skyrag(
        query="Learn Python",
        stores=stores,
        backend=backend,
        config=Config.load(),
        embedder=_MockEmbedder(),
        **kwargs,
    )
    return res, backend


# ---------------------------------------------------------------------------
# 1. Defaults are the arms as they were
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_arm2_rewrites_plain(self, stores):
        res, backend = _run(stores, "Expanded: learn Python", arm="2")
        assert res.expansion_mode == EXPANSION_PLAIN
        assert backend.calls[0]["prompt"] == EXPAND_QUERY.format(query="Learn Python")

    def test_arm3_rewrites_under_its_schema(self, stores):
        res, backend = _run(stores, SCHEMA_EXPANSION, arm="3", sweep_version="v5")
        assert res.expansion_mode == EXPANSION_SCHEMA
        assert "Return ONLY JSON matching the schema" in backend.calls[0]["prompt"]
        assert res.expanded_query == "An introductory Python programming course."

    def test_no_rewrite_stamps_no_mode(self, stores):
        """A cell that skipped Stage 1 must not claim a rewrite it never ran."""
        res, _ = _run(stores, "python\nprogramming", arm="2", expand_query=False)
        assert res.expansion_mode == ""


# ---------------------------------------------------------------------------
# 2. Overrides touch Stage 1 and nothing after it
# ---------------------------------------------------------------------------

class TestOverrides:
    def test_arm2_with_the_schema_rewrite(self, stores):
        res, backend = _run(
            stores, SCHEMA_EXPANSION, arm="2", expansion_mode=EXPANSION_SCHEMA
        )
        assert res.expansion_mode == EXPANSION_SCHEMA
        assert "Schema:" in backend.calls[0]["prompt"]
        # The validated field reaches selection, exactly as on Arm 3.
        assert res.expanded_query == "An introductory Python programming course."
        # Everything after Stage 1 is still Arm 2.
        assert backend.calls[1]["prompt"] == STEP_BACK.format(query="Learn Python")
        assert "copied_verbatim" not in backend.calls[2]["prompt"]

    def test_arm3_with_the_plain_rewrite(self, stores):
        res, backend = _run(
            stores, "Expanded: learn Python", arm="3", expansion_mode=EXPANSION_PLAIN
        )
        assert res.expansion_mode == EXPANSION_PLAIN
        assert backend.calls[0]["prompt"] == EXPAND_QUERY.format(query="Learn Python")
        # A plain reply is not judged against a schema it was never shown.
        assert "query_expansion" not in res.schema_violation_stages
        # Selection keeps Arm 3's schema.
        assert "copied_verbatim" in backend.calls[2]["prompt"]

    def test_schema_wording_sends_the_v4_instructions_without_a_schema(self, stores):
        reply = "A beginner course in Python. No prerequisites. About four weeks."
        res, backend = _run(
            stores, reply, arm="2", sweep_version="v5",
            expansion_mode=EXPANSION_SCHEMA_WORDING,
        )
        prompt = backend.calls[0]["prompt"]
        assert prompt == EXPAND_QUERY_SCHEMA_WORDING.format(query="Learn Python")
        assert "Schema:" not in prompt and "JSON" not in prompt
        assert res.expansion_mode == EXPANSION_SCHEMA_WORDING
        assert res.expanded_query == reply
        assert res.schema_violations == 0

    def test_schema_wording_shares_the_schema_prompts_instruction(self):
        """The control is only a control while the wording matches.

        If someone edits EXPAND_QUERY_SCHEMA_V4's instruction and not this one,
        the two cells differ by more than the schema and the contrast no longer
        isolates it.
        """
        from skyrag.prompts import EXPAND_QUERY_SCHEMA_V4

        assert "3-sentence expanded query specifying prerequisite level and duration" in (
            EXPAND_QUERY_SCHEMA_V4
        )
        assert "3-sentence expanded query specifying prerequisite level and duration" in (
            EXPAND_QUERY_SCHEMA_WORDING
        )


# ---------------------------------------------------------------------------
# 3. Refusals
# ---------------------------------------------------------------------------

class TestRefusals:
    def test_unknown_mode_raises(self, stores):
        with pytest.raises(ValueError, match="unknown expansion mode"):
            _run(stores, "x", arm="2", expansion_mode="terse")

    def test_schema_wording_outside_the_v4_family_raises(self, stores):
        with pytest.raises(ValueError, match="defined only for sweep versions"):
            _run(stores, "x", arm="2", sweep_version="v3",
                 expansion_mode=EXPANSION_SCHEMA_WORDING)

    def test_the_sweep_refuses_before_the_first_call(self, stores):
        backend = EchoBackend(responses=["x"])
        with pytest.raises(ValueError, match="defined only for sweep versions"):
            run_multi_model_eval(
                queries=["Learn Python"],
                models=["placeholder-ignored"],
                pipelines=["skyrag"],
                backend=backend,
                config=Config.load(),
                stores=stores,
                embedder=_MockEmbedder(),
                arms=["2"],
                sweep_version="v3",
                expansion_mode=EXPANSION_SCHEMA_WORDING,
            )
        assert backend.calls == []

    def test_the_cli_constrains_the_flag(self, capsys):
        from skyrag.cli import run_eval_cli

        with pytest.raises(SystemExit):
            run_eval_cli(["--expansion-mode", "terse"])
        assert "terse" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 4. Export
# ---------------------------------------------------------------------------

class TestExport:
    def _sweep(self, stores, **kwargs):
        return run_multi_model_eval(
            queries=["Learn Python"],
            models=["placeholder-ignored"],
            pipelines=["skyrag"],
            backend=EchoBackend(responses=[
                SCHEMA_EXPANSION,
                "Step back: programming basics",
                "python\nprogramming",
                "A good Python course is COU1.",
            ]),
            config=Config.load(),
            stores=stores,
            embedder=_MockEmbedder(),
            arms=["2"],
            **kwargs,
        )

    def test_the_mode_and_the_rewrite_reach_the_row(self, stores):
        (row,) = self._sweep(stores, expansion_mode=EXPANSION_SCHEMA)
        assert row.expansion_mode == EXPANSION_SCHEMA
        assert row.expanded_query == "An introductory Python programming course."
        assert row.arm == "2"

    def test_the_default_is_stamped_too(self, stores):
        """A decomposition cell must never pool with its arm's ordinary rows."""
        (row,) = self._sweep(stores)
        assert row.expansion_mode == EXPANSION_PLAIN

    def test_a_multiline_rewrite_survives_the_csv(self):
        """Same lossy idiom `learning_path` is encoded against."""
        text = "Line one.\nLine two, with a comma."
        row = BenchmarkMetric(
            model="m", pipeline="skyrag", query="q", elapsed_seconds=0.0,
            expansion_mode=EXPANSION_PLAIN, expanded_query=text,
        )
        csv_text = export_benchmark_csv([row])
        (parsed,) = list(csv.DictReader(io.StringIO(csv_text).read().splitlines()))
        assert parsed["expansion_mode"] == EXPANSION_PLAIN
        assert json.loads(parsed["expanded_query"]) == text
