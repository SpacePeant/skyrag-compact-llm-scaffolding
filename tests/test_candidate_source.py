"""`--candidate-source` from the CLI flag to the ranker and back out into the artifact.

The 2026-10-21 pre-submission review (A7): the injected candidates were ranked against
the Stage 1 rewrite, so a different rewrite also changed the candidate list.
`--candidate-source query` ranks against the learner's original query instead,
holding the list fixed while the selector still reads the rewrite. If that
silently fell back to the rewrite, the A7 cells would claim a fixed list they
never had, so this file checks every hop.

Coverage:
  1. CLI: the flag exists, defaults to `rewrite`, refuses anything else.
  2. Fail-fast: an unknown source, and `query` on a sweep where nothing
     injects, are refused before the first API call.
  3. Threading: `select_keywords` ranks against `candidate_query` when given
     and still shows the selector `query`; `run_skyrag` passes the original
     query under `query` and the rewrite under `rewrite`.
  4. Provenance: stamped on injected rows, blank on rows that ranked nothing,
     and present in the CSV.
"""

from __future__ import annotations

import csv
import inspect
import io

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
from skyrag.selection import (
    CANDIDATE_SOURCE_QUERY,
    CANDIDATE_SOURCE_REWRITE,
    RANKER_LEXICAL,
    select_keywords,
)

QUERY = "Learn Python"
REWRITE = "Expanded: learn Python for data analysis with pandas"


class _MockEmbedder:
    def encode_one(self, text: str) -> np.ndarray:
        v = np.ones(384, dtype=np.float32)
        return v / np.linalg.norm(v)

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return np.vstack([self.encode_one(t) for t in texts])


@pytest.fixture
def sample_stores():
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


def _echo_responses():
    return [REWRITE, "Step back: programming basics", "python\nprogramming", "A good Python course is COU1."]


def _run(stores, backend=None, **kwargs):
    return run_multi_model_eval(
        queries=[QUERY],
        models=["placeholder-ignored"],
        pipelines=["skyrag"],
        backend=backend or EchoBackend(responses=_echo_responses()),
        config=Config.load(),
        stores=stores,
        embedder=_MockEmbedder(),
        arms=["2"],
        **kwargs,
    )


class TestCliFlag:
    def test_rejects_unknown_source(self, capsys):
        from skyrag.cli import run_eval_cli

        with pytest.raises(SystemExit):
            run_eval_cli(["--candidate-source", "step_back"])
        assert "step_back" in capsys.readouterr().err

    def test_default_is_the_rewrite(self):
        """Every published cell ranked against the rewrite; the default must not move."""
        assert (
            inspect.signature(run_multi_model_eval).parameters["candidate_source"].default
            == CANDIDATE_SOURCE_REWRITE
        )


class TestFailFast:
    def test_unknown_source_raises(self, sample_stores):
        with pytest.raises(ValueError, match="unknown candidate source"):
            _run(sample_stores, candidate_injection=True, candidate_source="step_back")

    def test_query_source_without_injection_raises(self, sample_stores):
        with pytest.raises(ValueError, match="no cell"):
            _run(sample_stores, candidate_source=CANDIDATE_SOURCE_QUERY)


class TestThreading:
    def _spy(self, monkeypatch, seen):
        import skyrag.selection as sel

        def _rank(query, eligible, top_k=100, *, strategy=RANKER_LEXICAL, embedder=None):
            seen.append(query)
            return {"python": 5}

        monkeypatch.setattr(sel, "rank_candidate_keywords", _rank)

    def test_select_keywords_ranks_against_candidate_query(self, monkeypatch):
        seen: list[str] = []
        self._spy(monkeypatch, seen)
        backend = EchoBackend(responses=["python"])
        select_keywords(
            backend,
            REWRITE,
            "coursera",
            {"python": 5, "programming": 3},
            candidate_injection=True,
            candidate_query=QUERY,
        )
        assert seen == [QUERY]
        # The selector itself still reads the rewrite.
        assert REWRITE in backend.calls[-1]["prompt"]

    @pytest.mark.parametrize(
        "source, expected", [(CANDIDATE_SOURCE_QUERY, QUERY), (CANDIDATE_SOURCE_REWRITE, REWRITE)]
    )
    def test_pipeline_passes_the_right_text(self, monkeypatch, sample_stores, source, expected):
        seen: list[str] = []
        self._spy(monkeypatch, seen)
        _run(sample_stores, candidate_injection=True, candidate_source=source)
        assert seen, "the ranker never ran"
        assert set(seen) == {expected}


class TestProvenanceColumn:
    def test_field_defaults_blank(self):
        """Blank, not 'rewrite': every earlier run is unrecorded."""
        m = BenchmarkMetric(model="m", pipeline="skyrag", query="q", elapsed_seconds=0.0)
        assert m.candidate_source == ""

    def test_stamped_when_injection_ran(self, sample_stores):
        metrics = _run(sample_stores, candidate_injection=True, candidate_source=CANDIDATE_SOURCE_QUERY)
        assert metrics[0].candidate_source == CANDIDATE_SOURCE_QUERY

    def test_blank_when_nothing_ranked(self, sample_stores):
        metrics = _run(sample_stores)
        assert metrics[0].candidate_source == ""

    def test_survives_into_csv(self, sample_stores):
        metrics = _run(sample_stores, candidate_injection=True, candidate_source=CANDIDATE_SOURCE_QUERY)
        rows = list(csv.DictReader(io.StringIO(export_benchmark_csv(metrics))))
        assert rows[0]["candidate_source"] == CANDIDATE_SOURCE_QUERY
