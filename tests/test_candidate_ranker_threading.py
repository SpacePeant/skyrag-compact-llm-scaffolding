"""C1-C3: `--candidate-ranker` from the CLI flag down to the provenance column.

`tests/test_candidate_ranker_strategies.py` covers what each ranker *does*.
This file covers whether the choice survives the trip from the command line to
the ranker and back out into the artifact -- the part that, if it silently
failed, would produce a Run C whose rows claim a ranker that never ran.

Coverage:
  1. CLI: the flag exists, is constrained to the three strategies, and
     defaults to lexical.
  2. Fail-fast: an unknown ranker, and a non-lexical ranker on a sweep where
     nothing injects, are both refused before the first API call.
  3. Threading: `select_keywords` hands the strategy and the embedder to the
     ranker, and only when injection is on.
  4. Provenance: `candidate_ranker` is stamped on injected rows, blank on
     rows that ranked nothing, and survives into the CSV.
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
    RANKER_DENSE,
    RANKER_HYBRID,
    RANKER_LEXICAL,
    RANKER_STRATEGIES,
    select_keywords,
)


class _MockEmbedder:
    """Accepts `show_progress=` because the semantic ranker passes it."""

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
    return [
        "Expanded: learn Python",
        "Step back: programming basics",
        "python\nprogramming",
        "A good Python course is COU1.",
    ]


# ---------------------------------------------------------------------------
# 1. CLI surface
# ---------------------------------------------------------------------------

class TestCliFlag:
    def test_help_lists_every_strategy(self, capsys):
        from skyrag.cli import run_eval_cli

        with pytest.raises(SystemExit):
            run_eval_cli(["--help"])
        out = capsys.readouterr().out
        for strategy in RANKER_STRATEGIES:
            assert strategy in out, f"--candidate-ranker help omits {strategy!r}"

    def test_rejects_unknown_strategy(self, capsys):
        from skyrag.cli import run_eval_cli

        with pytest.raises(SystemExit):
            run_eval_cli(["--candidate-ranker", "bm25"])
        err = capsys.readouterr().err
        assert "bm25" in err

    def test_default_is_lexical(self):
        """Every published cell ran the lexical ranker; the default must not move."""
        assert (
            inspect.signature(run_multi_model_eval).parameters["candidate_ranker"].default
            == RANKER_LEXICAL
        )


# ---------------------------------------------------------------------------
# 2. Fail-fast in run_benchmark
# ---------------------------------------------------------------------------

class TestFailFast:
    def _run(self, stores, **kwargs):
        return run_multi_model_eval(
            queries=["Learn Python"],
            models=["placeholder-ignored"],
            pipelines=["skyrag"],
            backend=EchoBackend(responses=_echo_responses()),
            config=Config.load(),
            stores=stores,
            embedder=_MockEmbedder(),
            **kwargs,
        )

    def test_unknown_ranker_raises(self, sample_stores):
        with pytest.raises(ValueError, match="unknown candidate ranker"):
            self._run(sample_stores, arms=["2"], candidate_ranker="bm25")

    def test_ranker_without_injection_raises(self, sample_stores):
        """A ranker nobody runs stamps blank and looks like it was never asked for."""
        with pytest.raises(ValueError, match="no cell"):
            self._run(sample_stores, arms=["2"], candidate_ranker=RANKER_DENSE)

    def test_lexical_without_injection_is_fine(self, sample_stores):
        """The default must not become a flag you are forced to think about."""
        metrics = self._run(sample_stores, arms=["2"], candidate_ranker=RANKER_LEXICAL)
        assert len(metrics) == 1


# ---------------------------------------------------------------------------
# 3. Threading into the ranker
# ---------------------------------------------------------------------------

class TestSelectKeywordsThreading:
    def _vocab(self):
        return {"python": 5, "programming": 3, "databases": 2}

    def test_strategy_and_embedder_reach_the_ranker(self, monkeypatch):
        seen = {}

        import skyrag.selection as sel

        def _spy(query, eligible, top_k=100, *, strategy=RANKER_LEXICAL, embedder=None):
            seen["strategy"] = strategy
            seen["embedder"] = embedder
            return {"python": 5}

        monkeypatch.setattr(sel, "rank_candidate_keywords", _spy)
        embedder = _MockEmbedder()
        select_keywords(
            EchoBackend(responses=["python"]),
            "Learn Python",
            "coursera",
            self._vocab(),
            candidate_injection=True,
            candidate_ranker=RANKER_HYBRID,
            embedder=embedder,
        )
        assert seen["strategy"] == RANKER_HYBRID
        assert seen["embedder"] is embedder

    def test_ranker_not_consulted_without_injection(self, monkeypatch):
        import skyrag.selection as sel

        def _boom(*a, **k):
            raise AssertionError("ranker ran with candidate_injection=False")

        monkeypatch.setattr(sel, "rank_candidate_keywords", _boom)
        select_keywords(
            EchoBackend(responses=["python"]),
            "Learn Python",
            "coursera",
            self._vocab(),
            candidate_injection=False,
            candidate_ranker=RANKER_DENSE,
        )

    def test_dense_without_embedder_raises_not_falls_back(self):
        """A silent fallback would export a stamp its own numbers contradict."""
        with pytest.raises(ValueError, match="needs an embedder"):
            select_keywords(
                EchoBackend(responses=["python"]),
                "Learn Python",
                "coursera",
                self._vocab(),
                candidate_injection=True,
                candidate_ranker=RANKER_DENSE,
                embedder=None,
            )


# ---------------------------------------------------------------------------
# 4. Provenance column
# ---------------------------------------------------------------------------

class TestProvenanceColumn:
    def test_field_exists_and_defaults_blank(self):
        """Blank, not 'lexical': every pre-column run is unrecorded, not lexical."""
        m = BenchmarkMetric(model="m", pipeline="skyrag", query="q", elapsed_seconds=0.0)
        assert m.candidate_ranker == ""

    def _run(self, stores, **kwargs):
        return run_multi_model_eval(
            queries=["Learn Python"],
            models=["placeholder-ignored"],
            pipelines=["skyrag"],
            backend=EchoBackend(responses=_echo_responses()),
            config=Config.load(),
            stores=stores,
            embedder=_MockEmbedder(),
            arms=["2"],
            **kwargs,
        )

    def test_stamped_when_injection_ran(self, sample_stores):
        metrics = self._run(
            sample_stores,
            candidate_injection=True,
            candidate_ranker=RANKER_HYBRID,
        )
        assert metrics[0].candidate_ranker == RANKER_HYBRID
        assert metrics[0].candidate_injection is True

    def test_blank_when_nothing_ranked(self, sample_stores):
        metrics = self._run(sample_stores, candidate_ranker=RANKER_LEXICAL)
        assert metrics[0].candidate_ranker == "", (
            "a row that ranked nothing must not claim a ranker"
        )

    def test_survives_into_csv(self, sample_stores):
        metrics = self._run(
            sample_stores,
            candidate_injection=True,
            candidate_ranker=RANKER_HYBRID,
        )
        rows = list(csv.DictReader(io.StringIO(export_benchmark_csv(metrics))))
        assert "candidate_ranker" in rows[0]
        assert rows[0]["candidate_ranker"] == RANKER_HYBRID
