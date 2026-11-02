"""Regression tests for the metrics completed on 2026-08-19.

Cross-reference:
- project_knowledge/research_directions/metric_glossary.md
- project_knowledge/task_log/backlog_2026-08-19_dev-task_metric-completion.md

Five metrics that needed no research decision, only wiring:

* **M2.x denominators** -- the family is specified as rates and was exported as
  counts, so 30 inventions out of 35 emitted and 30 out of 300 were the same
  number in the artifact.
* **M3.2 citation coverage** -- grounding accuracy is silent about how much of the
  context was used; a row citing 1 of 82 retrieved courses scores 1.0.
* **M1.3 distinct keywords** -- breadth, counted across platforms rather than
  summed over them.
* **M1.2 selection rank profile** -- where in the frequency-sorted window the
  model's picks came from. Computable without a reference set, because the window
  order is a property of the corpus.
* **O4 generation-stage duration** -- the only strictly like-for-like slice across
  arms.

Assertions are on the exported CSV wherever the number could be reported.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from skyrag.config import Config  # noqa: E402
from skyrag.data.adapters import Vocabulary  # noqa: E402
from skyrag.data.schema import Course  # noqa: E402
from skyrag.embedding import EmbeddingStore  # noqa: E402
from skyrag.llm import EchoBackend  # noqa: E402
from skyrag.pipelines import (  # noqa: E402
    export_benchmark_csv,
    run_multi_model_eval,
    summarize_metrics,
)
from skyrag.selection import select_keywords  # noqa: E402

DIM = 8


class FixedEmbedder:
    def encode_one(self, text: str) -> np.ndarray:
        return np.ones(DIM, dtype=np.float32)


def _config() -> Config:
    cfg = Config()
    for attr in ("model", "expansion_model", "keyword_model", "generation_model"):
        setattr(cfg.llm, attr, "stub")
    return cfg


def _stores(*platforms: str) -> dict[str, EmbeddingStore]:
    out = {}
    for p in platforms:
        course = Course(
            id_=f"{p[:3].upper()}1", platform=p, title="Py", keywords=["python"]
        )
        out[p] = EmbeddingStore(
            courses=[course], matrix=np.ones((1, DIM), dtype=np.float32)
        )
    return out


def _rows(tmp_path, *, responses, platforms=("coursera",), pipelines=("skyrag",)):
    metrics = run_multi_model_eval(
        queries=["learn python"],
        models=["arm"],
        pipelines=list(pipelines),
        backend=EchoBackend(responses=list(responses)),
        config=_config(),
        stores=_stores(*platforms),
        embedder=FixedEmbedder(),
    )
    path = tmp_path / "bench.csv"
    export_benchmark_csv(metrics, path)
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    return metrics, rows


# --------------------------------------------------------------------------
# M2.x -- the denominator
# --------------------------------------------------------------------------

def test_emitted_counts_every_bucket_a_keyword_can_land_in():
    """Emitted = kept + invented + near-miss + out-of-window + cleaned-out.

    Built by addition because the model's raw reply reaches neither export, so
    this is the only place the total still exists.
    """
    vocab = Vocabulary(
        eligible={"python": 10},
        corpus={"python": 10, "java": 5, "amazon web services": 3},
        removed={"amazon web services": 3},
    )
    backend = EchoBackend(responses=[
        '{"python": 0.9, "javas": 0.8, "amazon-web-services": 0.7, "unicornology": 0.6}'
    ])
    sel = select_keywords(backend, "learn", "coursera", vocab, max_vocabulary=1)

    assert set(sel.keywords) == {"python"}                       # kept
    assert sel.near_miss == {"javas": "java",
                             "amazon-web-services": "amazon web services"}
    assert sel.hallucinated == ["unicornology"]                  # invented
    total = (len(sel.keywords) + len(sel.near_miss) + len(sel.hallucinated)
             + len(sel.out_of_window) + len(sel.removed_by_cleaning))
    assert total == 4, "every emitted keyword must land in exactly one bucket"


def test_the_denominator_reaches_the_exported_row(tmp_path):
    metrics, rows = _rows(
        tmp_path, responses=['{"python": 0.9, "unicornology": 0.5}'] * 32
    )
    row = rows[0]
    assert int(row["keywords_kept"]) == 1
    assert int(row["keywords_emitted"]) == 2
    assert int(row["hallucinated_keywords"]) == 1


def test_the_summary_reports_a_rate_not_only_a_count(tmp_path):
    """The whole point: 1 invention of 2 emitted is 50%, and the count alone
    cannot say so."""
    metrics, _ = _rows(tmp_path, responses=['{"python": 0.9, "unicornology": 0.5}'] * 32)
    summary = summarize_metrics(metrics)

    assert summary["keywords_emitted_total"] == 2
    assert summary["invalid_keyword_rate"] == 0.5


def test_an_undefined_rate_is_none_rather_than_zero(tmp_path):
    """A run that emitted no keywords has no invalid-keyword rate.

    0.0 would read as a model that emitted many and invented none -- the opposite
    finding. `n/a` in the CLI, None in the summary.
    """
    metrics, _ = _rows(tmp_path, responses=["not json at all"] * 32)
    summary = summarize_metrics(metrics)

    assert summary["keywords_emitted_total"] == 0
    assert summary["invalid_keyword_rate"] is None
    assert summary["near_miss_rate"] is None


# --------------------------------------------------------------------------
# M3.2 -- citation coverage
# --------------------------------------------------------------------------

def test_citation_coverage_is_cited_over_retrieved(tmp_path):
    metrics, rows = _rows(
        tmp_path, responses=['{"python": 0.9}'] * 32, pipelines=("naive",)
    )
    row = rows[0]
    retrieved, cited = int(row["retrieved_count"]), int(row["cited_count"])
    assert retrieved > 0
    expected = round(cited / retrieved, 4)
    assert float(row["citation_coverage"]) == pytest.approx(expected)


def test_coverage_is_absent_rather_than_zero_when_nothing_was_retrieved(tmp_path):
    """A ratio with a zero denominator is undefined, not 0.0."""
    metrics, rows = _rows(tmp_path, responses=["unparseable"] * 32)
    assert all(m.retrieved_count == 0 for m in metrics)
    assert all(row["citation_coverage"] == "" for row in rows)


def test_perfect_grounding_on_one_citation_is_visibly_thin(tmp_path):
    """The failure M3.2 exists to expose.

    On 2026-08-18 a live row cited 1 of 82 retrieved courses and scored
    `grounding_accuracy = 1.0`, because inventing nothing out of a single citation
    is arithmetically perfect. Grounding must not be readable alone.
    """
    metrics, rows = _rows(
        tmp_path, responses=['{"python": 0.9}'] * 32, pipelines=("naive",)
    )
    row = rows[0]
    if row["grounding_accuracy"] and float(row["grounding_accuracy"]) == 1.0:
        assert row["citation_coverage"] != "", (
            "a row claiming perfect grounding must also say how much context it used"
        )


# --------------------------------------------------------------------------
# M1.3 -- breadth, distinct across platforms
# --------------------------------------------------------------------------

def test_distinct_keywords_are_counted_across_platforms_not_summed(tmp_path):
    """A keyword chosen for both platforms is one keyword the model thought of.

    Summing would score breadth for what is really platform overlap.
    """
    metrics, rows = _rows(
        tmp_path,
        responses=['{"python": 0.9}'] * 32,
        platforms=("coursera", "edx"),
    )
    row = rows[0]
    assert int(row["keywords_kept"]) == 2, "one per platform, summed"
    assert int(row["distinct_keywords"]) == 1, "but one distinct keyword"


# --------------------------------------------------------------------------
# M1.2 -- selection rank profile
# --------------------------------------------------------------------------

def test_rank_records_the_position_in_the_window_the_model_saw():
    """Rank is knowable without a reference set: the window is frequency-sorted
    and deterministic, so position is a property of the corpus."""
    vocab = Vocabulary(
        eligible={"python": 100, "java": 50, "rust": 10},
        corpus={"python": 100, "java": 50, "rust": 10},
        removed={},
    )
    backend = EchoBackend(responses=['{"python": 0.9, "rust": 0.7}'])
    sel = select_keywords(backend, "learn", "coursera", vocab)

    assert sel.selected_ranks == [0, 2], "python is rank 0, rust is rank 2"


def test_the_rank_profile_reaches_the_export(tmp_path):
    metrics, rows = _rows(tmp_path, responses=['{"python": 0.9}'] * 32)
    row = rows[0]
    assert row["selection_rank_mean"] != ""
    assert int(row["selection_rank_max"]) >= 0


# --------------------------------------------------------------------------
# O4 -- the generation-stage slice
# --------------------------------------------------------------------------

def test_generation_seconds_is_a_slice_of_the_run_not_the_whole_run(tmp_path):
    metrics, rows = _rows(
        tmp_path, responses=['{"python": 0.9}'] * 32, pipelines=("naive",)
    )
    row = rows[0]
    generation = float(row["generation_seconds"])
    total = float(row["elapsed_seconds"])
    assert generation > 0
    assert generation <= total


def test_a_run_that_never_generated_reports_zero_not_a_duration(tmp_path):
    """An empty retrieval aborts before the generation call. Zero is the honest
    value: the stage did not run."""
    metrics, rows = _rows(tmp_path, responses=["unparseable"] * 32)
    assert all(m.status == "empty_retrieval" for m in metrics)
    assert all(float(row["generation_seconds"]) == 0.0 for row in rows)
