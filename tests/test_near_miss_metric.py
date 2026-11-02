"""Regression tests for M2.2, the near-miss rate.

Cross-reference:
- project_knowledge/task_log/backlog_2026-08-18_dev-task_m2-2-near-miss-metric.md

M2.2 separates "the model had the right concept and botched the transcription"
from "the model did not know the concept". M2.1 conflated the two, and a
capability that cannot be seen in the numbers cannot be reported.

Two properties carry the metric and both are pinned below:

1. **Order.** Near-misses must leave the hallucinated bucket. Computed after
   M2.1 is read, both numbers are wrong at once.
2. **A near-miss is measured, not repaired.** The keyword stays out of
   retrieval. Repairing it would delete the failure mode under study and move
   every downstream number toward the hypothesis.
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
from skyrag.selection import (  # noqa: E402
    normalise_keyword,
    select_keywords,
    split_near_misses,
)

DIM = 8


class FixedEmbedder:
    def encode_one(self, text: str) -> np.ndarray:
        return np.ones(DIM, dtype=np.float32)


def _vocab(*keywords: str) -> Vocabulary:
    counts = {k: 10 for k in keywords}
    return Vocabulary(eligible=dict(counts), corpus=dict(counts), removed={})


# --------------------------------------------------------------------------
# The normalisation ladder
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "emitted,corpus",
    [
        ("machine-learning", "machine learning"),      # punctuation
        ("machine  learning", "machine learning"),     # whitespace collapse
        ("machine/learning", "machine learning"),      # separator
        ("neural networks", "neural network"),         # plural
        ("data structures", "data structure"),         # plural
        ("deep-learning models", "deep learning model"),
    ],
)
def test_the_ladder_folds_a_transcription_difference(emitted, corpus):
    assert normalise_keyword(emitted) == normalise_keyword(corpus)


@pytest.mark.parametrize(
    "left,right",
    [
        ("machine learning", "deep learning"),
        ("python", "pytorch"),
        ("data science", "data analysis"),
    ],
)
def test_the_ladder_does_not_merge_different_concepts(left, right):
    assert normalise_keyword(left) != normalise_keyword(right)


@pytest.mark.parametrize("word", ["business", "access", "process", "wireless"])
def test_a_trailing_double_s_is_not_treated_as_a_plural(word):
    """`business` must not fold to `busines`, which matches nothing and would
    push a real keyword into the invented bucket -- the direction that inflates
    the headline hallucination rate."""
    assert normalise_keyword(word) == word


# --------------------------------------------------------------------------
# The split itself
# --------------------------------------------------------------------------

def test_a_mis_transcribed_keyword_leaves_the_hallucinated_bucket():
    """The ordering property. M2.1 must shed what M2.2 claims."""
    near_miss, still = split_near_misses(
        ["machine-learning", "unicorn studies"], _vocab("machine learning", "python")
    )

    assert near_miss == {"machine-learning": "machine learning"}
    assert still == ["unicorn studies"]


def test_an_invention_stays_an_invention():
    near_miss, still = split_near_misses(["quantum astrology"], _vocab("python"))
    assert near_miss == {}
    assert still == ["quantum astrology"]


def test_the_split_matches_against_the_whole_corpus_not_the_window():
    """A mis-transcription of a cleaned-out or below-cut keyword is still a
    transcription failure. `corpus` is the set of keywords that genuinely exist."""
    vocab = Vocabulary(
        eligible={"python": 10},
        corpus={"python": 10, "amazon web services": 5},
        removed={"amazon web services": 5},
    )
    near_miss, still = split_near_misses(["amazon-web-services"], vocab)

    assert near_miss == {"amazon-web-services": "amazon web services"}
    assert still == []


def test_nothing_hallucinated_means_no_index_is_built():
    """The lazy path. This sits inside the region finding 14 already measured."""
    assert split_near_misses([], _vocab("python")) == ({}, [])


# --------------------------------------------------------------------------
# Through select_keywords, and out to the export
# --------------------------------------------------------------------------

def test_a_near_miss_is_recorded_and_not_used_for_retrieval():
    """The measure-do-not-repair property.

    Silently correcting the spelling would remove the failure mode this
    experiment exists to measure, and would move the retrieved set, the
    citations and the grounding rate all toward the hypothesis.
    """
    backend = EchoBackend(responses=['{"machine-learning": 0.9, "python": 0.8}'])
    selection = select_keywords(
        backend, "learn ml", "coursera", _vocab("machine learning", "python")
    )

    assert selection.near_miss == {"machine-learning": "machine learning"}
    assert selection.hallucinated == []
    # The repaired spelling must NOT have been added to the usable set.
    assert set(selection.keywords) == {"python"}
    assert "machine learning" not in selection.keywords


def test_the_near_miss_count_reaches_the_exported_row(tmp_path):
    """Asserted on the CSV, per the discipline these backlogs established: a unit
    test calling the fixed function directly stayed green in this repository
    while the wiring feeding it was broken, three separate times."""
    cfg = Config()
    for attr in ("model", "expansion_model", "keyword_model", "generation_model"):
        setattr(cfg.llm, attr, "stub")

    course = Course(
        id_="COU1", platform="coursera", title="ML", keywords=["machine learning"]
    )
    metrics = run_multi_model_eval(
        queries=["learn ml"],
        models=["arm-8b"],
        pipelines=["skyrag"],
        backend=EchoBackend(responses=['{"machine-learning": 0.9}'] * 32),
        config=cfg,
        stores={
            "coursera": EmbeddingStore(
                courses=[course], matrix=np.ones((1, DIM), dtype=np.float32)
            )
        },
        embedder=FixedEmbedder(),
    )

    path = tmp_path / "bench.csv"
    export_benchmark_csv(metrics, path)
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))

    assert rows, "the sweep produced no rows"
    assert any(int(row["near_miss_keywords"]) == 1 for row in rows)
    # It must have left M2.1 on the same row it joined M2.2.
    for row in rows:
        if int(row["near_miss_keywords"]):
            assert int(row["hallucinated_keywords"]) == 0


def test_the_summary_totals_near_misses(tmp_path):
    cfg = Config()
    for attr in ("model", "expansion_model", "keyword_model", "generation_model"):
        setattr(cfg.llm, attr, "stub")

    course = Course(
        id_="COU1", platform="coursera", title="ML", keywords=["machine learning"]
    )
    metrics = run_multi_model_eval(
        queries=["learn ml"],
        models=["arm-8b"],
        pipelines=["skyrag"],
        backend=EchoBackend(responses=['{"machine-learning": 0.9}'] * 32),
        config=cfg,
        stores={
            "coursera": EmbeddingStore(
                courses=[course], matrix=np.ones((1, DIM), dtype=np.float32)
            )
        },
        embedder=FixedEmbedder(),
    )

    assert summarize_metrics(metrics)["near_miss_keywords_total"] == 1
