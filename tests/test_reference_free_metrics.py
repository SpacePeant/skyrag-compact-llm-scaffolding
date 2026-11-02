"""Regression tests for the metrics that replaced an answer key with a measurement.

Cross-reference:
- project_knowledge/task_log/backlog_2026-08-19_events_metric-design-audit.md

Two metrics built after auditing whether the unbuilt specification metrics were
good ideas:

* **`retrieval_score_mean` / `retrieval_score_min`** — did the selected keywords
  pull semantically relevant courses? Asked instead of M1.1's "did the model pick
  the keywords a curator would have picked", which has no non-circular answer.
* **`cited_redundancy`** — O2's redundancy half. Are the recommended courses about
  the same things? Answerable from the corpus's own topic labels.

Both are reference-free by construction, which is the property that made them
worth building: they compare arms without anyone having to author a key first.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from skyrag.config import Config  # noqa: E402
from skyrag.data.schema import Course  # noqa: E402
from skyrag.embedding import EmbeddingStore  # noqa: E402
from skyrag.llm import EchoBackend  # noqa: E402
from skyrag.pipelines import (  # noqa: E402
    BenchmarkMetric,
    export_benchmark_csv,
    run_multi_model_eval,
    summarize_metrics,
)
from skyrag.pipelines.eval import _mean_pairwise_overlap  # noqa: E402

DIM = 8


class FixedEmbedder:
    def encode_one(self, text: str) -> np.ndarray:
        return np.ones(DIM, dtype=np.float32)


def _course(cid: str, *keywords: str) -> Course:
    return Course(id_=cid, platform="coursera", title=cid, keywords=list(keywords))


# --------------------------------------------------------------------------
# O2 -- redundancy of the recommended set
# --------------------------------------------------------------------------

def test_identical_topics_score_total_redundancy():
    courses = [_course("A", "python", "data"), _course("B", "python", "data")]
    assert _mean_pairwise_overlap(courses) == 1.0


def test_disjoint_topics_score_no_redundancy():
    courses = [_course("A", "python"), _course("B", "welding")]
    assert _mean_pairwise_overlap(courses) == 0.0


def test_partial_overlap_is_the_jaccard_of_the_label_sets():
    """`{python, data}` vs `{python, welding}` share 1 of 3 distinct labels."""
    courses = [_course("A", "python", "data"), _course("B", "python", "welding")]
    assert _mean_pairwise_overlap(courses) == pytest.approx(1 / 3, abs=1e-4)


@pytest.mark.parametrize("courses", [
    [],
    [_course("A", "python")],
    [_course("A"), _course("B")],          # present, but carrying no labels
])
def test_redundancy_is_undefined_rather_than_zero_when_it_cannot_be_computed(courses):
    """Zero would claim perfect diversity for a set of one, or for a corpus with
    no topic labels at all. That is the reading that flatters the result."""
    assert _mean_pairwise_overlap(courses) is None


def test_redundancy_ignores_keyword_casing():
    courses = [_course("A", "Python"), _course("B", "python")]
    assert _mean_pairwise_overlap(courses) == 1.0


# --------------------------------------------------------------------------
# The M1.1 replacement -- retrieval yield
# --------------------------------------------------------------------------

def _run(tmp_path, *, responses, courses):
    cfg = Config()
    for attr in ("model", "expansion_model", "keyword_model", "generation_model"):
        setattr(cfg.llm, attr, "stub")
    store = EmbeddingStore(
        courses=list(courses),
        matrix=np.ones((len(courses), DIM), dtype=np.float32),
    )
    metrics = run_multi_model_eval(
        queries=["learn python"],
        models=["arm"],
        pipelines=["naive"],
        backend=EchoBackend(responses=list(responses)),
        config=cfg,
        stores={"coursera": store},
        embedder=FixedEmbedder(),
    )
    path = tmp_path / "b.csv"
    export_benchmark_csv(metrics, path)
    return metrics, list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))


def test_retrieval_yield_reaches_the_export(tmp_path):
    metrics, rows = _run(
        tmp_path,
        responses=["a path citing COU1"] * 8,
        courses=[_course("COU1", "python"), _course("COU2", "python")],
    )
    row = rows[0]
    assert row["retrieval_score_mean"] != ""
    assert row["retrieval_score_min"] != ""
    assert float(row["retrieval_score_min"]) <= float(row["retrieval_score_mean"])


def test_the_minimum_is_reported_because_a_mean_hides_the_tail(tmp_path):
    """A high mean with a low minimum means the model was shown junk alongside
    good material -- and junk is what a generation model cites when it runs out
    of good options."""
    metrics, _ = _run(
        tmp_path,
        responses=["a path"] * 8,
        courses=[_course("COU1", "python"), _course("COU2", "python")],
    )
    assert metrics[0].retrieval_score_min is not None
    assert metrics[0].retrieval_score_min <= metrics[0].retrieval_score_mean


def test_yield_is_absent_when_nothing_was_retrieved():
    """No retrieval means no yield to report. None, not 0.0 -- zero would read as
    a selection that pulled maximally irrelevant material."""
    metric = BenchmarkMetric(
        model="m", pipeline="skyrag", query="q", elapsed_seconds=0.0
    )
    assert metric.retrieval_score_mean is None
    assert metric.retrieval_score_min is None


def test_the_summary_carries_both_reference_free_signals(tmp_path):
    metrics, _ = _run(
        tmp_path,
        responses=["a path citing COU1 and COU2"] * 8,
        courses=[_course("COU1", "python"), _course("COU2", "python")],
    )
    summary = summarize_metrics(metrics)
    assert "retrieval_score_mean" in summary
    assert "cited_redundancy_mean" in summary


def test_redundancy_is_measured_on_the_cited_set_not_the_retrieved_set(tmp_path):
    """O2 asks about the *recommended* set. A retrieval that returned twenty
    near-duplicates is a retrieval finding; recommending five of them is a model
    finding, and only the second is what this column reports."""
    metrics, rows = _run(
        tmp_path,
        responses=["study COU1 then COU1 again"] * 8,
        courses=[
            _course("COU1", "python", "data"),
            _course("COU2", "welding"),
        ],
    )
    # Only COU1 is cited, so redundancy is undefined -- one course, no pairs --
    # even though the retrieved set holds two courses sharing nothing.
    assert metrics[0].cited_count == 1
    assert rows[0]["cited_redundancy"] == ""


def test_a_redundant_recommendation_reaches_the_export_as_a_number(tmp_path):
    """The reversion evidence for `cited_redundancy`.

    Written after the first attempt: the test above asserts an *absent* value, so
    stubbing the computation to `None` satisfied it and the reversion check passed
    against broken code. A test that asserts a blank cannot distinguish "correctly
    undefined" from "never computed" -- the same shape as finding 10 of the
    post-remediation sweep, where a test read a constant back and called it a fix.
    """
    metrics, rows = _run(
        tmp_path,
        responses=["study COU1 and then COU2"] * 8,
        courses=[
            _course("COU1", "python", "data"),
            _course("COU2", "python", "data"),
        ],
    )
    assert metrics[0].cited_count == 2, "both courses must be cited for a pair to exist"
    assert float(rows[0]["cited_redundancy"]) == 1.0, (
        "two courses carrying identical topic labels are maximally redundant"
    )
