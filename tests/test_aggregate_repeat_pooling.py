"""Repeats must be averaged per query before pairing, not pooled.

`Cell.rows` is keyed by `(query_id, repeat)` so CB6 -- repeats silently
discarded -- cannot recur. That key must not reach the paired tests: pairing on
it turns 50 queries x 3 passes into 150 "independent" observations, which
triples n while adding almost no information, shrinking the standard error by
roughly sqrt(3) and making every p smaller than the evidence supports.

The two defects are opposite errors introduced by the same commit: CB6's fix for
under-counting created over-counting one layer up. These tests pin both ends.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "aggregate_arms", Path(__file__).resolve().parents[1] / "scripts" / "aggregate_arms.py"
)
agg = importlib.util.module_from_spec(_SPEC)
sys.modules["aggregate_arms"] = agg
_SPEC.loader.exec_module(agg)


def _cell(label, values):
    """values: {(query_id, repeat): keywords_emitted_and_hallucinated_pair}"""
    rows = {
        (q, r): {
            "query_id": q,
            "repeat": r,
            "keywords_emitted": emitted,
            "hallucinated_keywords": hallucinated,
            "retrieved_count": 10,
            "cited_count": 1,
        }
        for (q, r), (emitted, hallucinated) in values.items()
    }
    return agg.Cell(label, rows)


def test_by_query_averages_passes_rather_than_pooling():
    cell = _cell("c", {
        ("q1", 1): (10, 2),   # 0.2
        ("q1", 2): (10, 4),   # 0.4
        ("q1", 3): (10, 6),   # 0.6
        ("q2", 1): (10, 0),
    })
    got = agg.by_query(cell, agg.itt)
    assert set(got) == {"q1", "q2"}, "one entry per query, not per row"
    assert got["q1"] == pytest.approx(0.4), "three passes averaged, not three entries"
    assert got["q2"] == pytest.approx(0.0)


def test_single_pass_runs_are_unchanged():
    """A mean over one value is that value.

    Every previously published one-repeat figure must be reproduced exactly, or
    this fix silently restates results that were never re-run.
    """
    cell = _cell("c", {("q1", 1): (10, 3), ("q2", 1): (10, 7)})
    assert agg.by_query(cell, agg.itt) == pytest.approx({"q1": 0.3, "q2": 0.7})


def test_undefined_passes_are_skipped_not_zeroed():
    """`invention_both_spoke` is None on a mute.

    Counting a mute as 0.0 would report a perfect score for a query the model
    never answered -- the flattering direction, and the one that matters.
    """
    cell = _cell("c", {
        ("q1", 1): (0, 0),    # mute -> both_spoke None
        ("q1", 2): (10, 8),   # 0.8
        ("q2", 1): (0, 0),    # every pass mute
    })
    got = agg.by_query(cell, agg.both_spoke)
    assert got["q1"] == pytest.approx(0.8), "the mute pass is skipped, not averaged in as 0"
    assert "q2" not in got, "a query with no defined pass drops out of the pairing"


def test_itt_still_scores_a_mute_as_total_failure():
    """CB3: a mute is a failure, not missing data. `by_query` must not soften it."""
    cell = _cell("c", {("q1", 1): (0, 0), ("q1", 2): (10, 0)})
    assert agg.by_query(cell, agg.itt)["q1"] == pytest.approx(0.5), "1.0 and 0.0 averaged"


def test_pairing_is_over_queries_not_rows(capsys):
    """The regression itself: three passes over three queries is n=3, never n=9."""
    # Differences must vary across queries: `tost_paired` refuses zero-variance
    # input rather than reporting a degenerate equivalence, which is correct.
    qs = ("q1", "q2", "q3")
    a = _cell("a", {(q, r): (10, 5) for q in qs for r in (1, 2, 3)})
    b = _cell("b", {(q, r): (10, i + r % 2) for i, q in enumerate(qs) for r in (1, 2, 3)})
    agg.compare(a, b, margin=0.05, provenance="test")
    out = capsys.readouterr().out
    assert "n=3  (of 9 and 9 rows)" in out, out
    assert "repeats averaged per query before pairing: 3 and 3" in out, out
    assert " n=9" not in out, "150-style pooling has returned"


def test_cb6_keying_survives():
    """The other end. Rows must stay keyed by (query_id, repeat) on load.

    If this regresses, repeats are discarded before `by_query` ever sees them and
    the averaging above silently becomes a no-op over one surviving pass.
    """
    src = (Path(__file__).resolve().parents[1] / "scripts" / "aggregate_arms.py").read_text(
        encoding="utf-8"
    )
    assert '(r["query_id"], int(r.get("repeat") or 1))' in src


def _describe_line(capsys, cell, metric="invention_itt"):
    agg.describe(cell)
    out = capsys.readouterr().out
    return next(line for line in out.splitlines() if line.strip().startswith(metric))


def test_describe_bootstraps_queries_not_rows_on_multi_pass_runs(capsys):
    """`describe` is where Table 4's single-arm intervals come from.

    It resampled rows, so a 4-pass run of 50 queries printed n=200 and an
    interval roughly half as wide as the queries support -- the same
    pseudo-replication `by_query` was added to stop in the paired path.
    """
    values = {}
    for i in range(6):
        for rep in (1, 2, 3):
            values[(f"q{i}", rep)] = (10, i)  # identical across passes
    line = _describe_line(capsys, _cell("c", values))
    assert "n=6/6" in line, f"n must count queries, not 18 rows: {line}"


def test_describe_single_pass_output_is_unchanged(capsys):
    line = _describe_line(capsys, _cell("c", {("q1", 1): (10, 3), ("q2", 1): (10, 7)}))
    assert "mean=   0.5000" in line and "n=2/2" in line
