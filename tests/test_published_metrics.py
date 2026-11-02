"""Tests for the metrics that have a published source, in `skyrag/metrics.py`.

Cross-reference:
- project_knowledge/research_directions/metric_literature_provenance.md
- project_knowledge/research_directions/research_backed_metrics_recommendations.md

Three properties are worth more than the arithmetic here, and each has a test
that fails loudly if it stops holding:

* **The ILS equivalence.** `eval.py::_mean_pairwise_overlap` and
  `metrics.label_ils` are the same computation reached from two directions --
  one designed here, one restated from Ziegler 2005. If they ever disagree, the
  claim that our `cited_redundancy` *is* ILS stops being true, and that claim is
  going into the paper.

* **The degeneracy is really avoided.** nDCG graded by the retriever's own cosine
  over the retriever's own sort is 1.0 for every input. `test_ndcg_over_a_sorted
  _list_is_degenerate` asserts that directly, so the reason the presented order
  is used is pinned in the suite rather than only in a comment.

* **Undefined is None, not zero.** Every metric here has a case where it cannot
  be computed, and 0.0 is a meaningful value in all of them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from skyrag.data.schema import Course  # noqa: E402
from skyrag.metrics import (  # noqa: E402
    cosine_similarity,
    dcg_at_k,
    embedding_ils,
    intra_list_similarity,
    jaccard,
    label_ils,
    ndcg_at_k,
    reciprocal_rank_of_ideal_top,
)
from skyrag.pipelines.eval import _mean_pairwise_overlap  # noqa: E402

# --------------------------------------------------------------------------
# Intra-List Similarity -- Ziegler et al., WWW 2005
# --------------------------------------------------------------------------


def test_jaccard_matches_the_definition():
    assert jaccard({"a", "b"}, {"a", "c"}) == pytest.approx(1 / 3)
    assert jaccard({"a"}, {"a"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_of_two_empty_sets_is_none_not_zero():
    """Zero would claim two unlabelled courses are maximally diverse."""
    assert jaccard(set(), set()) is None


def test_ils_below_two_items_is_none():
    assert label_ils([{"a"}]) is None
    assert label_ils([]) is None
    assert embedding_ils([np.array([1.0, 0.0])]) is None


def test_ils_of_identical_lists_is_one():
    assert label_ils([{"a", "b"}, {"a", "b"}, {"a", "b"}]) == 1.0


def test_ils_of_disjoint_lists_is_zero():
    """0.0 is a real measurement here and must not be confused with None."""
    assert label_ils([{"a"}, {"b"}, {"c"}]) == 0.0


def test_ils_skips_unscorable_pairs_rather_than_counting_them_as_zero():
    """A pair the similarity cannot score must not drag the mean toward 'diverse'.

    Note what is *not* unscorable: `jaccard({"a"}, set())` is a real 0.0 -- one
    course has labels, they share none. Only two label-less courses are
    unanswerable, and `label_ils` drops those before pairing. This exercises the
    `None` path in the generic function directly.
    """
    scored = intra_list_similarity(
        [{"a"}, {"a"}, "unscorable"],
        lambda x, y: None if "unscorable" in (x, y) else jaccard(x, y),
    )
    # The two real sets are identical; the unscorable item contributes no pair.
    assert scored == 1.0


def test_label_ils_drops_labelless_courses_before_pairing():
    """A course with no labels is unmeasurable, not maximally diverse.

    Pairing it at 0.0 would let an unlabelled catalogue entry manufacture
    diversity in a recommendation set that has none.
    """
    assert label_ils([{"a", "b"}, {"a", "b"}, set()]) == 1.0


def test_label_ils_agrees_with_the_existing_cited_redundancy():
    """The equivalence the ILS citation rests on. Pinned so it cannot drift.

    `_mean_pairwise_overlap` takes courses and lowercases their keywords;
    `label_ils` takes the label sets directly. Same numbers out.
    """
    courses = [
        Course(id_="C1", title="a", platform="coursera", keywords=["Python", "Data"]),
        Course(id_="C2", title="b", platform="coursera", keywords=["python", "ML"]),
        Course(id_="C3", title="c", platform="edx", keywords=["Data", "ML"]),
    ]
    from_eval = _mean_pairwise_overlap(courses)
    from_published = label_ils(
        [{k.strip().lower() for k in c.keywords if k.strip()} for c in courses]
    )
    assert from_eval == from_published
    assert from_eval is not None


def test_cosine_similarity_of_a_zero_vector_is_none():
    """An absent measurement, not an orthogonal one."""
    assert cosine_similarity(np.zeros(3), np.array([1.0, 0.0, 0.0])) is None


def test_cosine_similarity_ignores_magnitude():
    a = np.array([1.0, 0.0])
    assert cosine_similarity(a, a * 7.0) == pytest.approx(1.0)


def test_embedding_ils_sees_duplicates_that_labels_miss():
    """The reason the cosine half exists at all.

    Two courses with disjoint label sets but near-identical embeddings are a
    near-duplicate recommendation. Jaccard scores them perfectly diverse.
    """
    labels = [{"neural networks"}, {"deep learning"}]
    vectors = [np.array([1.0, 0.0, 0.0]), np.array([0.99, 0.14, 0.0])]

    assert label_ils(labels) == 0.0
    cosine = embedding_ils(vectors)
    assert cosine is not None and cosine > 0.9


def test_embedding_ils_drops_vectorless_courses():
    vectors = [np.array([1.0, 0.0]), None, np.array([1.0, 0.0])]
    assert embedding_ils(vectors) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# nDCG -- Jarvelin & Kekalainen, TOIS 2002
# --------------------------------------------------------------------------


def test_dcg_matches_a_hand_computed_value():
    # gains 3, 2 at ranks 1, 2 -> 3/log2(2) + 2/log2(3)
    assert dcg_at_k([3.0, 2.0]) == pytest.approx(3.0 + 2.0 / np.log2(3.0))


def test_dcg_clamps_negative_gains():
    """A negative cosine must not subtract from accumulated gain."""
    assert dcg_at_k([1.0, -5.0]) == pytest.approx(1.0)


def test_ndcg_over_a_sorted_list_is_degenerate():
    """The form that was refused, asserted so the refusal stays justified.

    Grading the retriever's own ranking with the retriever's own similarity
    compares a list against itself. This is why the exported column scores the
    *presented* order instead -- see the module docstring.
    """
    for scores in ([0.9, 0.8, 0.7, 0.6], [0.5, 0.4], [1.0, 0.0, 0.0]):
        assert ndcg_at_k(sorted(scores, reverse=True), k=10) == 1.0


def test_ndcg_falls_when_a_strong_item_is_pushed_down():
    ideal = ndcg_at_k([0.9, 0.8, 0.1], k=10)
    disturbed = ndcg_at_k([0.1, 0.8, 0.9], k=10)
    assert ideal == 1.0
    assert disturbed is not None and disturbed < ideal


def test_ndcg_detects_the_round_robin_interleave():
    """The case the exported column exists to measure.

    Two platforms, one scoring uniformly higher. Round-robin alternation inserts
    the weaker platform's courses between the stronger platform's, which is a
    real departure from the ideal ordering and the thing that varies with
    keyword selection.
    """
    strong, weak = [0.9, 0.85, 0.8], [0.4, 0.35, 0.3]
    interleaved = [strong[0], weak[0], strong[1], weak[1], strong[2], weak[2]]

    score = ndcg_at_k(interleaved, k=10)
    assert score is not None
    assert score < 1.0
    assert ndcg_at_k(sorted(interleaved, reverse=True), k=10) == 1.0


def test_ndcg_of_an_empty_list_is_none():
    assert ndcg_at_k([], k=10) is None


def test_ndcg_is_none_when_nothing_has_positive_gain():
    """No ideal to normalise against. Undefined, not perfect."""
    assert ndcg_at_k([0.0, 0.0, -0.2], k=10) is None


def test_ndcg_at_k_only_reads_the_first_k():
    """A change beyond the cutoff must not move nDCG@k."""
    head = [0.9, 0.8]
    assert ndcg_at_k(head + [0.1, 0.2], k=2) == ndcg_at_k(head + [0.2, 0.1], k=2)


# --------------------------------------------------------------------------
# Reciprocal rank -- Voorhees, TREC-8 1999
# --------------------------------------------------------------------------


def test_reciprocal_rank_is_one_when_the_best_course_leads():
    scores = [0.99] + [0.1] * 20
    assert reciprocal_rank_of_ideal_top(scores, top_n=10) == 1.0


def test_reciprocal_rank_falls_when_the_anchor_is_buried():
    """Nine weak courses, then the whole ideal top 10. First hit at rank 10."""
    scores = [0.01] * 9 + [0.9] * 10 + [0.02] * 5
    assert reciprocal_rank_of_ideal_top(scores, top_n=10) == pytest.approx(0.1)


def test_reciprocal_rank_is_none_when_the_list_is_too_short_to_mean_anything():
    """At or below top_n every course is in its own ideal top_n.

    The metric would report a constant 1.0 while measuring nothing, which is the
    degeneracy this module exists to avoid.
    """
    assert reciprocal_rank_of_ideal_top([0.5] * 10, top_n=10) is None
    assert reciprocal_rank_of_ideal_top([0.5] * 3, top_n=10) is None
    assert reciprocal_rank_of_ideal_top([], top_n=10) is None


def test_reciprocal_rank_counts_boundary_ties_as_relevant():
    """Threshold, not index set: identical scores must be treated identically."""
    # Eleven courses; the top-10 threshold value also appears at position 1.
    scores = [0.5, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.5]
    assert reciprocal_rank_of_ideal_top(scores, top_n=10) == 1.0
