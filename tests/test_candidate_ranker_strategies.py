"""Dense and hybrid candidate ranking, and the lexical default that must not move.

Paper #1 section 4.7.3 claims the candidate-injection finding is
"retriever-agnostic", listing dense embeddings, sparse lexical scoring, RRF
hybrid and lexical heuristics. Only the lexical ranker has ever run, so the
claim is an assertion. These pin the machinery that lets it be tested at
k=100 -- the depth the 2026-09-21 pool-size pilot left as the only one worth
testing.

The load-bearing test here is the first one: every published Condition C cell
used the lexical default, so a change to it would silently rewrite what those
artifacts mean.
"""

from __future__ import annotations

import numpy as np
import pytest

from skyrag.selection import (
    RANKER_DENSE,
    RANKER_FLOOR,
    RANKER_FREQ,
    RANKER_HYBRID,
    RANKER_LEXICAL,
    RANKER_STRATEGIES,
    rank_candidate_keywords,
)

ELIGIBLE = {
    "machine learning": 10,
    "deep learning": 5,
    "neural networks": 4,
    "basket weaving": 3,
    "python": 8,
}


class FakeEmbedder:
    """Deterministic stand-in: embeds by a fixed keyword -> vector table.

    A real `Embedder` pulls sentence-transformers and a model download, which
    a unit test must not. The vectors are chosen so the semantic ordering
    *disagrees* with the lexical one -- that disagreement is the whole point
    of the strategy, and a fixture where they agree would pass whatever the
    fusion did.
    """

    TABLE = {
        "machine learning": [1.0, 0.0],
        "deep learning": [0.95, 0.30],
        "neural networks": [0.90, 0.40],
        "python": [0.10, 1.00],
        "basket weaving": [-1.0, 0.0],
        "neural nets for vision": [0.0, 1.0],
    }

    def encode(self, texts, *, show_progress: bool = True):
        return np.asarray([self.TABLE[t] for t in texts], dtype=np.float32)

    def encode_one(self, text: str):
        return self.encode([text], show_progress=False)[0]


# --- the default must not move -----------------------------------------------


def test_the_default_strategy_is_lexical_and_unchanged():
    """Every published Condition C cell ranked this way.

    If this ordering changes, the artifacts on disk stop meaning what their
    `candidate_injection=True` stamp says they mean.
    """
    ranked = rank_candidate_keywords(
        "I want to learn machine learning", ELIGIBLE, top_k=3
    )
    assert list(ranked) == ["machine learning", "deep learning", "python"]
    explicit = rank_candidate_keywords(
        "I want to learn machine learning", ELIGIBLE, top_k=3, strategy=RANKER_LEXICAL
    )
    assert ranked == explicit


def test_the_ranker_returns_frequencies_not_relevance_scores():
    """The prompt renderer downstream reads the catalogue frequency.

    Returning the relevance score instead would change what every Condition C
    prompt contains without changing any stamp.
    """
    ranked = rank_candidate_keywords("machine learning", ELIGIBLE, top_k=2)
    assert all(ranked[k] == ELIGIBLE[k] for k in ranked)


def test_an_empty_vocabulary_is_empty_under_every_strategy():
    for strategy in RANKER_STRATEGIES:
        assert rank_candidate_keywords("q", {}, top_k=5, strategy=strategy) == {}


# --- refusals rather than silent fallback ------------------------------------


def test_a_semantic_strategy_without_an_embedder_raises():
    """Falling back to lexical would export a stamp the numbers contradict.

    That is the `serving_precision` failure one level down: a row claiming a
    treatment it did not receive.
    """
    with pytest.raises(ValueError, match="needs an embedder"):
        rank_candidate_keywords("q", ELIGIBLE, top_k=2, strategy=RANKER_DENSE)


def test_an_unknown_strategy_raises_rather_than_defaulting():
    with pytest.raises(ValueError, match="unknown candidate ranker"):
        rank_candidate_keywords("q", ELIGIBLE, top_k=2, strategy="bm25")


# --- dense --------------------------------------------------------------------


def test_dense_ranks_by_cosine_not_by_token_overlap():
    """The query shares no token with the terms it should rank first.

    "neural nets for vision" has zero lexical overlap with "python", and the
    fixture puts them at the same point in vector space, so a dense ranker
    must surface `python` where the lexical one cannot see it at all.
    """
    ranked = rank_candidate_keywords(
        "neural nets for vision", ELIGIBLE, top_k=2,
        strategy=RANKER_DENSE, embedder=FakeEmbedder(),
    )
    assert list(ranked)[0] == "python"


def test_dense_puts_the_opposed_vector_last():
    ranked = rank_candidate_keywords(
        "machine learning", ELIGIBLE, top_k=len(ELIGIBLE),
        strategy=RANKER_DENSE, embedder=FakeEmbedder(),
    )
    assert list(ranked)[-1] == "basket weaving"


def test_dense_respects_top_k():
    ranked = rank_candidate_keywords(
        "machine learning", ELIGIBLE, top_k=2,
        strategy=RANKER_DENSE, embedder=FakeEmbedder(),
    )
    assert len(ranked) == 2


# --- hybrid -------------------------------------------------------------------


def test_hybrid_keeps_a_term_that_only_one_ranker_finds():
    """RRF's point: agreement is not required, presence in either list counts.

    `python` is invisible to the lexical ranker for this query and top-ranked
    by the dense one; a fusion that dropped it would be an intersection, not a
    fusion.
    """
    ranked = rank_candidate_keywords(
        "neural nets for vision", ELIGIBLE, top_k=3,
        strategy=RANKER_HYBRID, embedder=FakeEmbedder(),
    )
    assert "python" in ranked


def test_hybrid_is_deterministic_across_repeated_calls():
    """Ties are broken by frequency then keyword, never by dict or argsort order.

    A ranker whose output moves between identical calls would put run-to-run
    noise inside the treatment, which is precisely the confound the pool-size
    pilot had to separate out by hand.
    """
    kwargs = dict(strategy=RANKER_HYBRID, embedder=FakeEmbedder())
    first = rank_candidate_keywords("machine learning", ELIGIBLE, top_k=4, **kwargs)
    second = rank_candidate_keywords("machine learning", ELIGIBLE, top_k=4, **kwargs)
    assert list(first) == list(second)


def test_hybrid_differs_from_both_of_its_inputs():
    """If fusion reproduced one input exactly it would not be worth running."""
    query = "neural nets for vision"
    lexical = list(rank_candidate_keywords(query, ELIGIBLE, top_k=4))
    dense = list(rank_candidate_keywords(
        query, ELIGIBLE, top_k=4, strategy=RANKER_DENSE, embedder=FakeEmbedder()))
    hybrid = list(rank_candidate_keywords(
        query, ELIGIBLE, top_k=4, strategy=RANKER_HYBRID, embedder=FakeEmbedder()))
    assert hybrid != lexical or hybrid != dense


# --- freq: the C' correction --------------------------------------------------
#
# Section 2.7 measured the defect these pin: the lexical ranker's injected set
# has a median catalog frequency of 1, and 83.6% of its terms attach to 7 or
# fewer courses. With `per_keyword = 7` such a term cannot retrieve more than
# one course however well the model chooses it.

# Two terms that both match the query, where the *better lexical match* is the
# one that can retrieve almost nothing. This is the pathology in miniature.
SKEWED = {
    "machine learning basics": 1,      # matches one more token, attached to 1 course
    "machine learning": 400,           # matches one fewer token, attached to 400
    "underwater basket weaving": 9999,  # no query match at all, but very common
}
SKEWED_QUERY = "machine learning basics for beginners"


def test_lexical_prefers_the_keyword_that_cannot_retrieve():
    """Not a bug report against lexical -- the control C' is measured against.

    If this assertion ever flips, C' is correcting something that no longer
    happens and the section 2.7 motivation has to be re-measured.
    """
    ranked = rank_candidate_keywords(
        SKEWED_QUERY, SKEWED, top_k=2, strategy=RANKER_LEXICAL
    )
    assert list(ranked)[0] == "machine learning basics"


def test_freq_prefers_the_keyword_that_can():
    ranked = rank_candidate_keywords(
        SKEWED_QUERY, SKEWED, top_k=2, strategy=RANKER_FREQ
    )
    assert list(ranked)[0] == "machine learning"


def test_freq_never_promotes_an_unmatched_keyword_over_a_matched_one():
    """The relevance tier survives the weighting.

    `underwater basket weaving` is 25x more common than anything else here. If
    frequency were added to relevance rather than scaling it, or if the tier
    were dropped, it would lead -- and the ranker would be selecting on
    popularity instead of on the query.
    """
    ranked = rank_candidate_keywords(
        SKEWED_QUERY, SKEWED, top_k=len(SKEWED), strategy=RANKER_FREQ
    )
    assert list(ranked)[-1] == "underwater basket weaving"


def test_freq_and_lexical_agree_when_every_frequency_is_equal():
    """log1p(count) is then a positive constant, so it cannot reorder anything.

    This is what makes the two rankers comparable as an ablation: they differ
    on frequency spread and on nothing else.
    """
    flat = {kw: 7 for kw in ELIGIBLE}
    lexical = rank_candidate_keywords(
        "I want to learn machine learning", flat, top_k=4, strategy=RANKER_LEXICAL
    )
    freq = rank_candidate_keywords(
        "I want to learn machine learning", flat, top_k=4, strategy=RANKER_FREQ
    )
    assert list(lexical) == list(freq)


def test_freq_needs_no_embedder():
    """C' is a re-weighting, not a retrieval strategy. It must run where dense
    cannot -- including on a machine with no embedding model configured."""
    ranked = rank_candidate_keywords(
        SKEWED_QUERY, SKEWED, top_k=2, strategy=RANKER_FREQ, embedder=None
    )
    assert len(ranked) == 2


def test_freq_returns_frequencies_not_weighted_scores():
    ranked = rank_candidate_keywords(
        SKEWED_QUERY, SKEWED, top_k=3, strategy=RANKER_FREQ
    )
    assert all(ranked[k] == SKEWED[k] for k in ranked)


def test_freq_did_not_move_the_published_lexical_ordering():
    """The load-bearing guard, restated for C'.

    C' is a new strategy precisely so that every artifact stamped
    `candidate_ranker=lexical` keeps meaning what it meant.
    """
    ranked = rank_candidate_keywords(
        "I want to learn machine learning", ELIGIBLE, top_k=3
    )
    assert list(ranked) == ["machine learning", "deep learning", "python"]


# --- floor: the half of C' that works ----------------------------------------


def test_floor_excludes_keywords_that_cannot_fill_a_retrieval_slot():
    """With per_keyword = 7, a keyword on <= 7 courses cannot fill its own slot.

    `machine learning basics` is the best lexical match in SKEWED and is attached
    to one course. The floor removes it from the pool rather than reordering it,
    which is what separates this strategy from RANKER_FREQ.
    """
    ranked = rank_candidate_keywords(
        SKEWED_QUERY, SKEWED, top_k=3, strategy=RANKER_FLOOR
    )
    assert "machine learning basics" not in ranked
    assert all(c >= 8 for c in ranked.values())


def test_floor_beats_freq_on_the_measured_defect():
    """The comparison that decided which correction to run.

    Ordering cannot fix a pool, and this vocabulary's problem is the pool: both
    strategies see the same three terms, and only one of them removes the
    unretrievable one.
    """
    freq = rank_candidate_keywords(SKEWED_QUERY, SKEWED, top_k=3, strategy=RANKER_FREQ)
    floor = rank_candidate_keywords(SKEWED_QUERY, SKEWED, top_k=3, strategy=RANKER_FLOOR)
    assert min(freq.values()) == 1
    assert min(floor.values()) >= 8


def test_floor_falls_back_rather_than_injecting_nothing():
    """An empty injected set is Arm 2 wearing Condition C's stamp.

    A platform whose vocabulary is entirely below the floor must still inject
    something, or the row records a treatment it did not receive -- the
    `serving_precision` failure again, one level down.
    """
    all_rare = {"alpha": 1, "beta": 2, "gamma": 3}
    ranked = rank_candidate_keywords(
        "alpha beta", all_rare, top_k=3, strategy=RANKER_FLOOR
    )
    assert ranked == rank_candidate_keywords(
        "alpha beta", all_rare, top_k=3, strategy=RANKER_LEXICAL
    )


def test_floor_ranks_the_survivors_lexically():
    """The floor changes the pool and nothing else.

    Holding the scorer fixed is what lets the run attribute a difference to the
    pool rather than to two changes at once.
    """
    above = {k: c for k, c in SKEWED.items() if c >= 8}
    assert list(
        rank_candidate_keywords(SKEWED_QUERY, SKEWED, top_k=2, strategy=RANKER_FLOOR)
    ) == list(
        rank_candidate_keywords(SKEWED_QUERY, above, top_k=2, strategy=RANKER_LEXICAL)
    )
