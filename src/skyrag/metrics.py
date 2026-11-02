"""Published information-retrieval and recommender metrics, implemented directly.

Every metric in this module has a citation. That is the whole point of the file:
the project's other fourteen metrics are bespoke diagnostics of one pipeline --
defensible, but they have to be *declared* as bespoke -- and these are the ones a
reviewer can look up. Cross-reference:
`project_knowledge/research_directions/metric_literature_provenance.md`.

The primitives are kept free of any SKYRAG type on purpose. A metric that takes
`list[float]` can be tested against the worked examples in the papers that define
it; one that takes a `PipelineResult` can only be tested against itself.

**Convention inherited from `eval.py`: an undefined measurement returns `None`,
never `0.0`.** Zero is a real value for every metric here and would read as the
opposite finding -- perfect diversity, no ranking loss -- in exactly the cases
where nothing was measured at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import log2

import numpy as np

__all__ = [
    "bounded_edit_distance",
    "cosine_similarity",
    "dcg_at_k",
    "embedding_ils",
    "intra_list_similarity",
    "jaccard",
    "label_ils",
    "ndcg_at_k",
    "nearest_within",
    "reciprocal_rank_of_ideal_top",
    "transcription_proximity",
]


# --------------------------------------------------------------------------
# Intra-List Similarity -- Ziegler, McNee, Konstan & Lausen, "Improving
# Recommendation Lists Through Topic Diversification", WWW 2005.
# https://dl.acm.org/doi/10.1145/1060745.1060754
#
# ILS is the mean pairwise similarity of the items in a recommendation list.
# High ILS = the list says the same thing several times. The paper leaves the
# similarity function open -- it used a taxonomy-based measure -- so "ILS" names
# the *shape* and an instantiation has to say which similarity it used.
#
# `eval.py::_mean_pairwise_overlap` was written here independently and is the
# Jaccard-over-corpus-labels instantiation. `label_ils` below is that same
# computation, restated in the published form; the two agree by construction and
# a test pins them together so the equivalence cannot drift.
# --------------------------------------------------------------------------


def intra_list_similarity(
    items: Sequence, similarity, *, min_items: int = 2
) -> float | None:
    """Mean pairwise similarity over a list. Ziegler 2005, generic in `similarity`.

    `None` below `min_items`: similarity within a set of one is undefined, and
    0.0 would claim perfect diversity for a recommendation of a single course --
    the reading that flatters the result.

    `similarity` may itself return `None` for a pair it cannot score (two courses
    carrying no labels, a course with no vector). Those pairs are skipped rather
    than counted as zero, so an unmeasurable pair cannot drag the mean toward
    "diverse".
    """
    if len(items) < min_items:
        return None

    total, pairs = 0.0, 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            s = similarity(items[i], items[j])
            if s is None:
                continue
            total += s
            pairs += 1
    return round(total / pairs, 4) if pairs else None


def jaccard(a: set[str], b: set[str]) -> float | None:
    """|A n B| / |A u B|. `None` when both sets are empty -- no labels, no answer."""
    union = a | b
    return len(a & b) / len(union) if union else None


def cosine_similarity(u, v) -> float | None:
    """Cosine of two vectors, or `None` if either carries no direction.

    A zero vector is an absent measurement rather than an orthogonal one --
    the same rule `retrieval._normalize` enforces, where it raises. Here it
    returns `None` instead: a metric that aborts a completed run because one
    course embedded badly destroys the run it exists to report on.
    """
    if u is None or v is None:
        return None
    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    if u.shape != v.shape:
        return None
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return None
    return float(u @ v / (nu * nv))


def label_ils(label_sets: Sequence[set[str]]) -> float | None:
    """ILS instantiated with Jaccard over the corpus's own topic labels.

    Interpretable in a way a cosine is not: 0.8 means "these two courses share
    80% of their topic labels". Sets that are empty after cleaning are dropped
    before pairing, because a course with no labels is unmeasurable rather than
    maximally diverse.
    """
    kept = [s for s in label_sets if s]
    return intra_list_similarity(kept, jaccard)


def embedding_ils(vectors: Sequence) -> float | None:
    """ILS instantiated with embedding cosine.

    The half `label_ils` cannot see. Two courses teaching the same material under
    disjoint vocabularies -- "neural networks" against "deep learning" -- score
    Jaccard 0.0 and are near-duplicates in the recommendation. Jaccard-over-labels
    is bounded by the catalogue's labelling consistency; cosine is not.

    Read the two together. They disagree exactly where the catalogue's labels are
    unreliable, which is itself worth knowing.
    """
    kept = [v for v in vectors if v is not None]
    return intra_list_similarity(kept, cosine_similarity)


# --------------------------------------------------------------------------
# nDCG -- Jarvelin & Kekalainen, "Cumulated Gain-Based Evaluation of IR
# Techniques", ACM TOIS 20(4):422-446, 2002.
# https://dl.acm.org/doi/10.1145/582415.582418
#
# READ THIS BEFORE USING IT. There is a way to apply nDCG here that is
# guaranteed to return 1.0, and it is the obvious one.
#
# `keyword_buckets` returns its courses `sorted(key=-score)` and `merge_platforms`
# re-sorts each platform queue by `-score` before interleaving. So scoring the
# *retriever's own ranking* with the *retriever's own cosine* as the relevance
# grade compares a list against itself: per platform, nDCG is exactly 1.0 for
# every query, every arm, every run. A number the architecture guarantees is not
# a finding.
#
# What is not guaranteed is the order the model actually reads. `merge_platforms`
# interleaves the platforms round-robin so neither leads the context, and that
# interleave departs from the score-ideal ordering by an amount that depends on
# how the two platforms' score distributions sit relative to each other -- which
# depends on which keywords were selected, which is the stage under treatment.
#
# So the metric implemented here scores the **presented order** against the ideal
# ordering of the same set. That is the faithful implementation of the argument
# that motivated adopting nDCG at all -- "a position-sensitive model should see
# good courses first, and `retrieval_score_mean` does not consider position" --
# and it measures the ordering rather than re-deriving the sort.
#
# Two things it is therefore NOT, and both must travel with it into any table:
#
#   1. It is **not recall** and does not address M1.1. There is still no answer
#      key. Grading with the retriever's own cosine is reference-free and
#      self-referential; a query whose embedding is a poor proxy for relevance
#      scores well by construction.
#   2. It is **not the cost of the context cap**. The ideal is computed over the
#      courses that survived `cap_context`, because the pre-cap list does not
#      reach `PipelineResult`. At the shipped default the cap binds on nothing.
#
# On the `naive` pipeline the presented order *is* `top_k`'s sort, so this is
# 1.0 by construction there. `naive` is a reference condition and not an arm
# (D3), but a table that ever puts them side by side must say so.
# --------------------------------------------------------------------------


def dcg_at_k(gains: Sequence[float], k: int | None = None) -> float:
    """Discounted cumulative gain, log2 discount, rank counted from 1.

    Negative gains are clamped to zero. Cosine is bounded in [-1, 1] and a
    negative gain would *subtract* from the accumulated total, so a badly matched
    course late in the list could raise nDCG by hurting the ideal more than the
    presented order -- an inversion that makes the metric unreadable.
    """
    k = len(gains) if k is None else min(k, len(gains))
    return sum(max(0.0, float(gains[i])) / log2(i + 2) for i in range(k))


def ndcg_at_k(presented_gains: Sequence[float], k: int = 10) -> float | None:
    """nDCG@k of a presented ordering against the ideal ordering of the same set.

    `presented_gains` is the relevance grade of each item **in the order the
    consumer sees it**. The ideal is those same grades sorted descending, so the
    result is 1.0 exactly when the presentation is already optimally ordered.

    `None` on an empty list, and `None` when the ideal DCG is zero -- every grade
    clamped to zero leaves nothing to normalise against, and the ratio is
    undefined rather than perfect.
    """
    if not presented_gains:
        return None
    ideal = sorted((max(0.0, float(g)) for g in presented_gains), reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return None
    return round(dcg_at_k(presented_gains, k) / idcg, 4)


# --------------------------------------------------------------------------
# Reciprocal rank -- Voorhees, "The TREC-8 Question Answering Track Report",
# NIST Special Publication 500-246, 1999.
#
# MRR is the mean over queries of 1/rank of the first relevant item. Per row this
# is the reciprocal rank itself; the mean belongs to the aggregation layer.
#
# Binary relevance needs a judgment, and this project has none. The definition
# used here is explicitly self-referential and stated rather than hidden:
# **relevant = in the score-ideal top `top_n` of the same retrieved set.** It
# answers the question that motivated adopting MRR -- does the pipeline put a
# strong anchor course early, or bury it -- without inventing an answer key.
# --------------------------------------------------------------------------


def reciprocal_rank_of_ideal_top(
    presented_scores: Sequence[float], *, top_n: int = 10
) -> float | None:
    """1/rank of the first presented course that belongs to the ideal top `top_n`.

    `None` when the list holds `top_n` courses or fewer: every course is then in
    its own ideal top `top_n`, the first position always qualifies, and the
    metric would report a constant 1.0 while measuring nothing. That is precisely
    the degeneracy this module exists to avoid, so it is refused rather than
    exported.
    """
    n = len(presented_scores)
    if n <= top_n:
        return None
    # The threshold, not the index set: ties at the boundary should all count as
    # relevant. Selecting indices instead would admit one tied course and reject
    # its identical twin on argsort order alone.
    threshold = sorted(presented_scores, reverse=True)[top_n - 1]
    for i, s in enumerate(presented_scores):
        if s >= threshold:
            return round(1.0 / (i + 1), 4)
    return None


# --------------------------------------------------------------------------
# Levenshtein edit distance -- Levenshtein, "Binary codes capable of correcting
# deletions, insertions, and reversals", Soviet Physics Doklady 10(8), 1966.
#
# Used here for one job only: to bound how much of metric M2.1 is character-level
# transcription error rather than parametric invention.
#
# **Why the existing M2.2 cannot answer this.** `selection.normalise_keyword` is
# a *fold*, not a distance: it collapses case, punctuation, separators and
# per-token plurals. It catches "Data-Science" and "data sciences"; it does not
# catch "data sceince", which folds to itself, matches nothing, and is therefore
# counted as an invention. Paper #1's F1 concedes exactly this, calling
# character-level transcription error "an unquantified upper-bound contribution"
# inside M2.1.
#
# **A diagnostic, never a control.** Nothing in this project may move a keyword
# between buckets, drop it, or repair it on the strength of these numbers. The
# near-miss docstring's rule -- measured, not repaired -- applies with more force
# here, because an edit-distance fold is looser than a normalisation fold and
# would move more keywords in the direction that flatters the hypothesis. This
# function reports a distance; the classification stays where `classify` and
# `split_near_misses` put it.
# --------------------------------------------------------------------------


def bounded_edit_distance(a: str, b: str, *, max_distance: int) -> int | None:
    """Levenshtein distance between `a` and `b`, or `None` if it exceeds the bound.

    Bounded because the caller only ever asks "is this within k?", and the full
    matrix over a corpus of tens of thousands of terms is the difference between
    an analysis that runs and one that does not. The length check is exact: two
    strings whose lengths differ by more than `max_distance` cannot be within it,
    since each edit changes length by at most one.

    Returns the true distance when it is `<= max_distance`, and `None` otherwise
    -- not the bound, and not a sentinel integer, because "farther than k" is an
    absence of measurement rather than a measurement of k+1.
    """
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_distance:
        return None

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,            # deletion
                    current[j - 1] + 1,         # insertion
                    previous[j - 1] + (ca != cb),  # substitution
                )
            )
        # Every cell in this row is already >= the row minimum, and each further
        # row can only add, so a row minimum above the bound settles it.
        if min(current) > max_distance:
            return None
        previous = current

    distance = previous[-1]
    return distance if distance <= max_distance else None


def nearest_within(
    keyword: str, candidates: Sequence[str], *, max_distance: int = 2
) -> tuple[str, int] | None:
    """The closest `candidate` to `keyword` within `max_distance`, and the distance.

    `None` when nothing is that close, which is the finding for a genuine
    invention: a fabricated term is not a typo of anything in the catalogue.

    Ties are broken by the smaller distance first and then by the order
    `candidates` arrives in, so the result is deterministic for a deterministic
    corpus ordering. It is not claimed to be *the* term the model meant -- only
    that a term this close exists, which is all the bound needs.
    """
    best: tuple[str, int] | None = None
    bound = max_distance
    for candidate in candidates:
        distance = bounded_edit_distance(keyword, candidate, max_distance=bound)
        if distance is None:
            continue
        if best is None or distance < best[1]:
            best = (candidate, distance)
            # Tighten as we go: once a distance-1 match exists, only a
            # distance-0 match can beat it, and that would have matched exactly
            # upstream and never reached this function.
            bound = distance - 1
            if bound < 0:
                return best
    return best


def transcription_proximity(
    invented: Sequence[str], corpus: Sequence[str], *, max_distance: int = 2
) -> dict[str, tuple[str, int]]:
    """For each invented keyword, the nearest real corpus term within `max_distance`.

    The return maps only the keywords that *have* a near neighbour; a keyword
    absent from the result is farther than the bound from every corpus term.
    So `len(result) / len(invented)` is the share of M2.1 that could be
    transcription error, which is the upper bound F1 leaves open.

    **Read it as an upper bound, not an estimate.** A short invented term can
    sit within two edits of an unrelated real one by coincidence
    ("java" -> "lava"), and this cannot tell that apart from a typo. The bound
    is useful because it is falsifiable in one direction: if it is small, F1's
    "strictly localized to parametric invention" is safe, and no amount of
    coincidence can rescue it if it is large.

    Candidates are bucketed by length so each keyword is compared only against
    terms it could possibly be within `max_distance` of. On the 10,092-course
    corpus this is the difference between seconds and minutes.
    """
    by_length: dict[int, list[str]] = {}
    for term in corpus:
        by_length.setdefault(len(term), []).append(term)

    found: dict[str, tuple[str, int]] = {}
    for keyword in invented:
        candidates: list[str] = []
        for length in range(len(keyword) - max_distance, len(keyword) + max_distance + 1):
            candidates.extend(by_length.get(length, ()))
        match = nearest_within(keyword, candidates, max_distance=max_distance)
        if match is not None:
            found[keyword] = match
    return found
