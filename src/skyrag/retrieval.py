"""Retrieval: the naive baseline and SKYRAG's keyword-bucketed variant.

The notebooks scored courses one row at a time inside a pandas loop, rebuilding
a torch tensor per row. That is replaced here by a single matrix product --
same result, orders of magnitude faster, and no torch requirement at query
time.

Note on the metric: all-mpnet-base-v2 returns L2-normalized vectors, so the
dot product used throughout *is* cosine similarity. Vectors are normalized
defensively anyway, so swapping in a non-normalizing embedding model does not
silently change the ranking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data.schema import Course
from .embedding import EmbeddingStore


@dataclass
class RetrievedCourse:
    course: Course
    score: float
    # Which selected keyword pulled this course in (None for naive retrieval).
    via_keyword: str | None = None
    # The course's own embedding row, carried so metrics computed downstream can
    # measure course-to-course similarity. `score` is similarity to the *query*
    # and cannot answer "are these two recommendations the same course twice" --
    # that needs the vectors themselves, and `PipelineResult` carries no handle
    # on the `EmbeddingStore` they came from.
    #
    # Stored unnormalised, exactly as the store holds it: `score_all` normalises
    # a copy at scoring time and does not write back, so normalising here would
    # be the only place in the pipeline where a stored vector and a retrieved one
    # differ. Consumers normalise -- `metrics.cosine_similarity` does.
    vector: np.ndarray | None = None


def _normalize(matrix: np.ndarray, *, subject: str = "corpus row") -> np.ndarray:
    """L2-normalize rows, refusing vectors that carry no direction.

    The clamp this replaced -- `np.maximum(norms, 1e-12)` -- kept a zero row from
    raising a divide-by-zero, and in doing so turned a dead embedding into a
    unit-length-looking row of zeros that scores ~0 against every query forever.
    That is indistinguishable from a course which is genuinely irrelevant to
    everything, so a corrupted matrix produces a complete, plausible, permanently
    wrong ranking with no anomalous number anywhere.

    A zero vector is not a low score; it is an absent measurement, so it raises.
    The clamp stays for the rows that pass, guarding ordinary floating-point
    underflow rather than substituting for a validity check.

    `subject` names which side of the comparison went dead. Both sides come
    through here -- `score_all` normalises the stored matrix and the query vector
    with the same call -- so a single message written for the corpus told a caller
    whose *query encoder* had failed to "re-embed the affected rows", which is
    both untrue and unactionable: there are no rows to re-embed, and the cache is
    fine. The guard was right and the instruction was pointed at the wrong file.
    """
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)

    dead = np.asarray(norms <= _MIN_VECTOR_NORM).reshape(-1)
    if dead.any():
        idx = np.flatnonzero(dead)
        raise ValueError(
            f"{idx.size} of {dead.size} {subject}(s) have effectively zero norm "
            f"(<= {_MIN_VECTOR_NORM:g}); first at position {int(idx[0])}. A zero "
            "embedding is a failed encode, not a low similarity: normalising it "
            "yields a vector that scores ~0 against everything and reads as "
            f"genuine irrelevance. {_REMEDY.get(subject, _REMEDY['corpus row'])}"
        )

    return matrix / np.maximum(norms, 1e-12)


# What to do about a dead vector, per side. Separated from the diagnostic above
# because the two failures share a symptom and share nothing else: one is a stale
# or damaged cache, the other is a live encoder returning nothing for the text it
# was just handed.
_REMEDY = {
    "corpus row": (
        "Re-embed the affected rows, or drop them explicitly, rather than "
        "scoring them."
    ),
    "query vector": (
        "This is the query encoder, not the cache: the embedder returned no "
        "direction for the query text just submitted. Check that the query is "
        "non-empty and that the embedding model loaded -- the stored corpus does "
        "not need re-embedding."
    ),
}


# Below this, a vector carries no usable direction. Well clear of float32 rounding
# on a unit vector, well under any real embedding's norm.
_MIN_VECTOR_NORM = 1e-8


# --- Total order (L1) -----------------------------------------------------
#
# Every ranking step below sorts on a float similarity, and float similarities
# tie. `sorted` is stable and `np.argsort`'s default quicksort is not, so before
# this the tie-breaker was, respectively, the order courses happened to sit in
# the store and whatever introsort left behind -- neither of which is a property
# of the data. Both are *deterministic within a process*, which is exactly why
# the defect survived: rerunning the identical input reproduced the identical
# arbitrary order, and nothing ever disagreed with itself.
#
# It matters at the cut, not in the middle. `keyword_buckets` keeps
# `per_keyword` courses and `cap_context` keeps `limit` of them, so when a tie
# straddles a boundary the tie-breaker decides *which course the model is shown*.
# Ordering by `(-score, course_id)` makes that choice a function of the corpus
# rather than of its row order, which is what lets two arms be compared at all.
#
# `id_` is the tie-breaker because it is the only field guaranteed present,
# unique and stable across runs -- `Course` has no other total key.


def _order_by_score_then_id(
    indices, scores: np.ndarray, courses: list[Course]
) -> list[int]:
    """Descending score, ascending course id. A total order over distinct rows."""
    return sorted(indices, key=lambda i: (-float(scores[i]), courses[i].id_))


def _top_indices(scores: np.ndarray, courses: list[Course], k: int) -> list[int]:
    """The `k` best rows under the total order above.

    `argpartition` is kept for the narrowing -- it is what makes this O(n) on a
    10,000-row corpus -- but its output is not trusted to decide the cut. The
    k-th best *score* is read off the partition and every row matching or beating
    it is re-collected, so any row tied with the boundary is a candidate before
    the tie-breaker runs. Slicing the partition directly would admit one tied
    course and drop its twin on partition order alone.
    """
    n = len(scores)
    if k >= n:
        return _order_by_score_then_id(range(n), scores, courses)
    part = np.argpartition(-scores, k - 1)[:k]
    threshold = float(scores[part].min())
    candidates = np.flatnonzero(scores >= threshold).tolist()
    return _order_by_score_then_id(candidates, scores, courses)[:k]


def score_all(query_vector: np.ndarray, store: EmbeddingStore) -> np.ndarray:
    """Cosine similarity of the query against every course. Shape: (n_courses,)."""
    query = _normalize(
        np.asarray(query_vector, dtype=np.float32).reshape(1, -1), subject="query vector"
    )
    matrix = _normalize(store.matrix, subject="corpus row")
    return (matrix @ query.T).ravel()


def top_k(
    query_vector: np.ndarray, store: EmbeddingStore, k: int = 20
) -> list[RetrievedCourse]:
    """Naive retrieval: single ranked list over the whole corpus.

    This is the Naive_RAG baseline.
    """
    scores = score_all(query_vector, store)
    k = min(k, len(scores))
    idx = _top_indices(scores, store.courses, k)
    return [
        RetrievedCourse(
            course=store.courses[i], score=float(scores[i]), vector=store.matrix[i]
        )
        for i in idx
    ]


def keyword_buckets(
    query_vector: np.ndarray,
    store: EmbeddingStore,
    selected_keywords: list[str],
    *,
    per_keyword: int = 7,
) -> list[RetrievedCourse]:
    """SKYRAG retrieval: filter symbolically by keyword, then rank semantically.

    For each selected keyword, keep only courses carrying that keyword and
    take the `per_keyword` best by similarity. This is what stops the result
    set from collapsing onto one subtopic: every selected keyword is
    guaranteed representation.

    A course matching several keywords appears once, attributed to the
    **first** of those keywords in `selected_keywords`. That list arrives in the
    model's own ranked order, so first-wins means highest-ranked-keyword-wins,
    which is the useful reading of `via_keyword` in the rendered report.

    This docstring used to promise attribution "to the keyword where it scored
    best", and the code carried a `score > best[i].score` comparison implementing
    that promise. The comparison could never fire: scores come from `score_all`,
    which ranks a course against the *query* and knows nothing about keywords, so
    a course's score is identical under every keyword that reaches it. What the
    code actually did was first-wins, and `via_keyword` is rendered to the reader
    as the explanation of why a course was retrieved -- so the explanation and
    the documented rule had to be made to agree. The dead comparison is gone
    rather than left as evidence of a rule that was never in force.
    """
    scores = score_all(query_vector, store)

    # keyword -> row indices carrying it (built once, not per keyword).
    #
    # Keys are normalised the same way the lookup below normalises, because the two
    # sides had drifted apart: the index used each course's raw casing while the
    # lookup lowercased, so a course carrying "Python" was unreachable by the
    # selection "python". `split_keywords` lowercases at ingest, which hid this for
    # adapter-loaded corpora -- but `Course.from_dict` does not go through it, so a
    # store built from an already-keyworded documents file kept whatever casing the
    # file had, and those courses silently never matched.
    index: dict[str, list[int]] = {}
    for i, course in enumerate(store.courses):
        for kw in course.keywords:
            index.setdefault(kw.strip().lower(), []).append(i)

    best: dict[int, RetrievedCourse] = {}
    for keyword in selected_keywords:
        rows = index.get(keyword.strip().lower())
        if not rows:
            continue
        order = _order_by_score_then_id(rows, scores, store.courses)[:per_keyword]
        for i in order:
            if i in best:
                continue          # first keyword to reach a course keeps it
            best[i] = RetrievedCourse(
                course=store.courses[i],
                score=float(scores[i]),
                via_keyword=keyword,
                vector=store.matrix[i],
            )

    return sorted(best.values(), key=lambda r: (-r.score, r.course.id_))


def merge_platforms(
    results: dict[str, list[RetrievedCourse]]
) -> list[RetrievedCourse]:
    """Interleave per-platform results so no single platform leads the context.

    Ordering matters: LLMs weight earlier context more heavily, so feeding
    Coursera's entire block first biases the generated path toward it. Round-
    robin interleaving is what makes the notebooks' all-caps
    "USE EDX AND UDEMY TOO" prompt unnecessary.

    Platforms are visited in **sorted name order**, not in `results`' insertion
    order (L1). The interleave equalises how much of each platform reaches the
    context, but it cannot equalise *position*, and position is the very thing
    the docstring above says biases generation: whoever is visited first holds
    rank 1. That was previously inherited from whatever order `stores` happened
    to be built in -- a caller passing `{"edx": ..., "coursera": ...}` produced a
    different context to one passing the same two platforms the other way round,
    with nothing in the run recording which had happened. Sorting makes the
    leader a function of the platform set alone, so two arms over the same
    platforms see the same ordering by construction rather than by luck.
    """
    queues = {
        p: sorted(results[p], key=lambda x: (-x.score, x.course.id_))
        for p in sorted(results)
        if results[p]
    }
    merged: list[RetrievedCourse] = []
    position = 0
    while queues:
        for platform in list(queues):
            queue = queues[platform]
            if position < len(queue):
                merged.append(queue[position])
            else:
                del queues[platform]
        position += 1
    return merged


def cap_context(
    results: list[RetrievedCourse], limit: int | None
) -> tuple[list[RetrievedCourse], int]:
    """Bound how many courses may reach a generation prompt. Returns (kept, dropped).

    A backstop, not a treatment knob. Nothing between `keyword_buckets` and
    `format_context` used to bound the count at all: `max_keywords` x
    `per_keyword` x platforms admits 560 courses, which at the *full* `render()`
    built a 441,000-token prompt against a 131,072-token endpoint limit. The
    consequence was an inversion rather than an outage -- an arm that selected
    valid keywords on both platforms retrieved more courses, overflowed and was
    refused, while an arm that hallucinated keywords retrieved less and passed,
    so selection quality was inversely correlated with completing the run.

    The trimmed `render_for_context` is what makes the configured ceiling fit; this
    cap exists so that no future combination of `max_keywords`, `per_keyword` and
    platform count can reach the endpoint limit unannounced. At the shipped default
    it binds on **no** arm, which is the property the methods section needs:
    a cap that binds on one arm and not another means the arms did not see the same
    task, and truncating context is itself a treatment.

    Applied after `merge_platforms`, never per platform: cutting each platform's
    list before the round-robin interleave would silently favour whichever platform
    sorted first, which is the confound the interleave exists to prevent. Taking a
    prefix of the merged list preserves the alternation.

    The dropped count is returned rather than logged, because "the model saw 150 of
    the 560 courses retrieved" is not recoverable from any other exported column.
    """
    if limit is None or limit < 0 or len(results) <= limit:
        return results, 0
    return results[:limit], len(results) - limit


def format_context(results: list[RetrievedCourse]) -> str:
    """Render retrieved courses as the context block sent to the LLM.

    Course IDs are included so the generated path cites real, checkable
    courses -- and so hallucinated IDs can be detected afterwards.

    `render_for_context`, not `render()`. The two are separate methods on purpose:
    `render()` is the embedder's input and shortening it would invalidate every
    cached matrix on disk while leaving the manifest -- which records no render
    format -- looking valid. See `Course.render_for_context`.

    The ID is no longer prefixed here. `render_for_context` emits it as its first
    line, and the old `f"Course ID: {id}\n{render()}"` printed it twice, since
    `render()` also carries one.
    """
    return "\n\n".join(r.course.render_for_context() for r in results)
