"""L1 -- the harness must return the same thing twice.

**Why this test exists.** Three of eight within-process pairs returned a
different recommended set at `per_keyword=7`, pinned provider, `temperature=0`.
Four causes were candidates and exactly one is fully inside this repository:
unstable sorts, `set`/`dict` iteration order, and unbroken ties. This file rules
that one in or out. Until it passes, every measured difference between two arms
carries an unknown amount of harness noise, and repeats would have *hidden* the
defect by averaging over it rather than exposing it.

**What is stubbed and why.** The LLM is an `EchoBackend` returning one fixed
keyword reply. That is the point: with the model's contribution held constant,
anything that still varies is ours. A test that let the model vary would be
measuring the provider.

**Four properties, in increasing order of strength.**

1. *Repetition.* The same call, 100x in one process, is byte-identical.
   Necessary, and much weaker than it looks -- see below.
2. *Permutation invariance.* Feeding the same courses in a shuffled order
   returns the same ranking. This is the property with teeth. Repetition is
   satisfied by any deterministic-but-arbitrary tie-break, because rerunning
   identical input reproduces the identical arbitrary answer; permutation is
   not, because it changes what "arbitrary" resolves to. **A tie broken by row
   order passes test 1 and fails test 2**, which is exactly how this class of
   defect survives a naive determinism check.
3. *The cut.* A tie straddling the `per_keyword` boundary decides which course
   the model is shown. Asserted directly rather than inferred from 1 and 2.
4. *Hash-seed independence.* Run in subprocesses under different
   `PYTHONHASHSEED` values. Within one process `str` hashing is fixed, so no
   in-process loop can ever catch a `set`-iteration order dependency -- only
   crossing a process boundary can.
"""

from __future__ import annotations

import random
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from skyrag.config import Config  # noqa: E402
from skyrag.data.adapters import build_vocabulary  # noqa: E402
from skyrag.data.schema import Course  # noqa: E402
from skyrag.embedding import EmbeddingStore  # noqa: E402
from skyrag.llm.echo import EchoBackend  # noqa: E402
from skyrag.pipelines.skyrag import run_skyrag  # noqa: E402
from skyrag.retrieval import (  # noqa: E402
    format_context,
    keyword_buckets,
    merge_platforms,
    top_k,
)
from skyrag.selection import KeywordSelection  # noqa: E402

REPEATS = 100
DIM = 8


# --------------------------------------------------------------------------
# A corpus built to tie
#
# Ties are not an edge case here. `score_all` returns cosine similarity in
# float32 over an 8-dimensional space in these fixtures and over a 768-dimensional
# one in production, and courses on the same subject genuinely land on the same
# score. The fixture makes them exact so the assertions are about ordering rather
# than about float tolerance.
# --------------------------------------------------------------------------

# Four vectors, each shared by several courses. Every course carrying a given
# vector scores *identically* against any query, so every ranking below has a
# tie in it, and several have a tie straddling a cut.
_VECTORS = {
    "a": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "b": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "c": [0.7071, 0.7071, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "d": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}

# (course id, vector key, keywords). Ten courses on "python" all sharing vector
# "a" is what puts a tie across the `per_keyword=3` boundary.
_CORPUS = [
    ("COU01", "a", ["python", "programming"]),
    ("COU02", "a", ["python"]),
    ("COU03", "a", ["python", "data science"]),
    ("COU04", "a", ["python"]),
    ("COU05", "a", ["python", "programming"]),
    ("COU06", "c", ["python", "machine learning"]),
    ("COU07", "c", ["machine learning"]),
    ("COU08", "b", ["data science", "statistics"]),
    ("COU09", "b", ["data science"]),
    ("COU10", "b", ["statistics"]),
    ("COU11", "d", ["programming"]),
    ("COU12", "d", ["programming", "python"]),
]

_EDX_CORPUS = [
    ("EDX01", "a", ["python", "computer science"]),
    ("EDX02", "a", ["python"]),
    ("EDX03", "b", ["data science"]),
    ("EDX04", "b", ["data science", "statistics"]),
    ("EDX05", "c", ["computer science"]),
]


def _build(rows, platform: str, order: list[int] | None = None) -> EmbeddingStore:
    """A store over `rows`, optionally permuted. Same courses, different row order."""
    picked = [rows[i] for i in order] if order is not None else list(rows)
    courses = [
        Course(id_=cid, platform=platform, title=f"Course {cid}", keywords=list(kws))
        for cid, _, kws in picked
    ]
    matrix = np.array([_VECTORS[vec] for _, vec, _ in picked], dtype=np.float32)
    return EmbeddingStore(courses=courses, matrix=matrix)


def _query() -> np.ndarray:
    """Deliberately equidistant from vectors "a" and "b", so both tie at "c"."""
    return np.array([0.7071, 0.7071, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


class _FixedEmbedder:
    def encode_one(self, text: str) -> np.ndarray:
        return _query()


def _fingerprint(retrieved) -> str:
    """The retrieved list as one comparable string: ids, order, attribution, score."""
    return "|".join(
        f"{r.course.id_}:{r.via_keyword}:{r.score:.9f}" for r in retrieved
    )


KEYWORDS = ["python", "data science", "programming", "machine learning"]


# --------------------------------------------------------------------------
# 1. Repetition -- the same call, 100x, in one process
# --------------------------------------------------------------------------


def test_keyword_buckets_is_identical_across_100_runs():
    store = _build(_CORPUS, "coursera")
    first = _fingerprint(keyword_buckets(_query(), store, KEYWORDS, per_keyword=3))

    for _ in range(REPEATS):
        assert _fingerprint(
            keyword_buckets(_query(), store, KEYWORDS, per_keyword=3)
        ) == first


def test_top_k_is_identical_across_100_runs():
    store = _build(_CORPUS, "coursera")
    first = _fingerprint(top_k(_query(), store, k=5))

    for _ in range(REPEATS):
        assert _fingerprint(top_k(_query(), store, k=5)) == first


def test_merged_context_is_byte_identical_across_100_runs():
    """The end of the retrieval chain: what actually reaches the prompt."""
    stores = {"coursera": _build(_CORPUS, "coursera"), "edx": _build(_EDX_CORPUS, "edx")}
    per_platform = {
        p: keyword_buckets(_query(), s, KEYWORDS, per_keyword=3)
        for p, s in stores.items()
    }
    first = format_context(merge_platforms(per_platform))

    for _ in range(REPEATS):
        again = {
            p: keyword_buckets(_query(), s, KEYWORDS, per_keyword=3)
            for p, s in stores.items()
        }
        assert format_context(merge_platforms(again)) == first


def test_full_pipeline_is_identical_across_100_runs():
    """The whole `run_skyrag` chain with the model held constant.

    `EchoBackend` is handed a long fixed reply list rather than one reply,
    because the pipeline calls it several times per run (expansion, step-back,
    one selection per platform) and a run that exhausted the list would fall
    back to echoing the prompt -- which changes per stage and would make this
    test pass for the wrong reason.
    """
    config = Config()
    for attr in ("model", "expansion_model", "keyword_model", "generation_model"):
        setattr(config.llm, attr, "stub")

    def run():
        return run_skyrag(
            "learn python",
            {
                "coursera": _build(_CORPUS, "coursera"),
                "edx": _build(_EDX_CORPUS, "edx"),
            },
            EchoBackend(
                responses=[
                    '{"python": 0.9, "data science": 0.9, "programming": 0.8}'
                ]
                * 64
            ),
            config,
            embedder=_FixedEmbedder(),
            per_keyword=3,
        )

    first = _fingerprint(run().retrieved)
    for _ in range(REPEATS):
        assert _fingerprint(run().retrieved) == first


# --------------------------------------------------------------------------
# 2. Permutation invariance -- the property repetition cannot reach
# --------------------------------------------------------------------------


def test_keyword_buckets_is_invariant_to_corpus_row_order():
    """Shuffling the corpus must not change the ranking.

    This is the assertion that fails when a tie is broken by row position.
    `sorted` is stable and `np.argsort`'s quicksort is not, so before the
    `(-score, course_id)` total order the winner of a tie was decided by where a
    course sat in the matrix -- reproducible on rerun, and different for the same
    courses loaded in a different order.
    """
    reference = _fingerprint(
        keyword_buckets(_query(), _build(_CORPUS, "coursera"), KEYWORDS, per_keyword=3)
    )

    rng = random.Random(20260820)
    order = list(range(len(_CORPUS)))
    for _ in range(50):
        rng.shuffle(order)
        shuffled = _build(_CORPUS, "coursera", order=list(order))
        assert (
            _fingerprint(keyword_buckets(_query(), shuffled, KEYWORDS, per_keyword=3))
            == reference
        ), f"ranking changed under corpus permutation {order}"


def test_top_k_is_invariant_to_corpus_row_order():
    """`argpartition` is unstable and its output decides the cut.

    Narrowing with `argpartition` and slicing its result directly admits one
    course tied at the boundary and drops its twin on partition order alone --
    which is a function of row position, not of the data.
    """
    reference = _fingerprint(top_k(_query(), _build(_CORPUS, "coursera"), k=5))

    rng = random.Random(11)
    order = list(range(len(_CORPUS)))
    for _ in range(50):
        rng.shuffle(order)
        assert (
            _fingerprint(top_k(_query(), _build(_CORPUS, "coursera", order=list(order)), k=5))
            == reference
        ), f"top-k changed under corpus permutation {order}"


def test_merge_platforms_is_invariant_to_platform_insertion_order():
    """Which platform leads the context must not depend on dict build order.

    The round-robin equalises how *many* courses each platform contributes; it
    cannot equalise position, and position is what the interleave exists to
    control. Whoever is visited first holds rank 1 in the prompt.
    """
    stores = {"coursera": _build(_CORPUS, "coursera"), "edx": _build(_EDX_CORPUS, "edx")}
    buckets = {
        p: keyword_buckets(_query(), s, KEYWORDS, per_keyword=3)
        for p, s in stores.items()
    }

    forward = merge_platforms({"coursera": buckets["coursera"], "edx": buckets["edx"]})
    reverse = merge_platforms({"edx": buckets["edx"], "coursera": buckets["coursera"]})

    assert _fingerprint(forward) == _fingerprint(reverse)


def test_selected_keyword_order_is_invariant_to_reply_key_order():
    """Equal scores must not be ordered by the model's JSON key order.

    `ranked()` feeds `keyword_buckets` directly and first-keyword-wins decides
    every course's exported `via_keyword`, so without a tie-break that column
    depends on reply formatting rather than on the scores the model assigned.
    Models emit round numbers, so these ties are the common case.
    """
    forward = KeywordSelection(
        platform="coursera", keywords={"python": 0.9, "data science": 0.9, "r": 0.5}
    )
    reverse = KeywordSelection(
        platform="coursera", keywords={"data science": 0.9, "python": 0.9, "r": 0.5}
    )

    assert forward.ranked() == reverse.ranked()
    assert forward.ranked() == ["data science", "python", "r"]


def test_vocabulary_window_is_invariant_to_course_order():
    """M1.2 exports a keyword's rank as evidence of position bias.

    The window is also cut at `max_vocabulary`, so a tie at the boundary decides
    whether a keyword is shown to the model at all. Both readings require the
    rank to be a property of the corpus rather than of its row order -- which is
    what `selected_ranks`' docstring already claims.
    """
    courses = [
        Course(id_=cid, platform="coursera", title=cid, keywords=list(kws))
        for cid, _, kws in _CORPUS
    ]
    reference = list(build_vocabulary(courses))

    rng = random.Random(7)
    for _ in range(50):
        shuffled = list(courses)
        rng.shuffle(shuffled)
        assert list(build_vocabulary(shuffled)) == reference


# --------------------------------------------------------------------------
# 3. The cut -- where a tie-break actually changes what the model is shown
# --------------------------------------------------------------------------


def test_the_per_keyword_cut_is_decided_by_the_tiebreaker_not_by_row_order():
    """Seven courses carry "python"; three survive `per_keyword=3`.

    Asserted as a concrete expected list rather than as invariance, so the rule
    is pinned in the suite and not merely consistent with itself.

    COU06 sits on vector "c", which *is* the query, so it scores 1.0 and wins on
    score alone -- the tie-break never runs for it. COU01-COU05 share vector "a"
    at 0.7071 and are genuinely tied, so the two remaining slots go to the two
    lowest ids among them. COU12 sits on "d" and scores 0.0.

    Both halves matter: the first position proves the tie-break has not
    displaced the actual ranking, and the next two prove it decides the cut.
    """
    store = _build(_CORPUS, "coursera")
    retrieved = keyword_buckets(_query(), store, ["python"], per_keyword=3)

    assert [r.course.id_ for r in retrieved] == ["COU06", "COU01", "COU02"]


def test_a_tie_at_the_cut_survives_permutation():
    """The same cut, under every permutation. The union of tests 2 and 3."""
    rng = random.Random(3)
    order = list(range(len(_CORPUS)))
    for _ in range(50):
        rng.shuffle(order)
        store = _build(_CORPUS, "coursera", order=list(order))
        retrieved = keyword_buckets(_query(), store, ["python"], per_keyword=3)
        assert [r.course.id_ for r in retrieved] == ["COU06", "COU01", "COU02"]


# --------------------------------------------------------------------------
# 4. Hash-seed independence -- the only part an in-process loop cannot test
# --------------------------------------------------------------------------

_SUBPROCESS_PROBE = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {src!r})
    sys.path.insert(0, {tests!r})

    from test_harness_determinism import (
        KEYWORDS, _CORPUS, _EDX_CORPUS, _build, _fingerprint, _query,
    )
    from skyrag.retrieval import format_context, keyword_buckets, merge_platforms

    stores = {{
        "coursera": _build(_CORPUS, "coursera"),
        "edx": _build(_EDX_CORPUS, "edx"),
    }}
    buckets = {{
        p: keyword_buckets(_query(), s, KEYWORDS, per_keyword=3)
        for p, s in stores.items()
    }}
    print(format_context(merge_platforms(buckets)))
    """
)


def _run_under_hash_seed(seed: str) -> str:
    root = Path(__file__).resolve().parent.parent
    probe = _SUBPROCESS_PROBE.format(
        src=str(root / "src"), tests=str(root / "tests")
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env={**_clean_env(), "PYTHONHASHSEED": seed},
        cwd=str(root),
    )
    if result.returncode != 0:
        pytest.fail(f"probe failed under PYTHONHASHSEED={seed}:\n{result.stderr}")
    return result.stdout


def _clean_env() -> dict:
    import os

    return {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}


def test_context_is_identical_under_different_hash_seeds():
    """`str` hashing is randomised per process, fixed within one.

    So no in-process loop, at any repeat count, can detect a `set`-iteration
    order dependency: every iteration shares the one seed the interpreter
    started with. Crossing a process boundary is the only way to vary it, and a
    `set` reached the ranking path would show up here and nowhere else.
    """
    outputs = {seed: _run_under_hash_seed(seed) for seed in ("0", "1", "12345", "99999")}

    distinct = set(outputs.values())
    assert len(distinct) == 1, (
        "retrieval output depends on PYTHONHASHSEED -- a set or dict built from "
        f"unordered iteration reached the ranking path. Seeds gave {len(distinct)} "
        "distinct contexts."
    )
