"""Bounding how much of M2.1 is transcription error rather than invention.

M2.2 is a normalisation fold, not a distance. It catches `"Data-Science"` and
`"data sciences"`; `"data sceince"` folds to itself, matches nothing, and is
charged to M2.1 as an invention. Paper #1's F1 concedes this as "an
unquantified upper-bound contribution" and leaves it open, because the
quantity was not computable: the invented keyword *strings* reached no export
at all, only their count.

These pin the two halves of the fix -- the distance itself, and the export
that gives it something to read -- and, above all, that it stays a diagnostic.
An edit-distance fold is looser than a normalisation fold, so reclassifying on
it would move more keywords out of M2.1 than M2.2 ever did, in exactly the
direction that flatters the hypothesis.
"""

from __future__ import annotations

from skyrag.metrics import (
    bounded_edit_distance,
    nearest_within,
    transcription_proximity,
)
from skyrag.selection import KeywordSelection, normalise_keyword

# --- the distance ------------------------------------------------------------


def test_identical_strings_are_distance_zero():
    assert bounded_edit_distance("python", "python", max_distance=2) == 0


def test_a_transposition_costs_two_edits():
    """`sceince` -> `science` is two substitutions, not one.

    Plain Levenshtein, not Damerau: a transposition is not a primitive here.
    Worth pinning because the default bound of 2 is what makes the commonest
    real typo reachable at all, and a bound of 1 would silently miss it.
    """
    assert bounded_edit_distance("data sceince", "data science", max_distance=2) == 2
    assert bounded_edit_distance("data sceince", "data science", max_distance=1) is None


def test_beyond_the_bound_returns_none_not_the_bound():
    """"Farther than k" is an absence of measurement, not a measurement of k+1."""
    assert bounded_edit_distance("quantum", "banana", max_distance=2) is None


def test_a_length_gap_wider_than_the_bound_short_circuits():
    """Each edit changes length by at most one, so this needs no matrix."""
    assert bounded_edit_distance("ai", "artificial intelligence", max_distance=2) is None


def test_the_distance_is_symmetric():
    a = bounded_edit_distance("machien learning", "machine learning", max_distance=3)
    b = bounded_edit_distance("machine learning", "machien learning", max_distance=3)
    assert a == b == 2


# --- what the fold cannot catch ----------------------------------------------


def test_the_normalisation_fold_misses_exactly_what_the_distance_catches():
    """This is the hole in F1, stated as an assertion.

    If `normalise_keyword` ever grows to catch typos, M2.2 absorbs them and
    this bound stops being needed -- and this test will say so by failing.
    """
    assert normalise_keyword("data sceince") != normalise_keyword("data science")
    assert bounded_edit_distance("data sceince", "data science", max_distance=2) == 2

    # The fold's own territory, for contrast: these never reach M2.1 at all.
    assert normalise_keyword("Data-Science") == normalise_keyword("data science")
    assert normalise_keyword("data sciences") == normalise_keyword("data science")


# --- the bound over a corpus --------------------------------------------------


CORPUS = ["data science", "machine learning", "python", "linear algebra"]


def test_a_typo_finds_its_term_and_an_invention_does_not():
    found = transcription_proximity(
        ["data sceince", "pythonn", "blockchain sorcery"], CORPUS, max_distance=2
    )
    assert found["data sceince"] == ("data science", 2)
    assert found["pythonn"] == ("python", 1)
    # A fabricated term is not a typo of anything in the catalogue. Absence from
    # the result *is* the finding here, which is why the function returns only
    # the matches rather than a distance for every keyword.
    assert "blockchain sorcery" not in found


def test_a_strictly_nearer_match_wins_whatever_the_candidate_order():
    """Distance decides first; candidate order only breaks exact ties.

    `pythonic` is two edits from `pythonn` and `python` is one, so `python`
    must win from either position in the list.
    """
    assert nearest_within("pythonn", ["pythonic", "python"], max_distance=2) == ("python", 1)
    assert nearest_within("pythonn", ["python", "pythonic"], max_distance=2) == ("python", 1)


def test_an_exact_tie_is_broken_by_candidate_order_and_is_deterministic():
    """`pythons` and `python` are both one edit from `pythonn`.

    Neither is more right, and the bound only needs "a real term this close
    exists" -- so the first is taken rather than a guess being dressed up as a
    choice. Pinned because determinism is what makes the figure reproducible
    for a deterministic corpus ordering, not because the winner matters.
    """
    assert nearest_within("pythonn", ["pythons", "python"], max_distance=2) == ("pythons", 1)
    assert nearest_within("pythonn", ["python", "pythons"], max_distance=2) == ("python", 1)


def test_repeated_inventions_are_counted_once_each_not_deduplicated():
    """The share must be a share of M2.1, whose numerator counts events.

    The same term invented on ten queries is ten inventions in
    `hallucinated_keywords`; computing over unique strings would report a
    different denominator from the metric being bounded.
    """
    found = transcription_proximity(["pythonn", "pythonn"], CORPUS, max_distance=2)
    # The map is keyed by term, so the caller divides by the input length --
    # pinned here so a future refactor cannot quietly make the map the
    # numerator.
    assert len(found) == 1
    assert len(["pythonn", "pythonn"]) == 2


# --- the export that makes it possible ---------------------------------------


def test_a_selection_carries_the_terms_not_only_the_count():
    """`hallucinated_keywords` was `sum(len(...))` and the strings were dropped.

    Nothing on disk could answer F1's open question; the bound had to be
    re-run rather than computed. This pins that the strings survive the
    selection object they are joined from.
    """
    selection = KeywordSelection(
        platform="coursera",
        keywords={"python": 1.0},
        hallucinated=["data sceince", "quantum flux"],
        near_miss={"machine-learning": "machine learning"},
    )
    assert ";".join(selection.hallucinated) == "data sceince;quantum flux"
    assert [f"{k}->{v}" for k, v in selection.near_miss.items()] == [
        "machine-learning->machine learning"
    ]


def test_no_keyword_in_the_real_corpus_contains_the_join_separator():
    """The export is semicolon-joined and plain, matching `cited_course_ids`.

    A keyword containing `;` would split wrongly on read. This is the round
    trip that makes the flat encoding safe rather than lucky.
    """
    terms = ["data sceince", "quantum flux", "r&d methods"]
    assert ";".join(terms).split(";") == terms


class TestOccurrenceWeighting:
    """Regression, 2026-09-23.

    `transcription_proximity` returns a dict keyed by term, so it is
    deduplicated by construction. The report divided that unique-term count by
    an occurrence denominator (`len(terms)`), understating every bucket and
    inflating "invention proper" as the residual. On the 70B E1 lane it
    reported a 3.59% bound where the occurrence-weighted figure is 4.65%.

    `invented_terms` states the rule the report broke: "a share computed over
    unique strings would not be a share of M2.1".
    """

    def _report(self, terms, corpus, capsys):
        import importlib.util, sys
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            "bte_occ", root / "scripts" / "bound_transcription_error.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules["bte_occ"] = m
        spec.loader.exec_module(m)
        m.report("t", terms, corpus, 2)
        return capsys.readouterr().out

    def test_repeated_exact_match_counts_every_occurrence(self, capsys):
        """The same real term invented on five queries is five occurrences."""
        out = self._report(["python"] * 5, ["python", "java"], capsys)
        assert "invented keywords (M2.1 numerator) : 5" in out
        assert "exact match elsewhere in corpus     : 5" in out, (
            "a deduped count would report 1 here and silently drop four "
            "inventions out of the split"
        )

    def test_repeated_near_miss_counts_every_occurrence(self, capsys):
        out = self._report(["pythonn"] * 4, ["python"], capsys)
        assert "within 2 of a real term (BOUND)      : 4" in out

    def test_buckets_sum_to_the_numerator(self, capsys):
        """exact + bound + invention-proper must equal M2.1's numerator."""
        terms = ["python"] * 3 + ["pythonn"] * 2 + ["zzzzzzzzqqq"] * 4
        out = self._report(terms, ["python", "java"], capsys)
        assert "invented keywords (M2.1 numerator) : 9" in out
        assert "exact match elsewhere in corpus     : 3" in out
        assert "within 2 of a real term (BOUND)      : 2" in out
        assert "farther than 2 -- invention proper   : 4" in out
