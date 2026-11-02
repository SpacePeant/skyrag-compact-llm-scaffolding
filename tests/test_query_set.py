"""B5 — the 50-query benchmark set, checked without spending anything.

The specification asks for n >= 50 and claim **C10** asks for domain spread; the
harness shipped with **5** queries, all computer-science. B6's result 3 promoted
this from "generalisability" to something sharper: reproducibility turned out to
be *query-dependent* -- all three long-prompt failures were one query and the
twice-as-long prompt reproduced 4 of 4 -- so with 2 queries "3 of 8 pairs" is
honestly "1 of 2 queries", and the query set binds the **reproducibility
characterisation itself**.

Drafting queries is free; running them is not. Everything asserted here is
offline: the file parses through the real loader, the ids are unique and stable,
and the domain spread is what the file claims it is. Nothing here calls a model.

**What these tests deliberately do not claim.** They do not establish that any
query retrieves well. That needs a paid run, and a cheap proxy would be worse
than nothing here -- a literal token match against the vocabulary window misses
`public speaking` for a query saying "public speaker", and flags `q03` and `q04`,
two queries that have already run successfully. The corpus-support evidence is
recorded in the backlog as a keyword-anchor probe, with its own limits stated.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skyrag.cli import load_query_set  # noqa: E402
from skyrag.pipelines.eval import HISTORICAL_BENCHMARK_QUERIES  # noqa: E402

QUERY_SET = REPO / "benchmarks" / "queries_v1.jsonl"


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return [
        json.loads(line)
        for line in QUERY_SET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_the_query_set_loads_through_the_loader_the_cli_actually_uses(rows):
    """Through `load_query_set`, not through `json.loads`.

    A file that parses as JSONL and not as a query set is the failure this
    catches: the loader requires a `query` field on every object and rejects the
    file otherwise, and finding that out at the top of a paid sweep is the
    expensive way to learn it.
    """
    queries, ids = load_query_set(QUERY_SET)

    assert len(queries) == len(rows) == 50
    assert len(ids) == 50


def test_the_set_meets_the_specifications_query_count():
    """n >= 50, from the active specification's evaluation protocol."""
    queries, _ = load_query_set(QUERY_SET)

    assert len(queries) >= 50, (
        f"the specification asks for n >= 50 and the set holds {len(queries)}"
    )


def test_every_id_is_unique_so_arms_can_be_paired(rows):
    """Ids are how the same question is matched across arms.

    The loader rejects duplicates already; this asserts it on the file that will
    actually be passed, because a duplicate introduced by an edit is caught here
    rather than at the start of a sweep.
    """
    ids = [r["id"] for r in rows]
    duplicates = [i for i, n in collections.Counter(ids).items() if n > 1]

    assert not duplicates, f"duplicate query ids: {duplicates}"


def test_the_five_historical_queries_keep_their_text_and_their_ids(rows):
    """`q01`-`q05` must stay byte-identical, or every prior run stops pairing.

    Every number this project has is from a run over these five. Their ids were
    assigned positionally as `q01`..`q05` by the old built-in set, so keeping
    both the id *and* the wording is what lets a 50-query sweep be compared
    against the runs already paid for. Editing the wording of one line breaks the
    pairing for that query and for no other -- silently, which is the failure
    mode `query_id` was introduced to prevent.
    """
    by_id = {r["id"]: r["query"] for r in rows}

    for i, historical in enumerate(HISTORICAL_BENCHMARK_QUERIES, start=1):
        qid = f"q{i:02d}"
        assert qid in by_id, f"{qid} is missing, so prior runs cannot be paired"
        assert by_id[qid] == historical, (
            f"{qid} was reworded; prior runs pair by id and text, and this "
            f"silently breaks the pairing for {qid} alone"
        )


def test_the_set_spans_more_than_one_domain(rows):
    """Claim C10. Five computer-science queries cannot support a generality claim.

    The source paper evaluated four domains including public speaking and
    mathematics, so a single-domain set also breaks structural comparability with
    the evaluation being reproduced.
    """
    domains = collections.Counter(r["domain"] for r in rows)

    assert len(domains) >= 4, f"only {len(domains)} domain(s): {sorted(domains)}"
    assert {"mathematics", "communication_leadership"} <= set(domains), (
        "the source paper's non-technical domains are the ones that test "
        "generality; a set that widens only within computing has not widened"
    )


def test_no_single_domain_dominates_the_set(rows):
    """A set that is 80% computer science with a garnish is still single-domain.

    A quarter is the ceiling: at 50 queries that leaves at least three other
    domains carrying real weight, and it is checked as a proportion so the bound
    survives the set growing.
    """
    domains = collections.Counter(r["domain"] for r in rows)
    largest, count = domains.most_common(1)[0]

    assert count <= len(rows) // 4, (
        f"{largest} carries {count} of {len(rows)} queries; the spread claim "
        "rests on no domain exceeding a quarter"
    )


def test_computer_science_stays_represented_well_enough_to_compare(rows):
    """The other direction. Widening must not strand the runs already paid for.

    Every existing number is computer-science, so the new set has to keep enough
    of that domain for a like-for-like slice against them -- the five historical
    queries at minimum.
    """
    cs = [r for r in rows if r["domain"] == "computer_science"]

    assert len(cs) >= 5


def test_every_row_carries_a_domain_the_loader_will_ignore(rows):
    """`domain` is documentation, and the loader reads only `id` and `query`.

    Asserted because that is easy to forget in both directions: a row missing
    `domain` loads fine and silently leaves a gap in the spread analysis, and a
    reader may assume the field reaches the export. It does not -- the domain is
    recoverable from the `query_id` prefix, which is what the export carries.
    """
    for row in rows:
        assert row.get("domain"), f"{row.get('id')} carries no domain"
        assert set(row) == {"id", "domain", "query"}, (
            f"{row['id']} carries unexpected keys {sorted(set(row) - {'id', 'domain', 'query'})}; "
            "the loader ignores them, so they document nothing and mislead"
        )


def test_no_query_is_blank_or_a_duplicate_of_another(rows):
    """Two identical questions under two ids are one measurement counted twice."""
    texts = [r["query"].strip().lower() for r in rows]

    assert all(texts), "a blank query would run an arm and measure nothing"
    duplicates = [t for t, n in collections.Counter(texts).items() if n > 1]
    assert not duplicates, f"duplicate query text: {duplicates}"
