#!/usr/bin/env python
"""How much of M2.1 could be transcription error rather than invention?

Paper #1's F1 claims the compact-model deficit is "strictly localized to
parametric invention rather than syntactic or transcription errors". Its
evidence is that near-miss rates (M2.2) are statistically identical across
scale -- 1.54% vs 1.41%, p = 0.99.

That evidence has a hole its own index concedes. M2.2 is
`selection.normalise_keyword`, a **fold**: case, punctuation, separators,
per-token plurals. It catches `"Data-Science"` and `"data sciences"`. It does
not catch `"data sceince"`, which folds to itself, matches no corpus term, and
is therefore counted by M2.1 as an invention. F1 calls this "an unquantified
upper-bound contribution" and leaves it there.

This script quantifies it, by asking of every keyword M2.1 scored as invented:
is there a real corpus term within `--max-distance` edits of it?

**The answer is an upper bound, not an estimate.** A short invented term can
land within two edits of an unrelated real one by coincidence. The bound is
still decisive in one direction: if it is small, F1 is safe and the caveat can
be dropped; no amount of coincidence rescues F1 if it is large.

**It reclassifies nothing.** Per the rule in `split_near_misses` -- measured,
not repaired -- no keyword moves bucket, no artifact is rewritten, and no
downstream number changes. This prints a diagnosis.

Requires `hallucinated_keyword_terms`, which runs from 2026-09-22 carry. Runs
before that exported counts only; this refuses them by name rather than
reporting a spuriously clean zero.

Usage:

    python scripts/bound_transcription_error.py outputs/run_dir [...] \
        --documents documents --platform coursera --platform edx
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from skyrag.data.adapters import build_vocabulary, load_all  # noqa: E402
from skyrag.metrics import transcription_proximity  # noqa: E402

TERMS_FIELD = "hallucinated_keyword_terms"


def load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads((path / "benchmark_eval.json").read_text(encoding="utf-8"))
    return [r for r in data if r.get("query_id")]


def invented_terms(rows: list[dict[str, Any]]) -> list[str]:
    """Every keyword M2.1 charged as an invention, across all rows.

    Flattened, and deliberately not de-duplicated: the same term invented on
    ten queries is ten inventions in `hallucinated_keywords`, and a share
    computed over unique strings would not be a share of M2.1.
    """
    terms: list[str] = []
    for row in rows:
        raw = str(row.get(TERMS_FIELD) or "")
        terms.extend(t for t in raw.split(";") if t)
    return terms


def corpus_terms(documents_dir: Path, platforms: list[str] | None) -> list[str]:
    """Every keyword the catalogues contain, lower-cased and de-duplicated.

    Built through `adapters.load_all` and `build_vocabulary` rather than by
    parsing the dataset files here, so the corpus this bound is measured
    against is the same object `classify` measured invention against. A
    bespoke parser that disagreed by one cleaning rule would move keywords
    across the bound for a reason that has nothing to do with the model.

    **Uncleaned, deliberately.** `classify` sends publisher terms to
    `removed_by_cleaning`, not to `hallucinated`, so they are not in this
    script's numerator; including them in the comparison set is therefore
    harmless, and excluding them would make a term that exists in the corpus
    read as unreachable.

    Lower-cased because the question is transcription and case is already
    M2.2's, folded upstream. Leaving case in would charge `"Python"` against
    `"python"` as an edit and inflate the bound.
    """
    seen: set[str] = set()
    for courses in load_all(documents_dir, platforms, allow_missing=True).values():
        for keyword in build_vocabulary(courses):
            seen.add(str(keyword).strip().lower())
    return sorted(seen)


def report(label: str, terms: list[str], corpus: list[str], max_distance: int) -> None:
    if not terms:
        print(f"\n### {label}\n    no invented keywords recorded")
        return

    lowered = [t.lower() for t in terms]
    matches = transcription_proximity(lowered, corpus, max_distance=max_distance)

    # **Count occurrences, not unique strings.** `transcription_proximity`
    # returns a dict keyed by term, so it is deduplicated by construction:
    # a term invented on ten queries appears once. `total` below is
    # `len(terms)` -- every occurrence. Counting the dict directly divided a
    # unique-term numerator by an occurrence denominator, understating every
    # bucket and inflating "invention proper" as the residual.
    #
    # `invented_terms` says why the denominator must stay occurrence-based:
    # "the same term invented on ten queries is ten inventions in
    # `hallucinated_keywords`, and a share computed over unique strings would
    # not be a share of M2.1." That is exactly the mistake this line made.
    #
    # Found 2026-09-23 when `keyword_provenance_split.py`, which iterates the
    # list, disagreed with this script on the same run (263 vs 230).
    distance_of = {term: d for term, (_, d) in matches.items()}
    by_distance = Counter(distance_of[t] for t in lowered if t in distance_of)
    total = len(terms)

    # **Distance 0 is not transcription error and must not enter the bound.**
    # `classify` decides validity against ONE platform's vocabulary, while this
    # script necessarily compares against the whole corpus -- the export joins
    # invented terms across platforms without recording which platform each
    # came from, so per-term attribution is not recoverable from the artifact.
    # A term that matches a real corpus keyword exactly was therefore not
    # mis-typed: it exists, just not in the catalogue the model was answering
    # for (or it was cleaned from that platform's window). That is a real and
    # separate finding -- cross-platform leakage -- and folding it into
    # "could be a typo" inflates the bound in the direction that weakens F1.
    exact = by_distance.get(0, 0)
    near = sum(n for d, n in by_distance.items() if d >= 1)

    print(f"\n### {label}")
    print(f"    invented keywords (M2.1 numerator) : {total}")
    if exact:
        print(f"    exact match elsewhere in corpus     : {exact}"
              f"   ({exact / total:.2%})  NOT transcription error --"
              " real on another platform, or cleaned from this window")
    for distance in range(1, max_distance + 1):
        count = by_distance.get(distance, 0)
        print(f"    within edit distance {distance}              : {count}"
              f"   ({count / total:.2%} of M2.1)")
    print(f"    within {max_distance} of a real term (BOUND)      : {near}"
          f"   ({near / total:.2%} of M2.1)")
    print(f"    farther than {max_distance} -- invention proper   : {total - near - exact}"
          f"   ({(total - near - exact) / total:.2%} of M2.1)")

    # Printed because the bound's weakness is coincidence on short strings, and
    # the reader can only discount it if the length distribution is visible.
    lengths = [len(t) for t in lowered if t in distance_of]
    if lengths:
        print(f"    matched-term length: median={st.median(lengths):.0f} "
              f"min={min(lengths)} max={max(lengths)}")
    examples = sorted(matches.items(), key=lambda kv: (kv[1][1], kv[0]))[:8]
    for emitted, (nearest, distance) in examples:
        print(f"      d={distance}  {emitted!r} -> {nearest!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--documents", type=Path, default=Path("documents"),
                    help="dataset directory the run loaded its catalogues from")
    ap.add_argument("--platform", action="append", default=None,
                    help="platform to include, repeatable; default all found")
    ap.add_argument("--max-distance", type=int, default=2)
    ap.add_argument("--model", default=None,
                    help="substring filter on keyword_model, to split a mixed run")
    args = ap.parse_args()

    corpus = corpus_terms(args.documents, args.platform)
    print(f"corpus keywords: {len(corpus)}   max edit distance: {args.max_distance}")

    for run in args.runs:
        rows = load_rows(run)
        if not any(TERMS_FIELD in r for r in rows):
            # Named rather than reported as zero: a run with no terms column is
            # unmeasurable here, and printing 0.00% would read as "no
            # transcription error", the opposite finding.
            print(f"\n### {run.name}\n    REFUSED: no `{TERMS_FIELD}` column. "
                  "This run predates 2026-09-22 and exported counts only; "
                  "the bound needs the strings and cannot be inferred.")
            continue
        if args.model:
            rows = [r for r in rows if args.model in str(r.get("keyword_model") or "")]
        label = run.name + (f"[{args.model}]" if args.model else "")
        report(label, invented_terms(rows), corpus, args.max_distance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
