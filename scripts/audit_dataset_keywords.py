"""Count index-time keyword-extraction failures in the built datasets.

Cross-reference:
- backlog_2026-08-17_bugfix_unified-code-audit-remediation.md, task 4.1
- backlog_2026-08-17_bugfix-fail-loud-class-sweep.md, finding S2

    python scripts/audit_dataset_keywords.py                 # from the repo root
    python scripts/audit_dataset_keywords.py --sample 5      # show example records

Finding S2 is that `extract_keywords_for_text` catches `LLMError` and returns `[]`,
documented in its own docstring as deliberate: "Returns [] on failure rather than
aborting a long run." A course whose extraction failed is therefore written to
`*_with_keywords.json` with zero keywords, indistinguishable from a course that
genuinely has none.

This is the least visible finding in that backlog, because it corrupts the
vocabulary permanently at index time: an affected course is unreachable by keyword
retrieval in every run, in every condition, for every model. It cannot be fixed
after the fact by changing the pipeline.

The datasets are already built, so the actionable question is not "should we make
it fail loudly" -- it is **whether failures already occurred, and how many**. That
is countable, and this script counts it.

What the count cannot tell us, and this script does not pretend to: a zero-keyword
course could be a failed LLM call or a course with genuinely nothing extractable
(an empty description). The two are indistinguishable in the artifact -- that is
precisely the defect. What the script can do is report how much source text each
zero-keyword course had, because a course with a long description and no keywords
is far more likely to be a swallowed error than one with no description at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skyrag.data.keywords import KEYWORD_FIELD, SOURCE_FIELDS, _source_text
from skyrag.data.schema import split_keywords

# platform -> the built file that carries extracted keywords
DATASETS = {
    "coursera": "CourseraDataset_EN_with_keywords.json",
    "edx": "EdxDataset_with_keywords.json",
    "udemy": "UdemyDataset_with_keywords.json",
}

# Above this many characters of source text, "nothing extractable" stops being a
# credible explanation for an empty keyword field.
SUBSTANTIAL_TEXT = 200


def audit(path: Path, platform: str, sample: int) -> dict:
    records = json.loads(path.read_text(encoding="utf-8"))
    field = KEYWORD_FIELD.get(platform, "keyword")

    empty_with_text: list[dict] = []
    empty_without_text = 0
    total_keywords = 0
    counts: dict[int, int] = {}

    for record in records:
        keywords = split_keywords(record.get(field))
        counts[len(keywords)] = counts.get(len(keywords), 0) + 1
        total_keywords += len(keywords)
        if keywords:
            continue
        text = _source_text(record, platform)
        if len(text) >= SUBSTANTIAL_TEXT:
            empty_with_text.append({"record": record, "chars": len(text)})
        else:
            empty_without_text += 1

    n = len(records)
    empty = len(empty_with_text) + empty_without_text

    print(f"\n{'=' * 78}")
    print(f"{platform}  --  {path.name}")
    print(f"{'=' * 78}")
    print(f"  courses                        : {n:,}")
    print(f"  keyword field                  : {field!r}")
    print(f"  total keywords                 : {total_keywords:,}")
    print(f"  mean keywords per course       : {total_keywords / n:.2f}" if n else "")
    print()
    print(f"  ZERO keywords                  : {empty:,} ({empty / n:.2%})" if n else "")
    print(f"    ...with < {SUBSTANTIAL_TEXT} chars of source text : {empty_without_text:,} "
          f"(plausibly nothing to extract)")
    print(f"    ...with >= {SUBSTANTIAL_TEXT} chars of source text: {len(empty_with_text):,} "
          f"(a swallowed LLMError is the likely cause)")
    print()
    print("  keyword-count distribution:")
    for k in sorted(counts):
        bar = "#" * min(50, round(50 * counts[k] / n)) if n else ""
        print(f"    {k:>2} keyword(s) : {counts[k]:>7,}  {bar}")

    if sample and empty_with_text:
        print()
        print(f"  Sample of zero-keyword courses that DID have source text "
              f"(showing {min(sample, len(empty_with_text))}):")
        worst = sorted(empty_with_text, key=lambda e: -e["chars"])[:sample]
        title_fields = ("Course Title", "course_title", "title")
        for entry in worst:
            record = entry["record"]
            title = next((record[f] for f in title_fields if record.get(f)), "<untitled>")
            print(f"    - {str(title)[:66]:<66} {entry['chars']:>6,} chars")

    return {
        "platform": platform,
        "courses": n,
        "zero_keyword": empty,
        "zero_with_text": len(empty_with_text),
        "zero_without_text": empty_without_text,
        "total_keywords": total_keywords,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--documents", default="documents",
                        help="Directory holding the built *_with_keywords.json files.")
    parser.add_argument("--sample", type=int, default=5,
                        help="How many suspicious records to show per platform (0 = none).")
    args = parser.parse_args()

    root = Path(args.documents)
    if not root.is_dir():
        print(f"No such directory: {root.resolve()}")
        return 2

    results = []
    for platform, filename in DATASETS.items():
        path = root / filename
        if not path.exists():
            print(f"\n  (skipping {platform}: {filename} not present)")
            continue
        results.append(audit(path, platform, args.sample))

    if not results:
        print("\nNo built keyword datasets found; nothing to audit.")
        return 2

    print(f"\n{'=' * 78}")
    print("VERDICT")
    print(f"{'=' * 78}")
    suspicious = sum(r["zero_with_text"] for r in results)
    courses = sum(r["courses"] for r in results)
    for r in results:
        print(f"  {r['platform']:<10} {r['zero_keyword']:>6,} of {r['courses']:>7,} courses "
              f"have no keywords; {r['zero_with_text']:,} of those had source text")

    print()
    if suspicious == 0:
        print("  No course with substantial source text lacks keywords, on any platform.")
        print("  So no evidence of swallowed index-time extraction failures in these files,")
        print("  and re-indexing to repair S2 is NOT indicated. The defect in the code is")
        print("  still real and still worth closing before the next index run -- this says")
        print("  only that it did not fire during the runs that produced these datasets.")
    else:
        share = suspicious / courses if courses else 0
        print(f"  {suspicious:,} course(s) ({share:.2%}) have source text but no keywords.")
        print("  These are unreachable by keyword retrieval in every run and condition.")
        print("  Re-extracting keywords for exactly these records repairs the vocabulary")
        print("  without paying to re-index the whole catalog. Whether it changes any")
        print("  reported number depends on how many are in the retrieved window, which is")
        print("  a separate measurement -- do not assume it is negligible or that it matters.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
