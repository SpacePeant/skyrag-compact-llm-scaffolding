#!/usr/bin/env python3
"""Rebuild the SKYRAG course corpus from its two public upstream datasets.

**Why this exists.** The catalogues under `documents/` cannot be redistributed:
the Coursera half is Apache 2.0 and could be, but the edX half
(`Knowledge-Innovation-Centre/edX-raw`) declares no licence at all, and absent
is not permissive. Both upstreams are public, so the reproducible artifact is
the *method*, not the data. Point a reviewer at this script and they can rebuild
the corpus themselves without anyone redistributing a byte.

**What it fixes about the original process.** The cleaning logic is a faithful
port of `legacy/mine/datacleaning.ipynb`, which is correct: Coursera reconciles
to the row. Three things around it were not.

1. **The fetch was unpinned.** `downloads.py` reads `/resolve/main/`, so two
   people running it get whatever upstream holds that day. Both revisions are
   pinned below by commit SHA. Upstream has not moved since 2024, so pinning
   makes the rebuild deterministic rather than merely likely.
2. **The fetch was fragile.** edX is 5,203 individually downloaded page files.
   An interrupted fetch leaves a contiguous hole and says nothing, which is
   exactly what happened on 2026-08-04: the shipped catalogue was cleaned from
   an incomplete download and is missing 26 courses (raw lines 3018-3053). This
   script fetches through `huggingface_hub`, which resumes and verifies, and
   asserts the page count before cleaning anything.
3. **The clean skipped silently.** `clean()` skipped any platform whose output
   already existed, so the stale catalogue survived a later complete fetch
   without a word. Here an existing target is reported with its delta and is
   not overwritten without `--force`.

**The verdict this script prints is the point.** It reports the count it built
*and* the count the published study used, and exits non-zero when a catalogue
matches neither. A rebuild that quietly produces a different corpus than the
paper measured is the failure mode worth engineering against.

Usage:

    python scripts/prepare_corpus.py --fetch --clean     # full rebuild
    python scripts/prepare_corpus.py --clean             # raw already present
    python scripts/prepare_corpus.py --verify            # check what is on disk
    python scripts/prepare_corpus.py --clean --as-published --force
                                                         # the 3,451-course edX
                                                         # catalogue the study measured

**`--as-published`.** A full rebuild gives 3,477 edX courses, the published
study measured 3,451. `benchmarks/edx_published_omissions.json` lists the 26
that the incomplete download dropped, by URL. With this flag they are removed
after cleaning, and the result equals the published catalogue record for record
(checked 2026-10-09 against `documents/EdxDataset-Clean.json`). The flag exists
so the measured corpus can be rebuilt exactly. It does not make 3,451 the
correct count.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html as html_mod
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "mine" / "raw"
DOCUMENTS = ROOT / "documents"

# Pinned by commit SHA, not `main`. Queried 2026-09-28; both upstreams have been
# unmodified since 2024, so these are stable rather than a moving snapshot.
COURSERA_REPO = "azrai99/coursera-course-dataset"
COURSERA_SHA = "ccd36c1660fee9610d5262e7517d8397c343168b"   # licence: apache-2.0
COURSERA_CSV = "coursera_course_2024.csv"

EDX_REPO = "Knowledge-Innovation-Centre/edX-raw"
EDX_SHA = "0d8ac6d3aa95a2484789c6ba82f597331d7deb62"        # licence: NONE DECLARED
EDX_PAGES = 5203        # 5,204 repo files, less `.gitattributes`
EDX_JSONL = "edx_pages.jsonl"

# What a correct rebuild produces, against what the published study measured.
# They differ on edX by 26 courses and the difference is understood: see
# `project_knowledge/audit/audit_2026-09-28_edx-corpus-does-not-reproduce-from-its-cleaner.md`.
# Recorded as two numbers rather than one so a rebuild can be told "you are
# right and the published corpus is 26 short" instead of an unhelpful mismatch.
REBUILD_COUNT = {"coursera": 6641, "edx": 3477}
SHIPPED_COUNT = {"coursera": 6641, "edx": 3451}
# The 26 edX courses absent from the published catalogue, keyed by URL, which is
# unique across the 3,477 rebuilt records. See `--as-published`.
EDX_OMISSIONS = ROOT / "benchmarks" / "edx_published_omissions.json"
KNOWN_EDX_DELTA = 26


# --------------------------------------------------------------- helpers
# Ported from legacy/mine/datacleaning.ipynb. Behaviour must not drift from the
# notebook: it is what produced the published Coursera catalogue exactly.

_BLOCK_END = re.compile(r"</(li|p|div|h[1-6]|tr)\s*>|<br\s*/?>", re.I)
_TAG = re.compile(r"<[^>]+>")


def strip_html(raw: Any) -> str:
    """Flatten an HTML fragment to plain text, one block element per line."""
    if not raw:
        return ""
    text = _BLOCK_END.sub("\n", str(raw))
    text = _TAG.sub("", text)
    text = html_mod.unescape(text).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def as_list(raw: Any) -> list[str]:
    """Coerce a keyword-ish field to a de-duplicated list of strings."""
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        text = str(raw).strip()
        if text.startswith("["):
            try:
                parsed = ast.literal_eval(text)
                items = list(parsed) if isinstance(parsed, (list, tuple)) else [text]
            except (ValueError, SyntaxError):
                items = [text.strip("[]")]
        else:
            items = text.split(",")
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item).strip()
        if value and value.lower() not in seen:
            seen.add(value.lower())
            out.append(value)
    return out


def _norm(value: Any) -> str:
    return re.sub(r"\W+", " ", str(value).lower()).strip()


def dedupe(records: list[dict], title_field: str, description_field: str) -> list[dict]:
    """Drop syndicated re-listings, keeping the first of each.

    Title *and* a 150-character description prefix, so three universities'
    "Introduction to Statistics" stay three courses.
    """
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for record in records:
        key = (_norm(record.get(title_field, "")),
               _norm(str(record.get(description_field, ""))[:150]))
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


# --------------------------------------------------------------- coursera

def normalize_coursera(row: dict) -> dict:
    """One CSV row to one record in the pipeline's Coursera schema.

    `What you will learn` is assigned the difficulty level because the upstream
    CSV carries no learning-outcomes column. That is faithful to the published
    corpus and is **not** a transcription slip; it is a documented asymmetry
    against edX, which does carry real outcomes. See
    `audit_2026-09-28_coursera-what-you-will-learn-holds-the-level.md`. Do not
    "fix" it here without re-indexing and re-running every retrieval figure.
    """
    skills = as_list(row.get("Skills"))
    level = str(row.get("Level") or "").strip()
    return {
        "Course Title": str(row.get("title") or "").strip(),
        "Keyword": ", ".join(skills),
        "Skill gain": ", ".join(skills),
        "What you will learn": level,
        "Modules": str(row.get("Modules/Courses") or "").strip(),
        "Course Summary": str(row.get("Description") or "").strip(),
        "Course Url": str(row.get("URL") or "").strip(),
        "Organization": str(row.get("Organization") or "").strip(),
        "Level": level,
    }


def clean_coursera(source: Path) -> list[dict]:
    with open(source, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = [normalize_coursera(row) for row in rows]
    titled = [r for r in records if r["Course Title"]]      # no title is not a course
    deduped = dedupe(titled, "Course Title", "Course Summary")
    print(f"  coursera: {len(rows)} rows -> {len(titled)} titled -> {len(deduped)} deduped")
    return deduped


# --------------------------------------------------------------- edx

def classify_edx(payload: dict) -> str:
    """Why a page was kept or dropped: kept / not-a-course / non-english.

    A missing language is *kept*. That is deliberate and matches the published
    corpus: one page in 5,203 has no declared language.
    """
    course = (payload.get("result") or {}).get("pageContext", {}).get("course")
    if not course or not course.get("title"):
        return "not-a-course"
    run = course.get("activeCourseRun") or {}
    language = str(run.get("contentLanguage") or run.get("language") or "").lower()
    if language and not language.startswith("en"):
        return "non-english"
    return "kept"


def normalize_edx(payload: dict) -> dict | None:
    if classify_edx(payload) != "kept":
        return None
    course = payload["result"]["pageContext"]["course"]
    subjects = list(course.get("skillNames") or [])
    primary = ((course.get("primarySubject") or {}).get("json") or {}).get("name")
    if primary:
        subjects.append(primary)
    description = "\n\n".join(
        part for part in (strip_html(course.get("shortDescription")),
                          strip_html(course.get("fullDescription"))) if part
    )
    path = course.get("prospectusPath") or payload.get("path") or ""
    return {
        "course_title": str(course.get("title")).strip(),
        "subject": ", ".join(as_list(subjects)),
        # .splitlines() gives a real list. Handed a string, the downstream
        # "\n".join() would iterate characters and garble every prompt.
        "what_you_will_learn_list": strip_html(course.get("outcome")).splitlines(),
        "description": description,
        "syllabus": strip_html(course.get("syllabusRaw")),
        "prerequisites": strip_html(course.get("prerequisitesRaw")),
        "url": f"https://www.edx.org{path}" if str(path).startswith("/") else str(path),
        "Organization": ", ".join(str(o.get("name", "")) for o in (course.get("owners") or [])
                                  if o.get("name")),
        "level": str(course.get("levelType") or ""),
    }


def clean_edx(source: Path) -> list[dict]:
    # Iterate the handle rather than `read_text().splitlines()`. `splitlines()`
    # also breaks on U+2028, U+2029 and U+0085, which occur inside JSON string
    # values, and it reported 5,224 "lines" in this 5,203-line file -- enough to
    # trip the completeness check below against a dump that was actually whole.
    with source.open(encoding="utf-8") as handle:
        lines = [ln for ln in handle if ln.strip()]
    # Assert completeness BEFORE cleaning. An incomplete dump is exactly how the
    # published catalogue lost 26 courses, and it produced a plausible file.
    if len(lines) != EDX_PAGES:
        raise SystemExit(
            f"\n  {source.name} holds {len(lines)} pages, expected {EDX_PAGES}.\n"
            f"  The upstream dump is {EDX_PAGES} pages at revision {EDX_SHA[:12]}.\n"
            f"  A short file means an incomplete fetch. Re-run with --fetch;\n"
            f"  do NOT clean a partial dump, it yields a plausible catalogue\n"
            f"  that is silently missing courses."
        )
    records: list[dict] = []
    outcomes: Counter[str] = Counter()
    for line in lines:
        payload = json.loads(line)
        outcomes[classify_edx(payload)] += 1
        record = normalize_edx(payload)
        if record:
            records.append(record)
    deduped = dedupe(records, "course_title", "description")
    print(f"  edx: {len(lines)} pages -> "
          + ", ".join(f"{n} {k}" for k, n in outcomes.most_common())
          + f" -> {len(deduped)} deduped")
    return deduped


# --------------------------------------------------------------- fetch

def fetch(dest: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        raise SystemExit(
            "--fetch needs `huggingface_hub`:  pip install huggingface_hub\n"
            "It is used rather than plain urllib because it resumes and "
            "verifies; the 5,203-file edX fetch is what failed silently before."
        )
    dest.mkdir(parents=True, exist_ok=True)

    print(f"  coursera <- {COURSERA_REPO}@{COURSERA_SHA[:12]}")
    path = hf_hub_download(COURSERA_REPO, COURSERA_CSV, revision=COURSERA_SHA,
                           repo_type="dataset")
    (dest / COURSERA_CSV).write_bytes(Path(path).read_bytes())

    print(f"  edx      <- {EDX_REPO}@{EDX_SHA[:12]}  ({EDX_PAGES} page files)")
    # `local_dir` writes real files. The default cache links them in with
    # symlinks, which Windows refuses without Developer Mode (WinError 1314,
    # found on a fresh-clone rebuild 2026-10-20). Four workers instead of the
    # default eight keep the 5,203 anonymous requests nearer the Hub's rate
    # limit; an HF token (`HF_TOKEN`) raises that limit.
    snap = Path(snapshot_download(EDX_REPO, revision=EDX_SHA, repo_type="dataset",
                                  allow_patterns=["*.json"], max_workers=4,
                                  local_dir=dest / "edx_pages"))
    pages = sorted(snap.glob("*.json"),
                   key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if len(pages) != EDX_PAGES:
        raise SystemExit(f"  fetched {len(pages)} page files, expected {EDX_PAGES}")
    with (dest / EDX_JSONL).open("w", encoding="utf-8") as out:
        for page in pages:
            out.write(page.read_text(encoding="utf-8").replace("\n", " ").strip() + "\n")
    print(f"  wrote {dest / EDX_JSONL}")


# --------------------------------------------------------------- verify

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def verify() -> int:
    """Report what is on disk against both reference counts. Returns an exit code."""
    problems = 0
    print("\n  platform   on disk   rebuild   published   verdict")
    for platform, name in (("coursera", "CourseraDataset_clean.json"),
                           ("edx", "EdxDataset-Clean.json")):
        path = DOCUMENTS / name
        if not path.exists():
            print(f"  {platform:<10} {'MISSING':>7}")
            problems += 1
            continue
        found = len(json.loads(path.read_text(encoding="utf-8")))
        rebuild, shipped = REBUILD_COUNT[platform], SHIPPED_COUNT[platform]
        if found == rebuild == shipped:
            verdict = "ok"
        elif found == shipped and rebuild - shipped == KNOWN_EDX_DELTA:
            verdict = f"published corpus, {KNOWN_EDX_DELTA} short of a clean rebuild (known)"
        elif found == rebuild:
            verdict = f"clean rebuild, {KNOWN_EDX_DELTA} MORE than the published study used"
        else:
            verdict = "UNRECOGNISED, matches neither reference"
            problems += 1
        print(f"  {platform:<10} {found:>7} {rebuild:>9} {shipped:>11}   {verdict}")
        print(f"  {'':<10} sha256[:16] = {_sha256(path)}")
    if problems:
        print("\n  FAIL: a catalogue matches neither the rebuild nor the published corpus.")
        print("  Do not run experiments against it. It is not a corpus this study describes.")
    return 1 if problems else 0


# --------------------------------------------------------------- main

def drop_published_omissions(records: list[dict]) -> list[dict]:
    """Remove the courses the published edX catalogue lacks.

    Refuses unless every listed URL is found exactly once, so a changed upstream
    or an edited list cannot quietly produce a third catalogue.
    """
    omitted = {row["url"] for row in json.loads(EDX_OMISSIONS.read_text(encoding="utf-8"))}
    present = [r["url"] for r in records if r["url"] in omitted]
    if len(present) != len(omitted) or len(set(present)) != len(omitted):
        raise SystemExit(
            f"\n  {EDX_OMISSIONS.name} lists {len(omitted)} courses, "
            f"{len(set(present))} of them are in the rebuilt catalogue.\n"
            f"  Not writing a catalogue that is neither the full rebuild nor the published one."
        )
    kept = [r for r in records if r["url"] not in omitted]
    print(f"  edx: --as-published removed {len(records) - len(kept)} courses -> {len(kept)}")
    return kept


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch", action="store_true", help="download both pinned datasets")
    parser.add_argument("--clean", action="store_true", help="build the catalogues")
    parser.add_argument("--verify", action="store_true", help="check what is on disk")
    parser.add_argument("--force", action="store_true", help="overwrite existing catalogues")
    parser.add_argument("--as-published", action="store_true",
                        help="drop the 26 edX courses the published catalogue lacks (3,451, not 3,477)")
    parser.add_argument("--raw", type=Path, default=RAW, help=f"raw directory (default {RAW})")
    parsed = parser.parse_args(argv)
    if not (parsed.fetch or parsed.clean or parsed.verify):
        parser.print_help()
        return 2

    if parsed.fetch:
        print("\nfetching pinned upstreams")
        fetch(parsed.raw)

    if parsed.clean:
        print("\ncleaning")
        targets = {
            "coursera": (parsed.raw / COURSERA_CSV,
                         DOCUMENTS / "CourseraDataset_clean.json", clean_coursera),
            "edx": (parsed.raw / EDX_JSONL,
                    DOCUMENTS / "EdxDataset-Clean.json", clean_edx),
        }
        for platform, (source, target, cleaner) in targets.items():
            if not source.exists():
                print(f"  {platform}: {source} not present, run with --fetch")
                return 1
            if target.exists() and not parsed.force:
                # Loud, not silent. A silent skip here is what let a catalogue
                # cleaned from an incomplete dump survive a later complete fetch.
                existing = len(json.loads(target.read_text(encoding="utf-8")))
                print(f"  {platform}: {target.name} already exists with {existing} records; "
                      f"a rebuild would produce {REBUILD_COUNT[platform]}. "
                      f"Not overwriting. Pass --force if that is what you want.")
                continue
            records = cleaner(source)
            if platform == "edx" and parsed.as_published:
                records = drop_published_omissions(records)
            # A fresh clone has no `documents/` (it is gitignored).
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(records, ensure_ascii=False, indent=1),
                              encoding="utf-8")
            print(f"  wrote {target.name} ({len(records)} courses)")

    if parsed.verify or parsed.clean:
        return verify()
    return 0


if __name__ == "__main__":
    sys.exit(main())
