"""Per-platform loaders: raw JSON -> list[Course].

Each adapter is tolerant of missing fields, because the scraped datasets are
uneven and a single absent key should not abort a run over thousands of
courses.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .schema import PLATFORM_PREFIX, Course, split_keywords


def _get(record: dict, *names, default: str = "") -> str:
    """First present, non-empty value among `names`.

    Absorbs the casing/underscore drift between the datasets (e.g.
    "Course Title" vs "course_title").
    """
    for name in names:
        value = record.get(name)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, list):
            return "\n".join(str(v) for v in value)
        return str(value)
    return default


def _course_id(platform: str, index: int) -> str:
    # Coursera was 1-based in the notebooks, the others 0-based. Unified to
    # 1-based: IDs are opaque labels, and consistency beats bug-compatibility.
    return f"{PLATFORM_PREFIX.get(platform, platform[:3].upper())}{index + 1}"


def load_coursera(records: list[dict]) -> list[Course]:
    return [
        Course(
            id_=_course_id("coursera", i),
            platform="coursera",
            title=_get(r, "Course Title", "course_title", "title"),
            keywords=split_keywords(r.get("Keyword") or r.get("keyword")),
            what_you_will_learn=_get(r, "What you will learn", "what_you_will_learn"),
            skills=_get(r, "Skill gain", "skill_gain", "skills"),
            modules=_get(r, "Modules", "modules"),
            description=_get(r, "Course Summary", "description"),
            url=_get(r, "Course Url", "course_url", "url"),
            # O3. "Beginner level" / "Intermediate level" / "Advanced level" on
            # 88.3% of this catalogue; blank on the rest.
            level=_get(r, "Level", "level"),
        )
        for i, r in enumerate(records)
    ]


def load_udemy(records: list[dict]) -> list[Course]:
    return [
        Course(
            id_=_course_id("udemy", i),
            platform="udemy",
            title=_get(r, "course_title", "Course Title", "title"),
            keywords=split_keywords(r.get("keyword") or r.get("Keyword")),
            what_you_will_learn=_get(r, "what_you_will_learn", "What you will learn"),
            description=_get(r, "description", "Course Summary"),
            url=_get(r, "url", "course_url", "Course Url"),
            level=_get(r, "level", "Level"),
        )
        for i, r in enumerate(records)
    ]


def load_edx(records: list[dict]) -> list[Course]:
    # edX has no keyword field; the notebooks repurpose `subject`.
    return [
        Course(
            id_=_course_id("edx", i),
            platform="edx",
            title=_get(r, "course_title", "Course Title", "title"),
            keywords=split_keywords(r.get("subject") or r.get("keyword")),
            what_you_will_learn=_get(
                r, "what_you_will_learn_list", "what_you_will_learn", "What you will learn"
            ),
            description=_get(r, "description", "Course Summary"),
            url=_get(r, "url", "course_url", "Course Url"),
            # O3. "Introductory" / "Intermediate" / "Advanced" on 100% of edx.
            level=_get(r, "level", "Level"),
        )
        for i, r in enumerate(records)
    ]


ADAPTERS: dict[str, Callable[[list[dict]], list[Course]]] = {
    "coursera": load_coursera,
    "udemy": load_udemy,
    "edx": load_edx,
}

# Default filenames produced by the Keyword_Gen_* stage.
DEFAULT_FILENAMES = {
    "coursera": "CourseraDataset_EN_with_keywords.json",
    "udemy": "UdemyDataset_with_keywords.json",
    "edx": "EdxDataset_with_keywords.json",
}


def load_platform(platform: str, path: Path) -> list[Course]:
    if platform not in ADAPTERS:
        raise ValueError(
            f"No adapter for platform {platform!r}. Known: {', '.join(ADAPTERS)}"
        )
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"{path} should contain a JSON array of course objects.")
    return ADAPTERS[platform](records)


@dataclass
class CorpusCoverage:
    """Which platforms a run asked for versus which it actually got.

    Recorded rather than inferred, because a two-platform corpus and a missing
    third dataset produce identical downstream numbers. The "Separated" in
    Separated Keyword RAG *is* the per-platform selection and merge, so how many
    platforms took part is a property of the result, not a detail of the load.
    """

    requested: list[str]
    loaded: list[str]
    missing: list[str]
    course_counts: dict[str, int]

    @property
    def complete(self) -> bool:
        return not self.missing

    @property
    def total_courses(self) -> int:
        return sum(self.course_counts.values())

    def describe(self) -> str:
        got = ", ".join(f"{p}: {self.course_counts[p]}" for p in self.loaded) or "none"
        text = f"{self.total_courses} courses across {len(self.loaded)} platform(s) ({got})"
        if self.missing:
            text += f"; MISSING: {', '.join(self.missing)}"
        return text


def load_all_with_coverage(
    documents_dir: Path,
    platforms: list[str] | None = None,
    *,
    allow_missing: bool = False,
) -> tuple[dict[str, list[Course]], CorpusCoverage]:
    """Load the requested platforms and report what was actually found.

    Raises when a requested platform has no dataset, unless `allow_missing` is
    set. Running on a reduced corpus is a legitimate choice, but it has to be an
    explicit one: silently skipping an absent file is how a run ends up reporting
    a two-platform result as though three had been intended.
    """
    requested = list(platforms or ADAPTERS)
    unknown = [p for p in requested if p not in DEFAULT_FILENAMES]
    if unknown:
        raise ValueError(
            f"Unknown platform(s) {unknown}. Known: {', '.join(DEFAULT_FILENAMES)}"
        )

    out: dict[str, list[Course]] = {}
    missing: list[str] = []
    for platform in requested:
        path = documents_dir / DEFAULT_FILENAMES[platform]
        if path.exists():
            out[platform] = load_platform(platform, path)
        else:
            missing.append(platform)

    if missing and not allow_missing:
        raise FileNotFoundError(
            f"Requested platform(s) {missing} have no dataset in {documents_dir}. "
            f"Loaded: {sorted(out) or 'none'}. Pass allow_missing=True to proceed "
            "on a reduced corpus -- that is a disclosed design decision, not a default."
        )

    coverage = CorpusCoverage(
        requested=requested,
        loaded=sorted(out),
        missing=missing,
        course_counts={p: len(c) for p, c in out.items()},
    )
    return out, coverage


def load_all(
    documents_dir: Path,
    platforms: list[str] | None = None,
    *,
    allow_missing: bool = False,
) -> dict[str, list[Course]]:
    """Load the requested platforms. See `load_all_with_coverage`."""
    courses, _ = load_all_with_coverage(
        documents_dir, platforms, allow_missing=allow_missing
    )
    return courses


def build_vocabulary(courses: list[Course]) -> dict[str, int]:
    """keyword -> frequency, for one platform.

    This is the controlled vocabulary the LLM must choose from during keyword
    selection, and the set its answers are validated against.

    Keys are case-folded, because this dict sits on both sides of a comparison
    whose other side already folds. `selection._coerce` lowercases every keyword a
    model returns, so a vocabulary carrying `Python` could never match the reply
    `python` -- and `classify` reads that miss as "absent from the corpus", i.e. an
    invention. A real corpus keyword was charged as a hallucination on the basis of
    its capitalisation alone, which inflates the headline metric in the direction
    that flatters the finding.

    `split_keywords` folds at ingest, so corpora built through the adapters were
    already safe; `Course.from_dict` does not, so any store built from an
    already-keyworded documents file was not. Folding here rather than trusting
    every producer is what makes the property hold for both.

    Counts are merged across casings: `Python` and `python` are one keyword with
    the summed frequency, which is what the vocabulary is for.
    """
    counter: Counter[str] = Counter()
    for course in courses:
        counter.update(kw.strip().lower() for kw in course.keywords if kw.strip())
    # Descending frequency, ties broken by keyword (L1). `most_common()` is
    # stable, so equal-frequency keywords used to be ordered by whichever course
    # mentioned one first -- a property of the file's row order, not of the
    # corpus. Two things read this order and neither tolerates that:
    # `select_keywords` takes a `[:max_vocabulary]` prefix, so a tie straddling
    # rank 1,000 decides whether a keyword is shown to the model at all; and
    # `selected_ranks` (M1.2) exports the position as evidence of position bias.
    # M1.2's docstring already claims the window is "a property of the corpus,
    # not of the run" -- that claim was true only up to ties until now.
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


# Institution names that leak into edX's `subject` field; they are publishers,
# not topics, and pollute keyword selection. (Deleted by hand in the notebook.)
NON_TOPIC_KEYWORDS = {"aws", "nyif", "nus", "stanfordonline", "ibm", "microsoft", "google"}


def clean_vocabulary(
    vocab: dict[str, int], *, min_count: int = 1, drop: set[str] | None = None
) -> dict[str, int]:
    """Drop publisher names and rare keywords from a platform's vocabulary.

    Comparison is case-folded on both sides. `NON_TOPIC_KEYWORDS` is written in
    lowercase, and a plain `k not in drop` therefore matched only keywords that
    were already lowercase -- so `AWS` and `IBM` reached the prompt while `aws` and
    `ibm` were removed. `split_keywords` lowercases at ingest, which hid this for
    corpora built through the adapters, but any vocabulary assembled from
    already-keyworded records keeps its original casing.

    `drop` entries are folded too, so a caller passing `{"AWS"}` gets the same
    behaviour as one passing `{"aws"}`.
    """
    drop = {d.strip().lower() for d in (drop if drop is not None else NON_TOPIC_KEYWORDS)}
    return {
        k: v for k, v in vocab.items()
        if v >= min_count and k.strip().lower() not in drop
    }


@dataclass(frozen=True)
class Vocabulary:
    """One platform's keywords, carrying the provenance of every exclusion.

    A keyword can be missing from the prompt shown to the model for three
    different reasons, and when the model names it anyway those three mean
    opposite things for the measurement:

    * **removed by cleaning** -- real in the corpus, deliberately excluded
      because it is a publisher rather than a topic (`aws`, `ibm`, ...) or falls
      under `min_count`. The model naming it is not an error at all.
    * **below the window cut** -- real and eligible, but outside the top
      `max_vocabulary` slice actually shown. A coverage limit of this harness.
    * **absent entirely** -- invented. The model-capability signal, and the only
      one of the three that belongs in a hallucination rate.

    Handing the pipeline a single already-cleaned dict destroys the first
    distinction: a cleaned-away keyword is in neither the window nor the
    "complete" vocabulary, so it reaches the invented bucket by elimination.
    That is how `aws` came to be charged as a hallucination against a corpus
    that contains it, and it biases edX hardest, since edX's keywords come from
    the `subject` field where institution names live. Keeping the three sets in
    one value is what stops a caller reconstructing the same conflation by
    passing the wrong dict twice.
    """

    eligible: dict[str, int]    # survived cleaning: the pool the model picks from
    corpus: dict[str, int]      # every keyword the platform's courses actually carry
    removed: dict[str, int]     # in `corpus`, excluded from `eligible` by cleaning

    @classmethod
    def uncleaned(cls, vocab: dict[str, int]) -> "Vocabulary":
        """Declare a plain dict as a vocabulary that had no cleaning applied.

        An accurate statement, not a default: with nothing removed, `corpus` and
        `eligible` genuinely coincide and `removed` is genuinely empty. Callers
        that *did* clean must build the value with `build_platform_vocabulary`
        instead, or the removals become invisible again.

        Keys are case-folded, for the same reason `build_vocabulary` folds them and
        in the same words: `selection._coerce` lowercases every keyword a model
        returns, so a vocabulary carrying `Python` cannot match the reply `python`,
        and `classify` reads that miss as *absent from the corpus* -- an invention.
        A real corpus keyword is then charged as a hallucination on the strength of
        its capitalisation.

        Folding `build_vocabulary` alone was an incomplete repair. That function is
        how the pipeline builds a vocabulary; this classmethod is how *any other
        caller* does, and it is the documented way to pass a plain dict to
        `select_keywords`. A new evaluation arm assembling its own vocabulary --
        exactly what an ablation or an embedder comparison does -- would have taken
        this path and silently manufactured hallucinations in the metric the central
        claim rests on. The fold has to sit at every entry point or it sits at none.
        """
        folded: dict[str, int] = {}
        for keyword, count in vocab.items():
            key = str(keyword).strip().lower()
            if key:
                folded[key] = folded.get(key, 0) + count
        return cls(eligible=dict(folded), corpus=dict(folded), removed={})

    def __len__(self) -> int:
        return len(self.eligible)

    def __bool__(self) -> bool:
        return bool(self.eligible)


def build_platform_vocabulary(
    courses: list[Course], *, min_count: int = 1, drop: set[str] | None = None
) -> Vocabulary:
    """The controlled vocabulary for one platform, with its exclusions recorded."""
    corpus = build_vocabulary(courses)
    eligible = clean_vocabulary(corpus, min_count=min_count, drop=drop)
    removed = {k: v for k, v in corpus.items() if k not in eligible}
    return Vocabulary(eligible=eligible, corpus=corpus, removed=removed)
