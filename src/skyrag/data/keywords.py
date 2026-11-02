"""Stage 1: LLM keyword extraction (port of Keyword_Gen_{Coursera,Edx,Udemy}.py).

Those three scripts were ~90% identical, differing only in input path and
which fields they fed the model. They collapse into one parameterized
function here.

This is the most expensive stage: one LLM call per course, over the entire
catalog. So it checkpoints after every batch and skips courses that already
have keywords -- an interrupted run resumes instead of paying twice.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from ..llm import GenerationParams, LLMBackend, LLMError
from .schema import split_keywords

# Field groups fed to the extractor, per platform.
#
# "Course Summary" matters more than it looks: the Coursera rows that reach
# this stage are precisely the ones whose `Skill gain` was empty, so without
# the description the model would be extracting from a title alone.
SOURCE_FIELDS = {
    "coursera": ("Course Title", "What you will learn", "Skill gain", "Modules", "Course Summary"),
    "udemy": ("course_title", "what_you_will_learn", "description"),
    "edx": ("course_title", "what_you_will_learn_list", "description"),
}

KEYWORD_FIELD = {"coursera": "Keyword", "udemy": "keyword", "edx": "subject"}

PROMPT = """Extract {n} topical keywords from the following course description.

Rules:
- Return ONLY the keywords, separated by commas, on a single line.
- Each keyword must be 1-2 words.
- Use subject topics (e.g. "machine learning", "organic chemistry").
- Do NOT return institution or provider names (e.g. "IBM", "AWS", "Stanford").
- No preamble, no numbering, no explanation.

Course:
{text}
"""


def _source_text(record: dict, platform: str) -> str:
    parts = []
    for field in SOURCE_FIELDS.get(platform, ()):
        value = record.get(field)
        if not value:
            continue
        if isinstance(value, list):
            value = "\n".join(str(v) for v in value)
        parts.append(f"{field}: {value}")
    return "\n".join(parts)


def extract_keywords_for_text(
    backend: LLMBackend,
    text: str,
    *,
    model: str | None = None,
    n: int = 5,
    params: GenerationParams | None = None,
) -> list[str]:
    """One extraction call. Returns [] on failure rather than aborting a long run."""
    if not text.strip():
        return []
    params = params or GenerationParams(temperature=0.0, max_tokens=128)
    try:
        response = backend.generate(
            PROMPT.format(n=n, text=text[:6000]), model=model, params=params
        )
    except LLMError:
        return []

    # Models sometimes prepend a line despite instructions; the keyword line is
    # the last non-empty one containing a comma.
    lines = [ln.strip() for ln in response.text.splitlines() if ln.strip()]
    line = next((ln for ln in reversed(lines) if "," in ln), lines[-1] if lines else "")
    return split_keywords(line)[:n]


# Fields tried, in order, to identify a record across runs. A URL is unique where
# it exists; a title is not guaranteed unique but is stable, and a collision costs
# a reused keyword line on a duplicate course rather than a wrong measurement.
# Both spellings of each field appear because the three datasets disagree on
# casing and separators, and the checkpoint carries whichever the source used.
_IDENTITY_FIELDS = (
    "Course Url", "course_url", "url",
    "Course Title", "course_title", "title",
)


def _record_key(record: dict) -> str | None:
    """A stable identity for one course record, or None if it carries none.

    The field name is part of the key, so a record matched on its URL can never
    be confused with a different record whose *title* happens to equal that URL.
    """
    for name in _IDENTITY_FIELDS:
        value = record.get(name)
        if isinstance(value, str) and value.strip():
            return f"{name}={value.strip()}"
    return None


def _adopt_checkpoint(
    records: list[dict], existing: list[dict], field: str, output_path: Path
) -> int:
    """Carry finished keyword lines from a checkpoint onto a changed catalog.

    Returns how many records were resumed.

    The guard this replaced was `if len(existing) == len(records)`, and when the
    two differed the checkpoint was not merely ignored: `records` stayed as the
    freshly loaded list and the final `_write` overwrote the checkpoint with it.
    One course added to a 6,641-course catalog therefore silently re-paid for the
    entire catalog -- at one LLM call per course, the most expensive stage in the
    project -- while the docstring promised a later run "picks up exactly where it
    left off".
    """
    done = {}
    for record in existing:
        key = _record_key(record)
        if key and split_keywords(record.get(field)):
            done[key] = record[field]

    resumed = 0
    for record in records:
        key = _record_key(record)
        if key and key in done and not split_keywords(record.get(field)):
            record[field] = done[key]
            resumed += 1

    if done and not resumed:
        # Every record in the checkpoint was paid for, and none of it can be
        # matched to the catalog now in hand. Continuing would re-pay for the
        # whole catalog and then overwrite the checkpoint with the result, which
        # destroys the evidence of what was already bought. A mismatch this total
        # is a wrong-file or wrong-platform mistake, not an incremental change.
        raise ValueError(
            f"{output_path} holds {len(done)} keyworded record(s), none of which "
            f"matches any of the {len(records)} record(s) now being processed. "
            "Resuming is impossible and continuing would re-pay for the entire "
            "catalog and then overwrite this checkpoint with the result. Check "
            "that the platform and the input file are the ones this checkpoint "
            "was built from; pass overwrite=True to discard it deliberately."
        )
    return resumed


def generate_keywords(
    backend: LLMBackend,
    records: list[dict],
    platform: str,
    *,
    output_path: Path,
    model: str | None = None,
    n: int = 5,
    checkpoint_every: int = 25,
    overwrite: bool = False,
    progress: bool = True,
    limit: int | None = None,
    workers: int = 8,
) -> list[dict]:
    """Add a keyword field to every record, checkpointing as it goes.

    Records that already have keywords are skipped unless `overwrite` -- which
    is what makes `limit` useful: it stops after N *extractions*, writes every
    record, and a later call without `limit` picks up exactly where it left off.
    Run a slice, check the cost and the output, then run the rest.

    Resume survives a catalog that has changed size. An unchanged record count
    adopts the checkpoint wholesale, as before; a changed one matches record by
    record on a stable identity (URL, else title) and carries the finished lines
    across. What was reused and what is about to be re-extracted is printed, so
    the cost of a run is visible before it is incurred rather than inferred from
    the bill.
    """
    field = KEYWORD_FIELD.get(platform, "keyword")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from a partial run if one exists.
    if output_path.exists() and not overwrite:
        with open(output_path, encoding="utf-8") as f:
            existing = json.load(f)
        if len(existing) == len(records):
            records = existing
        else:
            resumed = _adopt_checkpoint(records, existing, field, output_path)
            print(
                f"  keywords[{platform}]: checkpoint holds {len(existing)} record(s), "
                f"catalog holds {len(records)} -- resumed {resumed} by identity."
            )

    pending = [
        record for record in records
        if overwrite or not split_keywords(record.get(field))
    ]
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        _write(output_path, records)
        return records

    def extract(record: dict) -> str:
        return ", ".join(
            extract_keywords_for_text(backend, _source_text(record, platform), model=model, n=n)
        )

    # Each call is a second or two of waiting on the API, so running them one
    # at a time makes a 2,000-course catalog a 40-minute stage. The work is
    # IO-bound and every task touches only its own record, so a thread pool is
    # safe -- and results are consumed in order, keeping checkpoints coherent.
    results: Iterable = zip(pending, _pool_map(extract, pending, workers))
    if progress:
        try:
            from tqdm.auto import tqdm

            results = tqdm(results, desc=f"keywords[{platform}]", total=len(pending))
        except ImportError:
            pass

    for processed, (record, keywords) in enumerate(results, start=1):
        record[field] = keywords
        if processed % checkpoint_every == 0:
            _write(output_path, records)

    _write(output_path, records)
    return records


def _pool_map(fn, items: list, workers: int) -> Iterable:
    """Ordered, lazy map over a thread pool -- one worker means no pool at all.

    Kept separate from `_threaded_map` because a function containing `yield`
    is a generator throughout, and the single-worker path must return a plain
    iterator rather than becoming one.
    """
    return map(fn, items) if workers <= 1 else _threaded_map(fn, items, workers)


def _threaded_map(fn, items: list, workers: int) -> Iterable:
    """Ordered, lazy map over a thread pool, with deterministic teardown.

    `pool.map` submits every item up front, so abandoning this generator early --
    an exception in the consumer, a `break`, the object being collected -- leaves a
    queue of work behind. Each item here is a paid API call, so what happens next
    is a billing question, not just a tidiness one.

    `shutdown(wait=True, cancel_futures=True)` states both halves explicitly:
    queued-but-unstarted calls are cancelled, and the handful already in flight are
    waited for so the pool's threads are actually joined before this returns. The
    previous `shutdown(wait=False)` relied on `Executor.map`'s own generator
    cancelling pending futures as it was collected -- which it does, so nothing was
    leaking in practice, but it left the guarantee resting on a CPython
    implementation detail and the worker threads unjoined.
    """
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        yield from pool.map(fn, items)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _write(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    tmp.replace(path)  # atomic: a crash mid-write cannot corrupt the checkpoint
