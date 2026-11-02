"""B10 — the render fingerprint in `CacheManifest`.

`Course.render()` is what `Embedder.embed_courses` encodes. `CacheManifest`
recorded the embedder name, the dimension and the row count and **not** the
render format, so changing `render()` left every cached matrix on disk loading
clean while its vectors described text that no longer exists — and every check
in this repository passed. That is the residual risk the context-overflow
backlog closed *around* rather than closed, and it is why "do not modify
`Course.render()`" has been a rule enforced by convention.

**Why the existing guard is not enough.**
`test_render_is_left_alone_because_the_embedder_reads_it` asserts that certain
fields survive `render()`. It is blind to a reordering, a relabelling, or a
whitespace change — each of which changes the embedded text completely while
leaving every one of its assertions true. A digest over the rendered output is
the only check whose sensitivity matches what the embedder actually consumes.

**Why now rather than when `render()` first has to change.** A fingerprint
introduced at the same time as a render change stamps the *new* render onto
vectors built by the old one, certifying the corruption instead of catching it.
The guard only works if it is already on disk before the change.

**What it deliberately does not do.** An empty `render_id` — a manifest written
before this field existed — is "unknown", not "different", and keeps loading.
Treating the two alike would invalidate every cache on disk today and charge a
10,092-course re-embed for a check with nothing yet to check against.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skyrag.data.schema import Course, render_fingerprint  # noqa: E402
from skyrag.embedding import (  # noqa: E402
    CACHE_MISMATCH,
    CACHE_OK,
    MANIFEST_SUFFIX,
    CacheManifest,
    EmbeddingStore,
    inspect_cache,
)

DIM = 8
EMBEDDER = "test-embedder"


def _write_cache(directory: Path, name: str = "coursera", rows: int = 3) -> None:
    courses = [
        Course(
            id_=f"COU{i}",
            platform=name,
            title=f"title {i}",
            keywords=["python", "data"],
            what_you_will_learn="outcomes",
            description="description",
            skills="skills",
            modules="modules",
            url="https://example.invalid/c",
        )
        for i in range(rows)
    ]
    EmbeddingStore(
        courses=courses, matrix=np.ones((rows, DIM), dtype=np.float32)
    ).save(directory, name, model_name=EMBEDDER)


def _manifest(directory: Path, name: str = "coursera") -> CacheManifest:
    return CacheManifest.read(directory, name)


def _rewrite_manifest(directory: Path, name: str, **changes) -> None:
    replace(_manifest(directory, name), **changes).write(directory, name)


# --------------------------------------------------------------------------
# The fingerprint itself
# --------------------------------------------------------------------------


def test_the_fingerprint_changes_when_render_changes_in_a_way_the_field_guard_misses():
    """Reordering `render()`'s lines changes what was embedded, and nothing else sees it.

    Every field is still present, so
    `test_render_is_left_alone_because_the_embedder_reads_it` stays green — the
    text fed to the embedder is nonetheless a different document, and cosine
    similarity is computed over the text, not over the field list.

    Simulated by reversing the rendered lines rather than by editing `render()`,
    because a test that edits production code to prove a point is testing its own
    edit. What matters is that the digest is a function of the exact output.
    """
    before = render_fingerprint()

    original = Course.render

    def reordered(self):
        return "\n".join(reversed(original(self).split("\n")))

    Course.render = reordered
    try:
        after = render_fingerprint()
    finally:
        Course.render = original

    assert after != before, (
        "reordering render()'s output left the fingerprint unchanged, so it "
        "cannot detect the class of change it exists for"
    )
    assert render_fingerprint() == before, "the probe must be restored"


def test_the_fingerprint_is_stable_across_calls():
    """A cache check that returns a different answer each time is not a check."""
    assert render_fingerprint() == render_fingerprint()


def test_the_fingerprint_is_a_property_of_render_and_not_of_the_installed_corpus():
    """The probe course is frozen in `schema.py` rather than read from disk.

    A fingerprint drawn from the real corpus would change when the corpus
    changed, reporting `CACHE_MISMATCH` for a render that never moved — which
    trains a reader to ignore the status.
    """
    fingerprint = render_fingerprint()
    _ = [Course(id_="X1", platform="edx", title="unrelated", keywords=["z"])]
    assert render_fingerprint() == fingerprint


# --------------------------------------------------------------------------
# What reaches the manifest on disk
# --------------------------------------------------------------------------


def test_a_saved_cache_records_the_render_that_produced_it(tmp_path):
    """Asserted on the manifest file as written, not on the dataclass.

    The value has to survive `asdict` -> JSON -> `read`, and `read` filters to
    known fields, so a field added to the dataclass and not to the file would be
    invisible exactly where it is needed.
    """
    _write_cache(tmp_path)

    on_disk = json.loads(
        (tmp_path / f"coursera{MANIFEST_SUFFIX}").read_text(encoding="utf-8")
    )

    assert on_disk["render_id"] == render_fingerprint()
    assert _manifest(tmp_path).render_id == render_fingerprint()


def test_a_cache_whose_render_no_longer_matches_is_reported_mismatched(tmp_path):
    """The whole point: a silent invalidation becomes `CACHE_MISMATCH`.

    Written by rewriting the manifest to name a different render, which is
    exactly the state a real `render()` edit produces — the vectors are
    untouched and the stamp beside them stops describing them.
    """
    _write_cache(tmp_path)
    _rewrite_manifest(tmp_path, "coursera", render_id="render-v1:0000000000000000")

    status = inspect_cache(tmp_path, "coursera", model_name=EMBEDDER)

    assert status.state == CACHE_MISMATCH
    assert not bool(status), "a mismatched cache must be falsy, or callers reuse it"
    assert "render-v1:0000000000000000" in status.detail
    assert render_fingerprint() in status.detail


def test_a_manifest_written_before_the_fingerprint_existed_still_loads(tmp_path):
    """Empty is "unknown", not "different", and the two justify opposite actions.

    Every cache on disk today was written without this field. Failing them would
    charge a 10,092-course re-embed for a check that has nothing to compare
    against, so the guard arms itself on the next write instead.
    """
    _write_cache(tmp_path)
    _rewrite_manifest(tmp_path, "coursera", render_id="")

    status = inspect_cache(tmp_path, "coursera", model_name=EMBEDDER)

    assert status.state == CACHE_OK and bool(status)


def test_a_matching_render_is_reported_ok(tmp_path):
    """The check must not fire on the case it will meet on every healthy run."""
    _write_cache(tmp_path)

    assert inspect_cache(tmp_path, "coursera", model_name=EMBEDDER).state == CACHE_OK


def test_the_render_check_does_not_mask_a_corrupt_cache(tmp_path):
    """Ordering: corruption is diagnosed before identity.

    A cache that is both truncated *and* stamped with a stale render is corrupt
    first — that is the finding that stops the process, and reporting it as a
    mismatch would send someone to re-embed a file that is damaged on disk.
    """
    _write_cache(tmp_path)
    _rewrite_manifest(tmp_path, "coursera", render_id="render-v1:0000000000000000")
    jsonl = tmp_path / "coursera.jsonl"
    first = jsonl.read_text(encoding="utf-8").splitlines()[0]
    jsonl.write_text(first + "\n", encoding="utf-8")

    status = inspect_cache(tmp_path, "coursera", model_name=EMBEDDER)

    assert status.state == "corrupt", (
        f"a damaged cache was reported as {status.state!r}, which sends the "
        "reader to re-embed rather than to the file that is broken"
    )
