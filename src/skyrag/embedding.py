"""Embedding + persistence.

Persistence changed from the notebooks' CSV. They wrote NumPy arrays into a
CSV cell and parsed them back with
`np.fromstring(x.strip("[]"), sep=" ")`, which is slow and silently loses
float precision (values are re-read from their truncated repr). Here the
matrix goes to .npy and the metadata to .jsonl -- exact, and far faster.

A reader for the legacy CSV format is kept so existing artifacts still load.

Every cache entry also carries a manifest naming the embedder that produced it.
A cache keyed on platform name alone is valid by *presence*, which is not the
same property as being the cache that was asked for: swapping
SKYRAG_EMBEDDING_MODEL between two 768-dimensional models reloads the first
model's vectors under the second model's name, and nothing downstream can
notice. That failure mode has no anomalous-looking symptom and it defeats the
embedder comparison outright, so identity is checked here rather than hoped for.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import EmbeddingConfig
from .data.schema import Course, render_fingerprint

MANIFEST_SUFFIX = ".meta.json"

# Explicit opt-out for callers that genuinely want whatever is on disk (cache
# inspection, migration tooling). Spelled out at the call site so that reading a
# cache of unknown provenance is always a visible decision, never a default.
ANY_EMBEDDER = "*"

# What produced a manifest. "embedded" means it was written by the embedder that
# encoded the matrix, so identity is observed. "adopted" means a human declared
# it after the fact for a cache that predates manifests -- weaker evidence, and
# recorded as such so a result can state which of the two it rested on.
PROVENANCE_EMBEDDED = "embedded"
PROVENANCE_ADOPTED = "adopted"


class EmbeddingCacheError(RuntimeError):
    """The cache on disk cannot be shown to be the cache that was requested."""


@dataclass(frozen=True)
class CacheManifest:
    """Identity of an embedding cache entry, written beside the matrix."""

    model_name: str
    dim: int
    count: int
    provenance: str = PROVENANCE_EMBEDDED
    note: str = ""
    # B10. Which `Course.render()` produced the text these vectors encode.
    #
    # Empty means "this manifest predates the fingerprint", which is a different
    # claim from "the render disagrees" and is treated as such: caches written
    # before this field existed keep loading, and the guard arms itself the next
    # time anything is embedded. The alternative -- treating empty as a mismatch
    # -- would invalidate every cache on disk today and charge a 10,092-course
    # re-embed for a check that has nothing yet to check against.
    #
    # The asymmetry that makes this worth adding *before* `render()` ever changes:
    # a fingerprint introduced at the same time as a render change stamps the new
    # render onto vectors built by the old one, certifying the corruption instead
    # of catching it. It only works if it is already there.
    render_id: str = ""

    @classmethod
    def read(cls, directory: Path, name: str) -> "CacheManifest | None":
        path = directory / f"{name}{MANIFEST_SUFFIX}"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}          # tolerate added keys
        return cls(**{k: v for k, v in data.items() if k in known})

    def write(self, directory: Path, name: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{name}{MANIFEST_SUFFIX}").write_text(
            json.dumps(asdict(self), indent=2), encoding="utf-8"
        )


# Why a cache entry is not usable for a given embedder. Kept as distinct values
# because "no cache" and "someone else's cache" call for opposite responses, and
# a single boolean collapses them into the same silent re-embed or, worse, the
# same silent fallback to synthetic data.
CACHE_OK = "ok"
CACHE_MISSING = "missing"            # nothing on disk
CACHE_UNSTAMPED = "unstamped"        # matrix present, provenance unknown
CACHE_MISMATCH = "mismatch"          # present and stamped by a different embedder
CACHE_CORRUPT = "corrupt"            # manifest disagrees with the matrix beside it


@dataclass(frozen=True)
class CacheStatus:
    state: str
    detail: str
    manifest: "CacheManifest | None" = None

    def __bool__(self) -> bool:
        return self.state == CACHE_OK


def _count_records(jsonl: Path) -> int:
    """Non-blank lines in a cache's metadata file.

    A line count rather than a parse: this runs on every cache inspection,
    including `exists()`, and the question asked here is how many records are
    present, not whether each one is well-formed. `load` parses.
    """
    count = 0
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def inspect_cache(directory: Path, name: str, *, model_name: str) -> CacheStatus:
    """Report whether `directory/name` is a cache this embedder may reuse."""
    npy, jsonl = directory / f"{name}.npy", directory / f"{name}.jsonl"
    if not (npy.exists() and jsonl.exists()):
        return CacheStatus(CACHE_MISSING, f"no {name}.npy/.jsonl under {directory}")

    manifest = CacheManifest.read(directory, name)
    if manifest is None:
        return CacheStatus(
            CACHE_UNSTAMPED,
            f"{name} has no {MANIFEST_SUFFIX}, so the embedder that produced it is "
            "unknown; adopt it explicitly before reuse",
        )

    matrix = np.load(npy, mmap_mode="r")
    observed_dim = int(matrix.shape[1])
    if observed_dim != manifest.dim:
        return CacheStatus(
            CACHE_CORRUPT,
            f"{name}.npy is {observed_dim}-dimensional, manifest says {manifest.dim}",
            manifest,
        )

    # Row counts, both sides. `CACHE_CORRUPT` is defined as "manifest disagrees
    # with the matrix beside it" and this function checked only dimensionality, so
    # a cache whose .jsonl had been truncated returned `ok` -- readable as usable
    # by every caller -- and then raised out of `EmbeddingStore.load` several
    # frames later. `load` failing loud is right; the problem was upstream. The
    # eval CLI branches on `if status:` and builds its disclosed synthetic
    # placeholder in the `else`, so a corrupt cache took the *success* branch and
    # killed the process instead of degrading with a stated reason.
    observed_rows = int(matrix.shape[0])
    if observed_rows != manifest.count:
        return CacheStatus(
            CACHE_CORRUPT,
            f"{name}.npy holds {observed_rows} vector(s), manifest declares "
            f"{manifest.count}",
            manifest,
        )

    metadata_rows = _count_records(jsonl)
    if metadata_rows != manifest.count:
        return CacheStatus(
            CACHE_CORRUPT,
            f"{name}.jsonl holds {metadata_rows} course record(s), manifest "
            f"declares {manifest.count} -- the cache is internally inconsistent, "
            "so course i and vector i are not the same course",
            manifest,
        )

    # The render check sits with the identity checks, not the corruption ones: a
    # cache built by a different `render()` is not damaged, it is *someone else's*
    # -- the same relationship a cache from a different embedder has, and it takes
    # the same status. Skipped when the manifest predates the field, because
    # "unknown" and "different" are opposite findings and only one of them
    # justifies discarding paid work.
    #
    # It runs *last*, after both corruption checks, and that ordering is load-
    # bearing rather than incidental. A cache that is damaged on disk **and**
    # stamped with a stale render is damaged first: "mismatch" sends the reader to
    # re-embed, "corrupt" sends them to the file that is actually broken, and only
    # one of those is the next useful action. The first version of this check ran
    # before the `.jsonl` row count and reported a truncated cache as a mismatch;
    # its own regression test caught it.
    current_render = render_fingerprint()
    if manifest.render_id and manifest.render_id != current_render:
        return CacheStatus(
            CACHE_MISMATCH,
            f"{name} was embedded from {manifest.render_id}, but Course.render() "
            f"now produces {current_render}. The vectors encode text that no "
            "longer exists, so every similarity score against them is scored "
            "against the wrong document. Re-embed, or restore the render.",
            manifest,
        )

    if model_name != ANY_EMBEDDER and manifest.model_name != model_name:
        return CacheStatus(
            CACHE_MISMATCH,
            f"{name} was embedded by {manifest.model_name!r}, this run uses {model_name!r}",
            manifest,
        )

    return CacheStatus(CACHE_OK, f"{name}: {manifest.count} rows from {manifest.model_name}", manifest)


def adopt_cache(
    directory: Path, name: str, *, model_name: str, note: str = ""
) -> CacheManifest:
    """Declare, by hand, which embedder produced an existing unstamped cache.

    For caches written before manifests existed. The declaration is recorded as
    `provenance="adopted"` rather than passed off as observed, because it rests
    on someone's recollection of a configuration and that is a materially weaker
    claim than a manifest written at embed time.
    """
    npy = directory / f"{name}.npy"
    if not npy.exists():
        raise EmbeddingCacheError(f"nothing to adopt: {npy} does not exist")

    matrix = np.load(npy, mmap_mode="r")
    manifest = CacheManifest(
        model_name=model_name,
        dim=int(matrix.shape[1]),
        count=int(matrix.shape[0]),
        provenance=PROVENANCE_ADOPTED,
        note=note,
    )
    manifest.write(directory, name)
    return manifest


@dataclass
class EmbeddingStore:
    """Courses plus their embedding matrix, kept index-aligned."""

    courses: list[Course]
    matrix: np.ndarray  # shape (n_courses, dim), float32
    # Which embedder produced `matrix`. Set by `Embedder.embed_courses` and by
    # `load`; None means the vectors arrived from somewhere that did not say,
    # which is legal in memory and refused at save time.
    model_name: str | None = None

    def __post_init__(self) -> None:
        if len(self.courses) != self.matrix.shape[0]:
            raise ValueError(
                f"Mismatch: {len(self.courses)} courses vs {self.matrix.shape[0]} embeddings."
            )

    def __len__(self) -> int:
        return len(self.courses)

    def save(self, directory: Path, name: str, *, model_name: str | None = None) -> None:
        identity = model_name or self.model_name
        if not identity:
            raise EmbeddingCacheError(
                f"Refusing to write {name!r} without an embedder name. An anonymous "
                "cache is reusable by any later run, including one using a different "
                "embedder at the same dimensionality -- pass model_name=..., or build "
                "the store with Embedder.embed_courses, which records it."
            )
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / f"{name}.npy", self.matrix)
        with open(directory / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for course in self.courses:
                f.write(json.dumps(course.to_dict(), ensure_ascii=False) + "\n")
        CacheManifest(
            model_name=identity,
            dim=int(self.matrix.shape[1]),
            count=len(self.courses),
            provenance=PROVENANCE_EMBEDDED,
            # Observed at write time, from the same `render()` that built the
            # matrix in `embed_courses`.
            render_id=render_fingerprint(),
        ).write(directory, name)

    @classmethod
    def load(cls, directory: Path, name: str, *, expect_model: str) -> "EmbeddingStore":
        """Load a cache entry, refusing one produced by a different embedder.

        `expect_model` has no default on purpose: a caller that does not state
        which embedder it wants cannot be given a meaningful answer, and the
        version of this method that quietly answered anyway is what allowed an
        e5-versus-mpnet comparison to compare mpnet against itself. Pass
        ANY_EMBEDDER to read a cache regardless of provenance.
        """
        status = inspect_cache(directory, name, model_name=expect_model)
        if not status:
            raise EmbeddingCacheError(
                f"cannot load embedding cache {name!r}: {status.detail}. "
                f"Re-embed with this model, or -- if you know what produced these "
                f"vectors -- record it with adopt_cache({str(directory)!r}, {name!r}, "
                f"model_name=...)."
            )

        matrix = np.load(directory / f"{name}.npy")
        courses = []
        with open(directory / f"{name}.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    courses.append(Course.from_dict(json.loads(line)))

        manifest = status.manifest
        if manifest is not None and manifest.count != len(courses):
            raise EmbeddingCacheError(
                f"{name}: manifest declares {manifest.count} courses, {name}.jsonl "
                f"holds {len(courses)} -- the cache is internally inconsistent."
            )
        return cls(
            courses=courses,
            matrix=matrix,
            model_name=manifest.model_name if manifest else None,
        )

    @classmethod
    def exists(cls, directory: Path, name: str, *, model_name: str) -> bool:
        """True only when a cache exists AND this embedder produced it.

        Use `inspect_cache` when the reason matters: a missing cache and another
        embedder's cache both return False here, and they are not the same event.
        """
        return bool(inspect_cache(directory, name, model_name=model_name))


class Embedder:
    """Wraps SentenceTransformer with config-driven device/batch handling.

    The model is loaded lazily so that importing this module (and running the
    tests) does not require torch to spin up a model.
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            device = self.config.resolved_device()
            self._model = SentenceTransformer(
                model_name_or_path=self.config.model_name, device=device
            )
        return self._model

    def encode(self, texts: list[str], *, show_progress: bool = True) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text], show_progress=False)[0]

    def embed_courses(self, courses: list[Course], *, show_progress: bool = True) -> EmbeddingStore:
        texts = [course.render() for course in courses]
        return EmbeddingStore(
            courses=courses,
            matrix=self.encode(texts, show_progress=show_progress),
            # Stamped here, where the identity is observed rather than asserted:
            # a store built this way can be saved without anyone having to
            # remember which embedder was configured.
            model_name=self.config.model_name,
        )


def load_legacy_csv(path: Path, platform: str) -> EmbeddingStore:
    """Read a CSV written by the original notebooks.

    Only for migrating existing artifacts; new runs use save()/load() above.
    """
    import pandas as pd

    from .data.schema import split_keywords

    def text(row, *names: str, default: str = "") -> str:
        """First present, non-NaN value among `names`, as a string.

        `row.get(name, default)` is the wrong tool on a pandas row: a *present*
        cell that is empty holds NaN, so `get` returns that NaN rather than the
        default and `str()` turns it into the literal "nan". Course IDs and titles
        went into the store reading "nan", which then matched nothing and displayed
        as a course actually called nan.
        """
        for name in names:
            if name in row and pd.notna(row[name]):
                value = str(row[name]).strip()
                if value:
                    return value
        return default

    df = pd.read_csv(path)
    if "embedding" not in df.columns:
        raise ValueError(
            f"{path} has no 'embedding' column; columns are {list(df.columns)}. "
            "This loader is only for CSVs written by the original notebooks."
        )

    vectors = df["embedding"].apply(
        lambda x: np.fromstring(str(x).strip("[]"), sep=" ") if pd.notna(x) else np.empty(0)
    )

    # Rows whose vector is blank or ragged are dropped rather than allowed into
    # np.array(), which either raises a shape error or -- on older numpy -- builds
    # an object-dtype array that fails much later and further away. Dropping is
    # reported, not silent: a migration that quietly loses rows produces a corpus
    # that is smaller than the file it came from with nothing to say so.
    widths = {int(v.size) for v in vectors if v.size}
    if not widths:
        raise ValueError(
            f"{path} contains no parseable embedding vectors ({len(df)} row(s) read)."
        )
    # The width most rows agree on wins; ties break toward the wider vector, so the
    # choice is deterministic. Picking by set-iteration order would make a migration
    # non-reproducible, which is the one property this loader exists to provide.
    dim = max(widths, key=lambda w: (sum(1 for v in vectors if v.size == w), w))

    keep = [i for i, v in enumerate(vectors) if v.size == dim]
    dropped = len(df) - len(keep)
    if dropped:
        blank = sum(1 for v in vectors if v.size == 0)
        print(
            f"  ! load_legacy_csv({path.name}): dropped {dropped} of {len(df)} row(s) "
            f"-- {blank} blank, {dropped - blank} of a width other than {dim} "
            f"(widths seen: {sorted(widths)})"
        )
    if not keep:
        raise ValueError(
            f"{path}: no row carries a {dim}-dimensional embedding; widths seen "
            f"were {sorted(widths)}."
        )

    courses = []
    for position, i in enumerate(keep):
        row = df.iloc[i]
        courses.append(
            Course(
                id_=text(row, "id_", default=f"{platform[:3].upper()}{position + 1}"),
                platform=platform,
                title=text(row, "course_title", "Course Title"),
                keywords=split_keywords(row.get("keyword")),
                what_you_will_learn=text(row, "what_you_will_learn"),
                description=text(row, "description"),
            )
        )

    matrix = np.array([vectors.iloc[i] for i in keep], dtype=np.float32)
    if matrix.shape[0] != len(courses):
        raise ValueError(
            f"{path}: built {matrix.shape[0]} vectors for {len(courses)} courses. "
            "A store whose matrix and course list disagree scores the wrong course "
            "for every query past the first mismatch."
        )
    return EmbeddingStore(courses=courses, matrix=matrix)
