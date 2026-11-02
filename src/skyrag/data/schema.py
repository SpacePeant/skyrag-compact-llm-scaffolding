"""Normalized course record.

The three source datasets use three incompatible schemas (Coursera uses
"Course Title"/"Keyword", Udemy uses "course_title"/"keyword", edX uses
"course_title"/"subject"). Everything downstream speaks `Course` instead, so
adding a fourth platform means writing one adapter and touching nothing else.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Platform -> course-id prefix, matching the notebooks (COU1.., UDE0.., EDX0..).
PLATFORM_PREFIX = {"coursera": "COU", "udemy": "UDE", "edx": "EDX"}


# How much of a course reaches the *generation* prompt. Not a style choice: at the
# full `render()` the configuration's own ceiling of 560 courses builds a 441,000-
# token prompt against a 131,072-token endpoint limit, so the arm that selected
# keywords well was the arm that got refused. See
# `project_knowledge/task_log/backlog_2026-08-18_bugfix_live-run-context-overflow.md`.
CONTEXT_SUMMARY_CHARS = 400
CONTEXT_KEYWORDS = 5
CONTEXT_TITLE_CHARS = 150

# A hard ceiling on the whole per-course block, applied last.
#
# Clipping the summary alone does not bound a course: titles in this corpus reach
# 202 characters and a five-keyword block reaches 241, and neither has any
# structural limit -- so "600 courses fit" would be a fact about the courses
# currently on disk rather than about the code, which is the precise shape of the
# reasoning error this whole backlog exists to correct ("~280 candidate courses per
# platform, far more than a generation prompt consumes"). An adversarial test built
# from unbounded titles and keywords overflowed the budget at 600 courses even with
# the summary clipped.
#
# 700 x the 600-course backstop is ~105,000 proxy tokens against a 126,976 budget,
# and the proxy overestimates by about a quarter. Measured on the real corpus it
# binds on 5 of 10,092 courses (0.05%) -- recorded because a ceiling that binds on
# anything is a treatment, however small, and the number belongs in the record
# rather than in someone's assumption.
CONTEXT_MAX_CHARS = 700

# Marks a clipped summary so the prompt does not present a truncated course as a
# short one.
CONTEXT_CLIP_MARKER = " ..."


def _clip(text: str, max_chars: int) -> str:
    """Trim to `max_chars`, on a word boundary where one is near the cut."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    space = cut.rfind(" ")
    if space > max_chars * 0.6:
        cut = cut[:space]
    return cut.rstrip() + CONTEXT_CLIP_MARKER


@dataclass
class Course:
    id_: str
    platform: str
    title: str
    keywords: list[str] = field(default_factory=list)
    what_you_will_learn: str = ""
    description: str = ""
    skills: str = ""
    modules: str = ""
    url: str = ""
    # O3. The publisher-assigned difficulty, verbatim from the source catalogue:
    # "Beginner level" / "Intermediate level" / "Advanced level" on coursera,
    # "Introductory" / "Intermediate" / "Advanced" on edx. Empty where the
    # catalogue carries none (778 coursera rows).
    #
    # **Deliberately NOT included in `render()`.** Adding it would change the text
    # embedded and shown to the model, which is a treatment change, not a
    # measurement -- every sweep already on disk would stop being comparable to
    # anything run afterwards. O3 reads this field to score the *output* ordering;
    # the model is never told a course's level, which is exactly the condition
    # under test.
    level: str = ""

    @property
    def level_ordinal(self) -> int | None:
        """Difficulty as 0 / 1 / 2, or `None` when the catalogue carries none.

        `None` rather than a default: an unlabelled course is an unknown, and
        defaulting it to "beginner" would manufacture ordering violations that
        the data does not support. Both platforms' vocabularies are folded onto
        one scale -- edx's "Introductory" and coursera's "Beginner level" are the
        same rung, which is what makes a cross-platform path scorable at all.
        """
        if not self.level or not isinstance(self.level, str):
            return None
        t = self.level.strip().lower()
        if not t:
            return None
        if "beginner" in t or "introductory" in t or "intro" in t:
            return 0
        if "intermediate" in t:
            return 1
        if "advanced" in t:
            return 2
        return None

    def render(self) -> str:
        """The text blob that gets embedded and shown to the LLM.

        Mirrors the notebooks' `course["text"]`: title, ID, learning outcomes
        and keywords in one block. Keywords are included deliberately -- they
        are part of what the embedding matches on.
        """
        lines = [
            f"Course Title: {self.title}",
            f"Course ID: {self.id_}",
            f"Platform: {self.platform}",
        ]
        if self.what_you_will_learn:
            lines.append(f"What you will learn in this course: {self.what_you_will_learn}")
        if self.skills:
            lines.append(f"Skill gained by the user: {self.skills}")
        if self.modules:
            lines.append(f"Modules in this course: {self.modules}")
        if self.description:
            lines.append(f"Course Description: {self.description}")
        if self.url:
            lines.append(f"Course URL: {self.url}")
        if self.keywords:
            lines.append("Keywords:")
            lines.extend(f"- {k}" for k in self.keywords)
        return "\n".join(lines)

    def render_for_context(
        self,
        *,
        max_chars: int = CONTEXT_SUMMARY_CHARS,
        max_keywords: int = CONTEXT_KEYWORDS,
    ) -> str:
        """The trimmed blob shown to the *generation* model.

        Deliberately a second method rather than a shorter `render()`. `render()`
        is the embedder's input -- `Embedder.embed_courses` calls it to build the
        matrix -- and `CacheManifest` records the model name, dimension and row
        count but **not** the render format. Shortening `render()` would therefore
        leave every cached matrix on disk looking valid while its vectors described
        text that no longer exists, and every check in this repository would pass.
        The two consumers want different things anyway: the embedder wants all the
        text there is to match on, the generation prompt wants the least text that
        still identifies a course.

        What survives: the ID (so citations stay checkable and hallucinated IDs
        stay detectable), the title, the platform, one clipped summary and a few
        keywords. What goes: `modules`, `skills`, the URL, the full description,
        and the long tail of keywords -- which together are ~90% of the bytes and
        none of what the model needs to name a course in a learning path.

        The clip is announced with a trailing marker rather than made silently, so
        a reader of the prompt can tell a short course from a truncated one.

        Every variable-length field is bounded, and the whole block is bounded
        again at the end. Clipping only the summary would leave the per-course size
        a property of the corpus rather than of this method -- titles here reach 202
        characters and keyword blocks 241, with no structural limit on either -- and
        "it fits with the data we happen to have" is exactly the reasoning that
        produced this defect.

        Field order is the degradation order: ID first, so a course stays citable
        no matter what the ceiling takes; keywords last, because they are the
        cheapest thing to lose.
        """
        lines = [
            f"Course ID: {self.id_}",
            f"Course Title: {_clip(self.title, CONTEXT_TITLE_CHARS)}",
            f"Platform: {self.platform}",
        ]
        # Learning outcomes first, description as the fallback: the outcomes field
        # is what the generation prompt is actually reasoning over, and courses
        # carrying both would otherwise spend the budget on marketing copy.
        summary = self.what_you_will_learn or self.description
        if summary:
            lines.append(f"About: {_clip(summary, max_chars)}")
        if self.keywords:
            keywords = ", ".join(self.keywords[:max_keywords])
            lines.append(f"Keywords: {keywords}")

        block = "\n".join(lines)
        if len(block) > CONTEXT_MAX_CHARS:
            block = block[:CONTEXT_MAX_CHARS].rstrip() + CONTEXT_CLIP_MARKER
        return block

    def to_dict(self) -> dict:
        data = asdict(self)
        data["text"] = self.render()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Course":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def split_keywords(raw) -> list[str]:
    """Normalize a keyword field to a lowercase, de-duplicated list.

    Source data is inconsistent: sometimes a comma-joined string, sometimes
    already a list. Order is preserved so output stays deterministic.
    """
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else str(raw).split(",")

    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        kw = str(item).strip().lower()
        if kw and kw not in seen:
            seen.add(kw)
            out.append(kw)
    return out


# --- The render fingerprint (B10) -----------------------------------------
#
# `render()` is the embedder's input. `CacheManifest` recorded the model name,
# the dimension and the row count and *not* the render format, so a change to
# `render()` left every cached matrix on disk loading clean while its vectors
# described text that no longer exists -- and every check in this repository
# passed. That is the one residual risk the context-overflow backlog closed
# around rather than closed, and it is why "do not modify `Course.render()`" has
# been a standing rule enforced by convention.
#
# A hash over the rendered text of one fixed course turns that convention into a
# runtime check. It is deliberately sensitive to *everything* `render()` emits --
# field order, labels, separators, whitespace -- because all of those change what
# was embedded. The existing guard,
# `test_render_is_left_alone_because_the_embedder_reads_it`, asserts that certain
# fields survive; it is blind to a reordering or a relabelling, both of which
# invalidate the vectors just as completely.
#
# The probe course is frozen here rather than drawn from the corpus so the
# fingerprint depends on `render()` alone and not on which dataset is installed.
# It exercises every optional branch, because a field that is never populated is
# a field whose removal the fingerprint would not see.
_FINGERPRINT_PROBE = Course(
    id_="COU1",
    platform="coursera",
    title="Probe",
    keywords=["alpha", "beta"],
    what_you_will_learn="outcomes",
    description="description",
    skills="skills",
    modules="modules",
    url="https://example.invalid/probe",
)


def render_fingerprint() -> str:
    """Short stable digest of what `Course.render()` produces.

    Recomputed on call rather than cached in a module constant: the value has to
    track the function, and a constant computed at import is one edit away from
    being stale in exactly the situation it exists to detect.
    """
    import hashlib

    digest = hashlib.sha256(_FINGERPRINT_PROBE.render().encode("utf-8")).hexdigest()
    return f"render-v1:{digest[:16]}"
