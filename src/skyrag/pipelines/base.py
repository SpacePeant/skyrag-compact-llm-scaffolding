"""Shared pipeline result type and Markdown reporting."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..data.adapters import CorpusCoverage
from ..data.schema import PLATFORM_PREFIX
from ..retrieval import RetrievedCourse
from ..selection import (
    OUTCOME_BACKEND_ERROR,
    OUTCOME_EMPTY,
    OUTCOME_NO_VOCABULARY,
    OUTCOME_OVERRUN,
    OUTCOME_SCHEMA_VIOLATION,
    OUTCOME_TRUNCATED,
    OUTCOME_UNPARSEABLE,
    KeywordSelection,
)


def sum_tokens(counts) -> int | None:
    """Sum optional token counts, returning None if **any** is unknown (X1).

    The rule blocker 3.4 established for `total_tokens`, factored out because
    per-stage attribution applies it in four more places and a rule reimplemented
    per call site is a rule that eventually differs at one of them.

    `sum(c or 0 for c in counts)` is the shape this exists to prevent: it turns
    "one of three selection calls did not report usage" into a confident total
    that is short by one call. On the O4 cost axis a plausible undercount is
    strictly worse than an absent value -- it can be averaged, quoted and
    compared, and nothing about it looks wrong.

    An empty sequence is 0, not None: a stage that made no calls spent no
    tokens, which is a measurement rather than an absence. `expand_query=False`
    is a legitimate configuration, not a missing reading.
    """
    total = 0
    for count in counts:
        if count is None:
            return None
        total += count
    return total


# Run outcomes. These reach the results table, so they are values rather than
# log lines: a run that could not be grounded must be *counted*, not inferred
# afterwards from an empty field.
STATUS_OK = "ok"
STATUS_EMPTY_RETRIEVAL = "empty_retrieval"
# Some platforms contributed, others contributed nothing. Its own status because
# the alternative is inferring corpus coverage from `platform_spread`, where a
# deliberately two-platform run and a three-platform run that lost one are the
# same two keys. "Separated" in Separated Keyword RAG *is* the per-platform
# merge, so how many platforms took part is a property of the result.
STATUS_PARTIAL_RETRIEVAL = "partial_retrieval"
# The cell never completed: a stage outside keyword selection raised. Its own
# status because the alternative -- letting the exception escape -- discarded
# every cell already computed and left no artifact at all, so a sweep that died
# could not say which cell died or why. A recorded failure is comparable with a
# recorded success; a missing file is comparable with nothing.
#
# --- T1: "arm" means one thing in this repository, and this is not it ------
#
# Renamed from `STATUS_ARM_FAILED = "arm_failed"` on 2026-08-20, value included.
# Pre-rename artifacts -- `outputs/eval_20260818_arm1-2_llama70b-8b_4q_live-openrouter/` -- carry the old string;
# nothing reads it programmatically, and it is recorded here rather than kept as
# an alias no code path would exercise.
#
# The word was carrying two meanings in one module. Here it meant a **cell** of
# the sweep grid: one (query x model x pipeline) unit. In the experiment
# specification and in every research document it means the **intervention
# axis** -- Arm 1 baseline 70B, Arm 2 naive 8B, Arm 3 skill schemas. B0
# introduces `--arm` with the second meaning, and two senses of one
# word in one results table is a defect that is cheap now and expensive after a
# sweep has run.
#
# "cell" wins here because the code had already chosen it everywhere else:
# `on_cell`, `total_cells`, `_print_cell`, `tests/test_cell_progress.py`. This
# constant and a handful of identifiers in `eval.py` were the holdouts.
#
# **The word `arm` is now reserved for the intervention axis.** Anything naming
# a grid unit is a cell.
STATUS_CELL_FAILED = "cell_failed"

# Why a platform contributed no courses. Distinct values because the remedies
# are unrelated -- one is the model, one is a threshold this harness sets, one is
# the corpus -- and because "contributed nothing" with no reason attached is the
# shape the defect had.
REASON_OK = "ok"
REASON_NO_USABLE_KEYWORDS = "no_usable_keywords"      # selection yielded none that validated
REASON_BELOW_MIN_SCORE = "below_min_score"           # keywords existed; the threshold took them all
REASON_NO_COURSES_MATCHED = "no_courses_matched"     # keywords survived; no course carries them
# A platform can also contribute nothing because the selection stage never completed,
# and those causes are not the model declining to select. `REASON_NO_USABLE_KEYWORDS`
# used to absorb all four: an API outage, a truncated reply and an unparseable reply
# all reached it, so `platforms_lost_detail` -- the column a reader consults precisely
# to ask *why* a platform dropped -- reported a 503 as a model choice. One reason per
# outcome, so the exported field agrees with the outcome column beside it.
REASON_BACKEND_ERROR = "backend_error"               # the backend never answered
REASON_NO_VOCABULARY = "no_vocabulary"               # the platform had no keywords to offer
REASON_TRUNCATED = "reply_truncated"                 # cut off below a compliant answer: ours
REASON_OVERRUN = "reply_overran_cap"                 # cut off after ignoring the cap: the model's
REASON_UNPARSEABLE = "reply_unparseable"             # answered fully, emitted garbage
REASON_SCHEMA_VIOLATION = "reply_violated_schema"    # readable, but not the shape asked for (B1)

# selection outcome -> the reason a platform contributed nothing because of it.
# Absent from this map means the selection completed normally, and the reason is then
# decided by what happened downstream (threshold, no matching courses).
REASON_FOR_OUTCOME = {
    OUTCOME_BACKEND_ERROR: REASON_BACKEND_ERROR,
    OUTCOME_NO_VOCABULARY: REASON_NO_VOCABULARY,
    OUTCOME_TRUNCATED: REASON_TRUNCATED,
    OUTCOME_OVERRUN: REASON_OVERRUN,
    OUTCOME_UNPARSEABLE: REASON_UNPARSEABLE,
    OUTCOME_SCHEMA_VIOLATION: REASON_SCHEMA_VIOLATION,
}




# Derived from PLATFORM_PREFIX rather than restated. The prefixes were spelled out
# inline in `cited_ids` below, making this the second place the same constant lived.
# A fourth platform added to PLATFORM_PREFIX would not have reached that pattern, so
# its courses would have been invisible to `cited_ids`: a run citing them reports zero
# citations, which suppresses its grounding score entirely, and any ID the model
# invented for that platform escapes the hallucination check that exists to catch it.
# Both errors land on the model under test, for a platform the harness simply could
# not see. Sorted longest-first so no prefix can shadow a longer one sharing its start.
_CITED_ID_PATTERN = (
    r"\b((?:"
    + "|".join(sorted(PLATFORM_PREFIX.values(), key=len, reverse=True))
    + r")\d+)\b"
)


@dataclass
class PlatformOutcome:
    """What one platform contributed to a run, and why, when the answer is none.

    Recorded per platform rather than derived from the retrieved set, because a
    platform that drops out leaves no trace in the retrieved set at all -- that
    is what made a half-corpus run indistinguishable from a whole-corpus one.
    """

    platform: str
    keywords_selected: int          # validated against the vocabulary
    keywords_used: int              # ...and surviving min_keyword_score
    courses_retrieved: int
    reason: str = REASON_OK
    # The threshold responsible, carried with the outcome it produced. A run
    # aborting because every score fell under it otherwise reports no keywords
    # and no cause.
    min_keyword_score: float = 0.0

    @property
    def contributed(self) -> bool:
        return self.courses_retrieved > 0


@dataclass
class PipelineResult:
    pipeline: str
    query: str
    learning_path: str
    retrieved: list[RetrievedCourse] = field(default_factory=list)
    expanded_query: str = ""
    # Which Stage 1 rewrite produced `expanded_query`: "plain", "schema" or
    # "schema_wording" (see `skyrag.EXPANSION_MODES`). Empty on pipelines with
    # no rewrite stage.
    expansion_mode: str = ""
    principles: str = ""
    selections: list[KeywordSelection] = field(default_factory=list)
    platform_outcomes: list[PlatformOutcome] = field(default_factory=list)
    # What corpus this run actually got, as opposed to what it asked for. Distinct
    # from `platform_outcomes`, which records what each *loaded* platform then
    # contributed: a platform whose dataset file is absent never becomes a store,
    # so it cannot appear there at all. `load_all_with_coverage` computed this and
    # had no production caller, so the one failure it existed to catch -- running
    # against two platforms while reporting the paper's three -- reached no
    # artifact. Optional because a caller that built its stores by hand has no
    # coverage claim to make, and inventing one would be worse than leaving it None.
    coverage: CorpusCoverage | None = None
    model: str = ""
    backend: str = ""
    elapsed_seconds: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    status: str = STATUS_OK

    # --- What actually reached the generation prompt ---------------------
    #
    # `retrieved` above is the list the model was shown, capped included: grounding
    # is scored against it, so it has to be what the model could see. These three
    # record the trim itself, which is otherwise unrecoverable -- a run that
    # retrieved 700 courses and showed 600 exports identically to one that retrieved
    # 600, and "how many courses reached the prompt, and was that number capped" is
    # a methods-section obligation rather than a diagnostic.
    # Metric O4. The generation call alone, which the specification names as "the
    # only strictly like-for-like slice across arms" -- `elapsed_seconds` covers
    # retrieval, expansion, step-back, per-platform selection and the ~105 ms of
    # harness overhead finding 14 measured. 0.0 means the stage never ran, which is
    # a real state: an empty retrieval aborts before generation.
    generation_seconds: float = 0.0
    # The two upstream LLM stages, timed apart from generation. `elapsed_seconds`
    # covers the whole pipeline and `generation_seconds` covers the last call, so
    # without these the cost of the stage actually under substitution -- keyword
    # selection -- is only recoverable by subtraction, and subtraction also sweeps
    # in embedding and retrieval, which are not LLM work at all.
    selection_seconds: float = 0.0      # keyword selection, summed over platforms
    expansion_seconds: float = 0.0      # query expansion + step-back

    # --- X1: the same three stages, in tokens -----------------------------
    #
    # `prompt_tokens` / `completion_tokens` above are the **generation call
    # alone**. The three stages were timed apart and their tokens were counted
    # nowhere, so a token-efficiency ratio (R11) computed from `prompt_tokens`
    # priced one stage of four and called it the run's cost.
    #
    # Unrecoverable after the fact, like O5's counters: no analysis can recover
    # what a completed run did not count. That is what earns this a place ahead
    # of the arms under D12's carve-out rather than behind them.
    #
    # Each is None when *any* call in that stage failed to report a count --
    # never a partial sum. Same rule as `total_tokens` (blocker 3.4): a
    # plausible undercount of cost is worse than an absent value, because a
    # partial total can be averaged and an absent one cannot.
    expansion_prompt_tokens: int | None = None
    expansion_completion_tokens: int | None = None
    selection_prompt_tokens: int | None = None
    selection_completion_tokens: int | None = None

    context_courses_cap: int | None = None      # the backstop in force, None = none
    context_courses_dropped: int = 0            # how many it removed; 0 = did not bind
    # Size of the rendered context block. The defect that produced these fields was
    # invisible in every column the export carried and visible immediately in this
    # one, so it is measured rather than inferred from a course count: the courses
    # are not the same size as each other.
    context_chars: int = 0

    # --- Arm and treatment fields -------------------------------------------
    arm: str = ""
    intervention: str = ""

    # Telemetry (B18)
    llm_retries: int = 0
    llm_throttled_calls: int = 0

    # --- Arm 3 skill-schema fields (B1). Empty on every arm without a schema --
    #
    # A violation is a recorded outcome, never a crash and never a silent
    # retry: a retried failure becomes a latency figure instead of a failure
    # rate, and Arm 3's entire output is a claim about how often an 8B can hold
    # a shape.
    schema_violations: int = 0
    # Which stages produced them, comma-joined, in the order the stages ran.
    # The count alone cannot distinguish "the model cannot format keywords"
    # from "the model cannot format a learning path", and those are different
    # findings about what a schema does to a small model.
    schema_violation_stages: str = ""
    # M2.1's split by the model's own claim. `keywords_claimed_recalled` is the
    # interesting half: if invented keywords arrive marked as copied, the model
    # cannot tell recall from reading. These two are the model's *claim* about
    # its own provenance, not an observation of it.
    keywords_claimed_copied: int = 0
    keywords_claimed_recalled: int = 0

    @property
    def succeeded(self) -> bool:
        return self.status == STATUS_OK

    @property
    def context_capped(self) -> bool:
        """True when the backstop actually bound on this run."""
        return self.context_courses_dropped > 0

    def lost_platforms(self) -> list[PlatformOutcome]:
        """Platforms that were asked and returned nothing."""
        return [o for o in self.platform_outcomes if not o.contributed]

    def selection_errors(self) -> list[KeywordSelection]:
        """Selections where the backend failed, as opposed to selecting nothing."""
        return [s for s in self.selections if s.failed]

    @property
    def retrieved_ids(self) -> set[str]:
        return {r.course.id_ for r in self.retrieved}

    def cited_ids(self) -> list[str]:
        """Course IDs the model actually cited, in order of appearance.

        Matched case-insensitively and returned upper-cased. The prompt asks for
        exact IDs, but a model writing `cou1` has still cited COU1, and the
        case-sensitive pattern this replaced saw nothing there. That failure is
        silent and doubly wrong: the run reports zero citations, which suppresses
        its grounding score entirely, and any ID the model *did* invent in lower
        case escapes the hallucination check that exists to catch it. Both errors
        land on the model under test, for a formatting choice the metric was never
        meant to measure.
        """
        found = re.findall(_CITED_ID_PATTERN, self.learning_path, re.IGNORECASE)
        seen, out = set(), []
        for cid in found:
            cid = cid.upper()
            if cid not in seen:
                seen.add(cid)
                out.append(cid)
        return out

    def hallucinated_ids(self) -> list[str]:
        """Cited IDs that were never retrieved -- i.e. invented by the model.

        This is the concrete grounding check the notebooks' all-caps prompt was
        trying to enforce by intimidation.

        Both sides are upper-cased before comparison. `cited_ids` now normalises
        case, so comparing against raw store IDs would reintroduce the same
        mismatch one layer down and book a correctly-cited course as invented.
        """
        retrieved = {rid.upper() for rid in self.retrieved_ids}
        return [cid for cid in self.cited_ids() if cid.upper() not in retrieved]

    def platform_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.retrieved:
            counts[r.course.platform] = counts.get(r.course.platform, 0) + 1
        return counts


def to_markdown(result: PipelineResult) -> str:
    """Render a run as a self-contained Markdown report."""
    lines: list[str] = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append(f"# SKYRAG learning path — `{result.pipeline}`")
    lines.append("")
    lines.append(f"*Generated {ts}*")
    lines.append("")

    # --- Run metadata ---
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(f"| Pipeline | `{result.pipeline}` |")
    lines.append(f"| Backend | `{result.backend}` |")
    lines.append(f"| Model | `{result.model}` |")
    lines.append(f"| Courses retrieved | {len(result.retrieved)} |")
    if result.context_chars:
        lines.append(
            f"| Context sent | {len(result.retrieved)} courses, "
            f"{result.context_chars:,} chars (~{result.context_chars // 4:,} tokens) |"
        )
    if result.context_capped:
        lines.append(
            f"| Context cap | **bound** — {result.context_courses_dropped} course(s) "
            f"dropped at the {result.context_courses_cap}-course backstop |"
        )
    if result.platform_counts():
        spread = ", ".join(f"{k}: {v}" for k, v in sorted(result.platform_counts().items()))
        lines.append(f"| Platform spread | {spread} |")
    if result.platform_outcomes:
        contributing = len(result.platform_outcomes) - len(result.lost_platforms())
        lines.append(
            f"| Platform coverage | {contributing} of "
            f"{len(result.platform_outcomes)} contributed |"
        )
    if result.coverage:
        cov = result.coverage
        lines.append(
            f"| Corpus coverage | {len(cov.loaded)} of {len(cov.requested)} "
            f"platforms loaded ({sum(cov.course_counts.values())} courses) |"
        )
        if cov.missing:
            lines.append(f"| Corpus missing | {', '.join(cov.missing)} |")
    lines.append(f"| Elapsed | {result.elapsed_seconds:.1f}s |")
    # `is not None`, not truthiness: `if result.prompt_tokens:` was falsy for 0 as
    # well as None, so a backend genuinely reporting zero prompt tokens produced a
    # report with no token row -- identical to a backend that reported nothing. Same
    # falsy-versus-None class as the `or 0` total this repository already fixed once.
    if result.prompt_tokens is not None:
        lines.append(
            f"| Tokens | {result.prompt_tokens} in / {result.completion_tokens} out |"
        )
    lines.append("")

    # --- Run outcome -------------------------------------------------------
    if result.status == STATUS_EMPTY_RETRIEVAL:
        lines.append(
            "> **ABORTED — empty retrieval.** No platform yielded a usable "
            "keyword, so no courses were retrieved and the generation model was "
            "**not** called. Generating here would have produced a learning path "
            "with no grounding, which is the failure mode under study."
        )
        lines.append("")

    # Upstream of every per-platform outcome below: a platform with no dataset file
    # never became a store, so it cannot appear as a lost platform either.
    if result.coverage and result.coverage.missing:
        cov = result.coverage
        lines.append(
            f"> **INCOMPLETE CORPUS — {len(cov.loaded)} of {len(cov.requested)} "
            f"platforms had a dataset to load.** Missing: "
            + ", ".join(f"`{p}`" for p in cov.missing)
            + ". Every platform figure below is over the loaded subset, so this run "
            "is **not** comparable with one that loaded the full corpus."
        )
        lines.append("")

    if result.context_capped:
        lines.append(
            f"> **CONTEXT CAPPED — {result.context_courses_dropped} of "
            f"{len(result.retrieved) + result.context_courses_dropped} retrieved "
            f"courses were withheld from the generation prompt** at the "
            f"`{result.context_courses_cap}`-course backstop. Truncating the context "
            "is itself a treatment: a model prevented from seeing a course is being "
            "given a different task from one that sees it. If this bound on some "
            "arms and not others, those arms are not comparable."
        )
        lines.append("")

    if result.status == STATUS_PARTIAL_RETRIEVAL:
        lost = ", ".join(f"`{o.platform}` ({o.reason})" for o in result.lost_platforms())
        contributing = len(result.platform_outcomes) - len(result.lost_platforms())
        lines.append(
            f"> **PARTIAL RETRIEVAL — {contributing} of "
            f"{len(result.platform_outcomes)} platforms contributed.** Missing: "
            f"{lost}. The run produced courses and is usable, but it is **not** the "
            "run that was requested: comparing it against a run that kept every "
            "platform compares two different corpora."
        )
        lines.append("")

    for sel in result.selections:
        if sel.outcome == OUTCOME_TRUNCATED:
            lines.append(
                f"> **Reply truncated during keyword selection ({sel.platform}):** "
                f"`{sel.error}`. The token budget granted was below what a compliant "
                "answer needs, so this is a harness failure — **not** a model that "
                f"declined to select — and `{sel.platform}` is excluded from "
                "capability metrics."
            )
            lines.append("")
        elif sel.outcome == OUTCOME_OVERRUN:
            lines.append(
                f"> **Reply overran the keyword cap ({sel.platform}):** `{sel.error}`. "
                "The model had room for a compliant answer and used more, so the "
                "cutoff records a failure to follow the stated limit. Unlike a "
                "truncation this **counts against the model under test**, and it is "
                f"why `{sel.platform}` contributed nothing."
            )
            lines.append("")
        elif sel.outcome == OUTCOME_NO_VOCABULARY:
            lines.append(
                f"> **No vocabulary for {sel.platform}:** `{sel.error}`. The model "
                "was never asked. A corpus condition, not a model result."
            )
            lines.append("")
        elif sel.outcome == OUTCOME_BACKEND_ERROR:
            lines.append(
                f"> **Backend error during keyword selection ({sel.platform}):** "
                f"`{sel.error}`. This is an infrastructure failure, not a model "
                "that declined to select."
            )
            lines.append("")
        elif sel.outcome == OUTCOME_UNPARSEABLE:
            lines.append(
                f"> **Unparseable reply during keyword selection ({sel.platform}).** "
                "The model finished cleanly but returned nothing parseable. Unlike a "
                "truncation this **is** a model-capability failure and counts against "
                "the model under test."
            )
            lines.append("")
        elif sel.outcome == OUTCOME_SCHEMA_VIOLATION:
            # The reasons, not just the count. A schema violation reported as a
            # bare number is the one thing a reader cannot check, and this is
            # the arm whose entire result is a claim about shape compliance.
            detail = "; ".join(sel.schema_violations) or "no detail recorded"
            lines.append(
                f"> **Schema violation during keyword selection ({sel.platform}).** "
                "The model finished cleanly and returned something readable, but "
                f"not the shape the schema asked for: {detail}. This is a "
                "model-capability failure and counts against the model under "
                "test. It is **not** retried and **not** repaired."
            )
            lines.append("")
        elif sel.outcome == OUTCOME_EMPTY:
            lines.append(
                f"> **{sel.platform} — model selected nothing.** A well-formed empty "
                "answer. This is a genuine model result, not a failure."
            )
            lines.append("")

    # --- Grounding check ---
    hallucinated = result.hallucinated_ids()
    if hallucinated:
        lines.append(
            f"> **Warning — {len(hallucinated)} hallucinated course ID(s):** "
            + ", ".join(f"`{c}`" for c in hallucinated)
            + ". These were cited but never retrieved."
        )
    elif result.cited_ids():
        lines.append(
            f"> **Grounding OK** — all {len(result.cited_ids())} cited course IDs "
            "were present in the retrieved set."
        )
    lines.append("")

    lines.append("## Query")
    lines.append("")
    lines.append("```")
    lines.append(result.query.strip())
    lines.append("```")
    lines.append("")

    if result.expanded_query:
        lines.append("## Expanded query")
        lines.append("")
        lines.append(result.expanded_query.strip())
        lines.append("")

    if result.selections:
        lines.append("## Selected keywords")
        lines.append("")
        for sel in result.selections:
            kws = ", ".join(
                f"`{k}` ({v:.2f})"
                # Same total order as `KeywordSelection.ranked` (L1), so the
                # report lists keywords in the order retrieval consumed them.
                for k, v in sorted(sel.keywords.items(), key=lambda kv: (-kv[1], kv[0]))
            )
            if sel.failed:
                lines.append(f"**{sel.platform}** — _backend error: {sel.error}_")
            else:
                lines.append(f"**{sel.platform}** — {kws or '_none selected_'}")

            # Reported apart: one is a model defect, the other is a limit of the
            # window this harness showed the model.
            if sel.hallucinated:
                lines.append("")
                lines.append(
                    "  <sub>Hallucinated (absent from the platform vocabulary): "
                    + ", ".join(f"`{r}`" for r in sel.hallucinated)
                    + "</sub>"
                )
            if sel.near_miss:
                lines.append("")
                lines.append(
                    "  <sub>Near-miss (M2.2 — real keyword, mis-transcribed; "
                    "not used for retrieval): "
                    + ", ".join(
                        f"`{emitted}` → `{resolved}`"
                        for emitted, resolved in sel.near_miss.items()
                    )
                    + "</sub>"
                )
            if sel.out_of_window:
                lines.append("")
                lines.append(
                    "  <sub>Real but outside the vocabulary window shown: "
                    + ", ".join(f"`{r}`" for r in sel.out_of_window)
                    + "</sub>"
                )
            lines.append("")

    if result.principles:
        lines.append("## Educational principles (step-back)")
        lines.append("")
        lines.append(result.principles.strip())
        lines.append("")

    lines.append("## Learning path")
    lines.append("")
    lines.append(result.learning_path.strip())
    lines.append("")

    lines.append("## Retrieved courses")
    lines.append("")
    lines.append("| Score | Course ID | Platform | Title | Via keyword |")
    lines.append("|---|---|---|---|---|")
    for r in result.retrieved:
        title = r.course.title.replace("|", "\\|")[:80]
        via = f"`{r.via_keyword}`" if r.via_keyword else "—"
        lines.append(
            f"| {r.score:.4f} | `{r.course.id_}` | {r.course.platform} | {title} | {via} |"
        )
    lines.append("")

    return "\n".join(lines)


def write_report(result: PipelineResult, outputs_dir: Path, name: str | None = None) -> Path:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = name or f"{result.pipeline}-{stamp}.md"
    path = outputs_dir / filename
    path.write_text(to_markdown(result), encoding="utf-8")
    return path
