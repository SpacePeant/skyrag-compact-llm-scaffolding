"""Multi-model evaluation controller & paper benchmark exporter."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..data.adapters import CorpusCoverage
from ..data.schema import (
    CONTEXT_KEYWORDS,
    CONTEXT_MAX_CHARS,
    CONTEXT_SUMMARY_CHARS,
)
from ..embedding import Embedder, EmbeddingStore
from ..llm import LLMBackend
from ..llm.base import AuthError, LLMError
from ..metrics import embedding_ils, ndcg_at_k
from ..schemas import (
    SCHEMA_VERSION,
    VARIANT_SPEC_ONLY,
    VARIANT_SPEC_PLUS_CANDIDATE_INJECTION,
    VARIANT_SPEC_PLUS_EXEMPLAR,
)
from ..selection import (
    CANDIDATE_SOURCE_REWRITE,
    CANDIDATE_SOURCES,
    RANKER_LEXICAL,
    RANKER_STRATEGIES,
    OUTCOME_BACKEND_ERROR,
    OUTCOME_EMPTY,
    OUTCOME_NO_VOCABULARY,
    OUTCOME_OVERRUN,
    OUTCOME_SCHEMA_VIOLATION,
    OUTCOME_TRUNCATED,
    OUTCOME_UNPARSEABLE,
)
from .base import (
    STATUS_CELL_FAILED,
    STATUS_EMPTY_RETRIEVAL,
    STATUS_OK,
    STATUS_PARTIAL_RETRIEVAL,
    PipelineResult,
    sum_tokens,
)
from .naive import run_naive
from .skyrag import resolve_expansion_mode, run_skyrag

# Names the generation-context render in one string, so a change to the trim
# changes what every artifact reports rather than passing silently. Bump the
# version when the *shape* changes, not just the numbers.
CONTEXT_RENDER_ID = (
    f"trimmed-v1:{CONTEXT_SUMMARY_CHARS}c/{CONTEXT_KEYWORDS}kw/"
    f"{CONTEXT_MAX_CHARS}max"
)


# --- Arm axis constants (B0) -----------------------------------------------
#
# Model identifiers for the 70B-vs-8B axis. Declared once so `resolve_arm`
# and any test that checks them both read from the same source.
MODEL_70B = "meta-llama/llama-3.1-70b-instruct"
MODEL_8B = "meta-llama/llama-3.1-8b-instruct"

# Arm numbers accepted by resolve_arm(). A frozenset so membership testing is
# O(1) and the exhaustive match in resolve_arm does not need updating when a
# new arm is registered -- the guard here catches an unknown string first.
VALID_ARMS: frozenset[str] = frozenset({"1", "2", "3"})

# B16. Arms whose *design* is resolved but whose *treatment* does not exist in
# `src/` yet. `resolve_arm` still returns their ArmConfig -- the mapping from an
# arm number to its model assignment is real -- but `run_multi_model_eval`
# refuses to execute a cell for one, because that is where the damage is done:
# **the crash is the safer failure.** A crash cannot be mistaken for a result,
# while a mislabelled CSV row outlives everyone's memory of the caveat.
#
# Emptied 2026-08-26 by B1, which wired `schema_enabled` and defined S1-S3 in
# `schemas.py`. The set stays as the guard that stops a half-built arm
# exporting rows.
UNIMPLEMENTED_ARMS: frozenset[str] = frozenset()

# Default maximum consecutive cell failures before the sweep aborts to avoid
# burning resources / API spend on systemic outages (e.g., 404 endpoint/model not found).
MAX_CONSECUTIVE_CELL_FAILURES: int = 5

# `--sweep-version` selects the Stage 2 payload shape that Arms 1-3 select on.
# The earlier default of "v3" made Arm 3 need `--sweep-version v5` on every
# invocation to get the flat-map payload, and forgetting it silently produced
# the V1 three-field payload, whose measured cost is a 3.2x retrieval collapse
# against V5 on the same window. The default is therefore "v5".
#
# An explicitly passed version always wins, so every published recipe keeps
# meaning exactly what it meant.
SWEEP_VERSION_SELECTION_DEFAULT = "v5"
# What the run header shows when nothing was passed.
SWEEP_VERSION_AUTO_LABEL = "auto"


def resolve_sweep_version(requested: str | None) -> str:
    """The sweep version one cell actually runs under.

    `requested` is what the operator asked for, or None/"" / "auto" for the
    default.
    """
    if requested and requested != SWEEP_VERSION_AUTO_LABEL:
        return requested
    return SWEEP_VERSION_SELECTION_DEFAULT


class SweepCircuitBreakerError(RuntimeError):
    """Raised when an evaluation sweep exceeds the maximum tolerated error threshold."""


@dataclass
class ArmConfig:
    """Resolved configuration for one arm of the three-arm experiment.

    Separates the experiment-design layer (which arm?) from the execution layer
    (which model does each role use, and which intervention is active?). Created
    once per arm by `resolve_arm`, then applied to every cell of that arm's sweep.
    """

    arm: str                        # "1" | "2" | "3"
    model: str                      # cell label in the results table
    keyword_model: str
    expansion_model: str
    generation_model: str
    intervention: str               # "none" | "schemas"
    # Live as of 2026-08-26 (B1/B17). Read by `run_skyrag` and the single
    # source of truth for whether a cell runs the schema treatment, rather than
    # a string comparison on the arm label: the arm axis encoded twice could
    # disagree, which surfaces as a mislabelled results table rather than as an
    # exception.
    schema_enabled: bool = False    # B1: skill schemas S1-S3 (Arm 3)
    # Version string for the skill-schema text. The schema is part of the
    # treatment (arms_overview §5), so its version must travel with the row
    # exactly like a model revision. Empty on Arms 1 and 2 (no schema).
    # Set from `schemas.SCHEMA_VERSION` on the arms that run a schema (B1).
    schema_version: str = ""
    # Arm 3 prompt variant (2026-08-31). "spec_only" is B1's original prompt --
    # schema shown, no worked example. "spec_plus_exemplar" appends one valid
    # instance.
    #
    # This is a **level of the schema treatment, not a bug fix.** Measured live,
    # the 8B answers the spec-only prompt by echoing the schema back, so the arm
    # emits zero keywords. Prompt parity with Arm 2 is exact and neither arm
    # carries an example, so the behaviour is real -- but it makes Arm 3 measure
    # "can an 8B read a schema specification" rather than "can an 8B produce
    # structured output", and a reviewer would rightly say the negative is an
    # artifact of an un-exemplified schema. Both levels are therefore run and
    # the variant is stamped per row.
    schema_prompt_variant: str = VARIANT_SPEC_ONLY
    candidate_injection: bool = False
    candidate_limit: int = 100


def resolve_arm(arm: str, *, model: str | None = None) -> ArmConfig:
    """Map an arm number string to its full execution configuration.

    Arm 1: 70B on all roles, zero-shot (original paper baseline).
    Arm 2: 8B on all roles, zero-shot (unassisted small model).
    Arm 3: 8B + structured skill schemas (B1).

    If `model` is provided, it overrides the default model for all roles on the arm.
    Raises ValueError for unknown arm strings. The caller is responsible for
    parsing multi-arm strings ("1,2", "all") into individual arm strings before
    calling this.
    """
    arm = arm.strip()
    if arm not in VALID_ARMS:
        raise ValueError(
            f"resolve_arm: {arm!r} is not a valid arm. "
            f"Valid choices: {sorted(VALID_ARMS)}."
        )
    base_model = model or (MODEL_70B if arm == "1" else MODEL_8B)
    if arm == "1":
        return ArmConfig(
            arm="1",
            model=base_model,
            keyword_model=base_model,
            expansion_model=base_model,
            generation_model=base_model,
            intervention="none",
        )
    if arm == "2":
        return ArmConfig(
            arm="2",
            model=base_model,
            keyword_model=base_model,
            expansion_model=base_model,
            generation_model=base_model,
            intervention="none",
        )
    # arm == "3"
    return ArmConfig(
        arm="3",
        model=base_model,
        keyword_model=base_model,
        expansion_model=base_model,
        generation_model=base_model,
        intervention="schemas",
        schema_enabled=True,
        schema_version=SCHEMA_VERSION,
    )


def _count(res: PipelineResult, outcome: str) -> int:
    """How many of a run's per-platform selections ended in `outcome`."""
    return sum(1 for s in res.selections if s.outcome == outcome)


def _violation_json(res: PipelineResult, pick, outcomes=(OUTCOME_SCHEMA_VIOLATION,)) -> str:
    """`pick(selection)` for each selection whose outcome is in `outcomes`, as JSON by platform."""
    violated = {s.platform: pick(s) for s in res.selections if s.outcome in outcomes}
    return json.dumps(violated, ensure_ascii=False) if violated else ""


def _stage_total(
    metrics: list[BenchmarkMetric], prompt_field: str, completion_field: str
) -> dict[str, int]:
    """One stage's token total across a sweep, with the rows it is over (X1).

    Returns `{"tokens": n, "rows": k}` rather than a bare integer, because a
    bare integer is the aggregate form of the defect X1 exists to fix: it looks
    like the sweep's cost for that stage and silently omits every row whose
    backend reported no usage. `k` is what makes it quotable.

    A row counts only when **both** halves reported. Summing prompts over 20
    rows and completions over 17 produces a total describing no actual set of
    runs.
    """
    tokens = rows = 0
    for metric in metrics:
        prompt = getattr(metric, prompt_field)
        completion = getattr(metric, completion_field)
        if prompt is None or completion is None:
            continue
        tokens += prompt + completion
        rows += 1
    return {"tokens": tokens, "rows": rows}

HISTORICAL_BENCHMARK_QUERIES = [
    "I want to learn data science in 6 months",
    "Machine learning and deep learning path with Python",
    "Full-stack web development with Python and JavaScript",
    "Cybersecurity, network defense, and ethical hacking",
    "Cloud computing infrastructure and DevOps deployment",
]


@dataclass
class RunProvenance:
    """How a sweep was configured, so a row can be read months later.

    Every field here is something that changes a reported number and that no
    measurement column records. The exports previously carried scores and nothing
    else: no seed, no temperature, no backend, no embedder, no device, no version,
    no timestamp. Two sweeps run under different settings produced byte-comparable
    artifacts with nothing to tell them apart, and the standing mitigation was to
    "record the model ids, corpus size and date alongside the numbers" by hand.

    Stamped onto every row rather than written once at the top of the file: rows
    from several sweeps get merged into one frame for analysis, and the moment that
    happens a run-level header no longer travels with the row it describes.
    """

    run_id: str
    run_timestamp: str
    skyrag_version: str
    backend: str
    provider: str               # "" -- automatic routing, recorded as such
    temperature: float | None
    seed: int | None
    max_tokens: int | None
    embedding_model: str
    embedding_device: str       # resolved, not requested: "auto" is not an answer
    # Finding 18's other half. `embedding_device` says *what* the run got; this says
    # *why*, when what it got is not what was asked for. Empty means the resolution
    # was clean -- an accelerator was found, or cpu was requested outright. The
    # reason previously reached a module global and nothing else, so a run that
    # asked for CUDA and settled for cpu exported identically to one that meant cpu,
    # and `elapsed_seconds` feeds the O4 cost axis.
    embedding_device_fallback: str = ""
    # How much corpus a generation prompt was allowed to carry. Both belong here
    # rather than only on the row: they are settings, they must be identical across
    # arms for the comparison to mean anything, and the run that produced the
    # 2026-08-18 inversion had neither recorded anywhere at all.
    max_context_courses: int | None = None
    # Retrieval depth: how many courses each selected keyword was allowed to pull,
    # and how many keywords the model was allowed to return.
    #
    # **R6r, 2026-08-20.** These decide `retrieved_count` and therefore the size of
    # every generation prompt, and neither was stamped anywhere. Closing the
    # outstanding "re-run the two-arm sweep at `per_keyword=7`" item required
    # *inferring* the depth of `outputs/ablation_20260819_kw-sub_llama8b-kw-70b-gen_10q_coreweave/` from its
    # retrieved counts -- 253 courses is impossible at `per_keyword=1` and ordinary
    # at 7 -- which is an argument rather than a record. A setting that changes
    # every reported number and cannot be read off the artifact is the same defect
    # as the provider pin, one level down.
    per_keyword: int | None = None
    max_keywords: int | None = None
    # The context render format, as a string that changes when the trim changes.
    # `CacheManifest` records no render format, which is the precedent this avoids:
    # a silently altered render leaves every artifact looking valid while describing
    # text that no longer exists.
    context_render: str = ""
    sweep_version: str = "v3"
    # The serving precision the weights were run at: "bf16", "fp8", "q4", or "" when
    # nobody declared one.
    #
    # **Operator-declared, never inferred.** Hosted APIs do not report the dtype they
    # serve, so this cannot be captured from `config` the way every other field here
    # is -- it has to be passed in by whoever pinned the provider. That is a weaker
    # guarantee than the rest of this dataclass offers, and the empty default is the
    # honest encoding of "unstated": a row that does not say bf16 is a row where the
    # precision was never written down, not a row that ran at some assumed default.
    #
    # Why it earns a column at all: precision moved invention by +0.194 between the
    # fp8 and bf16 cells of the Arms 1/2/3 leg -- larger than any effect the schema
    # intervention was measured to produce -- so a cross-arm comparison whose arms
    # differ in precision is confounded by the larger of the two factors. Arm 1's
    # precision was inferred from its provider pin after the fact and appears in no
    # artifact, which is exactly the argument-rather-than-a-record failure that
    # `per_keyword` above was added to end.
    serving_precision: str = ""
    # **C14, 2026-09-25.** Whether this run's query vectors came from a real
    # embedder, and why not when they did not.
    #
    # The failure this closes: when `Embedder` cannot load -- on this machine
    # `OSError: [WinError 1455]` under concurrent lanes, each holding its own copy
    # of torch plus all-mpnet-base-v2 -- the harness substituted a constant query
    # vector, every course scored identically, and the sweep ran to completion. It
    # wrote a full 50-row CSV, exited 0, and said so in exactly one line on stdout.
    # `fact_20260923_v5fp8_i_l0_r1` is that artifact, and it survives only as
    # `outputs/INVALID_embedder-fallback_...` because nothing *in the file* said it
    # was invalid; the quarantine came from a human re-reading a lane log.
    #
    # The refusal in `cli.py` is the primary fix. This is the second line of
    # defence, covering the one path that still reaches a row: an operator who
    # typed `--allow-degraded-embedder`. A row that cannot be a measurement now
    # carries that fact itself rather than in a log that may not travel with it --
    # the same argument every other field in this dataclass was added on.
    #
    # Two columns rather than one string that is empty when clean. An empty string
    # reads as falsy in Python but arrives from `read_csv` as NaN, and `bool(nan)`
    # is True -- so a filter for degraded rows would select every clean one. That
    # is the C10 defect class exactly, and not repeating it is cheaper than
    # documenting it.
    embedder_degraded: bool = False
    embedder_degraded_reason: str = ""

    @classmethod
    def capture(
        cls,
        config: Config,
        backend_name: str,
        *,
        run_id: str | None = None,
        sweep_version: str = "v3",
        serving_precision: str = "",
        embedder_degraded: bool = False,
        embedder_degraded_reason: str = "",
    ) -> RunProvenance:
        from .. import __version__

        now = datetime.now(UTC)
        return cls(
            run_id=run_id or now.strftime("%Y%m%dT%H%M%SZ"),
            run_timestamp=now.isoformat(timespec="seconds"),
            skyrag_version=__version__,
            backend=backend_name,
            provider=config.llm.provider or "",
            temperature=config.llm.temperature,
            seed=config.llm.seed,
            max_tokens=config.llm.max_tokens,
            embedding_model=config.embedding.model_name,
            embedding_device=config.embedding.resolved_device(),
            embedding_device_fallback=config.embedding.device_fallback() or "",
            max_context_courses=config.retrieval.max_context_courses,
            per_keyword=config.retrieval.per_keyword,
            max_keywords=config.retrieval.max_keywords,
            context_render=CONTEXT_RENDER_ID,
            sweep_version=sweep_version,
            serving_precision=serving_precision,
            embedder_degraded=embedder_degraded,
            embedder_degraded_reason=embedder_degraded_reason,
        )

    def as_columns(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkMetric:
    model: str
    pipeline: str
    query: str
    elapsed_seconds: float
    # Metric O4's like-for-like slice: the generation call alone, without
    # retrieval, expansion, step-back, selection or harness overhead.
    generation_seconds: float = 0.0
    # The stage under substitution, timed. In a run that pins generation and varies
    # only keyword selection, this is the only latency column that moves with the
    # thing being tested.
    selection_seconds: float = 0.0
    expansion_seconds: float = 0.0
    # **The generation call alone.** Named without a stage prefix for
    # compatibility with every export already on disk; read them as
    # `generation_prompt_tokens` / `generation_completion_tokens`. The per-stage
    # columns below are what X1 added, and `pipeline_*` is the run's real total.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    # --- X1: per-stage token attribution ----------------------------------
    #
    # Before these, expansion, step-back and every per-platform selection call
    # were timed and their tokens counted nowhere, so `total_tokens` was one
    # stage of four presented as a run's cost, and **R11 (Token Efficiency
    # Ratio)** priced a quarter of the pipeline.
    #
    # None means at least one call in that stage reported no usage, never a
    # partial sum (`sum_tokens`, and blocker 3.4's rule). `selection_*` is None
    # on the `naive` pipeline throughout, because it has no keyword stage at all
    # -- distinct from 0, which would say the stage ran and cost nothing.
    expansion_prompt_tokens: int | None = None
    expansion_completion_tokens: int | None = None
    selection_prompt_tokens: int | None = None
    selection_completion_tokens: int | None = None
    # Every stage, summed. This is the number a cost-per-run comparison between
    # arms actually needs, and it is the denominator R11 was always meant to
    # have.
    pipeline_prompt_tokens: int | None = None
    pipeline_completion_tokens: int | None = None
    pipeline_total_tokens: int | None = None
    # False when any stage's count was unknown, so a blank `pipeline_total_tokens`
    # can be told from a genuine zero -- the same service `tokens_complete` does
    # for the generation call.
    pipeline_tokens_complete: bool = False

    # --- R11: Token Efficiency Ratio --------------------------------------
    #
    # **PROJECT-DEFINED. Not a published metric, and it does not come from
    # Self-RAG** -- R4's correction applies here as much as to FPD, and this is
    # the file where the label has to sit, because a column name in a CSV is
    # what a reader copies. There is no literature definition of "token
    # efficiency" this instantiates; what follows is a construction, and its
    # arithmetic is stated so it can be disagreed with.
    #
    # `token_efficiency_ratio` = generation completion tokens / pipeline total
    # tokens: **the fraction of everything the run spent that became the
    # artifact anyone reads.** Expansion, step-back and per-platform keyword
    # selection are all machinery -- necessary, but nobody is handed their
    # output. O4 says what a run cost; this says how much of that cost was the
    # deliverable.
    #
    # It could not be built before X1: the denominator was the generation call
    # alone and the ratio would have been ~1.0 on every arm by construction.
    #
    # **Expect a small number, and do not read it as a percentage of waste.**
    # The selection prompt carries the 1,000-keyword window, so it dominates
    # every run: values in the low percent are normal and the figure is only
    # meaningful *between arms holding the window fixed*, which the shared
    # protocol does.
    #
    # `None` whenever `pipeline_tokens_complete` is False. A ratio over a
    # denominator missing a stage is inflated by exactly the stage that went
    # missing, and on a cost axis a plausible overstatement is worse than a
    # blank -- it can be averaged and quoted, and nothing about it looks wrong.
    token_efficiency_ratio: float | None = None
    # The same question asked per unit of delivered curriculum: what did one
    # cited course cost, end to end? `token_efficiency_ratio` can be flattered
    # by a verbose path that recommends nothing -- R22 exists because length and
    # quality come apart -- so the two are reported together and neither is the
    # headline alone. `None` when the path cited nothing, never 0: a run that
    # recommended no courses has an undefined cost-per-course, and 0.0 would
    # read as free.
    tokens_per_cited_course: float | None = None

    # --- R22: how long the generated path is ------------------------------
    #
    # **A verbosity control on O1, and it only works if it exists before the
    # sweep runs.** LLM judges reward longer answers -- documented for MT-Bench
    # (Zheng et al. 2023) and catalogued in `metric_literature_provenance.md`
    # §E, where it sat in no action list until 2026-08-20. **An O1 gain that
    # tracks length is verbosity, not capability**, and with no length column
    # there is no way to separate the two after the fact.
    #
    # This is the same class of item as O5's counters and X1: near-zero cost
    # now, unrecoverable once a sweep has run.
    #
    # The token half of "generation length" is `completion_tokens` above, which
    # is why only the character and word counts are new here. Characters are not
    # redundant with it: `completion_tokens` is None whenever a backend reports
    # no usage, and a verbosity check with holes in it cannot be run at all.
    # Both are recorded so the correlation survives a provider that stops
    # reporting usage mid-sweep.
    #
    # Words are whitespace-delimited -- deliberately crude, and stated as such.
    # A real tokenizer would be a second, model-specific instrument to version
    # and pin; what the correlation needs is a monotone proxy for length, and
    # this is one.
    path_chars: int = 0
    path_words: int = 0

    retrieved_count: int = 0
    cited_count: int = 0
    hallucinated_count: int = 0
    # None -- not 1.0 -- when the run cited nothing. A run that recommended no
    # courses has no grounding to measure; scoring it as perfect grounding
    # rewards producing nothing, and silently inflates the mean.
    grounding_accuracy: float | None = None
    # Metric M3.2. Distinct courses cited over courses retrieved -- how much of the
    # context the model actually used. The necessary companion to grounding
    # accuracy, which is silent about it: on 2026-08-18 a row cited 1 of 82
    # retrieved courses and scored 1.0, because inventing nothing out of one
    # citation is arithmetically perfect grounding and tells nobody anything.
    # None when nothing was retrieved -- a ratio with a zero denominator is absent,
    # not zero.
    citation_coverage: float | None = None
    # --- Reference-free selection quality (M1.1's replacement) -------------
    #
    # M1.1 as specified needs an answer key of "the right keywords for this
    # query", and every legal way to build one is weak: human curation is one
    # person's opinion on an underdetermined question, a strong external model
    # measures agreement with that model rather than correctness, and deriving it
    # from judged-relevant courses moves the same judgement one step upstream.
    #
    # These two ask a question that has an answer in the data: did the selected
    # keywords pull *semantically relevant* courses? A keyword that retrieves
    # nothing close to the query was a bad pick, whoever would have chosen it.
    # Comparable across arms without a key, because both arms are scored against
    # the same query embedding.
    retrieval_score_mean: float | None = None
    # The weakest course the model was shown. A high mean hides a tail, and the
    # tail is what a generation model cites when it runs out of good options.
    retrieval_score_min: float | None = None
    # --- O2's objective half ----------------------------------------------
    #
    # Mean pairwise Jaccard overlap of the keyword sets of the courses actually
    # cited. 1.0 = every recommended course is about the same things; 0.0 = they
    # share no topic labels at all. Redundancy is the half of O2 that needs no
    # reference subtopic set: "did it recommend five versions of one course" is
    # answerable from the corpus's own labels.
    #
    # Jaccard over keywords rather than cosine over embeddings on purpose -- the
    # keywords are the corpus's own topic labels, so the number is interpretable
    # ("these two courses share 80% of their topics") where a cosine is not.
    # None when fewer than two courses were cited: redundancy within a set of one
    # is undefined, not zero.
    cited_redundancy: float | None = None
    # --- O3, the cheap form (2026-09-15) ------------------------------------
    #
    # Ordering violations in the recommended path, scored against the
    # publisher-assigned difficulty the source catalogues already carry
    # (`Course.level`, 92.3% coverage). Read `path_level_inversions` and
    # `path_level_pairs` as a pair: the count alone cannot be compared between
    # rows, because a longer path has more pairs to get wrong.
    #
    # **Scored in CITATION order, not retrieval order.** O3 is a claim about the
    # sequence the model wrote, and `cited_courses` above is in the order
    # retrieval returned -- using it here would score the harness instead.
    #
    # **The model is never shown a course's level** (`Course.render()` omits it,
    # pinned by test), so this measures whether the model supplied pedagogical
    # ordering from its own knowledge. That is the capability under test.
    #
    # `None`, not 0, when fewer than two cited courses carry a level: no
    # comparable pair means no measurement, and 0 would read as a flawless path.
    path_level_inversions: int | None = None
    path_level_pairs: int | None = None
    path_level_inversion_rate: float | None = None
    # --- The published forms. See `skyrag/metrics.py` for citations ---------
    #
    # These three are the only metrics in this export with a literature source a
    # reviewer can look up. Everything else here is a bespoke diagnostic of this
    # pipeline, which is defensible and must be *declared* rather than left for a
    # reviewer to discover.
    #
    # `cited_redundancy` above already **is** Intra-List Similarity (Ziegler et
    # al., WWW 2005) -- the Jaccard-over-corpus-labels instantiation. It was
    # designed here independently and needs no code change, only the citation.
    # This is its embedding-cosine sibling, and it exists because Jaccard over
    # labels is bounded by the catalogue's labelling consistency: two courses
    # teaching the same material under disjoint vocabularies score 0.0 and are
    # near-duplicates in the recommendation. Read the pair together -- where they
    # disagree, the corpus's labels are unreliable.
    cited_redundancy_cosine: float | None = None
    # nDCG@10 (Jarvelin & Kekalainen, TOIS 2002) over the order the generation
    # model actually reads, graded by query cosine, normalised against the ideal
    # ordering of the same set. It measures what `merge_platforms`'s round-robin
    # interleave costs in ranking terms.
    #
    # **It is not recall and does not close M1.1** -- there is still no answer
    # key, and grading with the retriever's own cosine is self-referential.
    # Scoring the retriever's *own sort* this way returns 1.0 by construction;
    # the module docstring records why that form was refused.
    context_ndcg_at_10: float | None = None
    # Whether this row contributed to the grounding mean: 1 if scored, 0 if not.
    # Exported per row precisely because `grounding_accuracy` is nullable -- the
    # blank is invisible to every averaging tool a reader will reach for, so the
    # count has to be a column rather than something recoverable only from the
    # summary object.
    grounding_accuracy_n: int = 0
    # False when either token count was missing, so `total_tokens=None` can be told
    # apart from a backend that genuinely reported zero.
    tokens_complete: bool = True
    platform_spread: dict[str, int] = field(default_factory=dict)

    # The states that must stay distinguishable in the results table. Several of
    # these present identically upstream -- an empty keyword dict -- while meaning
    # opposite things, so each gets its own column rather than being inferred.
    status: str = STATUS_OK
    # Why a cell never completed, when it did not. Empty for every cell that ran.
    failure_detail: str = ""
    zero_citation: bool = False
    selection_errors: int = 0           # backend never answered (infrastructure)
    selection_truncated: int = 0        # budget was below a compliant answer (harness)
    selection_overrun: int = 0          # ignored the stated cap (capability)
    selection_unparseable: int = 0      # answered fully, garbage out (capability)
    selection_no_vocabulary: int = 0    # never asked: platform had no keywords
    empty_selections: int = 0           # model chose nothing (capability)

    # Corpus coverage actually achieved, recorded rather than inferred from
    # `platform_spread`: a run designed for two platforms and a run that asked
    # for two and lost one produce the same spread dict minus a key.
    platforms_requested: int = 0
    platforms_contributing: int = 0
    platforms_lost: int = 0
    platforms_lost_detail: str = ""     # "edx:no_usable_keywords", semicolon-joined

    # Corpus coverage: which platforms were *asked for* versus which had a dataset
    # to load at all. Kept apart from the `platforms_*` block above, which counts
    # only platforms that became stores. A platform whose file is missing is absent
    # from that block entirely, so without these columns a two-platform run against
    # the paper's three exports exactly like a three-platform run.
    corpus_platforms_requested: int = 0
    corpus_platforms_loaded: int = 0
    corpus_platforms_missing: str = ""      # "udemy", comma-joined
    corpus_courses: int = 0
    # False when the run had no coverage record at all -- stores built by hand,
    # nothing claimed. Distinguishes "complete" from "unknown", which the zeros
    # above cannot do on their own.
    corpus_coverage_known: bool = False

    # --- What reached the generation prompt ---------------------------------
    #
    # `retrieved_count` above is what the model was *shown*. These say whether that
    # was the whole retrieved set and how large the prompt actually was. Exported
    # because the 2026-08-18 inversion -- an unbounded prompt refused by the
    # endpoint, so the arm that selected keywords well was the arm that failed --
    # was invisible in every column this export carried, and because §3.4 of the
    # methods commitments obliges the paper to state both numbers.
    context_courses_dropped: int = 0
    context_chars: int = 0

    min_keyword_score: float | None = None

    # --- The denominators. Without these the M2.x family is not a rate ------
    #
    # The specification defines M2.1 and M2.2 as ratios over total keywords
    # emitted, and until 2026-08-19 the export carried numerators only: 30
    # hallucinations out of 35 emitted and 30 out of 300 were the same number in
    # the CSV, and they are opposite findings. Comparing raw counts across arms is
    # sound only if the arms emitted similar numbers of keywords -- exactly what
    # was not recorded, and demonstrably false on the 2026-08-18 sweep, where the
    # 8B arm overran the keyword cap and the 70B did not.
    keywords_emitted: int = 0           # every keyword the model returned
    keywords_kept: int = 0              # ...that survived validation, all platforms
    # Metric M1.3. Distinct across platforms, not summed: a keyword chosen for both
    # coursera and edx is one keyword the model thought of, and summing would score
    # breadth for what is really platform overlap.
    distinct_keywords: int = 0
    # Metric M1.2. Position of each usable keyword in the frequency-sorted window,
    # 0 = most frequent. A mean far below the window midpoint is position bias.
    selection_rank_mean: float | None = None
    selection_rank_max: int | None = None

    hallucinated_keywords: int = 0      # invented, absent from the corpus
    # Metric M2.2. Emitted keywords that fail exact match but resolve to a real
    # corpus keyword once case, punctuation and plurals are folded -- the model
    # had the concept and botched the transcription. Its own column because M2.1
    # previously absorbed it, and the two measure different capabilities:
    # conflated with invention, exact-string fidelity cannot be seen at all.
    near_miss_keywords: int = 0
    out_of_window_keywords: int = 0     # real, outside the window shown
    # Real in the corpus and deliberately excluded from the vocabulary as a
    # publisher rather than a topic. Its own column because it was previously
    # summed into `hallucinated_keywords` by elimination, and because a large
    # value here is a finding about the cleaning list, not about the model.
    removed_by_cleaning_keywords: int = 0
    # **R24(a), 2026-08-21. The denominator M2.3 should have had all along.**
    #
    # Keywords the model named that the corpus actually contains:
    # `keywords_kept + out_of_window + removed_by_cleaning`, equivalently
    # `keywords_emitted - hallucinated - near_miss`. Both of the subtracted
    # buckets are absent from the corpus (see `selection.py`'s bucket
    # docstrings); the three added ones are real in it and differ only in
    # whether the harness showed them.
    #
    # **Why the column exists rather than being recomputed downstream.** M2.3 as
    # specified divides by `keywords_emitted`, and that denominator is
    # contaminated by exactly the quantity the arms differ most on. On the
    # 2026-08-20 sweep the 8B invented 65.6% of its emissions against the 70B's
    # 24.7%, so dividing by emissions dilutes the 8B's real keywords 2.7x harder
    # than the 70B's. The specified form reported 0.126 vs 0.144 -- no gap, and
    # read naively it says the small model has *no* long-context recall deficit.
    # Conditioning on the model having named something real reports 0.381 vs
    # 0.204, a 2.1x gap at p = 7.2e-06.
    #
    # **M2.3 as specified is not arm-comparable whenever invalid rates differ**,
    # and it reverses capability 1's conclusion. The specified form is kept and
    # still exported -- it is what the glossary defines and dropping it would
    # rewrite history -- but it may not be quoted arm-against-arm without this
    # one beside it.
    keywords_real: int = 0

    # --- Arm and treatment fields -------------------------------------------
    arm: str = ""
    intervention: str = ""
    # Version string for the skill-schema text in force for this cell. Empty
    # on Arms 1 and 2, which use no schema. Set when B1 finalises the schema
    # text. The schema is part of the treatment (arms_overview §5) and its
    # version must be stamped per row exactly like a model revision: a schema
    # edited mid-sweep without a version bump is indistinguishable, in the
    # artifact, from an unedited one. Set to `schemas.SCHEMA_VERSION` on the
    # arms that run one, as of B1 (2026-08-26).
    schema_version: str = ""
    # Arm 3 prompt variant: 'spec_only' or 'spec_plus_exemplar'. Empty on arms
    # that run no schema. The spec-only prompt is B1's original; the exemplar
    # variant appends one valid instance. **Never compare an Arm 3 number across
    # variants without naming which** -- they are different treatments, and the
    # 8B's spec-only behaviour (echoing the schema back) makes the difference
    # large rather than marginal.
    schema_prompt_variant: str = ""
    # Which Stage 1 rewrite fed keyword selection: "plain", "schema" or
    # "schema_wording" (2026-10-01, `skyrag.EXPANSION_MODES`). Stamped on every
    # row the rewrite ran on, not only where it departs from the arm's default,
    # so a decomposition cell can never pool with its arm's ordinary rows. Rows
    # written before this column existed are unrecorded, not "plain": they ran
    # the arm's own rewrite, which `arm` and `schema_prompt_variant` imply.
    expansion_mode: str = ""
    # Condition C provenance (2026-09-08). Stamped on **every** arm, not only
    # the schema arms: the first Condition C sweep ran injection on Arm 2 as
    # well, and `schema_prompt_variant` is empty there, so the variant label
    # alone cannot record what that arm ran. Without these two columns a
    # Condition C row is byte-identical in provenance to a Condition A row and
    # the two pool silently on aggregation. `candidate_limit` travels with the
    # flag because the size of the injected candidate set is the treatment's
    # only free parameter -- a run at 100 and a run at 500 are different
    # conditions, and neither is recoverable from the rows without it.
    candidate_injection: bool = False
    candidate_limit: int = 0
    # Which ranker built the injected candidate set. Empty when injection did
    # not run -- the ranker is not a property of a row that never ranked
    # anything, and stamping the `lexical` default on every Condition A row
    # would make the column unusable as a filter for the runs that did.
    #
    # Every published Condition C cell predates this column and so carries no
    # value at all. **An absent stamp is therefore not `lexical`; it is
    # unrecorded**, exactly as `serving_precision` is, and the two must be
    # read the same way -- `lexical` is what those runs almost certainly used,
    # but almost certainly is an inference and this column exists to stop
    # inferences being read as records.
    candidate_ranker: str = ""
    # Which text the injected candidates were ranked against (`rewrite` or
    # `query`, see `selection.CANDIDATE_SOURCES`). Same rules as
    # `candidate_ranker`: blank on rows that injected nothing, and absent on
    # every row that predates 2026-10-21, all of which ranked against the
    # rewrite because nothing else existed.
    candidate_source: str = ""

    # --- Arm 3 skill-schema columns (B1) ----------------------------------
    #
    # Zero on every arm that runs no schema, which is not the same as absent:
    # a schema-less arm genuinely violated nothing, so 0 is the measurement.
    #
    # `selection_schema_violation` is deliberately its own column rather than
    # folded into `selection_unparseable`. "Could not be read at all" and
    # "could be read but was the wrong shape" are different findings about what
    # a schema does to an 8B, and Arm 3 exists to tell them apart.
    selection_schema_violation: int = 0
    schema_violations: int = 0
    schema_violation_stages: str = ""
    # The reasons and the reply behind `selection_schema_violation`, added
    # 2026-09-30. The count said LLaMA 8B broke the schema on 61% of injected
    # runs but not which rule, so the silence mechanism could not be named and a
    # salvage sensitivity analysis could not be run. Both are filled only for
    # selections that violated, as JSON keyed by platform; empty otherwise.
    # Record-only: nothing reads them back into any metric.
    selection_violation_detail: str = ""
    selection_violation_raw: str = ""
    # C16: the raw reply of any selection cut off by the completion budget, so
    # "syntax overhead or too many items" can be counted from the artifact
    # instead of inferred. Record-only, like the two columns above.
    selection_cutoff_raw: str = ""
    # M2.1 split by the model's own claim about each keyword. The pair answers
    # "when this model invents a keyword, does it know it is inventing?" --
    # which is a finding about the failure mode, not about the rate.
    keywords_claimed_copied: int = 0
    keywords_claimed_recalled: int = 0

    # Telemetry (B18)
    llm_retries: int = 0
    llm_throttled_calls: int = 0

    # --- Which cell of the design this row is, and how it was configured ---

    #
    # `model` above is the arm label. These three are what each stage actually
    # ran, which stopped being the same thing once the roles became pinnable:
    # a cell holding generation fixed at 70B while selection varies has one arm
    # label and two different models inside it, and only these columns can say so.
    keyword_model: str = ""
    expansion_model: str = ""
    generation_model: str = ""
    # Stable identity for the query, so the same question can be paired across
    # arms. Row text alone cannot do it once queries are loaded from a file and
    # someone edits the wording of one.
    query_id: str = ""
    # Which pass over the grid this row came from. One run per cell gives a number
    # with unknown run-to-run spread; the outcome metrics ask for that spread, and
    # it cannot be recovered after the fact from a single row.
    repeat: int = 1

    run_id: str = ""
    run_timestamp: str = ""
    skyrag_version: str = ""
    backend: str = ""
    provider: str = ""
    temperature: float | None = None
    seed: int | None = None
    max_tokens: int | None = None
    embedding_model: str = ""
    embedding_device: str = ""
    # Stamped from provenance like the rest of the block. On the export rather than
    # only in the run header because rows from several sweeps get merged into one
    # frame, and a header does not travel with the row it describes.
    embedding_device_fallback: str = ""
    # Stamped from provenance, so it reaches `cell_failed` rows too -- which is where
    # it is needed most: an arm that died carries no pipeline result to read the
    # setting off, and the failure this column exists to explain is precisely a
    # generation prompt that was too large.
    max_context_courses: int | None = None
    # R6r. Stamped per row for the same reason `max_context_courses` is: retrieval
    # depth must be identical across arms for a comparison to mean anything, and a
    # run-level header does not survive rows from several sweeps being merged into
    # one frame for analysis.
    per_keyword: int | None = None
    max_keywords: int | None = None
    context_render: str = ""
    sweep_version: str = "v3"
    # Mirrors `RunProvenance.serving_precision`, stamped from it like the rest of
    # this block. Empty means undeclared, never an assumed default -- see the
    # field's documentation on `RunProvenance` for why that distinction is the
    # whole point of the column.
    serving_precision: str = ""
    # **C14.** Mirrors `RunProvenance.embedder_degraded`, stamped from it like the
    # rest of this block. True means the query vectors for this row came from a
    # constant, not from the embedder named in `embedding_model`, so its course
    # ranking is arbitrary and no metric computed downstream of retrieval is a
    # measurement. Reaching `cell_failed` rows matters here for the usual reason:
    # a row with no pipeline result has nothing else to infer the condition from.
    #
    # `cli.py` refuses such a run outright, so in practice this is True only where
    # an operator typed `--allow-degraded-embedder`. The column exists because the
    # 2026-09-23 artifact proved that a warning which reaches only stdout does not
    # survive the walk from the lane log to the analysis.
    embedder_degraded: bool = False
    embedder_degraded_reason: str = ""

    # --- The artifact the metrics describe ---------------------------------
    #
    # Kept last on purpose. `learning_path` is kilobytes where every column above
    # is a scalar, and dataclass field order is CSV column order: at the end, the
    # metric columns can still be eyeballed with a column slice or `head`, and
    # anyone appending a column later does not have to step over it.
    #
    # Until 2026-08-20 the generated path reached **no artifact at all**. It was
    # used to compute `cited_ids()` and then discarded when `run_multi_model_eval`
    # dropped the `PipelineResult`, so every number in every CSV described a text
    # nobody could ever read back. That blocked every claim-level metric (USR,
    # citation precision, RAGAS faithfulness) and it blocked **B3**: a judge
    # harness cannot score post-hoc from an export that carries no output. It also
    # meant a run already paid for could never be re-analysed under a metric
    # invented after it -- the analysis had to be decided before the spend.
    #
    # **Read this out of the CSV with `json.loads`.** The dataframe writes it
    # JSON-encoded, the same treatment `platform_spread` gets and for a sharper
    # reason: a generated path contains newlines, and `csv.DictReader(text.
    # splitlines())` -- the idiom this project's own test helpers use everywhere --
    # silently *drops* an embedded newline and concatenates the lines either side
    # of it. The row count survives, so the corruption is invisible. Encoding here
    # means the export is lossless through that idiom rather than only through
    # pandas. The JSON export carries the raw string, unencoded.
    learning_path: str = ""
    # The Stage 1 rewrite text keyword selection actually read (2026-10-01).
    # Until this column the export recorded only its token count, so whether
    # invented keywords are copied out of the rewrite could be checked only by
    # re-running. JSON-encoded in the CSV for the same reason as `learning_path`.
    expanded_query: str = ""
    # The IDs parsed out of `learning_path`, in order of appearance, upper-cased --
    # `cited_count` is their length and says nothing about *which*. Exported
    # because recommendation-set overlap between two runs is the measurement the
    # determinism work did by hand in `compare_context_render.py`, and per **D8**
    # anything entering the results table has to come from this instrument. With
    # this column that comparison is a groupby over the standard CSV.
    cited_course_ids: str = ""           # "COU1;COU2;EDX7", semicolon-joined
    # The invented subset, so M3.1 is auditable rather than a bare count. A run
    # reporting `hallucinated_count=2` cannot tell you whether the model invented
    # a plausible-looking ID or emitted garbage, and those are different failures.
    hallucinated_course_ids: str = ""
    # **The same argument, one stage upstream, and it took until 2026-09-22.**
    # `hallucinated_keywords` is `sum(len(s.hallucinated))` and the terms
    # themselves reached no export at all -- not the CSV, not the JSON, not
    # `run.log`, which carries only totals. So M2.1 was a number nobody could
    # check, and the one question Paper #1's F1 leaves open -- how much of it is
    # character-level transcription error rather than invention -- was not
    # answerable from any artifact on disk, only by re-running.
    #
    # `metrics.transcription_proximity` answers it from these strings. Without
    # them it has nothing to read.
    #
    # Semicolon-joined and plain, matching `hallucinated_course_ids` exactly. A
    # keyword containing a semicolon would split wrongly here; none does, because
    # `_coerce` has already stripped the string and the corpus vocabulary is
    # built from single metadata fields, and a test pins the round trip.
    hallucinated_keyword_terms: str = ""
    # M2.2's pairs, same argument: "emitted -> resolved" is the diagnosis where
    # the count is not. Written as "a->b;c->d" so the fold that was applied is
    # visible per keyword, which is what makes a near-miss claim checkable
    # against `normalise_keyword` after the fact.
    near_miss_keyword_pairs: str = ""
    # **X2, IDs half. 2026-08-21, per DEC-5's own recommendation: "IDs now, text
    # on demand".** The courses the model was *shown*, in presented order --
    # merged round-robin and capped, never re-sorted -- semicolon-joined and
    # upper-cased to match `cited_course_ids`.
    #
    # `retrieved_count` said how many; this says which, and the difference is
    # what several blocked metrics need. It makes retrieval sets diffable across
    # arms with a groupby -- the measurement the determinism work did by hand --
    # and it lets `cited_course_ids` be checked against what was actually
    # available rather than against the whole corpus.
    #
    # **The rendered text is deliberately NOT here, and that half stays open.**
    # Up to ~560 rendered courses per row is a materially larger artifact than
    # the path, and the text is identical across every row that retrieved the
    # same course, so putting it in the metric row duplicates it hundreds of
    # times. R8 (USR) and R15 (citation precision) both verify a claim against
    # its *source text* and are therefore still blocked on DEC-5's second half,
    # not unblocked by this column.
    #
    # **Plain, not JSON-encoded**, deliberately -- unlike `learning_path`, and
    # matching `cited_course_ids` exactly. The encoding on `learning_path` earns
    # itself because a generated path contains newlines that
    # `csv.DictReader(text.splitlines())` silently swallows. A course ID
    # contains neither a newline nor a comma. Encoding this one and not its
    # sibling would mean `row["cited_course_ids"].split(";")` worked while
    # `row["retrieved_course_ids"].split(";")` returned one quoted blob -- two
    # columns of the same kind, read two different ways, which is a trap laid
    # for whoever writes the first diff between them.
    retrieved_course_ids: str = ""

    @classmethod
    def from_pipeline_result(cls, res: PipelineResult) -> BenchmarkMetric:
        cited = len(res.cited_ids())
        hallucinated = len(res.hallucinated_ids())

        retrieved = len(res.retrieved)

        # B15. Grounding is a comparison against the retrieved set, so with an
        # empty retrieved set there is no comparison to report -- only an empty
        # one. Both available answers are wrong in that state: 1.000 rewards
        # citing nothing, and 0.000 reads as "the model cited badly" when what
        # actually happened is that it was never given anything to cite. `None`
        # is the only honest value, and `grounding_accuracy_n` below carries the
        # denominator into the CSV so the exclusion is visible rather than
        # inferred. Same discipline X1 applies to token sums.
        if retrieved == 0:
            accuracy = None
        else:
            # hallucinated_ids() is a subset of cited_ids(), so cited == 0 implies
            # hallucinated == 0. The old `1.0 if hallucinated == 0` branch therefore
            # scored every empty run as perfectly grounded.
            accuracy = (
                round(max(0.0, (cited - hallucinated) / cited), 4) if cited else None
            )
        coverage = round(cited / retrieved, 4) if retrieved else None

        # Emitted = validated + every bucket a keyword can be rejected into. Built
        # by addition rather than recorded at the source because the model's raw
        # reply reaches neither export, so this is the only place the total still
        # exists.
        kept = sum(len(s.keywords) for s in res.selections)
        emitted = kept + sum(
            len(s.hallucinated) + len(s.near_miss) + len(s.out_of_window)
            + len(s.removed_by_cleaning)
            for s in res.selections
        )
        distinct = len({k for s in res.selections for k in s.keywords})

        # R24(a). Built by addition from the three real buckets rather than by
        # subtracting the two unreal ones from `emitted`: the subtraction is
        # arithmetically identical today, but it stays correct only for as long
        # as no sixth bucket is added, and a bucket added to `emitted` without a
        # thought for this line would silently be counted as real.
        real_keywords = kept + sum(
            len(s.out_of_window) + len(s.removed_by_cleaning) for s in res.selections
        )

        scores = [r.score for r in res.retrieved]
        cited_set = {c.upper() for c in res.cited_ids()}
        cited_courses = [r.course for r in res.retrieved if r.course.id_.upper() in cited_set]
        # O3. Citation order, not retrieval order -- `cited_courses` above is
        # filtered out of `res.retrieved` and carries that ordering instead.
        _by_id = {r.course.id_.upper(): r.course for r in res.retrieved}
        _path_levels = [
            _by_id[c.upper()].level_ordinal
            for c in res.cited_ids()
            if c.upper() in _by_id
        ]
        _inversions, _level_pairs = level_inversions(_path_levels)
        ranks = [r for s in res.selections for r in s.selected_ranks]

        # A total is only a total when both halves are known. `or 0` turned a
        # missing prompt count into zero, so a run reporting 500 completion tokens
        # and an unknown prompt count exported `total_tokens=500` -- a number that
        # looks like a complete cost and is short by however much the prompt was.
        # The O4 cost axis is built on these sums, and a plausible undercount is
        # worse there than an absent value, which at least cannot be averaged.
        both_known = res.prompt_tokens is not None and res.completion_tokens is not None
        total_toks = (res.prompt_tokens + res.completion_tokens) if both_known else None
        tokens_complete = both_known

        # X1. The same rule applied across stages: a pipeline total is only a
        # total when every stage reported. `sum_tokens` returns None the moment
        # one does not, so a run whose expansion call came back without a usage
        # block exports a blank here rather than a confident figure short by one
        # stage. `selection_*` is None on `naive` by construction -- no keyword
        # stage exists -- which correctly makes its pipeline total unavailable
        # rather than silently equal to its generation cost.
        pipeline_prompt = sum_tokens(
            (res.expansion_prompt_tokens, res.selection_prompt_tokens, res.prompt_tokens)
        )
        pipeline_completion = sum_tokens(
            (
                res.expansion_completion_tokens,
                res.selection_completion_tokens,
                res.completion_tokens,
            )
        )
        pipeline_complete = pipeline_prompt is not None and pipeline_completion is not None

        # R11. Both are gated on the *pipeline* total being complete, not on the
        # generation call alone: the numerator is known whenever the generation
        # call reported usage, and pairing a known numerator with a short
        # denominator is precisely how a ratio overstates itself.
        pipeline_total = (
            pipeline_prompt + pipeline_completion if pipeline_complete else None
        )
        ter = (
            round(res.completion_tokens / pipeline_total, 6)
            if pipeline_total and res.completion_tokens is not None
            else None
        )
        per_cited = (
            round(pipeline_total / cited, 2) if pipeline_total and cited else None
        )

        return cls(
            model=res.model or "unknown",
            pipeline=res.pipeline,
            query=res.query,
            elapsed_seconds=round(res.elapsed_seconds, 6),
            prompt_tokens=res.prompt_tokens,

            completion_tokens=res.completion_tokens,
            total_tokens=total_toks,
            expansion_prompt_tokens=res.expansion_prompt_tokens,
            expansion_completion_tokens=res.expansion_completion_tokens,
            selection_prompt_tokens=res.selection_prompt_tokens,
            selection_completion_tokens=res.selection_completion_tokens,
            pipeline_prompt_tokens=pipeline_prompt,
            pipeline_completion_tokens=pipeline_completion,
            pipeline_total_tokens=pipeline_total,
            pipeline_tokens_complete=pipeline_complete,
            token_efficiency_ratio=ter,
            tokens_per_cited_course=per_cited,
            # R22. Measured on the same string that is exported as
            # `learning_path` and scored by the judge, so the length column and
            # the judged artifact cannot describe different texts.
            path_chars=len(res.learning_path),
            path_words=len(res.learning_path.split()),
            retrieved_count=len(res.retrieved),
            cited_count=cited,
            hallucinated_count=hallucinated,
            citation_coverage=coverage,
            retrieval_score_mean=round(sum(scores) / len(scores), 4) if scores else None,
            retrieval_score_min=round(min(scores), 4) if scores else None,
            cited_redundancy=_mean_pairwise_overlap(cited_courses),
            path_level_inversions=_inversions if _level_pairs else None,
            path_level_pairs=_level_pairs or None,
            path_level_inversion_rate=(
                round(_inversions / _level_pairs, 4) if _level_pairs else None
            ),
            # The cited set, matching `cited_redundancy`'s subject exactly: the
            # two halves of an ILS pair have to measure the same list or they
            # cannot be read together.
            cited_redundancy_cosine=embedding_ils(
                [r.vector for r in res.retrieved if r.course.id_.upper() in cited_set]
            ),
            # `res.retrieved` is the presented order -- merged round-robin and
            # capped, never re-sorted -- which is the whole point. Sorting here
            # would collapse both of these to their degenerate values.
            context_ndcg_at_10=ndcg_at_k(scores, k=10),
            # 6 decimals, not the 4 `elapsed_seconds` uses: a real generation call is
            # seconds long, but on the echo backend it is microseconds, and 4 dp
            # rounds a stage that ran to 0.0 -- which is the value reserved for a
            # stage that never ran at all.
            generation_seconds=round(res.generation_seconds, 6),
            selection_seconds=round(res.selection_seconds, 6),
            expansion_seconds=round(res.expansion_seconds, 6),
            keywords_emitted=emitted,
            keywords_kept=kept,
            distinct_keywords=distinct,
            selection_rank_mean=round(sum(ranks) / len(ranks), 1) if ranks else None,
            selection_rank_max=max(ranks) if ranks else None,
            grounding_accuracy=accuracy,
            # 1 when this row carries a grounding score, 0 when it does not, so the
            # denominator travels with the numerator into the CSV. Without it a
            # reader reaches for `df.grounding_accuracy.mean()`, pandas drops the
            # blanks under its default skipna=True, and a mean over 2 scored rows
            # of 20 prints identically to a mean over all 20. Summing this column
            # reproduces `summarize_metrics`'s `grounding_accuracy_n` exactly.
            grounding_accuracy_n=0 if accuracy is None else 1,
            tokens_complete=tokens_complete,
            platform_spread=res.platform_counts(),
            status=res.status,
            zero_citation=cited == 0,
            selection_errors=_count(res, OUTCOME_BACKEND_ERROR),
            selection_truncated=_count(res, OUTCOME_TRUNCATED),
            selection_overrun=_count(res, OUTCOME_OVERRUN),
            selection_unparseable=_count(res, OUTCOME_UNPARSEABLE),
            selection_no_vocabulary=_count(res, OUTCOME_NO_VOCABULARY),
            # Strictly the model answering "nothing relevant". Counting every
            # keyword-less selection here -- the previous behaviour -- silently
            # absorbed truncations and garbage replies into a capability figure.
            empty_selections=_count(res, OUTCOME_EMPTY),
            hallucinated_keywords=sum(len(s.hallucinated) for s in res.selections),
            # Flattened across platforms, like the count directly above it. A
            # keyword invented for both platforms is two inventions in
            # `hallucinated_keywords`, so listing it twice here keeps the string
            # and the count describing the same event.
            hallucinated_keyword_terms=";".join(
                k for s in res.selections for k in s.hallucinated
            ),
            near_miss_keyword_pairs=";".join(
                f"{emitted}->{resolved}"
                for s in res.selections
                for emitted, resolved in s.near_miss.items()
            ),
            near_miss_keywords=sum(len(s.near_miss) for s in res.selections),
            keywords_real=real_keywords,
            out_of_window_keywords=sum(len(s.out_of_window) for s in res.selections),
            removed_by_cleaning_keywords=sum(
                len(s.removed_by_cleaning) for s in res.selections
            ),
            arm=res.arm,
            intervention=res.intervention,
            selection_schema_violation=_count(res, OUTCOME_SCHEMA_VIOLATION),
            schema_violations=res.schema_violations,
            schema_violation_stages=res.schema_violation_stages,
            selection_violation_detail=_violation_json(res, lambda s: s.schema_violations),
            selection_violation_raw=_violation_json(res, lambda s: s.raw_response),
            selection_cutoff_raw=_violation_json(
                res, lambda s: s.raw_response, (OUTCOME_OVERRUN, OUTCOME_TRUNCATED)
            ),
            keywords_claimed_copied=res.keywords_claimed_copied,
            keywords_claimed_recalled=res.keywords_claimed_recalled,
            platforms_requested=len(res.platform_outcomes),

            platforms_contributing=sum(1 for o in res.platform_outcomes if o.contributed),
            platforms_lost=len(res.lost_platforms()),
            platforms_lost_detail="; ".join(
                f"{o.platform}:{o.reason}" for o in res.lost_platforms()
            ),
            min_keyword_score=(
                res.platform_outcomes[0].min_keyword_score if res.platform_outcomes else None
            ),
            corpus_platforms_requested=(
                len(res.coverage.requested) if res.coverage else 0
            ),
            corpus_platforms_loaded=len(res.coverage.loaded) if res.coverage else 0,
            corpus_platforms_missing=(
                ",".join(res.coverage.missing) if res.coverage else ""
            ),
            corpus_courses=(
                sum(res.coverage.course_counts.values()) if res.coverage else 0
            ),
            corpus_coverage_known=res.coverage is not None,
            context_courses_dropped=res.context_courses_dropped,
            # Also set here, not only stamped from provenance, so a caller that
            # passes no provenance still exports the cap that was in force. The two
            # sources read the same config field, so they cannot disagree.
            max_context_courses=res.context_courses_cap,
            context_chars=res.context_chars,
            # Exported verbatim. Not truncated, not summarised: a length cap on
            # the artifact would silently corrupt exactly the runs worth looking
            # at, and an over-long path is itself a finding.
            learning_path=res.learning_path,
            expanded_query=res.expanded_query,
            expansion_mode=res.expansion_mode,
            # Recomputed here rather than threaded through, because `cited_ids()`
            # is the same call `cited` above counts -- one source, so the column
            # and the count cannot disagree.
            cited_course_ids=";".join(res.cited_ids()),
            # X2. Presented order, upper-cased to match `cited_course_ids` so the
            # two are set-comparable without either side normalising first.
            retrieved_course_ids=";".join(
                r.course.id_.upper() for r in res.retrieved
            ),
            hallucinated_course_ids=";".join(res.hallucinated_ids()),
            llm_retries=res.llm_retries,
            llm_throttled_calls=res.llm_throttled_calls,
        )

    @classmethod
    def cell_failure(
        cls, *, model: str, pipeline: str, query: str, error: BaseException
    ) -> BenchmarkMetric:
        """A row for a **cell** that never completed -- one (query x model x
        pipeline) unit of the sweep grid, not an intervention arm (T1).

        Every measured field stays at its "nothing was measured" value -- notably
        `grounding_accuracy=None` and `grounding_accuracy_n=0`, so a failed cell
        cannot enter the grounding mean, and `tokens_complete=False`, so it cannot
        enter a token total either. Only `status` and `failure_detail` carry
        information, which is the whole point: the cell is *counted* rather than
        silently absent from the export.
        """
        return cls(
            model=model,
            pipeline=pipeline,
            query=query,
            elapsed_seconds=0.0,
            status=STATUS_CELL_FAILED,
            failure_detail=f"{type(error).__name__}: {error}"[:500],
            tokens_complete=False,
            grounding_accuracy=None,
            grounding_accuracy_n=0,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# B22 -- resume. Statuses that mean "this cell did not produce a measurement and
# should be run again". Everything else -- `ok`, and equally `partial_retrieval`
# and `empty_retrieval` -- is a **result about the model** and is kept: re-running
# an empty retrieval until it stops being empty would be selecting on the outcome,
# which is a different experiment from the one that was specified.
#
# `arm_failed` is the pre-T1 spelling of `cell_failed` and is accepted so a 2026-08-20
# export can be resumed by today's code. `compare_arms.py` already carries the same
# alias for the same reason.
RERUNNABLE_STATUSES: frozenset[str] = frozenset({"cell_failed", "arm_failed"})


def cell_key(
    *, repeat: int, query_id: str, model: str, pipeline: str, arm: str
) -> tuple[int, str, str, str, str]:
    """Identity of one cell, for resume matching (B22).

    All five fields are needed. `arm` in particular: an arm sweep runs the same
    (query, model, pipeline) triple under different interventions, so a key
    without it would treat an Arm 2 row as evidence that the Arm 3 cell had run.
    """
    return (int(repeat), str(query_id), str(model), str(pipeline), str(arm or ""))


def completed_cells(
    rows: Sequence[dict],
    *,
    rerunnable: frozenset[str] = RERUNNABLE_STATUSES,
) -> set[tuple[int, str, str, str, str]]:
    """Cell keys in `rows` that already carry a measurement and must not be paid for twice."""
    done = set()
    for r in rows:
        if str(r.get("status", "")) in rerunnable:
            continue
        done.add(
            cell_key(
                repeat=r.get("repeat", 1) or 1,
                query_id=r.get("query_id", ""),
                model=r.get("model", ""),
                pipeline=r.get("pipeline", ""),
                arm=r.get("arm", ""),
            )
        )
    return done


def metric_from_row(row: dict) -> BenchmarkMetric:
    """Rebuild a `BenchmarkMetric` from an exported row (B22).

    Only fields the dataclass declares are taken, so an export written by a newer
    or older revision does not explode the constructor -- a resumed sweep across a
    schema change keeps the columns both revisions agree on and drops the rest,
    which is visible in the merged export rather than fatal at load.
    """
    field_map = {f.name: f for f in fields(BenchmarkMetric)}
    kwargs = {}
    for k, v in row.items():
        if k not in field_map:
            continue
        f_type = str(field_map[k].type)
        if v is None or v == "" or (isinstance(v, str) and v.strip() == "None"):
            if "None" in f_type:
                kwargs[k] = None
            elif "int" in f_type:
                kwargs[k] = 0
            elif "float" in f_type:
                kwargs[k] = 0.0
            elif "bool" in f_type:
                kwargs[k] = False
            elif "dict" in f_type:
                kwargs[k] = {}
            else:
                kwargs[k] = ""
            continue

        if "bool" in f_type and isinstance(v, str):
            kwargs[k] = v.strip().lower() in ("true", "1", "t", "yes")
        elif "int" in f_type and isinstance(v, (str, float)):
            try:
                kwargs[k] = int(float(v))
            except ValueError:
                kwargs[k] = None if "None" in f_type else 0
        elif "float" in f_type and isinstance(v, str):
            try:
                kwargs[k] = float(v)
            except ValueError:
                kwargs[k] = None if "None" in f_type else 0.0
        elif "dict" in f_type and isinstance(v, str):
            try:
                import ast
                kwargs[k] = ast.literal_eval(v)
            except Exception:
                kwargs[k] = {}
        else:
            kwargs[k] = v

    return BenchmarkMetric(**kwargs)


def run_multi_model_eval(
    queries: Sequence[str],
    models: Sequence[str],
    pipelines: Sequence[str],
    backend: LLMBackend,
    config: Config,
    stores: dict[str, EmbeddingStore],
    *,
    embedder: Embedder | None = None,
    coverage: CorpusCoverage | None = None,
    query_ids: Sequence[str] | None = None,
    repeats: int = 1,
    keyword_model: str | None = None,
    expansion_model: str | None = None,
    generation_model: str | None = None,
    seed_per_repeat: bool = False,
    provenance: RunProvenance | None = None,
    on_cell: Callable[[dict], None] | None = None,
    arm: str | None = None,
    intervention: str | None = None,
    arms: Sequence[str] | None = None,
    model_override: str | Sequence[str] | None = None,
    output_csv_path: Path | str | None = None,
    skip_cells: set[tuple[int, str, str, str, str]] | None = None,
    sweep_version: str | None = None,
    schema_exemplar: bool = False,
    max_consecutive_failures: int | None = MAX_CONSECUTIVE_CELL_FAILURES,
    candidate_injection: bool = False,
    candidate_limit: int = 100,
    candidate_ranker: str = RANKER_LEXICAL,
    expansion_mode: str | None = None,
    candidate_source: str = CANDIDATE_SOURCE_REWRITE,
) -> list[BenchmarkMetric]:

    """Run every (query x model x pipeline) cell and return one row per cell.

    **Vocabulary (T1).** A *cell* is one (query x model x pipeline) unit of this
    grid. An *arm* is a condition of the experiment -- Arm 1 baseline 70B, Arm 2
    naive 8B, Arm 3 skill schemas.

    **B0 -- arm mode.** When `arms` is provided, the function iterates over the
    arm list instead of `models`, applying each `ArmConfig`'s full model assignment
    and intervention to every cell. The `models` list is ignored in arm mode unless
    `model_override` is passed; it exists solely for the legacy `models × pipelines` path.
    Backward compatibility: when `arms` is None, the existing behaviour is preserved exactly.

    By default every role -- expansion, keyword selection, generation -- runs on
    the cell's own model, which is the whole-model comparison Arms 1 and 2
    describe.

    The three role arguments pin a stage across all cells instead. Pinning
    generation is what the condition grid requires: "the final generation stage is
    identical in every cell -- same prompt, same model, one call. Only keyword
    selection varies." Without it, a cell that produces a better learning path
    could owe that to picking better keywords or to writing better prose, and the
    export cannot separate the two. Defaulting to the cell's model keeps the
    existing behaviour for callers that pass nothing.

    `repeats` runs the whole grid N times. The outer loop is the repeat, not the
    cell, so a sweep killed partway leaves complete passes behind rather than a
    grid that is deeper in one corner than another.

    `seed_per_repeat` walks the seed with the pass -- base, base+1, base+2 -- using
    the *same* sequence for every cell, so cells stay paired while each pass
    becomes a genuinely different draw. It is opt-in and off by default, because at
    `temperature=0.0` (this project's setting) decoding is greedy and the seed has
    almost nothing to steer: varying it would produce near-identical output and an
    error bar close to zero, which would read as precision rather than as the
    absence of anything to measure. Turn it on for a study that raises the
    temperature; leave it off for the headline comparison, where a single fixed
    seed shared by every cell is exactly what fairness requires.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    if query_ids is not None and len(query_ids) != len(queries):
        raise ValueError(
            f"query_ids has {len(query_ids)} entries for {len(queries)} queries; "
            "they must correspond one to one."
        )

    stamp = provenance.as_columns() if provenance else {}
    metrics: list[BenchmarkMetric] = []
    consecutive_failures: int = 0

    # B0: arm-mode slot resolution.
    #
    # Each slot is an 11-tuple:
    #   (cell_model, keyword_model, expansion_model, generation_model,
    #    arm_str, intervention_str, schema_version_str, schema_enabled,
    #    schema_variant, candidate_injection, candidate_limit)
    #
    # `schema_enabled` joined the slot in B1/B17 (2026-08-26). Before that the
    # treatment flag never left `ArmConfig` and the pipeline re-derived which
    # arm it was in from a string comparison on the label -- two encodings of
    # one axis, free to disagree, and disagreeing silently.
    #
    # In arm mode (`arms` is not None) the slots are assembled from the
    # resolved ArmConfig objects. In legacy mode (`arms` is None) they are
    # assembled from `models` and the per-role kwargs, preserving the existing
    # models × pipelines behaviour exactly -- the loop body below is identical
    # for both paths.
    if arms:
        # B16. Refuse before the first cell rather than after the hundredth.
        # Checked over the whole requested list so that `--arm all` fails at the
        # top with a name, instead of producing 100 good rows and 50 mislabelled
        # ones in the same artifact.
        _unimplemented = [a for a in (x.strip() for x in arms) if a in UNIMPLEMENTED_ARMS]
        if _unimplemented:
            raise ValueError(
                f"run_multi_model_eval: arm(s) {sorted(set(_unimplemented))} have no "
                "implementation -- the treatment is not built, so the run would "
                "execute the Arm 2 zero-shot pipeline and label the row with this "
                "arm number (B16). Request the arms that exist explicitly, "
                "e.g. --arm 1,2,3."
            )
        # Model selection and schema variant are independent axes, and the
        # merge that produced this block dropped one of them each way. They
        # compose: `-m` chooses which checkpoint fills every role on the arm,
        # `--schema-exemplar` chooses which prompt variant the schema arms use.
        if model_override:
            if isinstance(model_override, str):
                _models_to_run = [m.strip() for m in model_override.split(",") if m.strip()]
            else:
                _models_to_run = list(model_override)
            _arm_cfgs: list[ArmConfig] = [
                resolve_arm(a, model=m)
                for m in _models_to_run
                for a in arms
            ]
        else:
            _arm_cfgs = [resolve_arm(a) for a in arms]
        if schema_exemplar:
            # The variant is an argument, not a property of the arm, so it is
            # applied here rather than inside `resolve_arm` -- an arm's identity
            # must not depend on a CLI flag, or two runs of "--arm 3" would mean
            # different things with nothing in `resolve_arm` saying so.
            _arm_cfgs = [
                replace(ac, schema_prompt_variant=VARIANT_SPEC_PLUS_EXEMPLAR)
                if ac.schema_enabled else ac
                for ac in _arm_cfgs
            ]
        def _variant_for(ac: ArmConfig) -> str:
            """The condition label for this arm's rows.

            Empty on arms that run no schema -- the label is an Arm 3 taxonomy,
            while `candidate_injection` on the row is the machine truth and is
            stamped everywhere. Injection promotes `spec_only` to Condition C.
            Injection *with* the exemplar keeps `spec_plus_exemplar`: that
            combination is not a named condition, and inventing a fourth letter
            for it here would put a label in the artifact that no protocol
            document defines.
            """
            if not ac.schema_enabled:
                return ""
            if (
                (ac.candidate_injection or candidate_injection)
                and ac.schema_prompt_variant == VARIANT_SPEC_ONLY
            ):
                return VARIANT_SPEC_PLUS_CANDIDATE_INJECTION
            return ac.schema_prompt_variant

        _slots = [
            (ac.model, ac.keyword_model, ac.expansion_model,
             ac.generation_model, ac.arm, ac.intervention, ac.schema_version,
             ac.schema_enabled, _variant_for(ac),
             ac.candidate_injection or candidate_injection,
             ac.candidate_limit if ac.candidate_injection else candidate_limit)
            for ac in _arm_cfgs
        ]
    else:
        _slots = [
            (m, keyword_model or m, expansion_model or m,
             generation_model or m, arm or "", intervention or "", "",
             None, "", candidate_injection, candidate_limit)
            for m in models
        ]

    # Fail before the first API call, not on the first cell. Both checks are
    # about the artifact rather than the code: a run that asked for a ranker
    # nothing ran would stamp `candidate_ranker` empty on every row and look,
    # afterwards, exactly like a run that never asked -- so the request has to
    # be refused while the operator is still at the keyboard.
    if candidate_ranker not in RANKER_STRATEGIES:
        raise ValueError(
            f"unknown candidate ranker {candidate_ranker!r}; "
            f"expected one of {RANKER_STRATEGIES}"
        )
    if candidate_ranker != RANKER_LEXICAL and not any(slot[9] for slot in _slots):
        raise ValueError(
            f"--candidate-ranker {candidate_ranker!r} was requested but no cell "
            "in this sweep enables candidate injection, so no ranker will run. "
            "Pass --candidate-injection (or select an arm that sets it)."
        )
    # Same two refusals for the candidate source, for the same reason.
    if candidate_source not in CANDIDATE_SOURCES:
        raise ValueError(
            f"unknown candidate source {candidate_source!r}; "
            f"expected one of {CANDIDATE_SOURCES}"
        )
    if candidate_source != CANDIDATE_SOURCE_REWRITE and not any(slot[9] for slot in _slots):
        raise ValueError(
            f"--candidate-source {candidate_source!r} was requested but no cell "
            "in this sweep enables candidate injection, so no candidates are ranked. "
            "Pass --candidate-injection (or select an arm that sets it)."
        )
    if expansion_mode is not None:
        for _slot in _slots:
            resolve_expansion_mode(
                expansion_mode,
                _slot[7],
                resolve_sweep_version(sweep_version),
            )

    _skip = skip_cells or set()
    total_cells = sum(
        1
        for _rep in range(1, repeats + 1)
        for _qi in range(len(queries))
        for _slot in _slots
        for _pipe in pipelines
        if cell_key(
            repeat=_rep,
            query_id=(query_ids[_qi] if query_ids else f"q{_qi + 1:02d}"),
            model=_slot[0],
            pipeline=_pipe,
            arm=_slot[4],
        )
        not in _skip
    )
    skipped_cells = (repeats * len(queries) * len(_slots) * len(pipelines)) - total_cells
    if skipped_cells:
        print(f"  resume: skipping {skipped_cells} cell(s) already measured")

    out_csv = Path(output_csv_path) if output_csv_path else None
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        if out_csv.exists():
            out_csv.unlink()

    merged_courses = []
    matrices = []
    for store in stores.values():
        merged_courses.extend(store.courses)
        matrices.append(store.matrix)

    if matrices:
        merged_matrix = np.vstack(matrices)
    else:
        merged_matrix = np.empty((0, 384), dtype=np.float32)

    merged_store = EmbeddingStore(courses=merged_courses, matrix=merged_matrix)


    for rep in range(1, repeats + 1):
        pass_seed = config.llm.seed
        if seed_per_repeat and pass_seed is not None:
            pass_seed = pass_seed + (rep - 1)
        pass_stamp = {**stamp, "seed": pass_seed} if stamp else {}

        for qi, q in enumerate(queries):
            qid = query_ids[qi] if query_ids else f"q{qi + 1:02d}"
            for (
                cell_model, eff_kw, eff_exp, eff_gen,
                eff_arm, eff_intv, eff_schema_ver,
                eff_schema_on, eff_schema_variant,
                eff_candidate_inj, eff_candidate_lim,
            ) in _slots:
                # Stamped onto the row below rather than taken from the run
                # header, so the artifact records what this cell ran, not what
                # the run was launched with.
                eff_sweep = resolve_sweep_version(sweep_version)
                cell_stamp = {**pass_stamp, "sweep_version": eff_sweep}
                # For rows whose cell died before `run_skyrag` could report its
                # own mode. Legacy mode infers the schema from labels inside
                # `run_skyrag`, so with nothing explicit the honest stamp is empty.
                _sweep_expansion_mode = (
                    resolve_expansion_mode(expansion_mode, eff_schema_on, eff_sweep)
                    if expansion_mode is not None or eff_schema_on is not None
                    else ""
                )
                for pipe in pipelines:
                    _cell_expansion_mode = (
                        _sweep_expansion_mode if pipe == "skyrag" else ""
                    )
                    cfg = replace(config, llm=replace(config.llm))
                    cfg.llm.model = cell_model
                    cfg.llm.expansion_model = eff_exp
                    cfg.llm.keyword_model = eff_kw
                    cfg.llm.generation_model = eff_gen
                    cfg.llm.seed = pass_seed

                    if pipe not in ("skyrag", "naive"):
                        raise ValueError(f"Unknown pipeline: {pipe}")

                    if cell_key(
                        repeat=rep, query_id=qid, model=cell_model,
                        pipeline=pipe, arm=eff_arm or "",
                    ) in _skip:
                        continue

                    if on_cell is not None:
                        on_cell({"event": "start", "repeat": rep, "query_id": qid,
                                 "model": cell_model, "pipeline": pipe,
                                 "index": len(metrics) + 1, "total": total_cells})

                    try:
                        res = _run_cell(
                            pipe,
                            q,
                            cfg,
                            backend,
                            stores,
                            merged_store,
                            embedder,
                            coverage,
                            arm=eff_arm,
                            intervention=eff_intv,
                            schema_enabled=eff_schema_on,
                            schema_exemplar=(
                                eff_schema_variant == VARIANT_SPEC_PLUS_EXEMPLAR
                            ),
                            sweep_version=eff_sweep,
                            candidate_injection=eff_candidate_inj,
                            candidate_limit=eff_candidate_lim,
                            candidate_ranker=candidate_ranker,
                            expansion_mode=expansion_mode,
                            candidate_source=candidate_source,
                        )
                    except AuthError:
                        raise
                    # B14, second layer. `LLMError` alone was too narrow: a
                    # `ValueError`/`TypeError` raised inside a pipeline matched no
                    # handler here and propagated out of the sweep driver,
                    # discarding every cell already completed in this pass. The
                    # sweep must not depend on every pipeline being internally
                    # perfect: one bad cell is a failed
                    # row, never a lost run. `AuthError` still re-raises above,
                    # because retrying 250 cells against rejected credentials
                    # helps nobody.
                    except (LLMError, ValueError, TypeError, KeyError, AttributeError) as exc:
                        consecutive_failures += 1
                        metrics.append(
                            _stamp(
                                BenchmarkMetric.cell_failure(
                                    model=cell_model, pipeline=pipe, query=q, error=exc
                                ),
                                cell_model=cell_model, cfg=cfg, qid=qid, rep=rep,
                                stamp=cell_stamp,
                                arm=eff_arm, intervention=eff_intv,
                                schema_version=eff_schema_ver,
                                schema_prompt_variant=eff_schema_variant,
                                candidate_injection=eff_candidate_inj,
                                candidate_limit=eff_candidate_lim,
                                candidate_ranker=candidate_ranker,
                                candidate_source=candidate_source,
                                expansion_mode=_cell_expansion_mode,
                            )
                        )
                        if out_csv:
                            _append_metric_to_csv(metrics[-1], out_csv)
                        if on_cell is not None:
                            on_cell({"event": "failed", "repeat": rep, "query_id": qid,
                                     "model": cell_model, "pipeline": pipe,
                                     "index": len(metrics), "total": total_cells,
                                     "error": f"{type(exc).__name__}: {exc}"[:160]})
                        if (
                            max_consecutive_failures is not None
                            and consecutive_failures >= max_consecutive_failures
                        ):
                            raise SweepCircuitBreakerError(
                                f"Sweep halted by circuit breaker: encountered {consecutive_failures} consecutive cell failure(s) "
                                f"(latest error: {type(exc).__name__}: {exc}). Partial results ({len(metrics)} cells) saved to CSV."
                            )
                        continue

                    consecutive_failures = 0
                    metrics.append(
                        _stamp(
                            BenchmarkMetric.from_pipeline_result(res),
                            cell_model=cell_model, cfg=cfg, qid=qid, rep=rep,
                            stamp=cell_stamp,
                            arm=eff_arm, intervention=eff_intv,
                            schema_version=eff_schema_ver,
                            schema_prompt_variant=eff_schema_variant,
                            candidate_injection=eff_candidate_inj,
                            candidate_limit=eff_candidate_lim,
                            candidate_ranker=candidate_ranker,
                            candidate_source=candidate_source,
                            expansion_mode=_cell_expansion_mode,
                        )
                    )
                    if out_csv:
                        _append_metric_to_csv(metrics[-1], out_csv)
                    if on_cell is not None:
                        m = metrics[-1]
                        on_cell({"event": "done", "repeat": rep, "query_id": qid,
                                 "model": cell_model, "pipeline": pipe,
                                 "index": len(metrics), "total": total_cells,
                                 "status": m.status, "retrieved": m.retrieved_count,
                                 "cited": m.cited_count,
                                 "selection_seconds": m.selection_seconds,
                                 "generation_seconds": m.generation_seconds,
                                 "elapsed_seconds": m.elapsed_seconds})

    return metrics


def _append_metric_to_csv(metric: BenchmarkMetric, path: Path) -> None:
    header = not path.exists()
    df_row = metrics_to_dataframe([metric])
    path.parent.mkdir(parents=True, exist_ok=True)
    # `newline=""` per C15, same reason as `export_benchmark_csv` below: `to_csv`
    # terminates rows with `os.linesep` itself, and a text-mode handle translates
    # the trailing newline a second time. This is the incremental writer, so the
    # doubling landed in partially-written runs as well as finished ones.
    with open(path, "a", encoding="utf-8", newline="") as f:
        df_row.to_csv(f, index=False, header=header)
        f.flush()


def _stamp(
    metric: BenchmarkMetric,
    *,
    cell_model: str,
    cfg: Config,
    qid: str,
    rep: int,
    stamp: dict[str, Any],
    arm: str | None = None,
    intervention: str | None = None,
    schema_version: str = "",
    schema_prompt_variant: str = "",
    candidate_injection: bool = False,
    candidate_limit: int = 0,
    candidate_ranker: str = "",
    candidate_source: str = "",
    expansion_mode: str = "",
) -> BenchmarkMetric:
    """Attach identity and provenance to a row.

    Applied to the failure path too. A cell that died is exactly the row someone
    will come back to asking which configuration produced it, and it is the row
    that carries no measured field to infer anything from.

    `model` is forced back to the cell's model label. `from_pipeline_result` takes
    it from `res.model`, which is whichever model served the *generation* call --
    the same thing as the cell label only while every role runs on the cell's own
    model. Pin generation and every row in the sweep reports the pinned model as
    its own, so the axis under comparison vanishes from the results table
    entirely. What each stage actually ran is in the three role columns beside it.
    """
    return replace(
        metric,
        model=cell_model,
        query_id=qid,
        repeat=rep,
        keyword_model=cfg.llm.model_for("keyword"),
        expansion_model=cfg.llm.model_for("expansion"),
        generation_model=cfg.llm.model_for("generation"),
        arm=metric.arm or arm or "",
        intervention=metric.intervention or intervention or "",
        schema_version=metric.schema_version or schema_version or "",
        schema_prompt_variant=(
            metric.schema_prompt_variant or schema_prompt_variant or ""
        ),
        expansion_mode=metric.expansion_mode or expansion_mode or "",
        candidate_injection=metric.candidate_injection or candidate_injection,
        # Zero unless injection actually ran. Carrying the CLI default (100)
        # onto a row that injected nothing would read, in the artifact, as a
        # treatment that was never applied -- and `--candidate-limit` has a
        # default precisely so it does not have to be typed, so its presence on
        # a row must mean the flag was active, not merely available.
        candidate_limit=(
            (metric.candidate_limit or candidate_limit)
            if (metric.candidate_injection or candidate_injection)
            else 0
        ),
        # Same rule as `candidate_limit` above, and for the same reason.
        candidate_ranker=(
            (metric.candidate_ranker or candidate_ranker or "")
            if (metric.candidate_injection or candidate_injection)
            else ""
        ),
        candidate_source=(
            (metric.candidate_source or candidate_source or "")
            if (metric.candidate_injection or candidate_injection)
            else ""
        ),
        **stamp,
    )


def _run_cell(
    pipe: str,
    query: str,
    cfg: Config,
    backend: LLMBackend,
    stores: dict[str, EmbeddingStore],
    merged_store: EmbeddingStore,
    embedder: Embedder | None,
    coverage: CorpusCoverage | None,
    arm: str | None = None,
    intervention: str | None = None,
    schema_enabled: bool | None = None,
    schema_exemplar: bool = False,
    sweep_version: str | None = None,
    candidate_injection: bool = False,
    candidate_limit: int = 100,
    candidate_ranker: str = RANKER_LEXICAL,
    expansion_mode: str | None = None,
    candidate_source: str = CANDIDATE_SOURCE_REWRITE,
) -> PipelineResult:
    """One cell of the sweep. Extracted so the guard above has a single call to wrap.

    The treatment flags are tri-state: `None` means "infer from the arm label",
    which is what legacy mode passes and what preserves its pre-B17 behaviour.

    `sweep_version` is resolved here as well as by the caller, so a direct call
    that omits it still runs the default payload.
    """
    start_retries = getattr(backend, "retries", 0)
    start_throttled = getattr(backend, "throttled_calls", 0)
    sweep_version = resolve_sweep_version(sweep_version)

    if pipe == "skyrag":
        res = run_skyrag(
            query=query,
            stores=stores,
            backend=backend,
            config=cfg,
            embedder=embedder,
            coverage=coverage,
            arm=arm or "",
            intervention=intervention or "",
            schema_enabled=schema_enabled,
            schema_exemplar=schema_exemplar,
            sweep_version=sweep_version,
            candidate_injection=candidate_injection,
            candidate_limit=candidate_limit,
            candidate_ranker=candidate_ranker,
            expansion_mode=expansion_mode,
            candidate_source=candidate_source,
        )
    else:
        res = run_naive(
            query=query,
            store=merged_store,
            backend=backend,
            config=cfg,
            embedder=embedder,
            coverage=coverage,
        )

    res.llm_retries = getattr(backend, "retries", 0) - start_retries
    res.llm_throttled_calls = getattr(backend, "throttled_calls", 0) - start_throttled
    return res


def level_inversions(levels) -> tuple[int, int]:
    """O3, the cheap form. Returns (inversions, comparable_pairs).

    **BESPOKE. This is NOT RefD** (Liang et al., EMNLP 2015) and must not be
    cited as it: RefD infers prerequisite relations between concepts, this
    counts violations of a publisher-assigned three-rung difficulty scale.
    It lives here rather than in `metrics.py` for that reason -- that module's
    contract is that every metric in it has a citation.

    An *inversion* is a pair (i, j) with i before j in the recommended path where
    course i is harder than course j: advanced material placed before the
    foundational material it presumes. Counted over every ordered pair, not just
    adjacent ones, because a path that opens with its hardest course is wrong
    about that course relative to everything after it, not just the next entry.

    `None` levels are skipped rather than defaulted. An unlabelled course is an
    unknown, and defaulting it to "beginner" would manufacture violations the
    catalogue does not support -- so `comparable_pairs` is the honest
    denominator and is returned beside the count rather than left implicit. The
    same obligation `grounding_accuracy_n` carries.

    **What it cannot see.** Two courses on the same rung where one genuinely
    requires the other score as fine. This is a floor on ordering quality, not a
    prerequisite check, and must be described as one.
    """
    ordinals = list(levels)
    inversions = comparable = 0
    for i in range(len(ordinals)):
        if ordinals[i] is None:
            continue
        for j in range(i + 1, len(ordinals)):
            if ordinals[j] is None:
                continue
            comparable += 1
            if ordinals[i] > ordinals[j]:
                inversions += 1
    return inversions, comparable


def _percentile(values, pct: int) -> float | None:
    """Nearest-rank percentile. `None` on an empty list, per the module convention.

    Added with the nDCG distribution columns: a metric demoted for sitting at its
    ceiling was demoted on the strength of a mean, and the mean was hiding a tail
    reaching 0.7169. No interpolation -- the rank is an index into observed
    values, so every number reported is one a real run actually produced.
    """
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = max(0, min(len(vals) - 1, (pct * len(vals)) // 100))
    return round(vals[k], 4)


def _mean_pairwise_overlap(courses) -> float | None:
    """Mean Jaccard overlap of the keyword sets of every pair of courses.

    O2's redundancy half. 1.0 means every recommended course carries the same
    topic labels -- five spellings of one course, which is a bad recommendation
    set however well grounded each entry is. 0.0 means they share no labels.

    None below two courses, and None when no course carries keywords: overlap
    within a set of one is undefined, and a corpus with no labels cannot answer
    the question. Zero would claim perfect diversity in both cases, which is the
    reading that flatters the result.
    """
    sets = [{k.strip().lower() for k in c.keywords if k.strip()} for c in courses]
    sets = [s for s in sets if s]
    if len(sets) < 2:
        return None

    total, pairs = 0.0, 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            if union:
                total += len(sets[i] & sets[j]) / len(union)
                pairs += 1
    return round(total / pairs, 4) if pairs else None


def _rate(numerator: int, denominator: int) -> float | None:
    """A ratio, or None when the denominator is zero.

    None rather than 0.0 on purpose: a run that emitted no keywords has an
    *undefined* invalid-keyword rate, and 0.0 would read as a model that emitted
    many and invented none -- the opposite finding.
    """
    return round(numerator / denominator, 4) if denominator else None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _ceiling_fraction(
    values: list[float | None], *, ceiling: float = 1.0, tolerance: float = 1e-9
) -> float | None:
    """R24(c). Fraction of scored rows sitting at a metric's maximum.

    A saturation warning, not a metric. nDCG@10 and the reciprocal rank both
    returned their ceiling on every scorable row of both arms on 2026-08-20, so
    they described the presentation order rather than the arms. This says so in
    the summary rather than leaving it to be noticed in a results table.

    `None` -- not 0.0 -- when nothing was scorable: no rows means no evidence of
    saturation, and 0.0 would assert the opposite.
    """
    scored = [v for v in values if v is not None]
    if not scored:
        return None
    at_ceiling = sum(1 for v in scored if abs(v - ceiling) <= tolerance)
    return round(at_ceiling / len(scored), 4)


def summarize_metrics(metrics: list[BenchmarkMetric]) -> dict[str, Any]:
    """Aggregate a benchmark run without letting failures average into success.

    `grounding_accuracy` is averaged only over runs that actually cited
    something. Runs that cited nothing are reported as their own count, so a
    high mean cannot be produced by runs that recommended no courses at all.
    `grounding_accuracy_n` states how many runs the mean is over, so a mean
    computed from two of twenty runs cannot be mistaken for a mean over twenty.
    """
    scored = [m.grounding_accuracy for m in metrics if m.grounding_accuracy is not None]

    return {
        "runs": len(metrics),
        "grounding_accuracy_mean": (
            round(sum(scored) / len(scored), 4) if scored else None
        ),
        "grounding_accuracy_n": len(scored),
        "zero_citation_runs": sum(1 for m in metrics if m.zero_citation),
        "empty_retrieval_runs": sum(
            1 for m in metrics if m.status == STATUS_EMPTY_RETRIEVAL
        ),
        "ok_runs": sum(1 for m in metrics if m.status == STATUS_OK),
        # Ran and produced courses, but not from every platform asked. Counted
        # apart from ok_runs because averaging the two together is what let a
        # half-corpus run pass as a whole-corpus one.
        "partial_retrieval_runs": sum(
            1 for m in metrics if m.status == STATUS_PARTIAL_RETRIEVAL
        ),
        "platforms_lost_total": sum(m.platforms_lost for m in metrics),
        # Runs that were asked for more platforms than had a dataset to load. A
        # corpus-level shortfall, upstream of anything the model did, and reported
        # separately for that reason.
        "runs_on_incomplete_corpus": sum(
            1 for m in metrics
            if m.corpus_coverage_known and m.corpus_platforms_missing
        ),
        "runs_without_coverage_record": sum(
            1 for m in metrics if not m.corpus_coverage_known
        ),
        # Token totals are only summable over rows where both halves were reported;
        # mixing in the partial ones is the `or 0` defect at the aggregate level.
        "runs_with_incomplete_tokens": sum(1 for m in metrics if not m.tokens_complete),
        # --- X1: where a run's tokens actually went -------------------------
        #
        # Summed over the rows that reported each stage, with the count of such
        # rows beside every total. Without the denominator these totals are the
        # aggregate form of the defect X1 fixed: a figure that looks like a
        # run's cost and silently omits whichever rows had no usage block.
        #
        # `sum_tokens` cannot be reused here -- it refuses a partial sum, which
        # is right per run and wrong per sweep, where the honest report is "this
        # much, over this many rows".
        "expansion_tokens_total": _stage_total(
            metrics, "expansion_prompt_tokens", "expansion_completion_tokens"
        ),
        "selection_tokens_total": _stage_total(
            metrics, "selection_prompt_tokens", "selection_completion_tokens"
        ),
        "generation_tokens_total": _stage_total(
            metrics, "prompt_tokens", "completion_tokens"
        ),
        "pipeline_tokens_total": sum(
            m.pipeline_total_tokens for m in metrics if m.pipeline_total_tokens is not None
        ),
        "pipeline_tokens_n": sum(1 for m in metrics if m.pipeline_tokens_complete),
        # --- R22: the verbosity control ------------------------------------
        #
        # Reported at sweep level so a between-arm O1 difference can be checked
        # against a between-arm length difference without re-reading the CSV.
        # A mean alone would hide the case that matters -- one arm running long
        # on a few queries -- so the max travels with it.
        "path_chars_mean": (
            round(sum(m.path_chars for m in metrics) / len(metrics), 1)
            if metrics
            else None
        ),
        "path_words_mean": (
            round(sum(m.path_words for m in metrics) / len(metrics), 1)
            if metrics
            else None
        ),
        "path_chars_max": max((m.path_chars for m in metrics), default=0),
        # Cells that raised rather than producing a result. Counted, because a sweep
        # that lost cells to infrastructure is not the sweep that was requested, and
        # the alternative was an exception that left no artifact to count from.
        "failed_cells": sum(1 for m in metrics if m.status == STATUS_CELL_FAILED),
        "runs_with_selection_errors": sum(1 for m in metrics if m.selection_errors),
        "runs_with_empty_selections": sum(1 for m in metrics if m.empty_selections),
        # Harness-attributable: a run carrying either of these lost a platform for
        # reasons that are ours, not the model's, and must not be read as
        # capability. Kept separate from each other because one is a token budget
        # and the other is an absent corpus.
        "runs_with_truncated_selections": sum(
            1 for m in metrics if m.selection_truncated
        ),
        "runs_with_no_vocabulary": sum(
            1 for m in metrics if m.selection_no_vocabulary
        ),
        # Model-attributable: either finished cleanly and produced nothing
        # usable, or ran past a cap it had room to respect. Both are the
        # instruction-following signal a scale comparison is built on, so
        # neither may be filed beside the truncations above.
        "runs_with_unparseable_selections": sum(
            1 for m in metrics if m.selection_unparseable
        ),
        "runs_with_overrun_selections": sum(
            1 for m in metrics if m.selection_overrun
        ),
        # Rates, now that the denominator exists. None rather than 0 when nothing
        # was emitted: no keywords means the rate is undefined, and 0.0 would read
        # as a model that emitted many and invented none.
        "keywords_emitted_total": sum(m.keywords_emitted for m in metrics),
        "keywords_kept_total": sum(m.keywords_kept for m in metrics),
        "invalid_keyword_rate": _rate(
            sum(m.hallucinated_keywords for m in metrics),
            sum(m.keywords_emitted for m in metrics),
        ),
        "near_miss_rate": _rate(
            sum(m.near_miss_keywords for m in metrics),
            sum(m.keywords_emitted for m in metrics),
        ),
        # M2.3 as the glossary specifies it. **Not arm-comparable when invalid
        # rates differ** -- see `keywords_real` and the conditional form below.
        "out_of_window_rate": _rate(
            sum(m.out_of_window_keywords for m in metrics),
            sum(m.keywords_emitted for m in metrics),
        ),
        # **R24(a).** The conditional form: of the keywords the model named that
        # the corpus really contains, what fraction lay outside the window it was
        # shown? This is the question capability 1 is actually asking, and it is
        # the form that must be quoted in any between-arm comparison.
        "out_of_window_rate_conditional": _rate(
            sum(m.out_of_window_keywords for m in metrics),
            sum(m.keywords_real for m in metrics),
        ),
        "keywords_real_total": sum(m.keywords_real for m in metrics),
        "retrieval_score_mean": _mean(
            [m.retrieval_score_mean for m in metrics if m.retrieval_score_mean is not None]
        ),
        "cited_redundancy_mean": _mean(
            [m.cited_redundancy for m in metrics if m.cited_redundancy is not None]
        ),
        "cited_redundancy_cosine_mean": _mean(
            [
                m.cited_redundancy_cosine
                for m in metrics
                if m.cited_redundancy_cosine is not None
            ]
        ),
        "context_ndcg_at_10_mean": _mean(
            [m.context_ndcg_at_10 for m in metrics if m.context_ndcg_at_10 is not None]
        ),
        # Every mean above skips `None`, so the denominators differ between them
        # and none is recoverable from `runs`. `grounding_accuracy_n` exists for
        # exactly this reason; these three are nullable for reasons that vary by
        # arm -- a row citing one course scores no redundancy, a row retrieving
        # ten or fewer scores no reciprocal rank -- so the count has to travel.
        "cited_redundancy_n": sum(1 for m in metrics if m.cited_redundancy is not None),
        "cited_redundancy_cosine_n": sum(
            1 for m in metrics if m.cited_redundancy_cosine is not None
        ),
        "context_ndcg_at_10_n": sum(
            1 for m in metrics if m.context_ndcg_at_10 is not None
        ),
        # **R24(c). Saturation, reported beside the mean rather than discovered
        # in the results table.** On the 2026-08-20 sweep nDCG@10 was 0.99 and
        # the reciprocal rank was exactly 1.000 on *every* scorable row of
        # *both* arms. A metric that returns its maximum on every row of every
        # arm is measuring the harness -- here, that the presented order is
        # already close to the score-sorted order -- and it cannot separate two
        # arms no matter how different they are.
        #
        # These fractions make that visible in the summary that produces the
        # table, not three months later. A value at or near 1.0 means the metric
        # has no discriminating power on this data and must not be reported as
        # an arm comparison. `_ceiling_fraction` counts rows at the maximum;
        # `None` when nothing was scorable, per the module convention.
        "context_ndcg_at_10_ceiling_fraction": _ceiling_fraction(
            [m.context_ndcg_at_10 for m in metrics]
        ),
        # **The mean was hiding the distribution (2026-09-15).** R24c demoted
        # nDCG@10 as "saturated" on the strength of two means, 0.987 against
        # 0.992. Measured over all 1,371 scored rows on disk the column is not
        # saturated at all: only 32.4% sit at exactly 1.0, the 5th percentile is
        # 0.9495 and the minimum is 0.7169. A mean of 0.99 over a distribution
        # with a 0.72 tail is an average doing what averages do.
        #
        # The tail is the interesting part and it is the only part that can
        # answer the interleave question: `merge_platforms` is positional and
        # score-blind across platforms, so a weak edx course at that platform's
        # rank 0 is presented above a strong coursera course at rank 1. These
        # two report where that costs something instead of averaging it away.
        "context_ndcg_at_10_p5": _percentile(
            [m.context_ndcg_at_10 for m in metrics if m.context_ndcg_at_10 is not None],
            5,
        ),
        "context_ndcg_at_10_below_0_95": sum(
            1
            for m in metrics
            if m.context_ndcg_at_10 is not None and m.context_ndcg_at_10 < 0.95
        ),
        # **R24(b). M3.2 moves with its denominator, and the denominator is an
        # arm property.** `citation_coverage` is cited / retrieved. On the
        # 2026-08-20 sweep the 8B "won" it 0.228 to 0.071 while citing *fewer*
        # courses in absolute terms -- it scored higher purely because its cap
        # overruns lost it a platform and left it retrieving 74 courses where the
        # 70B retrieved 152. A ratio whose denominator is what the arm retrieved
        # cannot be read as a citation-behaviour finding.
        #
        # The mean is kept and the two quantities it is built from now travel
        # with it, so the column cannot be quoted bare. A reader who sees 0.228
        # beside 74 retrieved and 17 cited can see what happened; a reader who
        # sees 0.228 alone cannot.
        "citation_coverage_mean": _mean(
            [m.citation_coverage for m in metrics if m.citation_coverage is not None]
        ),
        "citation_coverage_n": sum(
            1 for m in metrics if m.citation_coverage is not None
        ),
        "cited_count_mean": _mean([float(m.cited_count) for m in metrics]),
        "retrieved_count_mean": _mean([float(m.retrieved_count) for m in metrics]),
        # R11, summarised. The `_n` counts travel for the same reason every
        # other nullable mean's does: both are None on any row whose stages did
        # not all report usage, so their denominators differ from `runs` and
        # from each other, and neither is recoverable afterwards.
        "token_efficiency_ratio_mean": _mean(
            [
                m.token_efficiency_ratio
                for m in metrics
                if m.token_efficiency_ratio is not None
            ]
        ),
        "token_efficiency_ratio_n": sum(
            1 for m in metrics if m.token_efficiency_ratio is not None
        ),
        "tokens_per_cited_course_mean": _mean(
            [
                m.tokens_per_cited_course
                for m in metrics
                if m.tokens_per_cited_course is not None
            ]
        ),
        "tokens_per_cited_course_n": sum(
            1 for m in metrics if m.tokens_per_cited_course is not None
        ),
        "generation_seconds_total": round(sum(m.generation_seconds for m in metrics), 3),
        "selection_seconds_total": round(sum(m.selection_seconds for m in metrics), 3),
        "expansion_seconds_total": round(sum(m.expansion_seconds for m in metrics), 3),
        "hallucinated_keywords_total": sum(m.hallucinated_keywords for m in metrics),
        "near_miss_keywords_total": sum(m.near_miss_keywords for m in metrics),
        "out_of_window_keywords_total": sum(m.out_of_window_keywords for m in metrics),
        # Reported beside the hallucination total rather than folded into it: both
        # of these are keywords the corpus really contains, and a hallucination
        # rate that absorbs them measures the harness, not the model.
        "removed_by_cleaning_keywords_total": sum(
            m.removed_by_cleaning_keywords for m in metrics
        ),
        "hallucinated_ids_total": sum(m.hallucinated_count for m in metrics),
    }


def metrics_to_dataframe(metrics: list[BenchmarkMetric]) -> pd.DataFrame:
    records = [m.to_dict() for m in metrics]
    df = pd.DataFrame(records)
    if "platform_spread" in df.columns:
        df["platform_spread"] = df["platform_spread"].apply(
            lambda d: json.dumps(d) if isinstance(d, dict) else str(d)
        )
    # The generated path is the one column that can contain a newline, and a
    # newline inside a quoted CSV field is legal but not survivable: every reader
    # in this repository's tests does `csv.DictReader(text.splitlines())`, which
    # keeps the row count correct and silently deletes the newline, welding the
    # lines either side of it together. The damage is invisible in the row count,
    # which is the only thing anyone checks.
    #
    # JSON-encoding it makes the field single-line, so it survives that idiom
    # unchanged, and `json.loads` recovers the original exactly -- including
    # quotes and backslashes, which a bare newline-escape would corrupt. The JSON
    # export is untouched and carries the raw string; only the flat, line-oriented
    # format needs the escape. `test_the_exported_path_survives_a_csv_round_trip`
    # pins it through the lossy idiom specifically.
    for _text_col in ("learning_path", "expanded_query"):
        if _text_col in df.columns:
            df[_text_col] = df[_text_col].apply(
                lambda t: json.dumps(t if isinstance(t, str) else "")
            )
    return df


def export_benchmark_csv(metrics: list[BenchmarkMetric], path: Path | None = None) -> str:
    df = metrics_to_dataframe(metrics)
    csv_str = df.to_csv(index=False)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        # **C15.** `to_csv` already terminates rows with `os.linesep`, which is
        # CRLF here. `write_text` opens in text mode, where the default newline
        # handling translates that trailing newline again, so every
        # `benchmark_eval.csv` written on this machine carries CR CR LF and
        # renders with a blank line between every row. Readers tolerate it, which
        # is why it survived this long; `scripts/merge_lane_repeats.py` documents
        # it and deliberately declines to reproduce it. `newline=""` hands the
        # terminator back to `to_csv`, which is the only layer that should be
        # choosing one.
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(csv_str)
    return csv_str


def export_benchmark_json(metrics: list[BenchmarkMetric], path: Path | None = None) -> str:
    records = [m.to_dict() for m in metrics]
    json_str = json.dumps(records, indent=2)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_str, encoding="utf-8")
    return json_str
