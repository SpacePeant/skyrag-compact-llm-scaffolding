"""SKYRAG pipeline (port of rag_edu_keyword.ipynb).

Stages:
  1. Expand the terse user request into a searchable paragraph.
  2. Elicit educational principles (step-back).
  3. Per platform: pick keywords from that platform's controlled vocabulary,
     validating every choice against it.
  4. Per platform: filter courses by each selected keyword, rank semantically
     inside each bucket, keep the best few.
  5. Round-robin merge across platforms, then generate.

Step 3 running *per platform* is what guarantees every platform contributes;
step 4 is what stops the result set collapsing onto a single subtopic.
"""

from __future__ import annotations

from time import perf_counter

from ..config import Config
from ..data.adapters import CorpusCoverage, build_platform_vocabulary
from ..embedding import Embedder, EmbeddingStore
from ..llm import LLMBackend, params_from_config
from ..prompts import (
    EXPAND_QUERY,
    EXPAND_QUERY_SCHEMA_V1,
    EXPAND_QUERY_SCHEMA_V4,
    EXPAND_QUERY_SCHEMA_WORDING,
    GENERATE_PATH,
    GENERATE_PATH_SCHEMA_V4,
    STEP_BACK,
    STEP_BACK_SCHEMA_V1,
    STEP_BACK_SCHEMA_V4 as STEP_BACK_PROMPT_SCHEMA_V4,
    build_generation_prompt,
)
from ..schemas import (
    LEARNING_PATH_SCHEMA,
    LEARNING_PATH_SCHEMA_V4,
    QUERY_EXPANSION_SCHEMA,
    QUERY_EXPANSION_SCHEMA_V4,
    STAGE_EXPANSION,
    STAGE_PATH,
    STAGE_SELECTION,
    STAGE_STEP_BACK,
    STEP_BACK_SCHEMA,
    STEP_BACK_SCHEMA_V4,
    parse_and_validate,
)

from ..schemas import render as render_schema

from ..retrieval import cap_context, format_context, keyword_buckets, merge_platforms
from ..selection import (
    CANDIDATE_SOURCE_QUERY,
    CANDIDATE_SOURCE_REWRITE,
    RANKER_LEXICAL,
    select_keywords,
)
from .base import (

    REASON_BELOW_MIN_SCORE,
    REASON_FOR_OUTCOME,
    REASON_NO_COURSES_MATCHED,
    REASON_NO_USABLE_KEYWORDS,
    REASON_OK,
    STATUS_EMPTY_RETRIEVAL,
    STATUS_OK,
    STATUS_PARTIAL_RETRIEVAL,
    PipelineResult,
    PlatformOutcome,
    sum_tokens,
)


# Stage 1 rewrite modes (2026-10-01). Arm 3's schema switches every stage at
# once, and the rewrite's `expanded_query` is the text keyword selection reads,
# so an "Arm 3 vs Arm 2" contrast cannot say whether a selection effect came
# from the selection schema or from the rewrite it was handed. These modes set
# the rewrite on its own; everything after Stage 1 still follows
# `schema_enabled`.
#   plain          -- EXPAND_QUERY, Arm 2's prompt.
#   schema         -- the schema prompt, validated; Arm 3's rewrite.
#   schema_wording -- the V4 schema prompt's instructions, plain text, no schema.
EXPANSION_PLAIN = "plain"
EXPANSION_SCHEMA = "schema"
EXPANSION_SCHEMA_WORDING = "schema_wording"
EXPANSION_MODES = (EXPANSION_PLAIN, EXPANSION_SCHEMA, EXPANSION_SCHEMA_WORDING)
_V4_FAMILY = ("v4", "v5")


def resolve_expansion_mode(
    expansion_mode: str | None, schema_enabled: bool | None, sweep_version: str = ""
) -> str:
    """The rewrite a cell runs. `None` keeps the arm's own: schema iff `schema_enabled`.

    Raises on an unknown mode, and on `schema_wording` outside the V4 family,
    whose prompt it copies -- a run that asked for a control nothing defines must
    stop before its first call, not export rows that look like the control.
    """
    if expansion_mode is None:
        return EXPANSION_SCHEMA if schema_enabled else EXPANSION_PLAIN
    if expansion_mode not in EXPANSION_MODES:
        raise ValueError(
            f"unknown expansion mode {expansion_mode!r}; expected one of {EXPANSION_MODES}"
        )
    if expansion_mode == EXPANSION_SCHEMA_WORDING and sweep_version not in _V4_FAMILY:
        raise ValueError(
            f"expansion mode {EXPANSION_SCHEMA_WORDING!r} copies the V4 schema "
            f"prompt and is defined only for sweep versions {_V4_FAMILY}; "
            f"got {sweep_version!r}"
        )
    return expansion_mode


def run_skyrag(
    query: str,
    stores: dict[str, EmbeddingStore],
    backend: LLMBackend,
    config: Config,
    *,
    embedder: Embedder | None = None,
    per_keyword: int | None = None,
    use_step_back: bool = True,
    expand_query: bool = True,
    coverage: CorpusCoverage | None = None,
    arm: str = "",
    intervention: str = "",
    schema_enabled: bool | None = None,
    schema_exemplar: bool = False,
    sweep_version: str = "v3",
    candidate_injection: bool = False,
    candidate_limit: int = 100,
    candidate_ranker: str = RANKER_LEXICAL,
    expansion_mode: str | None = None,
    candidate_source: str = CANDIDATE_SOURCE_REWRITE,
) -> PipelineResult:
    start = perf_counter()
    embedder = embedder or Embedder(config.embedding)
    per_keyword = per_keyword or config.retrieval.per_keyword
    arm_label = str(arm) if arm else ""
    intervention_label = intervention or ""

    # B17. The treatment flag decides what runs; the arm label only names it.
    if schema_enabled is None:
        schema_enabled = arm_label == "3" or intervention_label == "schemas"
    expansion_mode = resolve_expansion_mode(expansion_mode, schema_enabled, sweep_version)

    # Arm 3 schema tracking (B1). Empty when schema is not in force.
    # Every stage that runs under a schema appends to these lists; the Result
    # packs them for export.
    schema_violations: list[str] = []
    schema_violation_stages: list[str] = []

    def record(r) -> None:
        if not r.ok:
            schema_violations.extend(r.violations)
            schema_violation_stages.append(r.stage)


    # Carried through to the result so corpus shortfall reaches the artifact. When
    # the caller says nothing, the honest record is what the stores themselves
    # attest -- every platform present was requested and loaded, and nothing is
    # claimed about platforms that were never mentioned. Fabricating a `requested`
    # list from config here would assert a coverage claim the caller never made.
    if coverage is None and stores:
        coverage = CorpusCoverage(
            requested=list(stores),
            loaded=list(stores),
            missing=[],
            course_counts={p: len(s.courses) for p, s in stores.items()},
        )

    # --- 1. Query expansion ---------------------------------------------
    #
    # X1: both upstream calls are kept, not just their text, so their token
    # counts can be attributed to the expansion stage. `expansion_seconds`
    # already covers the pair; these make the cost side match the timing side.
    expansion_start = perf_counter()
    expansion_responses = []
    expanded = ""
    if not expand_query:
        expanded = query
    else:
        if sweep_version in _V4_FAMILY:
            exp_prompt_tmpl = EXPAND_QUERY_SCHEMA_V4
            exp_schema = QUERY_EXPANSION_SCHEMA_V4
        else:
            exp_prompt_tmpl = EXPAND_QUERY_SCHEMA_V1
            exp_schema = QUERY_EXPANSION_SCHEMA

        if expansion_mode == EXPANSION_SCHEMA:
            exp_prompt = exp_prompt_tmpl.format(
                query=query,
                schema=render_schema(
                    exp_schema,
                    exemplar_stage=STAGE_EXPANSION if schema_exemplar else None,
                ),
            )
        elif expansion_mode == EXPANSION_SCHEMA_WORDING:
            exp_prompt = EXPAND_QUERY_SCHEMA_WORDING.format(query=query)
        else:
            exp_prompt = EXPAND_QUERY.format(query=query)

        response = backend.generate(
            exp_prompt,
            model=config.llm.model_for("expansion"),
            params=params_from_config(config.llm, max_tokens=600),
        )
        expansion_responses.append(response)
        if expansion_mode == EXPANSION_SCHEMA:
            # S1a. On a violation the raw reply is used exactly as Arm 2 would
            # use it, so a rejected schema costs the run nothing beyond the
            # recorded violation. The alternative -- dropping the expansion --
            # would change what gets *retrieved* on schema failure, making
            # every downstream metric for that cell a measurement of the
            # harness's reaction rather than of the model.
            result = parse_and_validate(
                response.text, exp_schema, STAGE_EXPANSION
            )
            record(result)
            expanded = (
                result.value["expanded_query"].strip()
                if result.ok
                else response.text.strip()
            )
        else:
            expanded = response.text.strip()
    search_query = expanded or query

    # --- 2. Step-back principles ----------------------------------------
    principles = ""
    if not use_step_back:
        principles = ""
    else:
        if sweep_version in _V4_FAMILY:
            sb_prompt_tmpl = STEP_BACK_PROMPT_SCHEMA_V4
            sb_schema = STEP_BACK_SCHEMA_V4
        else:
            sb_prompt_tmpl = STEP_BACK_SCHEMA_V1
            sb_schema = STEP_BACK_SCHEMA


        response = backend.generate(
            sb_prompt_tmpl.format(
                query=query,
                schema=render_schema(
                    sb_schema,
                    exemplar_stage=STAGE_STEP_BACK if schema_exemplar else None,
                ),
            )
            if schema_enabled
            else STEP_BACK.format(query=query),
            model=config.llm.model_for("expansion"),
            params=params_from_config(config.llm, max_tokens=800),
        )
        expansion_responses.append(response)
        if schema_enabled:
            # S1b. `principles` is interpolated into the generation prompt as
            # text in both arms, so the validated list is rendered back to the
            # bullet form Arm 2 produces. Handing the generation stage a JSON
            # array here would make Arm 3 differ from Arm 2 in what the
            # *generation* prompt contains as well as in its own output shape.
            result = parse_and_validate(
                response.text, sb_schema, STAGE_STEP_BACK
            )
            record(result)
            principles = (
                "\n".join(f"- {p}" for p in result.value["principles"])
                if result.ok
                else response.text.strip()
            )
        else:
            principles = response.text.strip()

    expansion_seconds = perf_counter() - expansion_start
    expansion_prompt_tokens = sum_tokens(r.prompt_tokens for r in expansion_responses)
    expansion_completion_tokens = sum_tokens(
        r.completion_tokens for r in expansion_responses
    )

    # --- 3 & 4. Keyword selection and bucketed retrieval, per platform ---
    #
    # Arm 3 differs from Arm 2 inside this block in exactly one way: the
    # `schema_enabled` argument to `select_keywords` below. Same model, same
    # vocabulary window, same cap, same scoring, same corpus.
    # The embedder call sits outside `selection_seconds` deliberately: it is local
    # CPU work, identical across arms, and folding it in would credit the stage
    # under test with time no model spent.
    query_vector = embedder.encode_one(search_query)
    selection_seconds = 0.0
    selections, per_platform, platform_outcomes = [], {}, []
    min_score = config.retrieval.min_keyword_score

    for platform, store in stores.items():

        # Both vocabularies, in one value. The cleaned pool is what the model is
        # shown; the raw corpus is what its answers are scored against. Passing
        # only the cleaned dict -- as this line did -- left keywords like `aws`
        # in neither, so they were charged as inventions against a corpus that
        # contains them.
        vocabulary = build_platform_vocabulary(store.courses)

        selection_start = perf_counter()
        selection = select_keywords(
            backend,
            search_query,
            platform,
            vocabulary,
            model=config.llm.model_for("keyword"),
            max_vocabulary=config.retrieval.max_vocabulary,
            max_keywords=config.retrieval.max_keywords,
            params=params_from_config(config.llm),
            schema_enabled=schema_enabled,
            schema_exemplar=schema_exemplar,
            # S2's `platform` enum is every platform loaded, not just this
            # one: a model that attributes its keywords to the wrong platform
            # is making an error the enum should catch, and an enum of one
            # would make that error unrepresentable rather than absent.
            platforms=list(stores),
            sweep_version=sweep_version,
            candidate_injection=candidate_injection,
            candidate_limit=candidate_limit,
            candidate_ranker=candidate_ranker,
            # The same embedder the query vector came from. Passing it
            # unconditionally costs nothing on the lexical default -- which
            # never looks at it -- and means a dense run fails at the ranker
            # for a reason about the ranker, not for a missing argument.
            embedder=embedder,
            # The learner's own query, not the rewrite, when the candidate
            # list is to be held fixed across rewrites (review item A7).
            candidate_query=(query if candidate_source == CANDIDATE_SOURCE_QUERY else None),
        )

        selection_seconds += perf_counter() - selection_start
        schema_violations.extend(selection.schema_violations)
        if selection.schema_violations:
            schema_violation_stages.append(STAGE_SELECTION)
        selections.append(selection)

        keywords = selection.ranked(min_score=min_score)
        buckets = (
            keyword_buckets(query_vector, store, keywords, per_keyword=per_keyword)
            if keywords
            else []
        )
        if buckets:
            per_platform[platform] = buckets

        # Recorded for every platform, including -- especially -- the ones that
        # produced nothing. A platform used to be added to `per_platform` only
        # when it had courses, so dropping out left no trace anywhere: the run
        # reported status ok, and `platform_spread` simply had one fewer key,
        # which is also what a deliberately smaller corpus looks like.
        # The outcome decides first, when it says anything. `not selection.keywords`
        # is true for four unrelated events -- the model selected keywords and none
        # validated, the backend never answered, the reply was cut off, the reply was
        # garbage -- and collapsing them onto REASON_NO_USABLE_KEYWORDS exported a
        # 503 as a model choice in the one column a reader consults to ask why.
        if selection.outcome in REASON_FOR_OUTCOME:
            reason = REASON_FOR_OUTCOME[selection.outcome]
        elif not selection.keywords:
            reason = REASON_NO_USABLE_KEYWORDS
        elif not keywords:
            reason = REASON_BELOW_MIN_SCORE
        elif not buckets:
            reason = REASON_NO_COURSES_MATCHED
        else:
            reason = REASON_OK
        platform_outcomes.append(
            PlatformOutcome(
                platform=platform,
                keywords_selected=len(selection.keywords),
                keywords_used=len(keywords),
                courses_retrieved=len(buckets),
                reason=reason,
                min_keyword_score=min_score,
            )
        )

    # X1: the selection stage's own cost, summed across platforms.
    selection_prompt_tokens = sum_tokens(s.prompt_tokens for s in selections)
    selection_completion_tokens = sum_tokens(s.completion_tokens for s in selections)

    # --- 5. Merge, bound, and generate ----------------------------------
    #
    # The cap goes *after* the merge and never per platform: cutting each
    # platform's list first would favour whichever sorted first, which is the
    # confound the round-robin interleave exists to prevent. At the shipped
    # default it binds on nothing -- see `cap_context` for why that matters.
    retrieved, dropped = cap_context(
        merge_platforms(per_platform), config.retrieval.max_context_courses
    )

    # No platform produced a usable keyword, so there is nothing to ground on.
    # The upstream Algorithm 1 (p. 21439) has no branch here and generates
    # anyway, against an empty "Available courses" block -- which is exactly how
    # invented course IDs appear. Returning a recorded failure keeps that event
    # in the results table instead of converting it into a plausible answer.
    if not retrieved:
        return PipelineResult(
            pipeline="skyrag",
            query=query,
            learning_path="",
            retrieved=[],
            expanded_query=expanded,
            expansion_mode=expansion_mode if expand_query else "",
            principles=principles,
            selections=selections,
            platform_outcomes=platform_outcomes,
            coverage=coverage,
            # Still attributed to the model under test: a failure that cannot be
            # traced to a model is useless in a model comparison. The stage that
            # actually failed is recorded in `selections`.
            model=config.llm.model_for("generation"),
            backend=getattr(backend, "name", ""),
            elapsed_seconds=perf_counter() - start,
            status=STATUS_EMPTY_RETRIEVAL,
            context_courses_cap=config.retrieval.max_context_courses,
            selection_seconds=selection_seconds,
            expansion_seconds=expansion_seconds,
            expansion_prompt_tokens=expansion_prompt_tokens,
            expansion_completion_tokens=expansion_completion_tokens,
            selection_prompt_tokens=selection_prompt_tokens,
            selection_completion_tokens=selection_completion_tokens,
            arm=arm_label,
            intervention=intervention_label,
            schema_violations=len(schema_violations),
            schema_violation_stages=",".join(schema_violation_stages),
            keywords_claimed_copied=sum(
                1 for sel in selections for v in sel.copied_verbatim.values() if v
            ),
            keywords_claimed_recalled=sum(
                1 for sel in selections for v in sel.copied_verbatim.values() if not v
            ),
        )

    context = format_context(retrieved)
    generation_start = perf_counter()
    active_path_schema = (
        LEARNING_PATH_SCHEMA_V4
        if sweep_version in _V4_FAMILY
        else LEARNING_PATH_SCHEMA
    )
    if schema_enabled:
        if sweep_version in _V4_FAMILY:
            gen_prompt = GENERATE_PATH_SCHEMA_V4.format(
                query=query.strip(),
                context=context.strip(),
                principles=(principles.strip() or "Apply standard curriculum-design practice."),
                schema=render_schema(
                    active_path_schema,
                    exemplar_stage=STAGE_PATH if schema_exemplar else None,
                ),
            )
        else:
            gen_prompt = build_generation_prompt(
                query,
                context,
                principles,
                schema=render_schema(
                    active_path_schema,
                    exemplar_stage=STAGE_PATH if schema_exemplar else None,
                ),
            )
    else:
        gen_prompt = build_generation_prompt(query, context, principles)

    response = backend.generate(
        gen_prompt,
        model=config.llm.model_for("generation"),
        params=params_from_config(config.llm),
    )
    learning_path = response.text
    if schema_enabled:
        # S3. The path is validated but NOT rewritten. `learning_path` stays the
        # model's own text in both arms, because it is what the O1 judge scores
        # and what `cited_ids` reads course IDs out of; substituting a
        # re-rendered version would put the harness's formatting between the
        # model and its score.
        #
        # `cited_ids` finds IDs by regex over that text, and a JSON path still
        # carries them as literal strings, so grounding, `cited_count` and M3.2
        # keep working unchanged -- pinned by test_arm3_schemas.py rather than
        # assumed, because a silent break there suppresses a grounding score
        # rather than raising.
        record(parse_and_validate(response.text, active_path_schema, STAGE_PATH))


    # Some platforms contributed and others did not. The run is usable, but it is
    # not the run that was asked for, and the difference has to survive into the
    # results table -- an 8B arm that loses a platform must not be compared
    # against a 70B arm that kept both as though the corpora matched.
    lost = [o for o in platform_outcomes if not o.contributed]
    status = STATUS_PARTIAL_RETRIEVAL if lost else STATUS_OK

    return PipelineResult(
        pipeline="skyrag",
        query=query,
        learning_path=learning_path,
        retrieved=retrieved,
        expanded_query=expanded,
        expansion_mode=expansion_mode if expand_query else "",
        principles=principles,
        selections=selections,
        platform_outcomes=platform_outcomes,
        coverage=coverage,
        model=response.model,
        backend=response.backend,
        elapsed_seconds=perf_counter() - start,
        generation_seconds=perf_counter() - generation_start,
        selection_seconds=selection_seconds,
        expansion_seconds=expansion_seconds,
        expansion_prompt_tokens=expansion_prompt_tokens,
        expansion_completion_tokens=expansion_completion_tokens,
        selection_prompt_tokens=selection_prompt_tokens,
        selection_completion_tokens=selection_completion_tokens,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        status=status,
        context_courses_cap=config.retrieval.max_context_courses,
        context_courses_dropped=dropped,
        context_chars=len(context),
        arm=arm_label,
        intervention=intervention_label,
        schema_violations=len(schema_violations),
        schema_violation_stages=",".join(schema_violation_stages),
        keywords_claimed_copied=sum(
            1 for sel in selections for v in sel.copied_verbatim.values() if v
        ),
        keywords_claimed_recalled=sum(
            1 for sel in selections for v in sel.copied_verbatim.values() if not v
        ),
    )
