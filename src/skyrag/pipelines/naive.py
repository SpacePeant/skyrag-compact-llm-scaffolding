"""Naive RAG baseline (port of Naive_RAG.ipynb).

Single top-k similarity search over the whole corpus, then generate. No
keyword stage. Kept faithful because it is the control condition SKYRAG is
measured against -- improving it would invalidate the comparison.
"""

from __future__ import annotations

from time import perf_counter

from ..config import Config
from ..data.adapters import CorpusCoverage
from ..embedding import Embedder, EmbeddingStore
from ..llm import LLMBackend, params_from_config
from ..prompts import STEP_BACK, build_generation_prompt
from ..retrieval import cap_context, format_context, top_k
from .base import STATUS_EMPTY_RETRIEVAL, PipelineResult, sum_tokens


def run_naive(
    query: str,
    store: EmbeddingStore,
    backend: LLMBackend,
    config: Config,
    *,
    embedder: Embedder | None = None,
    k: int | None = None,
    use_step_back: bool = True,
    coverage: CorpusCoverage | None = None,
) -> PipelineResult:
    start = perf_counter()
    embedder = embedder or Embedder(config.embedding)
    k = k or config.retrieval.top_k

    # Step-back prompting (Naive_RAG.ipynb cell 30), kept as in the original.
    expansion_start = perf_counter()
    expansion_responses = []
    principles = ""
    if use_step_back:
        step_back = backend.generate(
            STEP_BACK.format(query=query),
            model=config.llm.model_for("expansion"),
            params=params_from_config(config.llm, max_tokens=800),
        )
        expansion_responses.append(step_back)
        principles = step_back.text

    expansion_seconds = perf_counter() - expansion_start
    # X1, on the control condition too. The baseline runs one upstream call
    # against SKYRAG's two, and that difference is part of what a cost comparison
    # between the two pipelines is *for* -- so it has to be counted here rather
    # than left blank and read as zero.
    expansion_prompt_tokens = sum_tokens(r.prompt_tokens for r in expansion_responses)
    expansion_completion_tokens = sum_tokens(
        r.completion_tokens for r in expansion_responses
    )

    # The same backstop the SKYRAG pipeline applies, for the same reason it has to
    # be the same: the control condition is measured on the terms the treatment is.
    # At `top_k=20` against a 600-course backstop this cannot bind, and that is the
    # point -- if it ever does bind on one pipeline and not the other, the export
    # says so per row instead of the difference being invisible.
    retrieved, dropped = cap_context(
        top_k(embedder.encode_one(query), store, k=k),
        config.retrieval.max_context_courses,
    )

    # Same guard as the SKYRAG pipeline. The baseline reaches this state by a
    # different route -- an empty store rather than an empty keyword selection
    # -- but the consequence is identical, and the control condition has to be
    # measured on the same terms as the treatment for the comparison to hold.
    if not retrieved:
        return PipelineResult(
            pipeline="naive",
            query=query,
            learning_path="",
            retrieved=[],
            principles=principles,
            coverage=coverage,
            model=config.llm.model_for("generation"),   # attributable, as above
            backend=getattr(backend, "name", ""),
            elapsed_seconds=perf_counter() - start,
            status=STATUS_EMPTY_RETRIEVAL,
            context_courses_cap=config.retrieval.max_context_courses,
            expansion_seconds=expansion_seconds,
            expansion_prompt_tokens=expansion_prompt_tokens,
            expansion_completion_tokens=expansion_completion_tokens,
            # `selection_*` stay None: the baseline has no keyword stage, so it
            # made no selection call and there is no count to report. Zero would
            # claim a stage ran for free.
        )

    context = format_context(retrieved)
    generation_start = perf_counter()
    response = backend.generate(
        build_generation_prompt(query, context, principles),
        model=config.llm.model_for("generation"),
        params=params_from_config(config.llm),
    )

    return PipelineResult(
        pipeline="naive",
        query=query,
        learning_path=response.text,
        retrieved=retrieved,
        principles=principles,
        coverage=coverage,
        model=response.model,
        backend=response.backend,
        elapsed_seconds=perf_counter() - start,
        generation_seconds=perf_counter() - generation_start,
        # `selection_seconds` stays 0.0: the baseline has no keyword stage at all,
        # which is the point of it. Zero here means "did not run", not "ran fast".
        expansion_seconds=expansion_seconds,
        expansion_prompt_tokens=expansion_prompt_tokens,
        expansion_completion_tokens=expansion_completion_tokens,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        context_courses_cap=config.retrieval.max_context_courses,
        context_courses_dropped=dropped,
        context_chars=len(context),
    )
