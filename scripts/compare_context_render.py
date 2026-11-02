"""Did trimming the generation context cost quality? Measure it, do not assume.

Cross-reference:
- project_knowledge/task_log/backlog_2026-08-18_bugfix_live-run-context-overflow.md

    python scripts/compare_context_render.py                       # free, offline
    python scripts/compare_context_render.py --live --openrouter   # SPENDS CREDIT

`format_context` now sends `Course.render_for_context()` -- ID, title, platform,
~400 characters of learning outcomes and five keywords -- instead of the full
`Course.render()`. That is what stops a 560-course prompt from reaching 441,000
tokens against a 131,072-token endpoint limit. It also removes about 89% of the
bytes the generation model used to see, and "the prompt got smaller" is not on
its own evidence that the answer stayed as good.

The comparison is controlled where it matters: **one** retrieval, so both arms are
handed the identical course list in the identical order, with the identical
step-back principles and the identical query. The only thing that varies is how
each course is rendered into the prompt. Anything that differs downstream is
attributable to the render and to nothing else.

`per_keyword` is deliberately small. The full render can only be measured at all
below roughly 138 courses -- past that it is the failure under repair rather than
a comparison -- so the baseline arm has to run inside a configuration where it
still works.

Reported per arm: prompt size, how many real course IDs were cited, how many were
invented, grounding accuracy, and the overlap between the two arms' recommended
sets. Overlap is the interesting number: two different-looking learning paths that
recommend the same courses are the same answer, and a trim that changed the prose
without changing the recommendation cost nothing that this experiment measures.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skyrag.config import Config, load_dotenv_if_present  # noqa: E402
from skyrag.data.schema import Course  # noqa: E402
from skyrag.embedding import Embedder, EmbeddingStore  # noqa: E402
from skyrag.llm import EchoBackend, get_backend, params_from_config  # noqa: E402
from skyrag.pipelines.base import PipelineResult  # noqa: E402
from skyrag.prompts import build_generation_prompt  # noqa: E402
from skyrag.retrieval import RetrievedCourse  # noqa: E402
from skyrag.pipelines.skyrag import run_skyrag  # noqa: E402

DEFAULT_QUERIES = [
    "I want to learn data science in 6 months",
    "Machine learning and deep learning path with Python",
]


def render_full(results: list[RetrievedCourse]) -> str:
    """The pre-fix context block, reproduced here rather than left in the codebase.

    Kept as a local function on purpose: `format_context` must have exactly one
    behaviour in `src/`, and a `full=True` switch on it is a second behaviour that
    a caller could reach by accident. This is a measurement instrument, so it lives
    with the measurement.
    """
    return "\n\n".join(
        f"Course ID: {r.course.id_}\n{r.course.render()}" for r in results
    )


def render_trimmed(results: list[RetrievedCourse]) -> str:
    return "\n\n".join(r.course.render_for_context() for r in results)


def renders_for(*, control: bool, control_only: bool) -> list:
    """Which arms run, as data rather than as branching buried in a loop.

    `control_only` drops the full render. That arm is the one that overflows
    context, and it is the only reason `per_keyword` had to be 1 -- so dropping it
    is what lets the reproducibility floor be measured at the shipped 7, which is
    the regime every reportable number actually lives in.
    """
    if control_only:
        return [("render_for_context()", render_trimmed),
                ("render_for_context() again", render_trimmed)]
    renders = [("full render()", render_full),
               ("render_for_context()", render_trimmed)]
    if control:
        renders.append(("render_for_context() again", render_trimmed))
    return renders


def control_pair(arms: list) -> tuple:
    """The control is always the LAST arm against the one before it.

    Indexing from the end rather than from position 2 is what keeps this correct
    in both modes: under `control_only` the pair is arms 0 and 1, otherwise 1 and
    2. Hard-coding 2 would silently compare the full render against the trimmed
    one and print the result under the word CONTROL -- an effect relabelled as a
    floor, which is the exact confusion the control exists to prevent.
    """
    return arms[-2], arms[-1]


@dataclass
class ArmResult:
    label: str
    prompt_chars: int
    prompt_tokens: int | None
    completion_tokens: int | None
    text: str
    cited: list[str]
    hallucinated: list[str]

    @property
    def grounding(self) -> float | None:
        if not self.cited:
            return None
        return round((len(self.cited) - len(self.hallucinated)) / len(self.cited), 4)


def _score(label: str, prompt: str, response, retrieved: list[RetrievedCourse]) -> ArmResult:
    """Reuse `PipelineResult`'s own citation logic rather than re-implementing it.

    Re-implementing the ID regex here would measure this script's idea of a
    citation instead of the harness's, and the two drifting apart is precisely how
    an earlier defect in this project stayed invisible.
    """
    probe = PipelineResult(
        pipeline="skyrag", query="", learning_path=response.text, retrieved=retrieved
    )
    return ArmResult(
        label=label,
        prompt_chars=len(prompt),
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        text=response.text,
        cited=probe.cited_ids(),
        hallucinated=probe.hallucinated_ids(),
    )


def compare_one(query: str, cfg: Config, backend, stores, embedder,
                *, control: bool = False, control_only: bool = False) -> tuple:
    """One retrieval, two renders, two generation calls -- three with the control.

    Under `control_only` it is one render sent twice and nothing else: the
    full-render arm is what forces `per_keyword` down, so dropping it is the only
    way to measure the reproducibility floor at a realistic retrieval size.
    """
    retrieval = run_skyrag(
        query=query, stores=stores, backend=backend, config=cfg, embedder=embedder
    )
    if not retrieval.retrieved:
        return retrieval, []

    params = params_from_config(cfg.llm)
    model = cfg.llm.model_for("generation")
    arms = []
    # The third entry, when asked for, is the control: the *same* prompt sent
    # twice. Without it an overlap below 100% cannot be attributed to the render at
    # all. Temperature is 0.0 and the seed is fixed, but
    # `backlog_2026-08-18_events_seed-determinism-and-repeat-semantics.md` records
    # that seeds are honoured best-effort through a hosted API and that routing is
    # free unless pinned -- so the harness's own reproducibility is an empirical
    # question, and reading a difference as an effect before measuring the floor is
    # how a routing artifact becomes a finding.
    renders = renders_for(control=control, control_only=control_only)
    for label, renderer in renders:
        prompt = build_generation_prompt(query, renderer(retrieval.retrieved),
                                         retrieval.principles)
        response = backend.generate(prompt, model=model, params=params)
        arms.append(_score(label, prompt, response, retrieval.retrieved))

    return retrieval, arms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--openrouter", action="store_true")
    parser.add_argument("--per-keyword", type=int, default=1,
                        help="Small on purpose. The baseline arm is the render that "
                             "overflows, so it has to be run where it still fits: "
                             "past ~138 courses there is no comparison to make. At 1 "
                             "the ceiling is 40 x 1 x 2 = 80 courses (~64k tokens at "
                             "the full render); at 2 it is 160, which exceeds the "
                             "budget and would measure the defect instead.")
    parser.add_argument("--control", action="store_true",
                        help="Also send the trimmed prompt a second time. The "
                             "self-overlap of that pair is the floor any "
                             "between-render overlap must be read against.")
    parser.add_argument("--control-only", action="store_true",
                        help="Drop the full-render arm and measure ONLY the "
                             "trimmed prompt sent twice. This is what makes the "
                             "control runnable at the shipped per_keyword=7: the "
                             "full render is the arm that overflows context, so "
                             "while it is present the whole comparison is pinned "
                             "to a small retrieval and the reproducibility floor "
                             "is only ever measured at 1.5k-3k tokens -- not the "
                             "5.5k-21.5k the reportable sweeps actually run at. "
                             "Implies --control. No between-render overlap is "
                             "reported, because there is only one render.")
    parser.add_argument("--queries", nargs="*", default=DEFAULT_QUERIES)
    args = parser.parse_args()
    # --control-only IS a control run; requiring both flags would let a run that
    # measures nothing else silently measure nothing at all.
    if args.control_only:
        args.control = True

    load_dotenv_if_present()
    cfg = Config.load()
    cfg.retrieval.per_keyword = args.per_keyword

    if args.live and args.openrouter:
        cfg.llm.backend = "openrouter"
        backend = get_backend(cfg.llm, backend="openrouter")
        print("  backend  : openrouter -- THIS SPENDS CREDIT")
    else:
        backend = EchoBackend(responses=['{"python": 0.9, "data science": 0.9}'] * 64)
        print("  backend  : echo (free). Prompt sizes are real; the replies are not,")
        print("             so the citation columns below are not a quality signal.")

    for role in ("expansion", "keyword", "generation"):
        print(f"      {role:<11}: {cfg.llm.model_for(role)}")
    # Stamp the run's own provenance into its only artifact. Every determinism
    # result this script has produced was evidenced by `.env` plus a separate
    # `Config.load()` check made at the same time -- which is one step short of
    # an artifact that proves what it ran under. `skyrag eval` puts provider on
    # every row; this had nothing.
    provider = cfg.llm.provider or None
    print(f"  provider : {provider!r}"
          + ("   <-- UNPINNED: routing is free and this run is not comparable "
             "to a pinned one" if not provider else ""))
    print(f"  temperature : {cfg.llm.temperature}   seed : {cfg.llm.seed!r}   "
          f"max_tokens : {cfg.llm.max_tokens}")
    print(f"  per_keyword : {cfg.retrieval.per_keyword}")
    if args.control_only:
        print("  mode     : CONTROL ONLY -- one render, sent twice. No "
              "between-render comparison.")
    print("  embedder : loading...", flush=True)

    embedder = Embedder(cfg.embedding)
    stores = {
        p: EmbeddingStore.load(cfg.paths.embeddings, p,
                               expect_model=cfg.embedding.model_name)
        for p in cfg.retrieval.platforms
    }
    print(f"  corpus   : {sum(len(s.courses) for s in stores.values())} courses "
          f"over {len(stores)} platforms\n")

    overlaps, controls = [], []
    for query in args.queries:
        print("=" * 78)
        print(f"  {query}")
        print("=" * 78)
        retrieval, arms = compare_one(query, cfg, backend, stores, embedder,
                                      control=args.control,
                                      control_only=args.control_only)
        if not arms:
            print(f"  no courses retrieved ({retrieval.status}) -- nothing to compare\n")
            continue

        print(f"  retrieved : {len(retrieval.retrieved)} courses "
              f"(identical list for every arm)\n")
        print(f"  {'arm':<28}{'prompt':>12}{'tok in':>9}{'tok out':>9}"
              f"{'cited':>7}{'invented':>10}{'grounding':>11}")
        for arm in arms:
            grounding = "n/a" if arm.grounding is None else f"{arm.grounding:.3f}"
            print(f"  {arm.label:<28}{arm.prompt_chars:>11,}c"
                  f"{arm.prompt_tokens if arm.prompt_tokens is not None else '?':>9}"
                  f"{arm.completion_tokens if arm.completion_tokens is not None else '?':>9}"
                  f"{len(arm.cited):>7}{len(arm.hallucinated):>10}{grounding:>11}")

        print()
        # Under --control-only there is one render, so there is no between-render
        # comparison to make. Reporting one would compare the trimmed arm against
        # itself -- the control floor, relabelled as an effect.
        if not args.control_only:
            full, trimmed = arms[0], arms[1]
            shared = set(full.cited) & set(trimmed.cited)
            union = set(full.cited) | set(trimmed.cited)
            overlap = len(shared) / len(union) if union else None
            if overlap is not None:
                overlaps.append(overlap)
            print(f"  recommendation overlap : {len(shared)} of {len(union)} distinct "
                  f"course IDs cited by both"
                  + (f"  ({overlap:.0%})" if overlap is not None else ""))
            only_full = [c for c in full.cited if c not in shared]
            only_trim = [c for c in trimmed.cited if c not in shared]
            if only_full:
                print(f"  only the full render cited  : {', '.join(only_full)}")
            if only_trim:
                print(f"  only the trimmed render cited: {', '.join(only_trim)}")
            print(f"  prompt reduction       : "
                  f"{100 * (1 - trimmed.prompt_chars / full.prompt_chars):.1f}%")

        if len(arms) > 1 and (args.control or args.control_only):
            first, repeat = control_pair(arms)
            shared_c = set(first.cited) & set(repeat.cited)
            union_c = set(first.cited) | set(repeat.cited)
            self_overlap = len(shared_c) / len(union_c) if union_c else None
            if self_overlap is not None:
                controls.append(self_overlap)
            print(f"  CONTROL, same prompt twice : {len(shared_c)} of {len(union_c)}"
                  + (f"  ({self_overlap:.0%})" if self_overlap is not None else ""))
            print("    Read the between-render overlap against this floor, not")
            print("    against 100%: whatever the control fails to reproduce was")
            print("    never the render's doing.")
        print()

    if overlaps:
        print("=" * 78)
        print(f"  mean recommendation overlap across {len(overlaps)} queries: "
              f"{sum(overlaps) / len(overlaps):.0%}")
        if controls:
            print(f"  mean self-overlap, SAME prompt sent twice          : "
                  f"{sum(controls) / len(controls):.0%}")
            print("  The second number is this harness's reproducibility floor at")
            print("  temperature 0. The first is only interpretable above it.")
        print("=" * 78)
        print("  Read this as the quality check it is: the arms saw the same courses,")
        print("  so a high overlap means the trim did not change which courses the")
        print("  model recommends. It says nothing about the prose, and nothing about")
        print("  queries not run here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
