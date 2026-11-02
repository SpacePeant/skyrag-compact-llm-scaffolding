"""SkyRag command-line interface: `skyrag eval` and `skyrag show`."""

import json
import os
import sys
from pathlib import Path
from typing import Sequence

_KNOWN_EMBEDDER_DIMS = {
    "all-mpnet-base-v2": 768,
    "all-MiniLM-L6-v2": 384,
    "multilingual-e5-base": 768,
    "BAAI/bge-m3": 1024,
}


def _configured_dim(config, model_name: str) -> int:
    """Vector width to use when no real cache was loaded.

    Only reached when every platform fell back to a placeholder store, so nothing
    real constrains the width and any self-consistent choice works. Keyed off the
    configured embedder anyway, so a later run that *does* load a cache finds the
    placeholder already in the right space instead of raising.

    An earlier version consulted `config.embedding.dimensions` first. `EmbeddingConfig`
    declares `model_name`, `device` and `batch_size` and no such field, so that branch
    could never be taken and the override it appeared to offer did not exist. A dead
    configuration path reads as coverage that is not there, so it is gone rather than
    backed by a new field nobody asked for.
    """
    short = str(model_name).rsplit("/", 1)[-1]
    return _KNOWN_EMBEDDER_DIMS.get(model_name) or _KNOWN_EMBEDDER_DIMS.get(short, 768)


def load_query_set(path: Path) -> tuple[list[str], list[str]]:
    """Read a benchmark query set, returning (queries, ids).

    Three accepted shapes, because the sets people already have are text files and
    the sets that carry a stable id are JSON:

      * `.txt`  -- one query per line; blank lines and `#` comments skipped. Ids are
                   assigned positionally as `q01`, `q02`, ...
      * `.jsonl`-- one JSON object per line, `{"id": ..., "query": ...}`
      * `.json` -- a list of such objects, or a plain list of strings

    Ids exist so the same question can be paired across arms. Matching on query
    text works only until someone edits the wording of one line, at which point the
    pairing silently breaks for that query and for no other -- the kind of failure
    this repository has now recorded three times under a different name.
    """
    import json

    if not path.exists():
        raise SystemExit(f"skyrag eval: query set not found: {path}")

    raw = path.read_text(encoding="utf-8")
    queries: list[str] = []
    ids: list[str] = []

    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        rows = json.loads(raw)
    else:
        rows = [
            line.strip()
            for line in raw.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    for i, row in enumerate(rows):
        if isinstance(row, str):
            queries.append(row)
            ids.append(f"q{i + 1:02d}")
            continue
        if not isinstance(row, dict) or not row.get("query"):
            raise SystemExit(
                f"skyrag eval: entry {i + 1} of {path} has no 'query' field: {row!r}"
            )
        queries.append(str(row["query"]))
        ids.append(str(row.get("id") or f"q{i + 1:02d}"))

    if not queries:
        raise SystemExit(f"skyrag eval: query set is empty: {path}")

    duplicates = {q for q in ids if ids.count(q) > 1}
    if duplicates:
        raise SystemExit(
            f"skyrag eval: duplicate query ids in {path}: {sorted(duplicates)}. "
            "Ids have to be unique or arms cannot be paired by query."
        )

    return queries, ids


# The **model axis** of the experiment this harness runs: one 70B, one 8B,
# same family, same provider pin. It separates Arm 1 from Arm 2 and nothing
# else -- Arm 3 shares Arm 2's model and differs by intervention, which
# is an axis this CLI does not yet have (B0). See T1 in `pipelines/base.py`
# for why `arm` is reserved for that meaning and grid units are `cell`s.
#
# The previous default named `meta-llama/llama-3.3-70b-instruct` and
# `mistralai/mistral-7b-instruct` and omitted the 8B entirely. Mistral-7B has
# **zero** providers on OpenRouter, so every cell carrying it fails -- and after
# the 2026-08-17 cell-guard work it fails as a *row*, not as a crash: an operator
# who forgot `-m` would get a complete-looking CSV in which one third of the grid
# is `cell_failed` for a reason that has nothing to do with any model's ability.
# A default that cannot run is not a default.


DEFAULT_EVAL_MODELS = (
    "meta-llama/llama-3.1-70b-instruct,meta-llama/llama-3.1-8b-instruct"
)


def _print_cell(event: dict) -> None:
    """Print one line per cell while a sweep is running.

    Without this a paid sweep is minutes of silence: the results table is built
    after the last cell returns, so a run that is producing twenty `cell_failed`
    rows looks exactly like a run that is going well, for as long as it takes to
    spend the money. That is the failure mode the 2026-08-18 sweep had -- both
    70B cells were rejected by the endpoint and nothing said so until the end.

    Every number printed here is read off the row that was just appended. None of
    it is recomputed, so this cannot disagree with the exported CSV.
    """
    head = (
        f"  [{event['index']:>3}/{event['total']}] "
        f"r{event['repeat']} {event['query_id']} "
        f"{event['model']} {event['pipeline']}"
    )
    kind = event.get("event")

    if kind == "start":
        # No newline: the outcome completes this line when the cell returns. A
        # cell in flight is therefore visibly distinct from one that finished,
        # which is the whole point of announcing it before the call.
        print(f"{head} ...", end="", flush=True)
        return

    if kind == "failed":
        print(f"  {event.get('error', 'failed')}", flush=True)
        return

    print(
        "  {status}  retrieved={retrieved} cited={cited}"
        "  sel={sel:.1f}s gen={gen:.1f}s total={total:.1f}s".format(
            status=event.get("status", "?"),
            retrieved=event.get("retrieved", 0),
            cited=event.get("cited", 0),
            sel=event.get("selection_seconds") or 0.0,
            gen=event.get("generation_seconds") or 0.0,
            total=event.get("elapsed_seconds") or 0.0,
        ),
        flush=True,
    )


def _pct(value: float | None) -> str:
    """A rate as a percentage, or 'n/a' when the denominator was zero.

    'n/a' rather than '0.0%': a run that emitted no keywords has an undefined
    invalid-keyword rate, and printing zero would read as a model that emitted
    many and invented none.
    """
    return "n/a" if value is None else f"{value * 100:.1f}%"


def parse_arms(raw: str) -> list[str]:
    """Expand an --arm value into a sorted list of individual arm strings.

    Accepts:
      "1"     → ["1"]
      "2,3"   → ["2", "3"]
      "all"   → ["1", "2", "3"]

    Raises SystemExit with a clear message for any unrecognised value. The
    valid set is hardcoded here (mirroring `VALID_ARMS` in eval.py) so the
    parser can run without importing the full evaluation stack.
    """
    _valid = {"1", "2", "3"}
    raw = raw.strip().lower()
    if raw == "all":
        return ["1", "2", "3"]
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    unknown = [p for p in parts if p not in _valid]
    if unknown or not parts:
        raise SystemExit(
            f"skyrag eval: --arm got unknown value(s): {unknown or [raw]!r}. "
            "Valid choices: 1, 2, 3, a comma-separated list, or 'all'."
        )
    # Deduplicate and sort so the sweep order is deterministic regardless of
    # how the user typed the list.
    return sorted(set(parts))


def arm_cell_models(arms: list[str], model_override: str | None) -> list[str]:
    """The model id each arm-cell will actually run, in executor order.

    Mirrors `run_multi_model_eval`'s arm branch exactly: without `-m` every arm
    supplies its own model (70B on arm 1, 8B on arms 2-3), and with `-m` the
    executor builds the **models x arms cross-product**, one ArmConfig per pair.

    The banner and the cell count are derived from this rather than from the
    `--models` default, which arm mode never reads. Both lines were previously
    computed from that default, which made them disagree with the run in two
    ways: the banner named a model that was never called, and -- because the
    count multiplied arms but not the `-m` cross-product -- an announced "100
    cells" could bill 200. A spend estimate that under-reports is the more
    expensive of the two, so the count is the reason this helper exists and the
    banner is the reason it returns ids rather than a bare length.
    """
    from skyrag.pipelines import resolve_arm

    if model_override:
        overrides = [m.strip() for m in model_override.split(",") if m.strip()]
        return [resolve_arm(a, model=m).model for m in overrides for a in arms]
    return [resolve_arm(a).model for a in arms]


def _canonical_model_slug(
    model: str | None,
    *,
    arms: Sequence[str] | None = None,
    arm: str | None = None,
) -> str:
    """Derive a canonical short identifier for models in directory naming."""
    if not model:
        if arms:
            if sorted(arms) == ["1"]:
                return "llama70b"
            if all(a in ("2", "3") for a in arms):
                return "llama8b"
            if "1" in arms:
                return "llama70b-8b"
        elif arm:
            if arm == "1":
                return "llama70b"
            return "llama8b"
        return "llama70b-8b"

    models = [m.strip() for m in model.split(",") if m.strip()]
    slugs = []
    for m in models:
        m_low = m.lower()
        if any(x in m_low for x in ("llama-3.1-8b", "llama3.1-8b", "llama-8b", "llama8b")):
            slugs.append("llama8b")
        elif any(x in m_low for x in ("llama-3.1-70b", "llama3.1-70b", "llama-70b", "llama70b")):
            slugs.append("llama70b")
        elif "qwen" in m_low and ("7b" in m_low or "2.5-7b" in m_low):
            slugs.append("qwen7b")
        elif "echo" in m_low:
            slugs.append("echo")
        else:
            name = m.split("/")[-1].replace("-instruct", "").replace("_instruct", "")
            slugs.append(name.replace(".", "-"))
    return "-".join(slugs) if slugs else "model"


def resolve_benchmark_output_dir(
    output_dir_arg: str | None,
    *,
    arms: Sequence[str] | None = None,
    arm: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    sweep_version: str | None = None,
    query_count: int = 1,
    run_type: str | None = None,
) -> Path:
    """Resolve destination directory for benchmark evaluation results.

    If an explicit custom directory is passed (and is not None, 'outputs', or 'auto'),
    it is honored directly. Otherwise, constructs a standardized semantic path:
      `outputs/<type>_<version>_<YYYYMMDD>_<arms>_<model-slug>_<N>q_[provider]`
    """
    from datetime import datetime
    from pathlib import Path

    from skyrag.pipelines.eval import SWEEP_VERSION_AUTO_LABEL

    if output_dir_arg and output_dir_arg.rstrip("/") not in ("outputs", "auto", ""):
        p = Path(output_dir_arg)
        p.mkdir(parents=True, exist_ok=True)
        return p

    date_stamp = datetime.now().strftime("%Y%m%d")

    if not run_type:
        if query_count == 1:
            run_type = "smoke"
        elif query_count == 10:
            run_type = "pilot"
        elif (arms and len(arms) > 1) or query_count > 10:
            run_type = "sweep"
        else:
            run_type = "eval"

    if arms:
        arms_slug = f"arm{'-'.join(sorted(arms))}"
    elif arm:
        arms_slug = f"arm{arm}"
    else:
        arms_slug = "arm1-2"

    model_slug = _canonical_model_slug(model, arms=arms, arm=arm)

    parts = [run_type, sweep_version or SWEEP_VERSION_AUTO_LABEL, date_stamp, arms_slug, model_slug, f"{query_count}q"]

    if provider:
        p_clean = provider.strip().lower()
        if p_clean and p_clean not in ("automatic", "none", "auto"):
            parts.append(p_clean.replace("_", "-"))

    dir_name = "_".join(parts)
    resolved = Path("outputs") / dir_name
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


# The per-cell results table, grouped by what each number answers. Printed as
# one wide pandas row it ran to ~400 columns, wrapped in any terminal, and
# glued adjacent values together (`0.45710.1407` was two columns).
RESULT_SUMMARY_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Timing (s)", ("elapsed_seconds", "generation_seconds")),
    ("Grounding", (
        "grounding_accuracy", "grounding_accuracy_n", "cited_count",
        "hallucinated_count", "citation_coverage", "cited_redundancy",
    )),
    ("Retrieval", ("retrieval_score_mean", "retrieval_score_min")),
    ("Keywords", (
        "keywords_emitted", "keywords_kept", "distinct_keywords",
        "selection_rank_mean", "hallucinated_keywords", "near_miss_keywords",
        "out_of_window_keywords", "removed_by_cleaning_keywords",
    )),
    # Present so a row's stated status has a visible cause in the same table.
    ("Selection failures", (
        "selection_errors", "selection_unparseable", "selection_overrun",
        "selection_truncated",
    )),
)

# Above this many cells, one column per cell stops fitting a terminal; the CSV
# is the place to read a sweep cell by cell, and "Run integrity" aggregates it.
RESULT_SUMMARY_MAX_COLUMNS = 4


def _summary_value(column: str, value) -> str:
    # Formatted by what the column is, not by the value: a grounding score of
    # 1.0 printed as `1` reads as a count, and pandas stores counts as floats
    # whenever any cell in the column is missing.
    if value is None or (isinstance(value, float) and value != value):
        return "n/a"
    if column.endswith("_seconds"):
        return f"{float(value):.2f}"
    if column in ("grounding_accuracy", "citation_coverage", "cited_redundancy",
                  "retrieval_score_mean", "retrieval_score_min"):
        return f"{float(value):.4f}"
    if column == "selection_rank_mean":
        return f"{float(value):.1f}"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def format_results_summary(df) -> str:
    """Render the per-cell results as metric rows x cell columns.

    Each column is headed by what identifies a cell (repeat, query, arm, model,
    pipeline) and its status, so a value is never read without knowing which
    cell and whether that cell ran. Sweeps larger than
    `RESULT_SUMMARY_MAX_COLUMNS` get a one-line pointer instead of a table
    nobody can read.
    """
    records = df.to_dict("records")
    if len(records) > RESULT_SUMMARY_MAX_COLUMNS:
        return (
            f"  {len(records)} cells -- too many to lay out side by side. Read them "
            f"per cell in the exported CSV; aggregates are under 'Run integrity'."
        )

    def cell_header(rec: dict) -> list[str]:
        repeat = rec.get("repeat")
        where = " ".join(
            part for part in (
                f"r{repeat}" if repeat not in (None, "") else "",
                str(rec.get("query_id") or ""),
                f"arm {rec['arm']}" if rec.get("arm") not in (None, "") else "",
            ) if part
        )
        return [where, str(rec.get("model", "")).split("/")[-1],
                f"{rec.get('pipeline', '')} -> {rec.get('status', '')}"]

    label_width = max(len(c) for _, cols in RESULT_SUMMARY_GROUPS for c in cols) + 4
    headers = [cell_header(r) for r in records]
    columns = [
        [_summary_value(c, r.get(c)) for _, cols in RESULT_SUMMARY_GROUPS for c in cols]
        for r in records
    ]
    widths = [max(len(s) for s in h + col) for h, col in zip(headers, columns)]

    lines = []
    for i in range(3):
        lines.append(" " * label_width + "  ".join(
            h[i].rjust(w) for h, w in zip(headers, widths)).rstrip())
    k = 0
    for title, cols in RESULT_SUMMARY_GROUPS:
        lines.append(f"  {title}")
        for c in cols:
            lines.append(f"    {c}".ljust(label_width) + "  ".join(
                col[k].rjust(w) for col, w in zip(columns, widths)))
            k += 1
    return "\n".join(lines)


def run_eval_cli(args: list[str]) -> None:

    """Execute head-to-head evaluation across models and export empirical metrics."""
    import argparse
    from pathlib import Path

    import numpy as np

    from skyrag.config import Config
    from skyrag.data.adapters import CorpusCoverage
    from skyrag.data.schema import Course
    from skyrag.embedding import EmbeddingStore, inspect_cache
    from skyrag.llm import get_backend
    from skyrag.pipelines.eval import (
        SWEEP_VERSION_AUTO_LABEL,
        SWEEP_VERSION_SELECTION_DEFAULT,
    )
    from skyrag.selection import (
        CANDIDATE_SOURCE_REWRITE,
        CANDIDATE_SOURCES,
        RANKER_LEXICAL,
        RANKER_STRATEGIES,
    )
    from skyrag.pipelines import (
        HISTORICAL_BENCHMARK_QUERIES,
        RunProvenance,
        SweepCircuitBreakerError,
        export_benchmark_csv,
        export_benchmark_json,
        metrics_to_dataframe,
        run_multi_model_eval,
        summarize_metrics,
        cell_key,
        completed_cells,
        metric_from_row,
    )

    parser = argparse.ArgumentParser(
        prog="skyrag eval",
        description="Run empirical multi-model benchmark evaluation across historical datasets.",
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        default=None,
        help="Custom query to evaluate. If omitted, uses default historical benchmark queries.",
    )
    parser.add_argument(
        "-m", "--models", "--model",
        type=str,
        default=None,
        help=(
            "Comma-separated model IDs to evaluate. When --arm is given, overrides "
            "the arm's default model IDs. When --arm is omitted, defaults to the "
            f"scale comparison ({DEFAULT_EVAL_MODELS})."
        ),
    )
    parser.add_argument(
        "-b", "--backend",
        type=str,
        default="echo",
        help="LLM backend to use (openrouter | echo | ollama). Default: echo (mock execution).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for benchmark reports. Default: outputs/",
    )
    parser.add_argument(
        "-Q", "--queries",
        type=str,
        default=None,
        help=(
            "Path to a query set: .txt (one query per line, '#' comments allowed) "
            "or .json/.jsonl (objects with 'query' and optional 'id'). Overrides "
            "--query. The benchmark specification asks for 50+ queries; -q takes one."
        ),
    )
    parser.add_argument(
        "-r", "--repeats",
        type=int,
        default=1,
        help=(
            "Run the whole grid this many times. One run per cell gives a number "
            "with no measurable run-to-run spread. Default: 1."
        ),
    )
    parser.add_argument(
        "--seed-per-repeat",
        action="store_true",
        help=(
            "Walk the seed with each pass (base, base+1, ...), the same sequence for "
            "every cell. Only useful above temperature 0; at temperature 0 decoding is "
            "greedy and the seed has almost nothing to steer."
        ),
    )
    parser.add_argument(
        "-P", "--pipelines",
        type=str,
        default="skyrag,naive",
        help=(
            "Comma-separated pipelines to run (skyrag | naive). Default: both. "
            "Per D3 every arm runs on `skyrag`, so an arm sweep should pass "
            "--pipelines skyrag rather than pay for `naive` rows it will drop."
        ),
    )
    parser.add_argument(
        "--keyword-model",
        type=str,
        default=None,
        help=(
            "Pin keyword selection to this model for every cell, instead of using "
            "the cell's own model."
        ),
    )
    parser.add_argument(
        "--expansion-model",
        type=str,
        default=None,
        help="Pin query expansion and step-back to this model for every cell.",
    )
    parser.add_argument(
        "--generation-model",
        type=str,
        default=None,
        help=(
            "Pin final generation to this model for every cell. Required by the "
            "condition grid, which varies only keyword selection between cells."
        ),
    )
    parser.add_argument(
        "--arm",
        type=str,
        default=None,
        help=(
            "Arm(s) to run: 1, 2, 3, a comma-separated list (e.g. '1,2'), "
            "or 'all'. Each arm self-configures its model IDs and intervention — "
            "arm 1 uses 70B on all roles, arms 2 and 3 use 8B. When given, --arm "
            "overrides --models and forces --pipelines to skyrag (D3). "
            "Omit to fall back to the --models x --pipelines grid."
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to a prior benchmark_eval.json (or .csv). Cells already carrying a "
            "measurement are skipped and their rows are carried into the new export; "
            "only cells that failed (cell_failed/arm_failed) or are absent are run. "
            "`partial_retrieval` and `empty_retrieval` are results, not failures, and "
            "are NOT re-run -- re-running until an empty set stops being empty selects "
            "on the outcome. Use for R1b-style top-ups and to restart a long sweep."
        ),
    )
    parser.add_argument(
        "--intervention",
        type=str,
        default=None,
        help="Intervention name (e.g. 'zero-shot', 'schemas').",
    )
    parser.add_argument(
        "--schema-exemplar",
        action="store_true",
        help=(
            "Arm 3 prompt variant. Without this flag the schema prompt shows the "
            "JSON Schema alone ('spec_only', B1's original). With it, one valid "
            "example instance is appended ('spec_plus_exemplar'). This is a LEVEL "
            "OF THE TREATMENT, not a fix. Both levels are run and the variant is "
            "stamped per row; never compare Arm 3 numbers across variants without "
            "naming which. Measured on the completed 50-query spec-only run "
            "(2026-08-31): 24 of 50 queries emitted zero keywords, and the "
            "dominant mechanism is REPLY OVERRUN, not schema echo -- 23 of the 24 "
            "mutes carry `reply_overran_cap`, and 10 of them have no schema "
            "violation of any kind. The schema makes the model enumerate further, "
            "that exhausts the completion budget, and a cut-off reply is discarded "
            "whole, losing the platform. (An earlier revision of this help text "
            "attributed the mutes to the 8B echoing the schema back; the completed "
            "run does not support that reading and it was withdrawn.)"
        ),
    )
    parser.add_argument(
        "--expansion-mode",
        choices=["plain", "schema", "schema_wording"],
        default=None,
        help=(
            "Override the Stage 1 query rewrite only; every later stage still "
            "follows the arm. Default: the arm's own rewrite (schema on Arm 3, "
            "plain otherwise). 'plain' is Arm 2's prompt, 'schema' is "
            "Arm 3's validated JSON prompt, 'schema_wording' is the V4 schema "
            "prompt's instructions answered as plain text with no schema "
            "(V4-family sweep versions only). Exists to separate the rewrite's "
            "share of the schema arm's effect from the selection schema's, and "
            "the schema from its wording. Stamped per row as `expansion_mode`."
        ),
    )
    parser.add_argument(
        "--allow-synthetic-corpus",
        action="store_true",
        help=(
            "Permit the run to proceed when a platform has no usable embedding "
            "cache and falls back to a 1-course placeholder store. WITHOUT this "
            "flag such a run is REFUSED (B24). Placeholder rows are not "
            "measurements: every course is scored against a constant vector, so "
            "ranking is arbitrary by construction and the vocabulary is "
            "~3 keywords. Use only for plumbing smoke tests, never for results."
        ),
    )
    parser.add_argument(
        "--allow-degraded-embedder",
        action="store_true",
        help=(
            "Permit the run to proceed when the sentence embedder cannot be "
            "loaded and query vectors fall back to a constant. WITHOUT this flag "
            "such a run is REFUSED (C14). A constant query vector scores every "
            "course identically, so course ranking is arbitrary by construction "
            "while retrieval, generation and every exported metric still look "
            "normal. Rows from an allowed run are stamped embedder_degraded=True. "
            "Use only for plumbing smoke tests, never for results."
        ),
    )
    parser.add_argument(
        "--sweep-version",
        type=str,
        choices=["v1", "v2", "v3", "v4", "v5", "auto"],
        default=None,
        help="Prompt and payload version for the schema stages (Arm 3, and any Stage 1 schema rewrite). v1, v2 and v3 use the V1 prompts and the array keyword payload; v4 uses the V4 prompts and the array payload; v5 uses the V4 prompts and the flat-map keyword payload. The paper's runs used v2 and v5. Default: auto, which resolves to v5.",
    )
    parser.add_argument(
        "--serving-precision",
        type=str,
        choices=["bf16", "fp16", "fp8", "q4", "q8"],
        # Defaulted from the environment so the CLI and the chat page read ONE
        # channel. The UI stamps the same variable onto a chat answer; if only
        # the UI honoured it, an operator who exported it would get a labelled
        # chat answer and an UNSTAMPED sweep row from the same shell -- two
        # provenance answers for one declared fact. An explicit flag still wins.
        default=os.environ.get("SKYRAG_SERVING_PRECISION", ""),
        help=(
            "Serving precision the weights run at, stamped onto every exported row. "
            "Operator-declared: hosted APIs do not report their dtype, so nothing can "
            "verify this and an unset value exports as empty rather than as an assumed "
            "default. Declare it whenever the provider is pinned -- precision moved "
            "invention by +0.194 between the fp8 and bf16 cells, so an unstamped arm "
            "cannot be compared against a stamped one without an inference the "
            "artifact does not contain."
        ),
    )

    parser.add_argument(
        "--candidate-injection",
        action="store_true",
        help="Enable in-context candidate keyword injection (RAG-for-Keywords).",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=100,
        help="Maximum candidate keywords to inject when --candidate-injection is active (default: 100).",
    )
    parser.add_argument(
        "--candidate-ranker",
        choices=list(RANKER_STRATEGIES),
        default=RANKER_LEXICAL,
        help=(
            "Which ranker builds the injected candidate set (default: lexical). "
            "Read only when --candidate-injection is active. Every published "
            "Condition C cell used the lexical default; 'dense' and 'hybrid' "
            "exist to test the retriever-agnostic claim and require an "
            "embedding model to be configured. 'freq' and 'floor' are the two "
            "C' corrections and need no embedder: 'freq' weights lexical "
            "relevance by log1p(catalog frequency), 'floor' instead drops "
            "candidates attached to 7 or fewer courses, which cannot fill a "
            "per_keyword slot. Measured on the 50-query set, 'floor' is much the "
            "stronger of the two."
        ),
    )
    parser.add_argument(
        "--candidate-source",
        choices=list(CANDIDATE_SOURCES),
        default=CANDIDATE_SOURCE_REWRITE,
        help=(
            "Which text the injected candidates are ranked against (default: "
            "rewrite, as in every published cell). 'query' ranks them against "
            "the learner's original query, so the candidate list stays the same "
            "whichever rewrite the selector reads. Read only when "
            "--candidate-injection is active."
        ),
    )

    parser.add_argument(
        "--max-queries",
        "--query-count",
        dest="max_queries",
        type=int,
        default=None,
        help="Maximum number of queries to evaluate from the query set.",
    )


    parsed = parser.parse_args(args)

    # `Config.load()`, not `Config()`. The two differ by whether `.env` is read, and
    # they are one attribute apart at the call site. With the bare constructor the
    # documented headline command failed on a machine that had the key in `.env` --
    # telling the user to do the thing they had already done -- and, far more quietly,
    # discarded every other declared setting: SKYRAG_MODEL, SKYRAG_TEMPERATURE,
    # SKYRAG_SEED, SKYRAG_MAX_TOKENS, SKYRAG_DEVICE. An echo run then completed
    # normally on library defaults rather than the declared configuration, with
    # nothing in any artifact to say the seed and temperature were not the ones the
    # methods section claims. Determinism is a stated property of this harness, so
    # that silence is the serious half of this defect.
    config = Config.load()
    config.llm.backend = parsed.backend

    backend = get_backend(config.llm, backend=parsed.backend)
    raw_models = DEFAULT_EVAL_MODELS if parsed.models is None else parsed.models
    models = [m.strip() for m in raw_models.split(",") if m.strip()]
    if not models:
        raise SystemExit(
            "skyrag eval: --models resolved to an empty list. Give at least one "
            "model id, comma-separated."
        )
    if parsed.repeats < 1:
        raise SystemExit(f"skyrag eval: --repeats must be at least 1, got {parsed.repeats}")

    # B0: arm-mode. When --arm is given, resolve each arm string into a
    # canonical ArmConfig (inside run_multi_model_eval via resolve_arm). The
    # CLI's job here is to (a) validate and expand the raw --arm string and
    # (b) enforce D3 by forcing pipelines to skyrag-only. The models list is
    # not replaced: run_multi_model_eval ignores it when arms is provided and
    # the slot list is assembled from the arm configs instead.
    parsed_arms: list[str] | None = None
    arm_models: list[str] | None = None
    if parsed.arm:
        parsed_arms = parse_arms(parsed.arm)
        # D3: every arm runs on `skyrag`; running naive rows for an arm sweep
        # pays for a second generation call per query and then drops the rows.
        # Overriding here rather than in the loop keeps the guard in one place
        # and ensures the "Pipelines" line below reflects what actually ran.
        parsed.pipelines = "skyrag"
        # Same reasoning, applied to the two lines that describe the run: the
        # "Models" banner and the cell count are resolved from the arm configs
        # so neither can describe a run the executor is not about to perform.
        # `models` above is left untouched -- run_multi_model_eval still
        # receives it and still ignores it in arm mode.
        arm_models = arm_cell_models(parsed_arms, parsed.models)

    # A file first, then a single query, then the built-in set. The built-in slice
    # was `[:2]` with no way to widen it: the specification asks for 50+ queries and
    # `-q` accepts one, so the only route to a real benchmark was 50 invocations --
    # each reloading the corpus and each overwriting the previous run's export,
    # since the output filename is fixed.
    query_ids: list[str] | None = None
    if parsed.queries:
        queries, query_ids = load_query_set(Path(parsed.queries))
    elif parsed.query:
        queries = [parsed.query]
    else:
        queries = list(HISTORICAL_BENCHMARK_QUERIES)

    if parsed.max_queries is not None and parsed.max_queries > 0:
        queries = queries[:parsed.max_queries]
        if query_ids:
            query_ids = query_ids[:parsed.max_queries]
    # `naive` is a separate reference condition, not an arm (D3): an arm sweep
    # that leaves it on pays for a second generation call per query and then
    # filters the rows out of the results table. Validated here rather than in
    # the loop, where an unknown name would surface as a ValueError after the
    # first cells had already been billed.
    pipelines = [p.strip() for p in parsed.pipelines.split(",") if p.strip()]
    unknown = [p for p in pipelines if p not in ("skyrag", "naive")]
    if unknown or not pipelines:
        raise SystemExit(
            f"skyrag eval: --pipelines must name skyrag and/or naive, got "
            f"{parsed.pipelines!r}."
        )

    # Load stored embeddings if this run's embedder produced them; otherwise fall
    # back to a synthetic store for safe CLI testing. The cache is checked by
    # identity, not by file presence: a cache left behind by a different embedder
    # at the same dimensionality would otherwise load silently and the run would
    # report an embedder comparison it never made.
    embedder_name = config.embedding.model_name
    stores = {}
    synthetic: list[tuple[str, str]] = []
    real_dims: set[int] = set()
    for platform in config.retrieval.platforms:
        status = inspect_cache(config.paths.embeddings, platform, model_name=embedder_name)
        if status:
            store = EmbeddingStore.load(
                config.paths.embeddings, platform, expect_model=embedder_name
            )
            stores[platform] = store
            if store.matrix.size:
                real_dims.add(int(store.matrix.shape[1]))
        else:
            synthetic.append((platform, status.detail))
            stores[platform] = None     # sized below, once the real width is known

    # A placeholder store has to live in the same vector space as the real ones or
    # the matrix product simply raises. The width was hardcoded to 384 while the
    # stored caches are 768, so `skyrag eval` could not run against the real corpus
    # at all -- it died inside score_all before producing a single row, which is why
    # the integration tests had to go through run_skyrag directly.
    if len(real_dims) > 1:
        raise SystemExit(
            f"Cached embeddings disagree on dimensionality: {sorted(real_dims)}. "
            "They cannot be compared in one run; re-embed so every platform shares "
            "an embedder."
        )
    # `next(iter(...))`, not `.pop()`: popping empties the set, and the emptied set
    # was then read as "no real caches loaded", which silently routed a run over the
    # real corpus through the constant placeholder vector.
    has_real_caches = bool(real_dims)
    dim = next(iter(real_dims)) if has_real_caches else _configured_dim(config, embedder_name)

    for platform, store in list(stores.items()):
        if store is not None:
            continue
        c = Course(
            id_=f"{platform[:3].upper()}1",
            platform=platform,
            title=f"Sample {platform.title()} Data Science Course",
            keywords=["data science", "python", "machine learning"],
        )
        stores[platform] = EmbeddingStore(
            courses=[c], matrix=np.ones((1, dim), dtype=np.float32)
        )

    degraded = {p for p, _ in synthetic}
    real = [p for p in config.retrieval.platforms if p not in degraded]

    # The coverage claim this run can actually support: every configured platform
    # was requested, and only the ones backed by a real cache count as loaded. A
    # synthetic placeholder is recorded as missing, because that is what it is --
    # counting it as loaded would export a full-corpus run built on 1-course stubs.
    coverage = CorpusCoverage(
        requested=list(config.retrieval.platforms),
        loaded=real,
        missing=sorted(degraded),
        course_counts={p: len(stores[p].courses) for p in real},
    )

    print("📊 Running SkyRag Evaluation Subsystem:")
    print(f"  Backend   : {parsed.backend}")
    print(f"  Models    : {', '.join(dict.fromkeys(arm_models)) if arm_models else ', '.join(models)}")
    print(f"  Embedder  : {embedder_name}")
    print(f"  Queries   : {len(queries)} query/queries")
    print(f"  Pipelines : {', '.join(pipelines)}")
    print(f"  Platforms : {', '.join(real) or 'none'} "
          f"({sum(len(stores[p].courses) for p in real)} courses)")
    if synthetic:
        # Loud on purpose, and per platform with the reason. A synthetic
        # single-course store produces plausible output from no data at all, and
        # results from such a run must never be mistaken for measurements. The
        # reason matters as much as the fact: "no cache" and "another embedder's
        # cache" are different problems with different remedies.
        print(f"  ⚠ SYNTHETIC : {len(synthetic)} platform(s) use a 1-course "
              f"placeholder. NOT VALID AS RESULTS.")
        for platform, why in synthetic:
            print(f"      {platform:<10} {why}")

        # B24, 2026-08-31. **A warning was not enough, and this is the evidence.**
        # The 150-cell 3-arm sweep and the 20-cell micro-sweep both ran to
        # completion against this placeholder, printed this exact banner, and
        # exported `corpus_courses = 0` with `corpus_platforms_missing =
        # coursera,edx` on every row -- and were still written up as results and
        # committed. The provenance columns worked; the human reading a 150-cell
        # scroll did not. Nine days of findings were withdrawn.
        #
        # So the refusal is the fix, not a louder banner. A run whose rows cannot
        # be a measurement should not produce rows a reader must later disqualify
        # -- DEC-9's carve-out again: a finished sweep cannot tell you afterwards
        # which of its rows were real.
        #
        # `--allow-synthetic-corpus` keeps the smoke-test path, because the
        # placeholder exists for a good reason (offline CLI plumbing tests). It
        # has to be typed, which is the whole point: an operator who types it has
        # decided, and an operator who does not cannot arrive here by accident.
        if not parsed.allow_synthetic_corpus:
            raise SystemExit(
                "\nREFUSED: "
                f"{len(synthetic)} platform(s) have no usable embedding cache, so this "
                "run would score every course against a constant vector and retrieve "
                "from a 1-course-per-platform stub.\n\n"
                "This is B24: the 2026-08-28 sweeps ran exactly this way, exported "
                "corpus_courses=0 on all 150 rows, and were written up as results "
                "before the artifacts were re-read.\n\n"
                "Fix the cache (check SKYRAG_EMBEDDINGS / the embeddings/ directory "
                "and that it was built by the configured embedder), or pass "
                "--allow-synthetic-corpus if you are deliberately smoke-testing "
                "plumbing and will not report the output."
            )

    # Sized from the stores this run actually loaded, so the query vector lands in
    # their space instead of a hardcoded 384. It is still a placeholder: a constant
    # vector makes every course equally similar, so the *ranking* from this path is
    # arbitrary and only the plumbing is under test. Said out loud below rather than
    # left for a reader to infer from a class named Dummy.
    class ConstantEmbedder:
        def encode_one(self, text: str) -> np.ndarray:
            return np.ones(dim, dtype=np.float32)

    real_embedder = None
    embedder_degraded_reason = ""
    if has_real_caches:
        # At least one real cache loaded, so a real query vector is both possible
        # and required -- ranking against real course vectors is meaningless
        # otherwise. Falls back loudly rather than silently if the model is absent.
        try:
            from skyrag.embedding import Embedder

            real_embedder = Embedder(config.embedding)
            real_embedder.encode_one("warm-up")
        except Exception as exc:  # noqa: BLE001
            real_embedder = None
            embedder_degraded_reason = f"{type(exc).__name__}: {exc}"
            print(f"  ⚠ EMBEDDER : {embedder_degraded_reason}")
            print(f"      Falling back to a constant {dim}-dim query vector. Course "
                  f"ranking from this run is ARBITRARY and NOT VALID AS RESULTS.")

            # **C14, 2026-09-25.** Refuse, on exactly the B24 argument above.
            #
            # The banner printed two lines up was the entire defence, and it was
            # not enough. On 2026-09-23 lane `i_l0` of the stamped factorial lost
            # torch to `WinError 1455` under three-way concurrency, printed this,
            # and went on to write 50 complete rows. Nothing downstream could tell
            # that directory from a valid one: the circuit breaker saw no failure,
            # no cell errored, the CSV was full and well-formed, and the process
            # exited 0. A human re-reading a lane log caught it, and
            # `scripts/launch_injection_lanes.sh` has since had to grep its own
            # logs for the banner above, which only works while the log survives
            # beside the artifact it describes.
            #
            # A degraded embedder is not graceful degradation. A constant query
            # vector makes every course equally similar, so the retrieval stage
            # returns an arbitrary slice of the catalogue and every metric
            # computed downstream describes that slice. These are not noisy
            # measurements, they are not measurements.
            #
            # So the run stops here, before any money is spent, rather than
            # producing rows a reader has to disqualify later -- DEC-9's
            # carve-out, and the same reasoning B24 settled for the corpus cache
            # one level up. The escape hatch has to be typed, for that reason too.
            if not parsed.allow_degraded_embedder:
                raise SystemExit(
                    "\nREFUSED: the sentence embedder could not be loaded, so "
                    "this run would score every course against a constant query "
                    "vector and rank the catalogue arbitrarily.\n\n"
                    f"  {embedder_degraded_reason}\n\n"
                    "This is C14: on 2026-09-23 a lane that failed exactly this "
                    "way ran to completion, exported 50 well-formed rows, exited "
                    "0, and was caught only because a human re-read its log. That "
                    "artifact is quarantined as outputs/"
                    "INVALID_embedder-fallback_fact_20260923_v5fp8_i_l0_r1/.\n\n"
                    "WinError 1455 here is the Windows commit limit, not RAM: "
                    "each concurrent lane loads its own torch and "
                    "all-mpnet-base-v2, and three lanes is the measured ceiling "
                    "on this machine. Retry with fewer concurrent lanes, or pass "
                    "--allow-degraded-embedder if you are deliberately "
                    "smoke-testing plumbing and will not report the output."
                )

    print(f"  Query vec : {'real embedder' if real_embedder else f'constant ({dim}-dim)'}")

    # Captured before the sweep and stamped onto every row. Printed too, because the
    # settings that decide a number should be visible in the terminal of the person
    # deciding whether the run is the one they meant to start.
    provenance = RunProvenance.capture(
        config,
        parsed.backend,
        sweep_version=parsed.sweep_version or SWEEP_VERSION_AUTO_LABEL,
        serving_precision=parsed.serving_precision,
        embedder_degraded=bool(embedder_degraded_reason),
        embedder_degraded_reason=embedder_degraded_reason,
    )
    print(f"  Run id    : {provenance.run_id}   skyrag {provenance.skyrag_version}")
    print(
        f"  Sweep ver : {parsed.sweep_version}"
        if parsed.sweep_version
        else f"  Sweep ver : {SWEEP_VERSION_AUTO_LABEL}"
        f"  ({SWEEP_VERSION_SELECTION_DEFAULT})"
    )
    # Printed as "UNSTAMPED" rather than blank. A precision nobody declared is the
    # state that produced Arm 1's inferred fp8, and a silent empty string in the
    # header is how that goes unnoticed until a reviewer asks.
    print(f"  Precision : {provenance.serving_precision or 'UNSTAMPED (not declared)'}")
    print(f"  Stage 1   : {parsed.expansion_mode or 'arm default'}")
    print(f"  Sampling  : temperature={provenance.temperature} seed={provenance.seed} "
          f"max_tokens={provenance.max_tokens}")
    print(f"  Embedder  : {provenance.embedding_model} on {provenance.embedding_device}")
    # Printed, not merely exported. A downgrade reason that reaches only the CSV is
    # discovered after the money is spent; this is the line that says the GPU run
    # you asked for is a CPU run before it starts. Set SKYRAG_DEVICE_STRICT=1 to
    # have it refuse instead of report.
    if provenance.embedding_device_fallback:
        print(f"              ⚠ {provenance.embedding_device_fallback} "
              "-- set SKYRAG_DEVICE_STRICT=1 to refuse rather than downgrade")
    print(f"  Routing   : {provenance.provider or 'automatic (provider not pinned)'}")
    # The dead variable says so out loud. SKYRAG_PROVIDER predates the routing pin,
    # is read by nothing, and holds a backend name rather than a provider slug --
    # so a reader who sets it and sees "Routing: automatic" would otherwise have no
    # way to tell that their setting does nothing.
    import os as _os
    if _os.environ.get("SKYRAG_PROVIDER") and not provenance.provider:
        print("      note: SKYRAG_PROVIDER is set but is not read by anything. "
              "Use SKYRAG_OPENROUTER_PROVIDER to pin upstream routing.")
    pinned = {
        "keyword": parsed.keyword_model,
        "expansion": parsed.expansion_model,
        "generation": parsed.generation_model,
    }
    if any(pinned.values()):
        for role, model_id in pinned.items():
            fallback_label = "(cell's own model)"
            print(f"  Pinned    : {role:<11}-> {model_id or fallback_label}")
    else:
        print("  Pinned    : none -- every role runs on the cell's own model")
    if parsed.repeats > 1:
        print(f"  Repeats   : {parsed.repeats} passes over the grid")
        # Said here rather than left for the reader of the CSV. Three passes under a
        # single fixed seed at temperature 0 are three plays of the same greedy
        # decode: whatever spread appears between them comes from routing, batching
        # and checkpoint drift, not from the model choosing differently. That is a
        # real and reportable property -- it is just not an error bar on model
        # behaviour, and the two are easy to confuse once the rows are in a
        # spreadsheet with nothing to say which one they are.
        if parsed.seed_per_repeat:
            base = provenance.seed
            walked = "unseeded" if base is None else f"{base}..{base + parsed.repeats - 1}"
            print(f"              seed walks per pass ({walked}), same sequence for every cell")
            if provenance.temperature == 0:
                print("              note: temperature is 0, so decoding is greedy and "
                      "walking the seed may change little")
        else:
            print(f"              fixed seed {provenance.seed} on every pass -- these "
                  "measure harness stability")
            print("              (routing, batching, checkpoint drift), NOT model "
                  "variability. Use --seed-per-repeat above temperature 0 for that.")
    prior_rows: list[dict] = []
    skip_cells = None
    if parsed.resume:
        resume_path = Path(parsed.resume)
        if not resume_path.exists():
            print(f"✗ --resume: no such file: {resume_path}")
            return 1
        if resume_path.suffix == ".json":
            prior_rows = json.loads(resume_path.read_text(encoding="utf-8"))
        else:
            import csv as _csv
            with resume_path.open(encoding="utf-8", newline="") as fh:
                prior_rows = list(_csv.DictReader(fh))
        skip_cells = completed_cells(prior_rows)
        print(f"  resume: {len(prior_rows)} prior row(s), "
              f"{len(skip_cells)} already measured -> {resume_path}")

    print()
    if parsed_arms:
        # `arm_models` carries one entry per ArmConfig the executor will build,
        # so its length already includes the `-m` x arms cross-product. Naming
        # the model factor separately keeps the doubling visible rather than
        # folding it silently into the arm count.
        _n_models = len(arm_models) // len(parsed_arms)
        _model_factor = f"x {_n_models} models " if _n_models > 1 else ""
        print(f"⏳ Running {parsed.repeats * len(queries) * len(arm_models) * len(pipelines)} "
              f"cells ({parsed.repeats} x {len(queries)} queries x {len(parsed_arms)} arms "
              f"{_model_factor}x {len(pipelines)} pipelines):")
    else:
        print(f"⏳ Running {parsed.repeats * len(queries) * len(models) * len(pipelines)} "
              f"cells ({parsed.repeats} x {len(queries)} queries x {len(models)} models "
              f"x {len(pipelines)} pipelines):")

    out_dir = resolve_benchmark_output_dir(
        parsed.output_dir,
        arms=parsed_arms,
        arm=parsed.arm if not parsed_arms else None,
        model=parsed.models,
        provider=provenance.provider or config.llm.provider or (parsed.backend if parsed.backend != "openrouter" else ""),
        sweep_version=parsed.sweep_version,
        query_count=len(queries),
    )
    csv_path = out_dir / "benchmark_eval.csv"
    json_path = out_dir / "benchmark_eval.json"


    try:
        metrics = run_multi_model_eval(
            queries=queries,
            models=models,
            pipelines=pipelines,
            backend=backend,
            config=config,
            stores=stores,
            embedder=real_embedder or ConstantEmbedder(),
            coverage=coverage,
            query_ids=query_ids,
            repeats=parsed.repeats,
            seed_per_repeat=parsed.seed_per_repeat,
            keyword_model=parsed.keyword_model,
            expansion_model=parsed.expansion_model,
            generation_model=parsed.generation_model,
            provenance=provenance,
            on_cell=_print_cell,
            arms=parsed_arms,
            model_override=parsed.models,
            arm=parsed.arm if not parsed_arms else None,
            intervention=parsed.intervention,
            output_csv_path=csv_path,
            skip_cells=skip_cells,
            sweep_version=parsed.sweep_version,
            schema_exemplar=parsed.schema_exemplar,
            expansion_mode=parsed.expansion_mode,
            candidate_injection=parsed.candidate_injection,
            candidate_limit=parsed.candidate_limit,
            candidate_ranker=parsed.candidate_ranker,
            candidate_source=parsed.candidate_source,
        )
    except SweepCircuitBreakerError as exc:
        print(f"\n❌ {exc}")
        print(f"   Partial results saved to: {csv_path}")
        print(f"   To resume after fixing the root cause, pass: --resume {csv_path}\n")
        sys.exit(1)


    # B22. Carry the prior measurements into the new export. Without this a resumed
    # run writes ONLY the re-run cells, so an R1b-style top-up of 11 cells would
    # replace a 100-row baseline with an 11-row file -- the resume would look like
    # it worked and would have destroyed the thing it was topping up.
    if parsed.resume and prior_rows:
        rerun = {
            cell_key(repeat=m.repeat, query_id=m.query_id, model=m.model,
                     pipeline=m.pipeline, arm=m.arm or "")
            for m in metrics
        }
        carried, undecodable = [], 0
        for row in prior_rows:
            key = cell_key(
                repeat=row.get("repeat", 1) or 1, query_id=row.get("query_id", ""),
                model=row.get("model", ""), pipeline=row.get("pipeline", ""),
                arm=row.get("arm", ""),
            )
            if key in rerun:
                continue  # superseded by this run's fresh measurement
            try:
                carried.append(metric_from_row(row))
            except (TypeError, ValueError):
                undecodable += 1
        if undecodable:
            print(f"  ⚠ resume: {undecodable} prior row(s) could not be decoded and "
                  f"were NOT carried. Resume from the .json, not the .csv.")
        # B22 guard. A resumed run whose backend differs from the prior rows'
        # produces one file mixing simulated and live measurements. The `backend`
        # column makes it *detectable*; nothing made it *refused*, and the first
        # real use of --resume merged 11 `echo` rows into an 89-row `openrouter`
        # baseline and printed "11 re-run" without complaint. The rows were
        # `empty_retrieval` with `selection_seconds=0.0`, which is what gave it
        # away -- an echo reply that happened to parse would not have.
        prior_backends = {str(r.get("backend")) for r in prior_rows if r.get("backend")}
        fresh_backends = {str(m.backend) for m in metrics if getattr(m, "backend", None)}
        crossed = prior_backends and fresh_backends and not (prior_backends & fresh_backends)
        if crossed:
            print(f"✗ --resume: backend mismatch. Prior rows ran on "
                  f"{sorted(prior_backends)}, this run on {sorted(fresh_backends)}. "
                  f"Merging them would put simulated and live cells in one file. "
                  f"Re-run with -b {sorted(prior_backends)[0]}, or resume a different export.")
            return 1

        metrics = carried + metrics
        print(f"  resume: {len(carried)} prior row(s) carried, "
              f"{len(rerun)} re-run -> {len(metrics)} total")

    export_benchmark_csv(metrics, csv_path)
    export_benchmark_json(metrics, json_path)


    df = metrics_to_dataframe(metrics)
    print("📈 Benchmark Execution Results Summary:")
    print(format_results_summary(df))

    # Printed separately and unconditionally: these are the states that must not
    # be inferred from an average. A run that produced nothing has no grounding
    # score, and an API failure is not a model that selected nothing.
    summary = summarize_metrics(metrics)
    print("\n🔍 Run integrity:")
    mean = summary["grounding_accuracy_mean"]
    print(f"  grounding accuracy (mean)  : "
          f"{'n/a' if mean is None else mean} "
          f"over {summary['grounding_accuracy_n']}/{summary['runs']} run(s) that "
          f"retrieved and cited something")
    print(f"  runs citing nothing        : {summary['zero_citation_runs']}"
          f"   (excluded from the mean, not scored 1.0)")
    print(f"  runs aborted, no retrieval : {summary['empty_retrieval_runs']}"
          f"   (generation LLM not called)")
    print(f"  runs w/ backend errors     : {summary['runs_with_selection_errors']}"
          f"   (infrastructure, not model capability)")
    print(f"  runs w/ empty selections   : {summary['runs_with_empty_selections']}"
          f"   (model chose nothing)")
    # These three were computed by summarize_metrics and never printed. On a real run
    # where three of six cells aborted, every visible cause read zero while the actual
    # cause -- unparseable -- was calculated and discarded before display. A counter
    # that is computed and not shown is indistinguishable, to the reader, from a
    # counter that is zero, which is the exact failure the outcome taxonomy exists to
    # remove. OUTCOME_OVERRUN in particular was added so a model ignoring the keyword
    # cap would be counted; the count existed and reached nobody.
    print(f"  runs w/ unparseable sel.   : {summary['runs_with_unparseable_selections']}"
          f"   (answered fully, garbage out -- counts against the model)")
    print(f"  runs w/ overrun selections : {summary['runs_with_overrun_selections']}"
          f"   (ignored the keyword cap -- counts against the model)")
    print(f"  runs w/ truncated sel.     : {summary['runs_with_truncated_selections']}"
          f"   (budget below a compliant answer -- ours, not the model's)")
    # A fourth hidden counter, found by the test written for the other three rather
    # than by the sweep that reported them. Same class: computed, never shown.
    print(f"  runs w/ no vocabulary      : {summary['runs_with_no_vocabulary']}"
          f"   (platform had no keywords -- corpus, not the model)")
    print(f"  cells that failed outright : {summary['failed_cells']}"
          f"   (backend raised; recorded as a row, not lost)")
    # X1. Printed for the same reason the four counters above are: a per-stage
    # cost that is computed and never shown is indistinguishable from one that
    # is zero. This block is also the only place the reader sees that generation
    # is a *minority* of a SKYRAG run's tokens -- the selection stage carries the
    # vocabulary window, which is the largest prompt in the pipeline.
    _stage = summary["expansion_tokens_total"]
    print(f"  tokens: expansion stage    : {_stage['tokens']}"
          f"   (over {_stage['rows']}/{summary['runs']} run(s) that reported usage)")
    _stage = summary["selection_tokens_total"]
    print(f"  tokens: selection stage    : {_stage['tokens']}"
          f"   (keyword selection, summed over platforms)")
    _stage = summary["generation_tokens_total"]
    print(f"  tokens: generation stage   : {_stage['tokens']}"
          f"   (O4's like-for-like slice; was the ONLY stage counted before X1)")
    print(f"  tokens: whole pipeline     : {summary['pipeline_tokens_total']}"
          f"   (over {summary['pipeline_tokens_n']}/{summary['runs']} run(s) with every stage reported)")
    # R22. Beside the token totals because it answers a different question about
    # the same output: an O1 gain that tracks length is verbosity, not capability.
    print(f"  path length (mean chars)   : {summary['path_chars_mean']}"
          f"   (max {summary['path_chars_max']}; R22's verbosity control on O1)")
    print(f"  path length (mean words)   : {summary['path_words_mean']}"
          f"   (crude whitespace count, by design -- a monotone length proxy)")
    # **R11. PROJECT-DEFINED, not a published metric** -- the label travels into the
    # terminal for the same reason it sits on the column. Printed immediately after
    # the token totals because it is the question those totals cannot answer: how
    # much of what a run spent became the thing anyone reads. Expect low single-digit
    # percentages -- the selection prompt carries the 1,000-keyword window -- and
    # read it only between arms holding that window fixed.
    print(f"  R11 token efficiency ratio : {_pct(summary['token_efficiency_ratio_mean'])}"
          f"   (project-defined: generation output / whole-pipeline spend)")
    print(f"  R11 tokens per cited course: {summary['tokens_per_cited_course_mean']}"
          f"   (project-defined: end-to-end cost of one recommendation)")
    print(f"  runs on incomplete corpus  : {summary['runs_on_incomplete_corpus']}"
          f"   (a requested platform had no dataset)")
    # The denominators, printed before the counts they divide. Until 2026-08-19 the
    # export carried numerators only, so "30 hallucinated keywords" could mean 30 of
    # 35 or 30 of 300 -- opposite findings that the artifact could not tell apart.
    print(f"  keywords emitted / kept    : {summary['keywords_emitted_total']}"
          f" / {summary['keywords_kept_total']}"
          f"   (the denominator for the three rates below)")
    print(f"  M2.1 invalid keyword rate  : {_pct(summary['invalid_keyword_rate'])}"
          f"   (invented -- absent from the corpus)")
    print(f"  M2.2 near-miss rate        : {_pct(summary['near_miss_rate'])}"
          f"   (real concept, botched transcription; NOT repaired)")
    print(f"  M2.3 out-of-window rate    : {_pct(summary['out_of_window_rate'])}"
          f"   (real, outside the window shown -- harness limit, not the model)")
    # **R24(a).** Printed directly beneath the specified form, and labelled, because
    # the two disagree in the direction that matters. Dividing by everything the
    # model emitted dilutes its real keywords in proportion to how much it invented,
    # so the arm that invents more is flattered: on 2026-08-20 the specified form
    # read 0.126 (8B) vs 0.144 (70B) -- the small model apparently *better* at long-
    # context recall -- while the conditional form read 0.381 vs 0.204. Anyone
    # reading one of these without the other reaches the opposite conclusion, which
    # is why they are adjacent rather than in separate blocks.
    print(f"  M2.3 conditional (R24a)    : {_pct(summary['out_of_window_rate_conditional'])}"
          f"   (of {summary['keywords_real_total']} REAL keywords named -- the arm-comparable form)")
    # R24(b). The two quantities M3.2 is built from, printed above it so the ratio
    # cannot be read bare. An arm that retrieves half as much scores higher here
    # while citing fewer courses in absolute terms.
    print(f"  cited / retrieved (mean)   : {summary['cited_count_mean']}"
          f" / {summary['retrieved_count_mean']}"
          f"   (M3.2's numerator and denominator -- the ratio moves with both)")
    # M3.2, printed next to grounding because grounding alone is silent about it: a
    # row citing 1 of 82 retrieved courses scores 1.0 and has used almost nothing.
    print(f"  M3.2 citation coverage     : {_pct(summary['citation_coverage_mean'])}"
          f"   (of the courses handed to the model, how many it cited)")
    # Reference-free selection quality, in place of M1.1's answer key. A low mean
    # here says the selected keywords pulled material unlike the query -- which is
    # a bad selection whoever would have chosen differently.
    print(f"  retrieval score (mean)     : {summary['retrieval_score_mean']}"
          f"   (similarity of what the keywords pulled; no reference set needed)")
    # O2's redundancy half. High means the recommended courses are about the same
    # things -- a bad set however well grounded each entry is.
    print(f"  cited-set redundancy       : {summary['cited_redundancy_mean']}"
          f"   (keyword overlap among cited courses; 1.0 = all the same topics)")
    # The published forms, printed together and labelled with their sources so the
    # distinction survives into the terminal: these three have a literature
    # citation and every other number in this block is a bespoke diagnostic.
    print(f"  ILS, embedding cosine      : {summary['cited_redundancy_cosine_mean']}"
          f"   (Ziegler 2005; the line above is the same metric over labels)")
    print(f"  context nDCG@10            : {summary['context_ndcg_at_10_mean']}"
          f"   (Jarvelin 2002; ordering the model reads vs the ideal. NOT recall)")
    # **R24(c). The saturation warning, printed where the two metrics it applies to
    # are read.** Both returned their ceiling on every scorable row of both arms on
    # 2026-08-20 -- nDCG 0.99, reciprocal rank exactly 1.000 -- which means they
    # were describing the presented order, not the arms. A fraction at or near 1.0
    # says this metric cannot separate anything on this data and must not appear in
    # a between-arm table; it is far cheaper to see that here than in review.
    print(f"  ^ fraction at ceiling      : nDCG {summary['context_ndcg_at_10_ceiling_fraction']}"
          f"   (R24c: near 1.0 = saturated, measures the harness not the arms)")
    # The mean hid a tail. Over every scored row on disk only 32.4% sit at 1.0,
    # p5 is 0.9495 and the minimum is 0.7169 -- so the ceiling fraction above is
    # the share at the top, and these two are where the interleave actually cost
    # something. The low tail is the only part that can answer whether
    # round-robin merging buries strong courses; the mean cannot.
    print(f"  ^ nDCG 5th pct / n < 0.95  : {summary['context_ndcg_at_10_p5']}"
          f" / {summary['context_ndcg_at_10_below_0_95']}"
          f"   (the tail, where positional interleave costs ranking)")
    print(f"  generation-stage seconds   : {summary['generation_seconds_total']}"
          f"   (O4's like-for-like slice; excludes retrieval and selection)")
    # Printed beside generation because the comparison is the reading. In a run that
    # pins generation and varies only keyword selection, `selection` is the only one
    # of the three that moves with the thing under test -- and it is 0.0 for every
    # `naive` row by construction, because that pipeline has no keyword stage.
    print(f"  selection-stage seconds    : {summary['selection_seconds_total']}"
          f"   (the stage a substitution experiment actually varies; 0 for naive)")
    print(f"  expansion-stage seconds    : {summary['expansion_seconds_total']}"
          f"   (query expansion + step-back)")
    print(f"  hallucinated keywords      : {summary['hallucinated_keywords_total']}"
          f"   (absent from the corpus, exactly and after folding)")
    # Printed beside M2.1 on purpose: the pair is the reading. A high near-miss
    # count against a low hallucination count is a transcription problem; the
    # reverse is a knowledge problem. One number without the other supports
    # neither claim.
    print(f"  near-miss keywords         : {summary['near_miss_keywords_total']}"
          f"   (M2.2 -- real concept, botched transcription; NOT repaired)")
    print(f"  out-of-window keywords     : {summary['out_of_window_keywords_total']}"
          f"   (real, outside the window shown)")
    print(f"  cleaned-out keywords       : {summary['removed_by_cleaning_keywords_total']}"
          f"   (real, removed as publishers -- not hallucinations)")

    print(f"\n✅ Exported evaluation CSV  -> {csv_path}")
    print(f"✅ Exported evaluation JSON -> {json_path}")


def _force_utf8_stdout() -> None:
    """Windows consoles default to cp1252, which cannot encode this CLI's emoji.

    Without this the eval summary dies with UnicodeEncodeError before printing
    a single row -- so the run-integrity counts below would never be seen on
    the platform this repository is developed on.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")    # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass                                    # already UTF-8, or redirected


def main():
    """Main CLI entry point for SkyRag."""
    _force_utf8_stdout()
    args = sys.argv[1:]

    from .help_text import resolve as _resolve_help

    _help_page = _resolve_help(args)
    if _help_page is not None:
        print(_help_page)
        return

    if len(args) > 0 and args[0] == "eval":
        run_eval_cli(args[1:])
        return

    if len(args) > 0 and args[0] == "show":
        from .show import run_show_cli

        run_show_cli(args[1:])
        return

    if not args:
        print(_resolve_help(["--help"]))
        return
    print(f"Unknown subcommand: {args[0]}")
    print("Run 'skyrag --help' for available commands.")
    sys.exit(1)


if __name__ == "__main__":
    main()

