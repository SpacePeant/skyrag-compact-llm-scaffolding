r"""Help text for the `skyrag` CLI.

Two levels, deliberately.

`skyrag --help` is a **clean command list** -- what you can type, and where to look
next. That is what a person expects from `--help`, and a page that scrolls is a page
that gets skimmed. `skyrag <command> --help` gives that command's flags in prose,
already rendered by argparse.

`skyrag --help-all` is the **full reference**: every flag, grouped, with the traps
spelled out. It exists so that detail has somewhere to live other than the front
page.
"""

from __future__ import annotations

# ---------------------------------------------------------------- short page

_SHORT = """\
SkyRag - Separated Keyword RAG Model Comparison Suite

Usage
  skyrag <command> [options]

Commands
  eval      Run an evaluation sweep
  show      Print the learning path a run produced

Arms
  1         70B, untreated                   the control
  2         8B, untreated                    the scale contrast
  3         8B + JSON schema                 the schema treatment

Before you spend anything
  -b defaults to `echo`, a mock.  Pass -b openrouter for a real run.
  Do not pass -m on --arm 1,2 -- it would collapse the 70B/8B contrast.
  -r 1 supports no variance claim.  Use -r 3 for anything to be compared.

More help
  skyrag eval --help        every eval flag, in full
  skyrag --help-all         the complete reference

README.md - setup, and which script produces which result.
"""

# ----------------------------------------------------------- full reference

_FULL_HEADER = """\
SkyRag - Separated Keyword RAG Model Comparison Suite
Complete reference.  Short version: skyrag --help
"""

_FULL_USAGE = """\
USAGE
  skyrag eval [flags]    Run a sweep   cells = arms x queries x repeats x pipelines
  skyrag show [run]      Print the learning path a run produced
"""

_FULL_TRAPS = """\
TRAPS
  -b defaults to `echo`, the MOCK -- the export will have no path in it
  -m overrides every arm's model AND multiplies cells; omit on --arm 1,2
  pin the provider AND declare --serving-precision, or both vary unrecorded
  -r 1 supports no variance claim; use -r 3 for anything to be compared
"""

_FULL_ARMS = """\
ARMS
  1  70B, untreated    2  8B, untreated    3  8B + JSON schema
  1 vs 2 is the scale contrast -- do not pass -m.  1 and 2 are ALREADY
  untreated: running "no treatment" means leaving flags off, not adding one.
"""

_FULL_EVAL = """\
EVAL -- core
  -b, --backend       echo   openrouter | echo | ollama.  THE DEFAULT IS A MOCK.
  --arm               none   comma list or `all`.  Overrides -m, forces -P skyrag.
  -m, --models        auto   Overrides EVERY arm's model; with --arm, cells x N.
  -P, --pipelines     both   skyrag | naive.  Ignored under --arm.
  -q / -Q, --queries  -      One query / a query set (-Q wins).  benchmarks/*.jsonl
  --max-queries       all    Cap queries taken from the set.
  -r, --repeats       1      See TRAPS.        --seed-per-repeat  moot at temp 0.
  -o, --output-dir    auto   Explicit path, else semantic.  Personal runs: scratch_
  --resume <prior>    -      Re-runs ONLY cell_failed/absent.  Repeat every flag.

EVAL -- treatment
  --candidate-injection      Condition C.  The largest measured effect here.
  --candidate-limit   100    Candidates injected.
  --candidate-ranker  lexical  How the injected candidates are ranked.
  --candidate-source  rewrite  Rank against the rewrite or the learner's query.
  --schema-exemplar          Arm 3: spec_only -> spec_plus_exemplar.  A LEVEL of
                             the treatment, not a fix.  Never compare across it.
  --expansion-mode    arm    plain | schema | schema_wording.  Stage 1 only.
  --sweep-version     v5     v1|v2|v3|v4|v5.  The paper's runs used v2 and v5.
                             PART OF THE TREATMENT -- v2 and v5 are not comparable.
  --serving-precision -      bf16|fp16|fp8|q4|q8.  Declared, unverifiable, stamped.

EVAL -- less common
  --keyword-model / --expansion-model / --generation-model   Pin one role's model.
  --intervention             Override the label.  Can mislabel a row's treatment.
  --allow-synthetic-corpus   1-course placeholder.  NEVER for a reported result.
  --allow-degraded-embedder  Constant query vector.  NEVER for a reported result.
"""

_FULL_SHOW = """\
SHOW
  skyrag show [run] [--list] [--row N] [--all]   Default: newest run WITH ROWS.
"""

_FULL_ENV = """\
NO --temperature, NO --seed
  Protocol constants are env vars in .env (temp 0.0, seed 34, max_tokens 4096,
  per_keyword 7, max_vocabulary 1000, ...), read automatically and stamped onto
  every row.  Changing one changes the experiment.  Template: .env.example.
"""

_FULL_FOOTER = """\
LADDER   echo  ->  1q smoke  ->  10q pilot  ->  50q -r 3.   Never skip to the end.
THEN     skyrag show outputs/<run>   |   scripts/aggregate_arms.py --compare ...

Every flag in prose: skyrag eval --help    Setup and scripts: README.md.
"""


def _join(*blocks: str) -> str:
    """One blank line between sections -- the blocks carry no spacing of their own."""
    return "\n\n".join(b.rstrip("\n") for b in blocks if b) + "\n"


def front_page() -> str:
    """The front page: a command list, not a reference."""
    return _SHORT


def everything() -> str:
    """The complete reference, for `--help-all`."""
    return _join(_FULL_HEADER, _FULL_USAGE, _FULL_TRAPS, _FULL_ARMS, _FULL_EVAL,
                 _FULL_SHOW, _FULL_ENV, _FULL_FOOTER)


def resolve(args: list[str]) -> str | None:
    """Return the help page `args` asks for, or None if it asks for no help.

    Accepts the spellings a user is likely to try rather than one canonical form.
    """
    if not args:
        return None
    first, rest = args[0], args[1:]
    if first in ("--help-all", "--helpall", "--help_all"):
        return everything()
    if first in ("-h", "--help", "help"):
        if rest and rest[0] in ("all", "--all"):
            return everything()
        return front_page()
    return None
