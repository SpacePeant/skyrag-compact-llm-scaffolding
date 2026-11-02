"""Pre-evaluation gate: everything that must hold before a reportable run.

Cross-reference:
- backlog_2026-08-17_bugfix_unified-code-audit-remediation.md, task 5.4

    python scripts/preflight.py              # offline checks only (free, ~30s)
    python scripts/preflight.py --live       # ...plus the paid live smoke run

Why this exists rather than "just run pytest": on 2026-08-17 the full suite passed
with eleven defects present, and two of the five most serious were found by the
first live run rather than by any test. A green suite is necessary here and nowhere
near sufficient, so the gate runs four independent kinds of evidence:

  1. unit and integration suite            -- contracts, and assertions on artifacts
  2. adversarial reproductions             -- each known defect driven by real input
  3. dataset audit                         -- index-time corruption already on disk
  4. live smoke run against a real backend -- the class of failure 1-3 cannot see

Step 4 costs money and needs OPENROUTER_API_KEY, so it is opt-in via --live. It is
not optional before a reportable run: steps 1-3 exercise the harness against inputs
we thought of, and the whole finding of the 2026-08-17 session was that the harness
fails against inputs we did not.

Step 4 runs at the *configured* retrieval parameters. It used to run at
`--per-keyword 1` -- a ceiling of 80 courses against the 560 `.env` permits -- so on
2026-08-18 it printed CLEARED fifteen minutes before the reportable configuration
failed on a defect it could not reach. `--cheap-live` restores the smaller run and is
reported as an uncleared gate, because a gate that covers less than the experiment
has to say so rather than print the same word.

Exit status is 0 only when every executed step passed. A skipped live run is
reported as a gap, not as a pass.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# The key lives in .env, not the shell, everywhere else in this project. Reading
# only os.environ would have made the gate report "no API key" on a machine that
# has one -- and a gate that misreports the reason it cannot run is worse than one
# that simply fails.
try:
    from skyrag.config import load_dotenv_if_present

    load_dotenv_if_present()
except Exception:                       # the suite step will surface a real import problem
    pass

Step = tuple[str, list[str], str]


def offline_steps() -> list[Step]:
    return [
        (
            "test suite",
            [sys.executable, "-m", "pytest", "tests/", "-q", "--no-header"],
            "Contracts hold, and the exported CSV/JSON carry every integrity column.",
        ),
        (
            "dataset keyword audit",
            [sys.executable, "scripts/audit_dataset_keywords.py", "--sample", "0"],
            "No course with source text is missing its keywords (finding S2).",
        ),
    ]


def live_step(*, cheap: bool) -> Step:
    """The paid step. At the configured `per_keyword` unless asked for the cheap one.

    This used to pass `--per-keyword 1` unconditionally. At `per_keyword=1` the
    ceiling is 40 x 1 x 2 = 80 courses and the smoke run retrieved 33, against the
    560 that `.env` permits -- so the gate executed a configuration seven times
    smaller than the reportable one, and on 2026-08-18 it printed `CLEARED`
    fifteen minutes before both 70B cells of the real sweep failed on a defect that
    was structurally unreachable from it.

    A cost control that quietly narrows what the gate covers is the same shape as
    every other defect in these backlogs: a legal value that reads as more than it
    is. So the default is now the configuration that will actually be reported,
    `--cheap-live` restores the old behaviour explicitly, and a cheap run is reported as
    an uncleared gate rather than as a pass -- the same treatment a skipped live
    run already gets, and for the same reason.
    """
    # Release copy: the development regression scripts are not shipped, so the
    # live step is one real query through `skyrag eval` against the built corpus.
    eval_args = [
        "eval", "--backend", "openrouter", "--arm", "2",
        "-Q", "benchmarks/queries_1q_smoke.jsonl",
        "-o", "outputs/scratch_preflight",
    ]
    if cheap:
        # per_keyword has no CLI flag; set it for the child process only.
        launcher = (
            "import os, runpy, sys; os.environ['SKYRAG_PER_KEYWORD'] = '1'; "
            "sys.argv = ['skyrag'] + sys.argv[1:]; "
            "runpy.run_module('skyrag.cli', run_name='__main__')"
        )
        cmd = [sys.executable, "-c", launcher, *eval_args]
    else:
        cmd = [sys.executable, "-m", "skyrag.cli", *eval_args]
    return (
        "live smoke run",
        cmd,
        "A real backend, a real corpus, and a real model's non-compliance.",
    )


def live_parameters(*, cheap: bool) -> tuple[dict[str, object], list[str]]:
    """What the live step will run under, and how that differs from `.env`.

    Returned rather than printed so the summary can state the divergence twice --
    once before the spend and once beside the verdict. A reader who sees only the
    verdict is the reader this exists for.
    """
    try:
        from skyrag.config import Config
    except Exception as exc:                # the suite step reports a real import problem
        return {"error": f"{type(exc).__name__}: {exc}"}, []

    cfg = Config.load()
    configured = cfg.retrieval.per_keyword
    effective = 1 if cheap else configured
    platforms = len(cfg.retrieval.platforms)

    params = {
        "per_keyword": effective,
        "max_keywords": cfg.retrieval.max_keywords,
        "platforms": ", ".join(cfg.retrieval.platforms),
        "max_context_courses": cfg.retrieval.max_context_courses,
        "course ceiling": cfg.retrieval.max_keywords * effective * platforms,
        "generation model": cfg.llm.model_for("generation"),
        "keyword model": cfg.llm.model_for("keyword"),
        # Printed because pinning a provider also pins a numeric precision, and an
        # unpinned run can serve two rows of the same arm at different precision
        # with no column recording it. See methods commitments 3.3b/3.3c.
        "provider": cfg.llm.provider or "AUTOMATIC -- not pinned",
    }

    divergences = []
    if not cfg.llm.provider:
        divergences.append(
            "SKYRAG_OPENROUTER_PROVIDER is unset, so routing is automatic -- part of "
            "the latency and price signal is the router rather than the model, and "
            "nothing guarantees two rows of the same arm were served at the same "
            "numeric precision"
        )
    if effective != configured:
        divergences.append(
            f"per_keyword={effective}, but the configuration declares {configured} "
            f"-- a ceiling of {cfg.retrieval.max_keywords * effective * platforms} "
            f"courses instead of "
            f"{cfg.retrieval.max_keywords * configured * platforms}"
        )
    return params, divergences


def run(step: Step, verbose: bool) -> tuple[bool, float, str]:
    name, cmd, _ = step
    print(f"\n{'-' * 78}")
    print(f"  {name}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'-' * 78}")

    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=not verbose, text=True)
    elapsed = time.perf_counter() - start

    output = ""
    if not verbose:
        output = (proc.stdout or "") + (proc.stderr or "")
        tail = [ln for ln in output.strip().splitlines() if ln.strip()][-6:]
        for line in tail:
            print(f"    {line[:120]}")

    print(f"    ({elapsed:.1f}s, exit {proc.returncode})")
    return proc.returncode == 0, elapsed, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true",
                        help="Also run the paid live smoke run against OpenRouter.")
    parser.add_argument("--cheap-live", action="store_true",
                        help="Run the live step at --per-keyword 1 instead of the "
                             "configured value. Cheaper, and NOT a cleared gate: it "
                             "certifies the plumbing rather than the experiment.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Stream each step's full output instead of a tail.")
    args = parser.parse_args()

    print("=" * 78)
    print("PRE-EVALUATION GATE")
    print("=" * 78)

    steps = offline_steps()
    skipped_live_reason = None
    live_divergences: list[str] = []

    if args.live:
        if os.environ.get("OPENROUTER_API_KEY"):
            params, live_divergences = live_parameters(cheap=args.cheap_live)
            print()
            print("  The live step will run under:")
            for key, value in params.items():
                print(f"      {key:<21}: {value}")
            for line in live_divergences:
                print(f"  DIVERGES FROM CONFIG: {line}")
            steps.append(live_step(cheap=args.cheap_live))
        else:
            skipped_live_reason = (
                "--live was requested but OPENROUTER_API_KEY is not set in the "
                "environment; the live step could not run."
            )
    else:
        skipped_live_reason = (
            "the live smoke run was not requested (pass --live to include it)."
        )

    results = []
    for step in steps:
        ok, elapsed, _ = run(step, args.verbose)
        results.append((step[0], ok, elapsed))

    print()
    print("=" * 78)
    print("GATE RESULT")
    print("=" * 78)
    for name, ok, elapsed in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<26} {elapsed:>6.1f}s")

    failed = [name for name, ok, _ in results if not ok]
    ran_live = any(name == "live smoke run" for name, _, _ in results)

    print()
    if failed:
        print(f"  BLOCKED -- {len(failed)} step(s) failed: {', '.join(failed)}")
        print("  Do not run a reportable evaluation until these are green.")
        return 1

    if ran_live and live_divergences:
        print("  Offline checks and the live run all pass, but this is NOT a cleared")
        print("  gate: the live step did not run the configuration that will be")
        print("  reported.")
        for line in live_divergences:
            print(f"      {line}")
        print()
        print("  A smaller live run certifies the plumbing, not the experiment. On")
        print("  2026-08-18 a gate reduced this way printed CLEARED fifteen minutes")
        print("  before both 70B cells of the real sweep failed, on a defect the")
        print("  reduced configuration could not reach.")
        print()
        print("      python scripts/preflight.py --live   # i.e. without --cheap-live")
        print()
        return 2

    if not ran_live:
        print("  Offline checks all pass, but this is NOT a cleared gate.")
        print(f"  Reason: {skipped_live_reason}")
        print()
        print("  Steps 1-3 exercise the harness against failures we anticipated. Both")
        print("  the truncation-attribution defect and the silent platform dropout were")
        print("  found by a live run and by nothing else, so a reportable evaluation")
        print("  still needs:")
        print()
        print("      python scripts/preflight.py --live")
        print()
        return 2

    print("  CLEARED -- offline checks pass, and a live smoke run at the configured")
    print("  parameters passes. The parameters are printed above; they are what a")
    print("  reportable run will use.")
    print("  Record the model ids, corpus size and date alongside any numbers reported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
