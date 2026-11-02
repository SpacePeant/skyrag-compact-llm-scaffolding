#!/usr/bin/env python
"""Merge several single-lane run directories into one, renumbering `repeat`.

**The problem.** A 3-repeat sweep run sequentially takes ~76 h. Split across
parallel lanes it takes a third of that, but every lane stamps `repeat = 1`,
because each lane is its own `skyrag eval` invocation and nothing tells it that
two siblings exist. `aggregate_arms.load` then keys rows on
`(query_id, repeat)`, finds duplicates, and exits:

    N rows collapse to M (query_id, repeat) keys -- duplicate cells that are
    not distinguished by the `repeat` column. Refusing to guess which row wins.

That refusal is correct and must stay. This script supplies the distinction it
is asking for, before the aggregator ever sees the rows.

**What it will not do.** Two quite different things get called "splitting a
sweep across lanes", and only one of them is a repeat split:

* **Repeat split** -- every lane runs *the same queries*. Lane 2 is a second
  independent pass. Renumbering is exactly right.
* **Query split** -- lane 1 runs q01-q25, lane 2 runs q26-q50. This is *one*
  pass over a partitioned query set. The lanes do not collide on
  `(query_id, repeat)` at all, so the aggregator already accepts them
  concatenated, and renumbering would state that 50 single-pass queries are
  25 queries measured twice. Every per-query mean would then be computed over
  a repeat axis that does not exist, and `_repeats_per_cell` would report 2,
  dropping the single-pass caveat from the TOST verdict.

So the query sets decide, and the script refuses anything it cannot read as
unambiguously one or the other. A partial overlap is refused by name rather
than resolved, because the two readings imply different denominators and
picking one silently is the failure this whole file exists to prevent.

**Lanes must be the same system.** Repeats of a cell are repeats of *one*
treatment. If two lanes disagree on arm, model, sweep version, schema variant,
candidate injection or serving precision, they are different cells, and
stacking them under one `repeat` axis fabricates a within-cell variance out of
a between-cell difference. Refused, with the field named.

**Nothing is recomputed.** Rows are copied verbatim but for `repeat`, and the
original value is preserved alongside in `lane_original_repeat` so the
renumbering is reversible from the artifact alone.

Usage:

    python scripts/merge_lane_repeats.py \
        outputs/lane_a outputs/lane_b outputs/lane_c \
        --output outputs/merged_3rep

Lane order on the command line is the repeat order. It is not sorted by name
or by mtime: those are not stable across machines, and a merge that renumbers
differently on a rerun is not reproducible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# The fields that define *which system ran*, as opposed to when it ran or how
# long it took. Two lanes may differ freely on `run_id`, `run_timestamp` and
# every timing column -- that is what makes them separate lanes. They may not
# differ on any of these, because then they are not repeats of one cell.
#
# It starts from the fields `Cell.stamp` prints in `aggregate_arms.py` and adds
# every other field that has ever been a *measured contrast* in this project.
# Each addition below was found by running this script over the real `outputs/`
# tree and seeing a pair it should have refused come back clean:
#
#   `serving_precision` -- fp8 against bf16 is the coreweave/deepinfra contrast.
#   `candidate_limit`   -- `scratch_inject100_10q` and `scratch_inject1000_10q`
#                          match on every other field and are the 2026-09-21
#                          pool-size pilot, where k=1000 collapsed retrieval
#                          172.4 -> 44.4. Merged as repeats, that finding would
#                          have been averaged away into one cell's variance.
#   `provider`          -- a smoke run and its `_deepinfra` twin can
#                          differ only here: one is the echo
#                          backend. Stacking a run that called nothing with one
#                          that called a model is not a repeat pair.
#
# `seed` is deliberately absent: `--seed-per-repeat` varies it *by design*, so
# refusing on it would refuse the exact case this script is built for.
TREATMENT_FIELDS = (
    # what answers
    "arm",
    "keyword_model",
    "backend",
    "provider",
    "serving_precision",
    # how it is asked
    "sweep_version",
    "schema_prompt_variant",
    "candidate_injection",
    "candidate_limit",
    # C17, added 2026-09-29. The ranker decides WHICH candidates reach the
    # model, so two lanes that differ on it are not repeats of one system --
    # they are the two halves of E3, whose entire question is floor against
    # lexical. It was omitted when injection was added, and the omission is
    # silent in the worst way: the merged artifact loads cleanly through
    # `aggregate_arms` and reports a repeat axis that was never run.
    #
    # The corpus carries four distinct stamped values (lexical 550 rows,
    # floor 203, freq 50, unstamped 659). Two lanes that BOTH predate the field
    # are now refused, exactly as they are for every other treatment here: the
    # artifacts do not record whether they match, and this script does not
    # guess. The refusal names the field and says to confirm from the task logs.
    # That cost is real but bounded -- every lane in the 2026-09-23 factorial
    # stamps it (`lexical` on the injection cells, a RECORDED empty string on
    # the static ones, which `_stamp` already distinguishes from missing).
    "candidate_ranker",
    # 2026-10-01. The Stage 1 rewrite feeds selection, so lanes that differ on
    # it are the decomposition's separate cells, not repeats of one.
    "expansion_mode",
    "temperature",
    "max_tokens",
    # what it retrieves over
    "embedding_model",
    "context_render",
    "max_keywords",
    "per_keyword",
    "max_context_courses",
)


def _repeat_of(row: dict[str, Any]) -> int:
    """The row's repeat under `aggregate_arms.load`'s rule.

    `int(r.get("repeat") or 1)` -- absent, `None` and `0` all mean the first
    pass. Duplicated here rather than imported because `aggregate_arms.py` is
    a script, not a module on the path; if that rule ever changes, this is the
    second place to change, and `test_merge_lane_repeats` pins the two
    together.
    """
    return int(row.get("repeat") or 1)


def load_lane(path: Path) -> list[dict[str, Any]]:
    """Rows of one lane, with the same `query_id` filter the aggregator applies."""
    artifact = path / "benchmark_eval.json"
    if not artifact.exists():
        raise SystemExit(f"{path}: no benchmark_eval.json. Not a run directory.")
    data = json.loads(artifact.read_text(encoding="utf-8"))
    rows = [r for r in data if r.get("query_id")]
    if not rows:
        raise SystemExit(
            f"{path}: no rows carry a query_id. Nothing here can be keyed, so "
            "nothing here can be merged."
        )
    return rows


UNRECORDED = object()
"""A field no row in the lane carries. Distinct from a recorded empty string.

The two are not the same answer and must not compare equal. `provider` is
genuinely `""` on the echo backend and `schema_prompt_variant` is genuinely
`""` on the unschema'd arms -- those are measurements. A *missing* key is the
run declining to say, which is what every artifact written before the
2026-09-08 provenance fix does for `candidate_injection`.

Folding the two together is not hypothetical. `arm3_A_spec_only_20260831` and
`eval_20260908_arm3_inject_v1_50q` both omit `candidate_injection` and
`candidate_limit`, so under a `.get(field, "")` fold they compare equal on
every treatment field and merge cleanly -- while their directory names say one
injected candidates and the other did not. The artifacts cannot settle it.
"""


def treatment_of(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The lane's treatment stamp, or a refusal if the lane is not one cell.

    A lane that is internally mixed (a `--arm 2,3` sweep, say) cannot be given
    a single repeat offset, because its two arms are different cells that each
    need their own repeat axis.
    """
    stamp: dict[str, Any] = {}
    for field in TREATMENT_FIELDS:
        if not any(field in r for r in rows):
            stamp[field] = UNRECORDED
            continue
        values = {str(r.get(field, "")) for r in rows}
        if len(values) > 1:
            raise SystemExit(
                f"lane is not a single cell: it carries {len(values)} distinct "
                f"values of {field!r} ({', '.join(sorted(values))}). Merge one "
                "cell at a time -- a mixed lane needs a repeat axis per arm, "
                "not one shared offset."
            )
        stamp[field] = values.pop()
    return stamp


def _keys(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {(str(r["query_id"]), _repeat_of(r)) for r in rows}


def classify_split(lanes: list[tuple[Path, list[dict[str, Any]]]]) -> None:
    """Refuse unless every lane covers exactly the same queries.

    See the module docstring: identical query sets are a repeat split and
    renumber correctly; disjoint ones are a query split that must be
    concatenated instead; anything between is ambiguous.
    """
    first_path, first_rows = lanes[0]
    reference = {str(r["query_id"]) for r in first_rows}

    for path, rows in lanes[1:]:
        ids = {str(r["query_id"]) for r in rows}
        if ids == reference:
            continue
        shared = ids & reference
        if not shared:
            raise SystemExit(
                f"{path.name} and {first_path.name} share no query_id.\n"
                "  These lanes are a QUERY split, not a repeat split: together "
                "they are one pass over a partitioned query set.\n"
                "  Renumbering would claim a repeat axis that was never run. "
                "They do not collide on (query_id, repeat), so pass the lane "
                "directories to the aggregator as they are."
            )
        only_ref = sorted(reference - ids)
        only_this = sorted(ids - reference)
        raise SystemExit(
            f"{path.name} and {first_path.name} overlap on {len(shared)} "
            f"query_id(s) but do not match.\n"
            f"  Only in {first_path.name}: {', '.join(only_ref) or '-'}\n"
            f"  Only in {path.name}: {', '.join(only_this) or '-'}\n"
            "  This is neither a clean repeat split nor a clean query split, "
            "and the two readings give different denominators. Resolve it by "
            "hand; this script will not pick one."
        )


def merge(
    lanes: list[tuple[Path, list[dict[str, Any]]]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Renumbered rows, plus a human-readable line per lane for the report.

    Lane `i` is offset by the highest repeat assigned so far, so a lane that
    already carries repeats 1-3 keeps its internal structure and the next lane
    starts at 4. Offsetting rather than assigning one number per lane is what
    makes the script correct for lanes that are themselves multi-repeat.
    """
    merged: list[dict[str, Any]] = []
    notes: list[str] = []
    offset = 0

    for path, rows in lanes:
        lane_repeats = sorted({_repeat_of(r) for r in rows})
        for row in rows:
            original = _repeat_of(row)
            new_row = dict(row)
            new_row["repeat"] = offset + original
            # Provenance, so the renumbering is reversible from the artifact
            # alone. A merged run that cannot say where a row came from is a
            # run whose repeat axis has to be taken on trust.
            new_row["lane_source"] = path.name
            new_row["lane_original_repeat"] = original
            merged.append(new_row)
        assigned = [offset + r for r in lane_repeats]
        notes.append(
            f"  {path.name}: {len(rows)} rows, repeat "
            f"{lane_repeats} -> {assigned}"
        )
        offset += max(lane_repeats)

    return merged, notes


def verify(merged: list[dict[str, Any]]) -> None:
    """The check the aggregator will perform, performed here where it is fixable."""
    keys = _keys(merged)
    if len(keys) != len(merged):
        raise SystemExit(
            f"merge produced {len(merged)} rows but only {len(keys)} distinct "
            "(query_id, repeat) keys. The aggregator would refuse this. This "
            "is a bug in the merge, not in the input."
        )


def write(out_dir: Path, merged: list[dict[str, Any]]) -> None:
    """Write `benchmark_eval.json` and `.csv`, mirroring `eval.export_benchmark_*`.

    The CSV goes through pandas with the union of the row keys as columns, in
    first-seen order, matching how `metrics_to_dataframe` lays a run out. Both
    files are written because the two are read by different tools -- the
    aggregator takes JSON, the UI and `--resume` take CSV -- and a merged run
    that carries only one of them is a run that works until someone reaches
    for the other.

    **Written with `newline=""`, which `export_benchmark_csv` does not.** That
    function does `path.write_text(df.to_csv(...))`; `to_csv` already emits
    `\r\n`, and text-mode `write_text` translates the `\n` again, so every
    committed `benchmark_eval.csv` on this machine carries `\r\r\n` and a blank
    line between every row. Readers tolerate it, which is why it has gone
    unnoticed. It is not reproduced here: a merged artifact should not inherit
    a defect to look authentic. `eval.py` is not this task's to edit, so the
    divergence is deliberate and recorded rather than silently matched.
    """
    artifact = out_dir / "benchmark_eval.json"
    if artifact.exists():
        raise SystemExit(
            f"{artifact} already exists. Refusing to overwrite a run artifact; "
            "choose an --output directory that does not hold one."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    columns: list[str] = []
    for row in merged:
        for key in row:
            if key not in columns:
                columns.append(key)

    import pandas as pd

    frame = pd.DataFrame(merged, columns=columns)
    with open(out_dir / "benchmark_eval.csv", "w", newline="", encoding="utf-8") as handle:
        handle.write(frame.to_csv(index=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "lanes",
        nargs="+",
        type=Path,
        help="run directories, in the order their repeats should be numbered",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="directory to write the merged run into; must not already hold one",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the renumbering and the checks, write nothing",
    )
    args = parser.parse_args(argv)

    if len(args.lanes) < 2:
        raise SystemExit("merge needs at least two lanes; one lane is already a run.")

    lanes = [(path, load_lane(path)) for path in args.lanes]

    stamps = [(path, treatment_of(rows)) for path, rows in lanes]
    reference_path, reference = stamps[0]
    # C17, narrowed. `candidate_ranker` ranks INJECTED candidates and nothing
    # else -- `eval.py` refuses `--candidate-ranker` outright when no cell in a
    # sweep enables injection, on the grounds that "no ranker will run". So when
    # every lane RECORDS injection as off, the ranker provably did not act and
    # cannot make two lanes different cells.
    #
    # Without this, the 70B scale control could not be merged at all: Arm 1 never
    # injects, pass 1 predates the field, and passes 2-4 stamp the no-ranker
    # empty string. The guard would have blocked a legitimate merge while
    # protecting nothing.
    #
    # The condition is deliberately strict. Every row of every lane must record
    # `candidate_injection` as a real False. An absent or None value means the
    # artifact does not say, and a run that does not say is not a run at the
    # default -- so the guard stays armed, exactly as it does for every other
    # field here.
    def _injection_recorded_off(rows: list[dict[str, Any]]) -> bool:
        return bool(rows) and all(
            "candidate_injection" in r
            and r.get("candidate_injection") is not None
            and bool(r.get("candidate_injection")) is False
            for r in rows
        )

    ranker_is_inert = all(_injection_recorded_off(rows) for _, rows in lanes)

    for path, stamp in stamps[1:]:
        for field in TREATMENT_FIELDS:
            if field == "candidate_ranker" and ranker_is_inert:
                continue
            mine, theirs = reference[field], stamp[field]
            # Unlike every other field here, absence on BOTH sides is an answer:
            # there was no way to override the rewrite before the column existed,
            # so each lane ran its arm's own, and `arm` is compared above. One
            # stamped lane against one unstamped still falls through to refusal.
            if field == "expansion_mode" and mine is UNRECORDED and theirs is UNRECORDED:
                continue
            if mine is not UNRECORDED and mine == theirs:
                continue

            # **Absent is unknown, never equal.** Three cases, three fixes, so
            # they are reported as three messages rather than one. This mirrors
            # `aggregate_arms.load`'s reasoning on `candidate_injection`: a row
            # that does not say is not a row that ran at the default.
            unrecorded = [
                p.name
                for p, s in ((reference_path, reference), (path, stamp))
                if s[field] is UNRECORDED
            ]
            if len(unrecorded) == 2:
                raise SystemExit(
                    f"neither {reference_path.name} nor {path.name} stamps "
                    f"{field!r}.\n"
                    "  Whether they are the same cell is therefore not "
                    "recorded in either artifact, and this script will not "
                    "assume they are. These runs predate the provenance stamp; "
                    "take the answer from their task logs and merge by hand if "
                    "it says they match."
                )
            if unrecorded:
                blank = unrecorded[0]
                other, value = (
                    (path.name, theirs) if blank == reference_path.name else (reference_path.name, mine)
                )
                raise SystemExit(
                    f"{blank} does not stamp {field!r} at all, and {other} "
                    f"records {value!r}.\n"
                    "  An unstamped run is not a run at the default. Whether "
                    "the lanes match on this is not in the artifact -- take it "
                    "from the run's task log, not from the export."
                )
            raise SystemExit(
                f"lanes disagree on {field!r}: {reference_path.name} has "
                f"{mine!r}, {path.name} has {theirs!r}.\n"
                "  These are different cells, not repeats of one. Stacking "
                "them on a shared repeat axis would report a between-cell "
                "difference as within-cell variance."
            )

    classify_split(lanes)
    merged, notes = merge(lanes)
    verify(merged)

    print(f"Merging {len(lanes)} lanes -> {len(merged)} rows")
    for note in notes:
        print(note)
    shown = " ".join(
        f"{k}={'<unrecorded>' if v is UNRECORDED else repr(v)}" for k, v in reference.items()
    )
    print(f"  treatment: {shown}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    write(args.output, merged)
    print(f"\nWrote {args.output}/benchmark_eval.json and .csv")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
