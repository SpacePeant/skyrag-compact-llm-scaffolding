r"""`skyrag show` -- print the learning path a run produced.

The path lives in the `learning_path` field: plain text in `benchmark_eval.json`,
JSON-encoded inside a CSV cell, which is why this reads the JSON.

This exists because the documented alternative (RUNNING.md A.3) is a
`python -c "import json,sys; ..."` one-liner with three sharp edges: it is written
with a bash `\` continuation that PowerShell forwards to Python as `sys.argv[1]`,
it says bare `python` rather than `uv run python`, and it invites `len()` on
`cited_course_ids`, which is a ';'-delimited STRING -- so the length is a character
count that looks like a course count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = PROJECT_ROOT / "outputs"


def latest_export() -> Path:
    """The newest export that HAS rows, not merely the newest directory.

    `outputs/` carries directories that were named and then never written to, and
    those sort first by mtime. Selecting on the presence of the export file means
    "the last thing I ran" cannot silently resolve to something that never ran.
    """
    exports = sorted(
        OUTPUTS.glob("*/benchmark_eval.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not exports:
        raise SystemExit(
            f"skyrag show: no benchmark_eval.json under {OUTPUTS}. "
            "Run an eval first, or name a directory."
        )
    return exports[0]


def resolve_export(target: str | None) -> Path:
    if target is None:
        return latest_export()
    path = Path(target)
    if path.is_dir():
        path = path / "benchmark_eval.json"
    if not path.exists():
        raise SystemExit(f"skyrag show: no such export: {path}")
    return path


def describe(row: dict, index: int, total: int) -> str:
    """The fields that say whether the path below can be trusted.

    Printed BEFORE the path, because a path reads as authoritative whether or not
    the courses in it exist.
    """
    cited_ids = row.get("cited_course_ids") or ""
    hallucinated = row.get("hallucinated_course_ids") or ""
    # A ';'-delimited string, so split -- never len().
    n_invented = len([c for c in hallucinated.split(";") if c.strip()])
    grounding = row.get("grounding_accuracy")
    lines = [
        f"row {index + 1} of {total}",
        f"  query     : {row.get('query')}",
        f"  model     : {row.get('model')}  arm={row.get('arm') or '-'}  "
        f"status={row.get('status')}",
        f"  grounding : {grounding if grounding is not None else 'n/a (cited nothing)'}"
        f"   cited {row.get('cited_count')} of {row.get('retrieved_count')} retrieved",
    ]
    if n_invented:
        lines.append(
            f"  INVENTED  : {n_invented} cited course(s) do not exist -- {hallucinated}"
        )
    if not cited_ids.strip():
        lines.append(
            "  NOTE      : nothing cited. An --backend echo run looks exactly like this."
        )
    return "\n".join(lines)


def run_show_cli(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(
        prog="skyrag show",
        description="Print the learning path a run produced.",
    )
    ap.add_argument(
        "run", nargs="?", default=None,
        help="run directory or benchmark_eval.json. Default: the newest run that has rows.",
    )
    ap.add_argument("--row", type=int, default=1,
                    help="1-indexed row to show (default: 1).")
    ap.add_argument("--all", action="store_true", help="show every row.")
    ap.add_argument("--list", action="store_true",
                    help="list runs that have rows, newest first, and exit.")
    args = ap.parse_args(argv)

    if args.list:
        exports = sorted(
            OUTPUTS.glob("*/benchmark_eval.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not exports:
            raise SystemExit(f"skyrag show: no runs with rows under {OUTPUTS}.")
        for export in exports[:25]:
            n = len(json.loads(export.read_text(encoding="utf-8")))
            print(f"  {n:>4} row(s)  {export.parent.name}")
        return

    export = resolve_export(args.run)
    rows = json.loads(export.read_text(encoding="utf-8"))
    if not rows:
        raise SystemExit(f"skyrag show: {export} has no rows.")

    print(f"# {export.parent.name}\n")
    wanted = range(len(rows)) if args.all else [args.row - 1]
    for i in wanted:
        if not 0 <= i < len(rows):
            raise SystemExit(f"skyrag show: --row {i + 1} out of range (1..{len(rows)}).")
        row = rows[i]
        print(describe(row, i, len(rows)))
        print()
        print(row.get("learning_path") or "(no learning path -- see NOTE above)")
        print()
