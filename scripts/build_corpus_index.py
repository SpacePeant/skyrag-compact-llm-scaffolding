#!/usr/bin/env python
"""Build the keyword tags and course embeddings from the cleaned catalogues.

The second half of the corpus rebuild. `prepare_corpus.py` fetches the two
pinned upstreams and writes `documents/*_clean.json`; this script takes those to
what the evaluation harness reads:

    keywords   documents/CourseraDataset_clean.json -> documents/CourseraDataset_EN_with_keywords.json
               documents/EdxDataset-Clean.json      -> documents/EdxDataset_with_keywords.json
    embed      documents/*_with_keywords.json       -> embeddings/{coursera,edx}.{npy,jsonl,meta.json}

**Why it exists.** Both stages were reachable only through `legacy/run_pipeline.py`,
which does not ship and no longer runs against the current `EmbeddingStore` API
(found on a fresh-clone rebuild, 2026-10-20). Without them the rebuild stopped at
the cleaned catalogues and the harness refused to start.

**Keywords.** Only courses that arrived with no tags get an LLM call; platform
tags are kept as they are. The study's tags were filled by LLaMA 3.1-8B through
OpenRouter with no provider pin (fingerprinted 2026-10-20 by
`fingerprint_keyword_fill_model.py`), so that is the default here. Hosted output
varies between identical requests, so a rebuild approximates the filled tags; it
does not reproduce them. The model used is printed and written beside the output
as `<output>.provenance.json`, because the study's own run recorded none.

**Embeddings.** `all-mpnet-base-v2` unless `SKYRAG_EMBEDDING_MODEL` says
otherwise, one vector per course over the text the course schema renders.

Usage:

    python scripts/build_corpus_index.py                       # both stages
    python scripts/build_corpus_index.py --only keywords
    python scripts/build_corpus_index.py --only embed --force
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skyrag.config import Config  # noqa: E402
from skyrag.data.adapters import DEFAULT_FILENAMES, load_platform  # noqa: E402
from skyrag.data.keywords import KEYWORD_FIELD, generate_keywords  # noqa: E402
from skyrag.data.schema import split_keywords  # noqa: E402
from skyrag.embedding import Embedder, EmbeddingStore  # noqa: E402
from skyrag.llm import get_backend  # noqa: E402

CLEAN_FILENAMES = {"coursera": "CourseraDataset_clean.json", "edx": "EdxDataset-Clean.json"}
STUDY_KEYWORD_MODEL = "meta-llama/llama-3.1-8b-instruct"
STUDY_FILLED = {"coursera": 1952, "edx": 25}


def stage_keywords(config: Config, platforms: list[str], model: str, force: bool) -> int:
    for platform in platforms:
        source = config.paths.documents / CLEAN_FILENAMES[platform]
        target = config.paths.documents / DEFAULT_FILENAMES[platform]
        if not source.exists():
            print(f"  {platform}: {source.name} not found, run scripts/prepare_corpus.py --fetch --clean first")
            return 1
        if target.exists() and not force:
            print(f"  {platform}: {target.name} exists, not rebuilding (pass --force)")
            continue

        records = json.loads(source.read_text(encoding="utf-8"))
        field = KEYWORD_FIELD[platform]
        todo = sum(1 for r in records if not split_keywords(r.get(field)))
        note = "" if todo == STUDY_FILLED[platform] else f"  (the study filled {STUDY_FILLED[platform]})"
        print(f"  {platform}: {len(records)} courses, {todo} without tags -> {todo} calls to {model}{note}")

        generate_keywords(get_backend(config), records, platform,
                          output_path=target, model=model, overwrite=force)

        provider = os.environ.get("SKYRAG_OPENROUTER_PROVIDER") or "unpinned"
        Path(f"{target}.provenance.json").write_text(json.dumps({
            "keyword_model": model,
            "provider": provider,
            "filled": todo,
            "source": source.name,
            "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=1), encoding="utf-8")
        empty = sum(1 for r in json.loads(target.read_text(encoding="utf-8")) if not split_keywords(r.get(field)))
        print(f"    wrote {target.name}; {empty} course(s) still without tags")
    return 0


def stage_embed(config: Config, platforms: list[str], force: bool) -> int:
    embedder = Embedder(config.embedding)
    model_name = config.embedding.model_name
    for platform in platforms:
        path = config.paths.documents / DEFAULT_FILENAMES[platform]
        if not path.exists():
            print(f"  {platform}: {path.name} not found, run the keywords stage first")
            return 1
        if EmbeddingStore.exists(config.paths.embeddings, platform, model_name=model_name) and not force:
            print(f"  {platform}: embeddings from {model_name} exist, not rebuilding (pass --force)")
            continue
        courses = load_platform(platform, path)
        print(f"  {platform}: embedding {len(courses)} courses with {model_name} "
              f"on {config.embedding.resolved_device()}")
        embedder.embed_courses(courses).save(config.paths.embeddings, platform)
        print(f"    wrote embeddings/{platform}.npy")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["keywords", "embed"], help="run one stage")
    ap.add_argument("--platforms", nargs="+", default=list(CLEAN_FILENAMES), choices=list(CLEAN_FILENAMES))
    ap.add_argument("--keyword-model", default=STUDY_KEYWORD_MODEL,
                    help=f"model that fills empty tags (default: the study's, {STUDY_KEYWORD_MODEL})")
    ap.add_argument("--force", action="store_true", help="rebuild outputs that already exist")
    args = ap.parse_args(argv)

    config = Config()
    config.paths.ensure()
    if args.only in (None, "keywords"):
        print("\nkeywords")
        if stage_keywords(config, args.platforms, args.keyword_model, args.force):
            return 1
    if args.only in (None, "embed"):
        print("\nembed")
        if stage_embed(config, args.platforms, args.force):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
