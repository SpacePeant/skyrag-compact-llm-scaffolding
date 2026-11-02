"""Which model filled the empty keyword tags? Regenerate a sample and compare.

Reads the shipped and cleaned catalogues (read-only), takes all 25 filled edX
records and 35 filled Coursera records (fixed seed), and re-extracts their tags
with each candidate model through the project's own extractor at the original
settings (temperature 0, seed 34, n=5). Prints exact-set and overlap matches.
"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
from skyrag.config import Config  # noqa: E402
from skyrag.data.keywords import KEYWORD_FIELD, _source_text, extract_keywords_for_text  # noqa: E402
from skyrag.data.schema import split_keywords  # noqa: E402
from skyrag.llm import get_backend  # noqa: E402
from skyrag.llm.base import GenerationParams  # noqa: E402

MODELS = sys.argv[2].split(",")
PAIRS = {
    "coursera": ("CourseraDataset_clean.json", "CourseraDataset_EN_with_keywords.json"),
    "edx": ("EdxDataset-Clean.json", "EdxDataset_with_keywords.json"),
}
TAKE = {"coursera": 35, "edx": 25}

sample = []
for plat, (clean, shipped) in PAIRS.items():
    a = json.loads((ROOT / "documents" / clean).read_text(encoding="utf-8"))
    b = json.loads((ROOT / "documents" / shipped).read_text(encoding="utf-8"))
    f = KEYWORD_FIELD[plat]
    filled = [(x, y[f]) for x, y in zip(a, b) if not split_keywords(x.get(f)) and split_keywords(y.get(f))]
    random.Random(34).shuffle(filled)
    sample += [(plat, x, shipped_tags) for x, shipped_tags in filled[: TAKE[plat]]]

backend = get_backend(Config())
params = GenerationParams(temperature=0.0, max_tokens=128, seed=34)

for model in MODELS:
    def one(item):
        plat, rec, shipped_tags = item
        got = extract_keywords_for_text(backend, _source_text(rec, plat), model=model, n=5, params=params)
        return plat, split_keywords(shipped_tags), got

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, sample))
    exact = sum(1 for _, s, g in res if s == g)
    sameset = sum(1 for _, s, g in res if set(s) == set(g))
    jac = [len(set(s) & set(g)) / len(set(s) | set(g)) if (s or g) else 1.0 for _, s, g in res]
    empty = sum(1 for _, _, g in res if not g)
    print(f"\n{model}: n={len(res)}  identical order {exact}  same set {sameset}  "
          f"mean Jaccard {sum(jac) / len(jac):.3f}  empty replies {empty}")
    for plat in PAIRS:
        sub = [j for (p, _, _), j in zip(res, jac) if p == plat]
        print(f"  {plat}: mean Jaccard {sum(sub) / len(sub):.3f} over {len(sub)}")
    for _, s, g in res[:4]:
        print(f"    shipped {s}\n    regen   {g}")
