# Floor ranker with the query rewrite, 2026-10-07

The hosted (DeepInfra, fp8) run behind the floor-ranker paragraph in Limitations: LLaMA 3.1-8B with candidate injection and the short JSON rewrite, ranking candidates with the lexical ranker or the frequency-floor ranker, three passes each, every pass run as two halves.

| File | What it is |
|---|---|
| `floor_hosted_lab_executed.ipynb` | The executed notebook, with its cell outputs. Keys were entered by `getpass` and show only as masked prompts |
| `BUNDLE_hosted.json` | Harness stamp of the run (dev commit `b3153b8`) |

Where the data is, at the top of `outputs/`:

- the raw halves, `floorhosted_20261007_v5fp8_i_a2_s1_{floor,lexical}_r{1,2,3}_h{1,2}/`, with their launcher logs `lane_floorhosted_*.log`;
- the stitched passes, `floorhosted_20261007_v5fp8_i_a2_s1_{floor,lexical}_r{1,2,3}/`;
- the merged cells and the comparisons, in `merged_floor_20261007/` (`hosted_llama_a2_s1_floor/`, `hosted_llama_a2_s1_lexical/`, `compare_*.txt`).

The local RTX 5090 half of this run backs no number in the paper and is not included.
