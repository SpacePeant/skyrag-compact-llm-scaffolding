# skyrag-compact-llm-scaffolding

Code and benchmark for the study **"When Scaffolding Beats Scale: A Compact Language Model Against an Unassisted Larger Model in Catalogue-Grounded Course Recommendation"** (manuscript under review).

This is an independent study that builds on the SKYRAG framework of Soekamto et al. (2025). It is not the official SKYRAG repository and is not maintained by the original authors.

The evaluation pipeline, the corpus build, the analysis scripts, the tests, the 50-query benchmark and the per-query results of every run the paper reports are here.

## What the study asks

A course recommender built on a language model must pick search keywords that exist in the course catalogue. Large models do this more reliably but cost more; compact models are cheap but invent keywords the catalogue does not contain. The study asks whether scaffolding around a compact model (candidate keywords injected into the prompt, a shorter query rewrite and a JSON schema on the output) can make it as faithful to the catalogue as a much larger model, and which part of that scaffolding is responsible.

## Reproducing the paper, step by step

There are three levels. Each one builds on the steps before it, so stop at the level you need.

| Level | What you get | Steps | API key | Cost | Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| A. Check the numbers | Every table, figure and inline number recomputed from the shipped per-query results | 1 to 4 | No | Free | About 10 minutes |
| B. Quick live check | The pipeline run live on 5 queries in every condition the paper reports | 1 to 8 | Yes | About $0.15 | A few hours, mostly the corpus build |
| C. Full rerun | Every hosted run at full size (50 queries, three passes) | 1 to 7, then 9 and 10 | Yes | About $6 to $7 | Several hours per run |

All commands are run from the repository root. Commands that set shell variables (`C=...`, `$C`) need a bash shell: on Windows use Git Bash or WSL, not PowerShell.

### Step 1. Install the prerequisites

* Python 3.12 or later.
* [uv](https://docs.astral.sh/uv/), which installs the exact package versions in `uv.lock`.
* git.
* For levels B and C only: an [OpenRouter](https://openrouter.ai/keys) API key with credit, and optionally a free Hugging Face token (it speeds up the corpus download).

### Step 2. Clone and install

```bash
git clone https://github.com/SpacePeant/skyrag-compact-llm-scaffolding.git
cd skyrag-compact-llm-scaffolding
uv sync
```

`pip install -e .` also works, but installs whichever versions are current.

### Step 3. Run the tests

```bash
uv run pytest -q
```

Every test passes on a fresh clone, before any corpus is built; the few that need files a clone lacks skip with a reason.

### Step 4. Check the paper's numbers (level A)

These commands read only the per-query results shipped in `outputs/`. They need no API key and no corpus.

```bash
# Recompute every number the paper reports and flag any difference
uv run python scripts/reproduce_paper.py

# Regenerate the key figures and compare them with the stored capture (no output means every number agrees)
uv run python scripts/key_figures.py | diff --strip-trailing-cr - outputs/key_figures_20261027.txt

# Redraw Figures 1 to 3 into figures/, and check the values they plot
uv run python scripts/plot_figures.py
uv run python scripts/plot_figures.py --check
```

`reproduce_paper.py` ends with a line such as `320 match, 0 drift, 0 missing`. Each claim is recomputed from the rows, traced to a stored analysis output, or checked against the code constant that sets it. Its last step, which lists any number in the paper that no claim covers, needs the manuscript (`--manuscript PATH`); the manuscript is not in this repository while it is under review, so that step is skipped.

Every reported number is computed by [`scripts/aggregate_arms.py`](scripts/aggregate_arms.py) or a script built on it. It averages each query's passes before pairing, so a three-pass run counts as 50 paired observations, not 150. Two examples on the shipped results:

```bash
# Describe one run: the 8B with candidate injection and the short JSON rewrite
uv run python scripts/aggregate_arms.py outputs/merged_s1dec_20261001/i_a2_s1/

# Paired contrast: Wilcoxon test, Cohen's d_z, bootstrap interval and TOST at +/-0.05
# (the rewrite effect under injection: 0.3860 -> 0.1803)
uv run python scripts/aggregate_arms.py \
  --compare "default=outputs/merged_s1dec_20261001/i_a2:2" \
  --compare "short_json=outputs/merged_s1dec_20261001/i_a2_s1:2"
```

Each `--compare` is `LABEL=path:arm`; give it once per side. Use relative paths, because a Windows drive letter is read as the `:arm` separator.

Which script produces which result:

| Result | Script | Stored output |
| :--- | :--- | :--- |
| Every number in the paper, checked | `scripts/reproduce_paper.py` | printed |
| Scale gap (8B against 70B), the Qwen rows, the schema arm on both payloads, factorial means and contrasts | `scripts/key_figures.py` | `outputs/key_figures_20261027.txt` |
| The factorial table: invalid rate, usable-output rate and silent runs per cell | `scripts/build_stamped_factorial_table.py` | printed |
| Factorial main effects and Holm-adjusted contrasts | `scripts/analyze_factorial.py` | `outputs/factorial_r3_invention_20260930.txt`, `outputs/factorial_r3_retrieved_20260930.txt` |
| Figures 1 to 3 | `scripts/plot_figures.py` | `figures/` (regenerated) |
| Stage decomposition, local replication on the RTX 5090, precision | `scripts/local_vllm/report_5090_runs.py` | `outputs/local_vllm_5090_summary/5090_summary.md` |
| Share of invented keywords copied from the rewrite | `scripts/local_vllm/copy_share_decomp.py` | `outputs/merged_s1dec_20261001/copy_share.txt` |
| Paired contrasts between cells (decomposition, array payload, precision, floor ranker, hosted 70B, fixed candidates) | `scripts/aggregate_arms.py --compare` | `compare_*.txt` in each `merged_*` folder; `outputs/a7fix_c70h_analysis_20261022/` |
| Why schema calls fail: budget overruns and cap violations (the "220 of the 223 failed calls") | `scripts/overrun_cause.py` | `outputs/merged_s1decA_20261002/overrun_cause.txt` |
| Cost per query and grounding of the final answer | `scripts/cost_and_answer.py` | `outputs/cost_and_answer/` |
| Table 8, cost by stage and per answered run | `scripts/cost_table.py` | `outputs/cost_table/` |
| Sample size for an equivalence test ("115 to 205 query pairs") | `scripts/power_tost.py` | printed |
| Robustness checks on the invalid-keyword metric | `scripts/selection_bias_test.py`, `scripts/bound_transcription_error.py` | printed |
| Run conditions per run folder | `scripts/describe_provenance.py` | `PROVENANCE.md` |

`scripts/merge_lane_repeats.py` merges passes run as separate directories into one multi-pass cell; the `merged_*` and `*_m3` folders were built with it.

### Step 5. Set the API key (levels B and C)

```bash
cp .env.example .env
```

Open `.env` and set `OPENROUTER_API_KEY`. `.env` is read automatically and is excluded from git. It already sets what the study used: temperature 0, seed 34, and the provider pinned to DeepInfra (`SKYRAG_OPENROUTER_PROVIDER=deepinfra`), which serves both LLaMA 3.1 models at fp8. Keep these to match the study's serving conditions. If you have a Hugging Face token, add it as `HF_TOKEN`.

### Step 6. Build the corpus

The course catalogues are not redistributed (see [What this repository does not contain](#what-this-repository-does-not-contain)), so they are rebuilt from their public sources:

```bash
uv run python scripts/prepare_corpus.py --fetch --clean --as-published   # download and clean both catalogues
uv run python scripts/build_corpus_index.py --only keywords               # fill missing keyword tags (calls the 8B, a few cents)
uv run python scripts/build_corpus_index.py --only embed                  # embed every course (runs locally)
uv run python scripts/prepare_corpus.py --verify                          # check the result
```

`--as-published` gives the 3,451-course edX catalogue the study measured; without it the rebuild gives the complete 3,477. The edX download is 5,203 small files and is much faster with `HF_TOKEN` set. [`CORPUS.md`](CORPUS.md) explains every step, the counts at each stage, and what a rebuild does and does not reproduce exactly. In short, the catalogue records rebuild exactly, but about 1,977 keyword tags are filled by a hosted model and come back only approximately, so a live rerun lands close to the reported rates, not on them.

Without a built corpus, a live `skyrag eval` stops with `REFUSED: 2 platform(s) have no usable embedding cache`, rather than produce numbers on a placeholder corpus.

### Step 7. Run the preflight gate

```bash
uv run python scripts/preflight.py --live
```

It runs the tests, a dataset audit and one live query (`benchmarks/queries_1q_smoke.jsonl`, under a cent), and exits 0 only when all of them pass. Do not start paid runs until it does.

### Step 8. Quick live check: 5 queries per condition (level B)

These commands run every condition the paper reports on the first five benchmark queries, one pass each. They show that the pipeline works end to end; five queries are far too few to reproduce the paper's rates, which is what Step 4 is for.

```bash
C="-Q benchmarks/queries_v1.jsonl --max-queries 5 -r 1 -b openrouter -P skyrag --serving-precision fp8 --sweep-version v5"
M8="-m meta-llama/llama-3.1-8b-instruct"
M70="-m meta-llama/llama-3.1-70b-instruct"
INJ="--candidate-injection --candidate-limit 100"

# Scale contrast: Arm 1 (70B, unassisted)
uv run skyrag eval $C --arm 1                                                 -o outputs/try/arm1_70b/

# Scaffolding factorial (8B): schema level x context condition
uv run skyrag eval $C $M8 --arm 2                                             -o outputs/try/s_l0/
uv run skyrag eval $C $M8 --arm 3                                             -o outputs/try/s_l1/
uv run skyrag eval $C $M8 --arm 3 --schema-exemplar                           -o outputs/try/s_l2/
uv run skyrag eval $C $M8 --arm 2 $INJ                                        -o outputs/try/i_l0/
uv run skyrag eval $C $M8 --arm 3 $INJ                                        -o outputs/try/i_l1/
uv run skyrag eval $C $M8 --arm 3 --schema-exemplar $INJ                      -o outputs/try/i_l2/

# Query rewrite (8B, injection): short rewrite (JSON) and its wording-only variant
uv run skyrag eval $C $M8 --arm 2 $INJ --expansion-mode schema                -o outputs/try/rw_short/
uv run skyrag eval $C $M8 --arm 2 $INJ --expansion-mode schema_wording        -o outputs/try/rw_wording/

# Candidates ranked against the original query instead of the rewrite
uv run skyrag eval $C $M8 --arm 2 $INJ --candidate-source query               -o outputs/try/cq_long/
uv run skyrag eval $C $M8 --arm 2 $INJ --candidate-source query --expansion-mode schema -o outputs/try/cq_short/

# The 70B with the same scaffolding
uv run skyrag eval $C $M70 --arm 2 $INJ                                       -o outputs/try/70b_inj/
uv run skyrag eval $C $M70 --arm 2 $INJ --expansion-mode schema               -o outputs/try/70b_inj_short/

# Qwen 2.5-7B, served by Phala (Arms 2 and 3 in one run)
Q="-m qwen/qwen-2.5-7b-instruct --serving-precision bf16"
SKYRAG_OPENROUTER_PROVIDER=phala uv run skyrag eval $C $Q --arm 2,3      -o outputs/try/qwen_static/
SKYRAG_OPENROUTER_PROVIDER=phala uv run skyrag eval $C $Q --arm 2,3 $INJ -o outputs/try/qwen_inject/
```

Each run writes `benchmark_eval.json` into its folder. To read one:

```bash
uv run python scripts/aggregate_arms.py outputs/try/i_l0/
```

To try a command for free first, replace `-b openrouter` with `-b echo` and add `--allow-synthetic-corpus`: the pipeline then runs on a placeholder model and corpus and calls nothing.

### Step 9. Full rerun (level C)

The same conditions at the study's size: all 50 queries, three passes (`-r 3`), the 70B unassisted with four. A 50-query, three-pass run can take several hours, depending on the provider's load; an interrupted run continues with `--resume`.

```bash
COMMON="-Q benchmarks/queries_v1.jsonl -b openrouter -P skyrag --serving-precision fp8 --sweep-version v5"
M8="-m meta-llama/llama-3.1-8b-instruct -r 3"
INJ="--candidate-injection --candidate-limit 100"

# Scale contrast: Arm 1 (70B, four passes); its 8B partner is s_l0 below
uv run skyrag eval $COMMON --arm 1 -r 4                                       -o outputs/full/arm1_70b/

# Scaffolding factorial (8B)
uv run skyrag eval $COMMON $M8 --arm 2                                        -o outputs/full/s_l0/
uv run skyrag eval $COMMON $M8 --arm 3                                        -o outputs/full/s_l1/
uv run skyrag eval $COMMON $M8 --arm 3 --schema-exemplar                      -o outputs/full/s_l2/
uv run skyrag eval $COMMON $M8 --arm 2 $INJ                                   -o outputs/full/i_l0/
uv run skyrag eval $COMMON $M8 --arm 3 $INJ                                   -o outputs/full/i_l1/
uv run skyrag eval $COMMON $M8 --arm 3 --schema-exemplar $INJ                 -o outputs/full/i_l2/

# Query rewrite (8B, injection)
uv run skyrag eval $COMMON $M8 --arm 2 $INJ --expansion-mode schema           -o outputs/full/rw_short/
uv run skyrag eval $COMMON $M8 --arm 2 $INJ --expansion-mode schema_wording   -o outputs/full/rw_wording/

# Qwen 2.5-7B (Phala; bf16 is the authors' assumption, see PROVENANCE.md)
Q="-m qwen/qwen-2.5-7b-instruct --serving-precision bf16 -r 3"
SKYRAG_OPENROUTER_PROVIDER=phala uv run skyrag eval $COMMON $Q --arm 2,3      -o outputs/full/qwen_static/
SKYRAG_OPENROUTER_PROVIDER=phala uv run skyrag eval $COMMON $Q --arm 2,3 $INJ -o outputs/full/qwen_inject/
```

The 70B with scaffolding and the fixed-candidate runs were run one pass per folder, then merged. Their launch scripts print every command with `--dry-run`, run one query per cell with `--smoke`, and run the full set without a flag:

```bash
uv run python scripts/launch_70b_hosted_injection.py --dry-run
uv run python scripts/launch_a7_fixed_candidates_hosted.py --dry-run
```

Each run directory holds `benchmark_eval.csv` (one row per query and pass, written as the run goes) and `benchmark_eval.json` (the same rows). Every row carries its run provenance: model, provider, serving precision, seed and the settings above. The study's own runs ship as `benchmark_eval.json` only, each file exactly as the run wrote it; see [`PROVENANCE.md`](PROVENANCE.md).

The local RTX 5090 runs (the local replication, the precision test and the constrained-decoding pilot) need a GPU rather than an API key. Their executed notebooks, with every command and its output, are in `outputs/local_vllm_*`.

### Step 10. Compare your rerun with the paper

Pair your run with the shipped one, query by query. For example, the short rewrite (JSON) under injection:

```bash
uv run python scripts/aggregate_arms.py \
  --compare "paper=outputs/merged_s1dec_20261001/i_a2_s1:2" \
  --compare "rerun=outputs/full/rw_short:2"
```

On a rebuilt corpus, expect a small difference from the filled keyword tags (Step 6). The 2026-10-20 end-to-end check in [`CORPUS.md`](CORPUS.md) reran the 8B static baseline on a rebuilt corpus and got an invalid rate of 0.6579 against 0.6666 reported, inside the spread of the study's own passes.

## The arms

| Arm | System |
| :--- | :--- |
| 1 | LLaMA 3.1-70B, unassisted |
| 2 | LLaMA 3.1-8B, unassisted |
| 3 | LLaMA 3.1-8B with a JSON schema on the query rewrite and the keyword selection |

Candidate injection (`--candidate-injection --candidate-limit 100`), the schema exemplar (`--schema-exemplar`) and the query rewrite (`--expansion-mode`) are switched on per run. `uv run skyrag eval --help` lists every flag.

## For reviewers: where each item is

| Item | Where |
| :--- | :--- |
| The 50 benchmark queries | `benchmarks/queries_v1.jsonl` (`q01` to `q05` are the original SKYRAG pipeline's; the paper's Methods describe how the other 45 were drafted) |
| Every prompt and schema | [`PROMPTS.md`](PROMPTS.md), which indexes `src/skyrag/prompts.py`, `selection.py` and `schemas.py` by run configuration |
| The lexical ranker and the near-miss rule | [`PROMPTS.md`](PROMPTS.md), sections "The lexical ranker" and "Invented keywords and near-misses" |
| Per-query results with their run conditions | `outputs/<run>/benchmark_eval.json`, one row per query and pass |
| Model, provider, declared precision and passes per run | [`PROVENANCE.md`](PROVENANCE.md), including the four runs with a missing stamp |
| The scripts behind each table and figure | The table in Step 4 |
| How records without keyword tags were filled | [`CORPUS.md`](CORPUS.md) |
| The course catalogues, and how to rebuild them | Not redistributed; [`CORPUS.md`](CORPUS.md) gives the sources, pinned versions and build steps |

## Repository layout

| Path | Contents |
| :--- | :--- |
| `src/skyrag/` | The pipeline (query rewrite, keyword selection, retrieval, generation) and the evaluation harness (`skyrag eval`) |
| `scripts/` | Corpus build, analysis and launch scripts |
| `benchmarks/queries_v1.jsonl` | The 50 benchmark queries across ten domains |
| `benchmarks/queries_1q_smoke.jsonl` | One query, for a cheap live check |
| `benchmarks/edx_published_omissions.json` | The 26 edX courses missing from the measured catalogue |
| `outputs/` | The per-query results of every run the paper reports, raw passes and merged cells, plus the analysis captures |
| `tests/` | The test suite |

## Licences

| Part | Licence |
| :--- | :--- |
| Evaluation pipeline and analysis scripts | MIT (see `LICENSE`) |
| The 50-query benchmark and the per-query results with their run provenance | CC BY 4.0 (see `LICENSE-DATA.md`) |
| Documentation of how the course corpus was built | CC BY 4.0 |

## What this repository does not contain

The course catalogues, and the keyword vocabulary drawn from them, are not redistributed, because the edX source declares no licence. Both sources are public on Hugging Face:

* the Coursera Course Dataset published by azrai99 (Apache 2.0), and
* the edX-raw dataset published by the Knowledge Innovation Centre (no licence declared).

[`CORPUS.md`](CORPUS.md) documents how the corpus was built from them, including the 26 edX courses it omits. See `THIRD_PARTY_NOTICES.md`.

The development notes are not included either. Some code comments and test docstrings cite files under `project_knowledge/` (task logs, audits and design notes kept while the study was built). Those references record where a decision or fix was written up; the code does not read them, and nothing in this repository needs them to run or to reproduce a result.

## Original SKYRAG paper

Y. S. Soekamto, L. C. Limanjaya, Y. K. Purwanto and D.-K. Kang, "From queries to courses: SKYRAG's revolution in learning path generation via keyword-based document retrieval," *IEEE Access*, vol. 13, pp. 21434–21455, 2025, doi: [10.1109/ACCESS.2025.3535618](https://doi.org/10.1109/ACCESS.2025.3535618).

## Citation

To cite the code and results, see [`CITATION.cff`](CITATION.cff). A citation for the paper will be added on publication.

## Contact

Sean Matthew, seanmatthew1509@gmail.com. See `CODE_OF_CONDUCT.md` for community standards.
