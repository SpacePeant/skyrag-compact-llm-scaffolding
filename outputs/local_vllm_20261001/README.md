# Local vLLM diagnostics, 2026-09-30 to 2026-10-01

Supplementary single-pass diagnostics run on lab GPUs with local vLLM. **These are not hosted (OpenRouter) results and must not be pooled with them.**

Common settings: greedy decoding, seed 34, `--generation-config vllm`, the real SKYRAG prompts, 50 benchmark queries × 2 platforms (25 queries for the 4070 Ti pilot).

| Folder | What ran | Used in the paper |
|---|---|---|
| `4070ti_pilot/` | Qwen 2.5-7B constrained-decoding pilot on an RTX 4070 Ti (25 queries, one pass) | Yes: the constrained-decoding pilot in Limitations (`stage2_summary.md`) |
| `5090_precision/` | LLaMA 3.1-8B and Qwen 2.5-7B at bf16 and fp8, keyword selection only | No: pilot, kept as a record |
| `5090_static_2x2/` | Query rewrite {plain, schema} × selection {arm2, schema}, 1,000-term window | No: pilot, superseded by the full-pipeline runs |
| `5090_injection_2x2/` | The same 2×2 under 100-candidate lexical injection | No: pilot, superseded by the full-pipeline runs |
| `5090_pipeline_decomp/` | `BUNDLE_*.json`: harness stamps of the full-pipeline decomposition runs (`s1dec5090_*`, `merged_s1dec_20261001/`) | Provenance only |

The `*_output.txt` and `*_itt_per_query.json` files beside this README are the captured outputs of the pilot scorers.

## Not runnable from this release

The pilots were scored by scripts that read the course catalogues and the keyword vocabulary packages sent to the GPU. Neither is redistributed (see `CORPUS.md` and `LICENSE-DATA.md`), and the scorers and notebook generators are not included. The files here are the record of what ran and what it gave; they cannot be re-scored from this repository alone.

## Notebooks

`*_executed.ipynb` are the copies run on the GPU, with their printed outputs. The Hugging Face token was entered with `getpass` and appears in no notebook.

## Known defects

- `5090_precision/precision_results.json` stores each reply cut at 3,000 characters. That affects 48 replies, mostly LLaMA Arm 2; the scorer excluded the affected queries from both sides of each pair. The 2×2 runs store full replies.
- The notebooks' in-notebook summary tables use an approximate scorer (not-eligible counts as invalid). Quote only the `*_output.txt` numbers.
- The retrieval figures are a ceiling (the sum of min(7, frequency) over kept keywords), not measured retrieval.
