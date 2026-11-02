# Run provenance

Every exported row in `outputs/` carries the conditions it ran under. This file lists them per run folder and names the runs where a stamp is missing. It is generated, not written by hand:

```bash
uv run python scripts/describe_provenance.py             # the full table below
uv run python scripts/describe_provenance.py --gaps-only # only the runs with a missing stamp
```

## How to read the stamps

* **Provider** is the OpenRouter provider the run was pinned to (`SKYRAG_OPENROUTER_PROVIDER`). Blank means it was not pinned, so OpenRouter chose the provider for each request. Local GPU runs read `local-vllm-rtx5090` here, although their `backend` column says `openrouter`, because they used the same OpenAI-compatible client.
* **Precision** is *declared*, not measured. `--serving-precision` records what the operator stated about the endpoint; it does not change the request, and no API reports the numeric format a hosted provider actually served.
* **Passes** are the pass numbers present. A merged cell (`merged_*`, `*_m3`) lists all of its passes; a raw lane lists one.
* The rows also carry `seed`, `temperature`, `max_tokens`, `max_keywords`, `per_keyword`, `context_render`, `embedding_model`, `skyrag_version` and `run_id`, which the table leaves out for width.

## Runs with a missing stamp

Four run folders lack a provider or a precision stamp. The rows are left as they ran; what is known about each comes from the run records, and is stated with its source.

| Run | Backs | Missing | What is known |
| :--- | :--- | :--- | :--- |
| `eval_20260827_arm1-2_llama70b-8b_100q_topup` | The 8B array-payload rows, the hosted drift figure (0.2851) and the power analysis | Precision (the run predates the precision stamp) | Provider DeepInfra, stamped. DeepInfra listed its LLaMA 3.1 endpoints at fp8; that is the provider's listing, not a row stamp |
| `arm3_A_spec_only_20260831` | The LLaMA array-payload Arm 3 rows | Precision (predates the stamp) | Provider DeepInfra, stamped; the same listing applies |
| `qwen_20260929_arm2-3_v5_static_50q_phala_r3` | The Qwen static-window rows | Precision (blank) | Provider Phala, stamped. bf16 is the authors' assumption: OpenRouter lists Phala's precision for this model as "unknown" and returns none with responses. The launch log records bf16 without a source |
| `sweep_v2_20260902_arm2-3_qwen7b_50q` | The Qwen array-payload rows | Provider and precision | The provider was not pinned, so OpenRouter routed each request; which provider and which precision served each row is **unknown** |

## Every run

| Run | Rows | Queries | Passes | Models | Provider | Precision | Sweep | Arm | Rewrite | Injection | Limit | Ranker | Candidate source | Dates |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `a7fix_v5fp8_i_a2_cq_m3` | 150 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | plain | True | 100 | lexical | query | 2026-10-21 |
| `a7fix_v5fp8_i_a2_cq_r1` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | plain | True | 100 | lexical | query | 2026-10-21 |
| `a7fix_v5fp8_i_a2_cq_r2` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | plain | True | 100 | lexical | query | 2026-10-21 |
| `a7fix_v5fp8_i_a2_cq_r3` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | plain | True | 100 | lexical | query | 2026-10-21 |
| `a7fix_v5fp8_i_a2_s1_cq_m3` | 150 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | lexical | query | 2026-10-21 |
| `a7fix_v5fp8_i_a2_s1_cq_r1` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | lexical | query | 2026-10-21 |
| `a7fix_v5fp8_i_a2_s1_cq_r2` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | lexical | query | 2026-10-21 |
| `a7fix_v5fp8_i_a2_s1_cq_r3` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | lexical | query | 2026-10-21 |
| `arm1_merged_70b_v5fp8_4pass_v2` | 200 | 50 | 1 | llama-3.1-70b-instruct | deepinfra | fp8 | v5 | 1 | (not stamped) | False | 0 | blank, (not stamped) | (not stamped) | 2026-09-22 to 2026-09-29 |
| `arm3_A_spec_only_20260831` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | (not stamped) | v2 | 3 | (not stamped) | (not stamped) | (not stamped) | (not stamped) | (not stamped) | 2026-08-31 |
| `c70h_v5fp8_i_a2_m3` | 150 | 50 | 1 | llama-3.1-70b-instruct | deepinfra | fp8 | v5 | 2 | plain | True | 100 | lexical | rewrite | 2026-10-21 |
| `c70h_v5fp8_i_a2_r1` | 50 | 50 | 1 | llama-3.1-70b-instruct | deepinfra | fp8 | v5 | 2 | plain | True | 100 | lexical | rewrite | 2026-10-21 |
| `c70h_v5fp8_i_a2_r2` | 50 | 50 | 1 | llama-3.1-70b-instruct | deepinfra | fp8 | v5 | 2 | plain | True | 100 | lexical | rewrite | 2026-10-21 |
| `c70h_v5fp8_i_a2_r3` | 50 | 50 | 1 | llama-3.1-70b-instruct | deepinfra | fp8 | v5 | 2 | plain | True | 100 | lexical | rewrite | 2026-10-21 |
| `c70h_v5fp8_i_a2_s1_m3` | 150 | 50 | 1 | llama-3.1-70b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | lexical | rewrite | 2026-10-21 |
| `c70h_v5fp8_i_a2_s1_r1` | 50 | 50 | 1 | llama-3.1-70b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | lexical | rewrite | 2026-10-21 |
| `c70h_v5fp8_i_a2_s1_r2` | 50 | 50 | 1 | llama-3.1-70b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | lexical | rewrite | 2026-10-21 |
| `c70h_v5fp8_i_a2_s1_r3` | 50 | 50 | 1 | llama-3.1-70b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | lexical | rewrite | 2026-10-21 |
| `diag_20260930_violog/i_l1_shard1` | 17 | 17 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 3 | (not stamped) | True | 100 | lexical | (not stamped) | 2026-09-29 |
| `diag_20260930_violog/i_l1_shard2` | 17 | 17 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 3 | (not stamped) | True | 100 | lexical | (not stamped) | 2026-09-29 |
| `diag_20260930_violog/i_l1_shard3` | 16 | 16 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 3 | (not stamped) | True | 100 | lexical | (not stamped) | 2026-09-29 |
| `eval_20260827_arm1-2_llama70b-8b_100q_topup` | 100 | 50 | 1 | llama-3.1-70b-instruct, llama-3.1-8b-instruct | deepinfra | (not stamped) | (not stamped) | blank | (not stamped) | (not stamped) | (not stamped) | (not stamped) | (not stamped) | 2026-08-20 to 2026-08-27 |
| `fact_20260923_v5fp8_i_l0_r1` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | (not stamped) | True | 100 | lexical | (not stamped) | 2026-09-22 |
| `fact_20260923_v5fp8_i_l0_r2` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | (not stamped) | True | 100 | lexical | (not stamped) | 2026-09-28 |
| `fact_20260923_v5fp8_i_l0_r3` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | (not stamped) | True | 100 | lexical | (not stamped) | 2026-09-29 |
| `fact_20260923_v5fp8_i_l1_r1` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 3 | (not stamped) | True | 100 | lexical | (not stamped) | 2026-09-22 |
| `fact_20260923_v5fp8_i_l1_r2` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 3 | (not stamped) | True | 100 | lexical | (not stamped) | 2026-09-28 |
| `fact_20260923_v5fp8_i_l1_r3` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 3 | (not stamped) | True | 100 | lexical | (not stamped) | 2026-09-29 |
| `fact_20260923_v5fp8_i_l2_r1` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 3 | (not stamped) | True | 100 | lexical | (not stamped) | 2026-09-22 |
| `fact_20260923_v5fp8_i_l2_r2` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 3 | (not stamped) | True | 100 | lexical | (not stamped) | 2026-09-28 |
| `fact_20260923_v5fp8_i_l2_r3` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 3 | (not stamped) | True | 100 | lexical | (not stamped) | 2026-09-29 |
| `fact_20260923_v5fp8_s_l0_r1` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | (not stamped) | False | 0 | blank | (not stamped) | 2026-09-22 |
| `fact_20260923_v5fp8_s_l0_r2` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | (not stamped) | False | 0 | blank | (not stamped) | 2026-09-28 |
| `fact_20260923_v5fp8_s_l0_r3_resumed` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | (not stamped) | False | 0 | blank | (not stamped) | 2026-09-29 |
| `fact_20260923_v5fp8_s_l1_r1` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 3 | (not stamped) | False | 0 | blank | (not stamped) | 2026-09-22 |
| `fact_20260923_v5fp8_s_l1_r2` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 3 | (not stamped) | False | 0 | blank | (not stamped) | 2026-09-28 |
| `fact_20260923_v5fp8_s_l1_r3_resumed` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 3 | (not stamped) | False | 0 | blank | (not stamped) | 2026-09-29 |
| `fact_20260923_v5fp8_s_l2_r1` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 3 | (not stamped) | False | 0 | blank | (not stamped) | 2026-09-22 |
| `fact_20260923_v5fp8_s_l2_r2` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 3 | (not stamped) | False | 0 | blank | (not stamped) | 2026-09-28 |
| `fact_20260923_v5fp8_s_l2_r3_resumed` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 3 | (not stamped) | False | 0 | blank | (not stamped) | 2026-09-29 |
| `final_20260922_arm1_llama70b_50q_fp8` | 50 | 50 | 1 | llama-3.1-70b-instruct | deepinfra | fp8 | v5 | 1 | (not stamped) | False | 0 | (not stamped) | (not stamped) | 2026-09-22 |
| `final_20260922_arm2_llama8b_50q_fp8` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | (not stamped) | False | 0 | (not stamped) | (not stamped) | 2026-09-22 |
| `floorhosted_20261007_v5fp8_i_a2_s1_floor_r1` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | floor | (not stamped) | 2026-10-07 |
| `floorhosted_20261007_v5fp8_i_a2_s1_floor_r1_h1` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | floor | (not stamped) | 2026-10-07 |
| `floorhosted_20261007_v5fp8_i_a2_s1_floor_r1_h2` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | floor | (not stamped) | 2026-10-07 |
| `floorhosted_20261007_v5fp8_i_a2_s1_floor_r2` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | floor | (not stamped) | 2026-10-07 |
| `floorhosted_20261007_v5fp8_i_a2_s1_floor_r2_h1` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | floor | (not stamped) | 2026-10-07 |
| `floorhosted_20261007_v5fp8_i_a2_s1_floor_r2_h2` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | floor | (not stamped) | 2026-10-07 |
| `floorhosted_20261007_v5fp8_i_a2_s1_floor_r3` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | floor | (not stamped) | 2026-10-07 |
| `floorhosted_20261007_v5fp8_i_a2_s1_floor_r3_h1` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | floor | (not stamped) | 2026-10-07 |
| `floorhosted_20261007_v5fp8_i_a2_s1_floor_r3_h2` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | floor | (not stamped) | 2026-10-07 |
| `floorhosted_20261007_v5fp8_i_a2_s1_lexical_r1` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | lexical | (not stamped) | 2026-10-07 |
| `floorhosted_20261007_v5fp8_i_a2_s1_lexical_r1_h1` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | lexical | (not stamped) | 2026-10-07 |
| `floorhosted_20261007_v5fp8_i_a2_s1_lexical_r1_h2` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | lexical | (not stamped) | 2026-10-07 |
| `floorhosted_20261007_v5fp8_i_a2_s1_lexical_r2` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | lexical | (not stamped) | 2026-10-07 |
| `floorhosted_20261007_v5fp8_i_a2_s1_lexical_r2_h1` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | lexical | (not stamped) | 2026-10-07 |
| `floorhosted_20261007_v5fp8_i_a2_s1_lexical_r2_h2` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | lexical | (not stamped) | 2026-10-07 |
| `floorhosted_20261007_v5fp8_i_a2_s1_lexical_r3` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | lexical | (not stamped) | 2026-10-07 |
| `floorhosted_20261007_v5fp8_i_a2_s1_lexical_r3_h1` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | lexical | (not stamped) | 2026-10-07 |
| `floorhosted_20261007_v5fp8_i_a2_s1_lexical_r3_h2` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | lexical | (not stamped) | 2026-10-07 |
| `merged_floor_20261007/hosted_llama_a2_s1_floor` | 150 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | floor | (not stamped) | 2026-10-07 |
| `merged_floor_20261007/hosted_llama_a2_s1_lexical` | 150 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | lexical | (not stamped) | 2026-10-07 |
| `merged_qprec_20261006/bf16_i_a2` | 150 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | plain | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `merged_qprec_20261006/bf16_i_a2_s1` | 150 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | schema | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `merged_qprec_20261006/bf16_i_a2_sw` | 150 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | schema_wording | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `merged_qprec_20261006/bf16_i_a3` | 150 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 3 | schema | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `merged_qprec_20261006/bf16_i_a3_p1` | 150 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 3 | plain | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `merged_qprec_20261006/bf16_s_a2` | 150 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | plain | False | 0 | blank | (not stamped) | 2026-10-06 |
| `merged_qprec_20261006/bf16_s_a2_s1` | 150 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-06 |
| `merged_qprec_20261006/bf16_s_a2_sw` | 150 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | schema_wording | False | 0 | blank | (not stamped) | 2026-10-06 |
| `merged_qprec_20261006/bf16_s_a3` | 150 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-06 |
| `merged_qprec_20261006/bf16_s_a3_p1` | 150 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-06 |
| `merged_qprec_20261006/fp8_i_a2` | 150 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | plain | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `merged_qprec_20261006/fp8_i_a2_s1` | 150 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | schema | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `merged_qprec_20261006/fp8_i_a2_sw` | 150 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | schema_wording | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `merged_qprec_20261006/fp8_i_a3` | 150 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 3 | schema | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `merged_qprec_20261006/fp8_i_a3_p1` | 150 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 3 | plain | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `merged_qprec_20261006/fp8_s_a2` | 150 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | plain | False | 0 | blank | (not stamped) | 2026-10-06 |
| `merged_qprec_20261006/fp8_s_a2_s1` | 150 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-06 |
| `merged_qprec_20261006/fp8_s_a2_sw` | 150 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | schema_wording | False | 0 | blank | (not stamped) | 2026-10-06 |
| `merged_qprec_20261006/fp8_s_a3` | 150 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-06 |
| `merged_qprec_20261006/fp8_s_a3_p1` | 150 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-06 |
| `merged_s1dec_20261001/i_a2` | 150 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | (not stamped) | True | 100 | lexical | (not stamped) | 2026-09-22 to 2026-09-29 |
| `merged_s1dec_20261001/i_a2_s1` | 150 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | lexical | (not stamped) | 2026-10-01 |
| `merged_s1dec_20261001/i_a2_sw` | 150 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema_wording | True | 100 | lexical | (not stamped) | 2026-10-01 |
| `merged_s1dec_20261001/i_a3` | 150 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 3 | (not stamped) | True | 100 | lexical | (not stamped) | 2026-09-22 to 2026-09-29 |
| `merged_s1dec_20261001/i_a3_p1` | 150 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 3 | plain | True | 100 | lexical | (not stamped) | 2026-10-01 |
| `merged_s1dec_20261001/s_a2` | 150 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | (not stamped) | False | 0 | blank | (not stamped) | 2026-09-22 to 2026-09-29 |
| `merged_s1dec_20261001/s_a2_s1` | 150 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `merged_s1dec_20261001/s_a2_sw` | 150 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema_wording | False | 0 | blank | (not stamped) | 2026-10-01 |
| `merged_s1dec_20261001/s_a3` | 150 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 3 | (not stamped) | False | 0 | blank | (not stamped) | 2026-09-22 to 2026-09-29 |
| `merged_s1dec_20261001/s_a3_p1` | 150 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `merged_s1decA_20261002/hosted_a2v2s1` | 150 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v2 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `merged_s1decA_20261002/hosted_a3v2` | 150 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v2 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `merged_s1decA_20261002/hosted_a3v2p1` | 150 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v2 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `merged_s1decA_20261002/llama_a2v2s1` | 150 | 50 | 1 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v2 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `merged_s1decA_20261002/llama_a3v2` | 150 | 50 | 1 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v2 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `merged_s1decA_20261002/llama_a3v2p1` | 150 | 50 | 1 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v2 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `merged_s1decA_20261002/qwen_a2v2s1` | 150 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v2 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `merged_s1decA_20261002/qwen_a3v2` | 150 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v2 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `merged_s1decA_20261002/qwen_a3v2p1` | 150 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v2 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `qprec5090_qwen_bf16_i_a2_r1` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | plain | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_i_a2_r2` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | plain | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_i_a2_r3` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | plain | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_i_a2_s1_r1` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | schema | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_i_a2_s1_r2` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | schema | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_i_a2_s1_r3` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | schema | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_i_a2_sw_r1` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | schema_wording | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_i_a2_sw_r2` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | schema_wording | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_i_a2_sw_r3` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | schema_wording | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_i_a3_p1_r1` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 3 | plain | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_i_a3_p1_r2` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 3 | plain | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_i_a3_p1_r3` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 3 | plain | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_i_a3_r1` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 3 | schema | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_i_a3_r2` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 3 | schema | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_i_a3_r3` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 3 | schema | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_s_a2_r1` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | plain | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_s_a2_r2` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | plain | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_s_a2_r3` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | plain | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_s_a2_s1_r1` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_s_a2_s1_r2` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_s_a2_s1_r3` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_s_a2_sw_r1` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | schema_wording | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_s_a2_sw_r2` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | schema_wording | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_s_a2_sw_r3` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | schema_wording | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_s_a3_p1_r1` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_s_a3_p1_r2` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_s_a3_p1_r3` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_s_a3_r1` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_s_a3_r2` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_bf16_s_a3_r3` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_i_a2_r1` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | plain | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_i_a2_r2` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | plain | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_i_a2_r3` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | plain | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_i_a2_s1_r1` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | schema | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_i_a2_s1_r2` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | schema | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_i_a2_s1_r3` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | schema | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_i_a2_sw_r1` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | schema_wording | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_i_a2_sw_r2` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | schema_wording | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_i_a2_sw_r3` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | schema_wording | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_i_a3_p1_r1` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 3 | plain | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_i_a3_p1_r2` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 3 | plain | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_i_a3_p1_r3` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 3 | plain | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_i_a3_r1` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 3 | schema | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_i_a3_r2` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 3 | schema | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_i_a3_r3` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 3 | schema | True | 100 | lexical | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_s_a2_r1` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | plain | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_s_a2_r2` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | plain | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_s_a2_r3` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | plain | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_s_a2_s1_r1` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_s_a2_s1_r2` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_s_a2_s1_r3` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_s_a2_sw_r1` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | schema_wording | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_s_a2_sw_r2` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | schema_wording | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_s_a2_sw_r3` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | schema_wording | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_s_a3_p1_r1` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_s_a3_p1_r2` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_s_a3_p1_r3` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_s_a3_r1` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_s_a3_r2` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qprec5090_qwen_fp8_s_a3_r3` | 50 | 50 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | fp8 | v5 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-06 |
| `qwen_20260929_arm2-3_v5_static_50q_phala_r3` | 300 | 50 | 1, 2, 3 | qwen-2.5-7b-instruct | phala | blank | v5 | 2, 3 | (not stamped) | False | 0 | blank | (not stamped) | 2026-09-29 |
| `qwen_20260930_arm2-3_v5_inject_50q_phala_r3` | 300 | 50 | 1, 2, 3 | qwen-2.5-7b-instruct | phala | bf16 | v5 | 2, 3 | (not stamped) | True | 100 | lexical | (not stamped) | 2026-09-29 |
| `s1dec5090_llama_i_a2` | 150 | 50 | 1, 2, 3 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | plain | True | 100 | lexical | (not stamped) | 2026-10-01 |
| `s1dec5090_llama_i_a2_s1` | 150 | 50 | 1, 2, 3 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | schema | True | 100 | lexical | (not stamped) | 2026-10-01 |
| `s1dec5090_llama_i_a2_sw` | 150 | 50 | 1, 2, 3 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | schema_wording | True | 100 | lexical | (not stamped) | 2026-10-01 |
| `s1dec5090_llama_i_a3` | 150 | 50 | 1, 2, 3 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v5 | 3 | schema | True | 100 | lexical | (not stamped) | 2026-10-01 |
| `s1dec5090_llama_i_a3_p1` | 150 | 50 | 1, 2, 3 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v5 | 3 | plain | True | 100 | lexical | (not stamped) | 2026-10-01 |
| `s1dec5090_llama_s_a2` | 150 | 50 | 1, 2, 3 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1dec5090_llama_s_a2_s1` | 150 | 50 | 1, 2, 3 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1dec5090_llama_s_a2_sw` | 150 | 50 | 1, 2, 3 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v5 | 2 | schema_wording | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1dec5090_llama_s_a3` | 150 | 50 | 1, 2, 3 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v5 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1dec5090_llama_s_a3_p1` | 150 | 50 | 1, 2, 3 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v5 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1dec5090_qwen_i_a2` | 150 | 50 | 1, 2, 3 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | plain | True | 100 | lexical | (not stamped) | 2026-10-01 |
| `s1dec5090_qwen_i_a2_s1` | 150 | 50 | 1, 2, 3 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | schema | True | 100 | lexical | (not stamped) | 2026-10-01 |
| `s1dec5090_qwen_i_a2_sw` | 150 | 50 | 1, 2, 3 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | schema_wording | True | 100 | lexical | (not stamped) | 2026-10-01 |
| `s1dec5090_qwen_i_a3` | 150 | 50 | 1, 2, 3 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 3 | schema | True | 100 | lexical | (not stamped) | 2026-10-01 |
| `s1dec5090_qwen_i_a3_p1` | 150 | 50 | 1, 2, 3 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 3 | plain | True | 100 | lexical | (not stamped) | 2026-10-01 |
| `s1dec5090_qwen_s_a2` | 150 | 50 | 1, 2, 3 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1dec5090_qwen_s_a2_s1` | 150 | 50 | 1, 2, 3 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1dec5090_qwen_s_a2_sw` | 150 | 50 | 1, 2, 3 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 2 | schema_wording | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1dec5090_qwen_s_a3` | 150 | 50 | 1, 2, 3 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1dec5090_qwen_s_a3_p1` | 150 | 50 | 1, 2, 3 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v5 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1dec_20261001_v5fp8_i_a2_s1_r1` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | lexical | (not stamped) | 2026-10-01 |
| `s1dec_20261001_v5fp8_i_a2_s1_r2` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | lexical | (not stamped) | 2026-10-01 |
| `s1dec_20261001_v5fp8_i_a2_s1_r3` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | True | 100 | lexical | (not stamped) | 2026-10-01 |
| `s1dec_20261001_v5fp8_i_a2_sw_r1` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema_wording | True | 100 | lexical | (not stamped) | 2026-10-01 |
| `s1dec_20261001_v5fp8_i_a2_sw_r2` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema_wording | True | 100 | lexical | (not stamped) | 2026-10-01 |
| `s1dec_20261001_v5fp8_i_a2_sw_r3` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema_wording | True | 100 | lexical | (not stamped) | 2026-10-01 |
| `s1dec_20261001_v5fp8_i_a3_p1_r1` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 3 | plain | True | 100 | lexical | (not stamped) | 2026-10-01 |
| `s1dec_20261001_v5fp8_i_a3_p1_r2` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 3 | plain | True | 100 | lexical | (not stamped) | 2026-10-01 |
| `s1dec_20261001_v5fp8_i_a3_p1_r3` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 3 | plain | True | 100 | lexical | (not stamped) | 2026-10-01 |
| `s1dec_20261001_v5fp8_s_a2_s1_r1` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1dec_20261001_v5fp8_s_a2_s1_r2` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1dec_20261001_v5fp8_s_a2_s1_r3` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1dec_20261001_v5fp8_s_a2_sw_r1` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema_wording | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1dec_20261001_v5fp8_s_a2_sw_r2` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema_wording | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1dec_20261001_v5fp8_s_a2_sw_r3` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 2 | schema_wording | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1dec_20261001_v5fp8_s_a3_p1_r1` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1dec_20261001_v5fp8_s_a3_p1_r2` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1dec_20261001_v5fp8_s_a3_p1_r3` | 50 | 50 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v5 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_llama_a2v2s1_r1_h1` | 25 | 25 | 1 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v2 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_llama_a2v2s1_r1_h2` | 25 | 25 | 1 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v2 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_llama_a2v2s1_r2_h1` | 25 | 25 | 1 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v2 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_llama_a2v2s1_r2_h2` | 25 | 25 | 1 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v2 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_llama_a2v2s1_r3_h1` | 25 | 25 | 1 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v2 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_llama_a2v2s1_r3_h2` | 25 | 25 | 1 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v2 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_llama_a3v2_r1_h1` | 25 | 25 | 1 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v2 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_llama_a3v2_r1_h2` | 25 | 25 | 1 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v2 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_llama_a3v2_r2_h1` | 25 | 25 | 1 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v2 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_llama_a3v2_r2_h2` | 25 | 25 | 1 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v2 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_llama_a3v2_r3_h1` | 25 | 25 | 1 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v2 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_llama_a3v2_r3_h2` | 25 | 25 | 1 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v2 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_llama_a3v2p1_r1_h1` | 25 | 25 | 1 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v2 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_llama_a3v2p1_r1_h2` | 25 | 25 | 1 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v2 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_llama_a3v2p1_r2_h1` | 25 | 25 | 1 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v2 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_llama_a3v2p1_r2_h2` | 25 | 25 | 1 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v2 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_llama_a3v2p1_r3_h1` | 25 | 25 | 1 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v2 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_llama_a3v2p1_r3_h2` | 25 | 25 | 1 | llama-3.1-8b-instruct | local-vllm-rtx5090 | fp8 | v2 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_qwen_a2v2s1_r1_h1` | 25 | 25 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v2 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_qwen_a2v2s1_r1_h2` | 25 | 25 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v2 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_qwen_a2v2s1_r2_h1` | 25 | 25 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v2 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_qwen_a2v2s1_r2_h2` | 25 | 25 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v2 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_qwen_a2v2s1_r3_h1` | 25 | 25 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v2 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_qwen_a2v2s1_r3_h2` | 25 | 25 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v2 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_qwen_a3v2_r1_h1` | 25 | 25 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v2 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_qwen_a3v2_r1_h2` | 25 | 25 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v2 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_qwen_a3v2_r2_h1` | 25 | 25 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v2 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_qwen_a3v2_r2_h2` | 25 | 25 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v2 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_qwen_a3v2_r3_h1` | 25 | 25 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v2 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_qwen_a3v2_r3_h2` | 25 | 25 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v2 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_qwen_a3v2p1_r1_h1` | 25 | 25 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v2 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_qwen_a3v2p1_r1_h2` | 25 | 25 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v2 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_qwen_a3v2p1_r2_h1` | 25 | 25 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v2 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_qwen_a3v2p1_r2_h2` | 25 | 25 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v2 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_qwen_a3v2p1_r3_h1` | 25 | 25 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v2 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA5090_qwen_a3v2p1_r3_h2` | 25 | 25 | 1 | qwen-2.5-7b-instruct | local-vllm-rtx5090 | bf16 | v2 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA_20261002_v2fp8_a2v2s1_r1_h1` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v2 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA_20261002_v2fp8_a2v2s1_r1_h2` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v2 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA_20261002_v2fp8_a2v2s1_r2_h1` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v2 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA_20261002_v2fp8_a2v2s1_r2_h2` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v2 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA_20261002_v2fp8_a2v2s1_r3_h1` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v2 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA_20261002_v2fp8_a2v2s1_r3_h2` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v2 | 2 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA_20261002_v2fp8_a3v2_r1_h1` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v2 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA_20261002_v2fp8_a3v2_r1_h2` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v2 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA_20261002_v2fp8_a3v2_r2_h1` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v2 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA_20261002_v2fp8_a3v2_r2_h2` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v2 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA_20261002_v2fp8_a3v2_r3_h1` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v2 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA_20261002_v2fp8_a3v2_r3_h2` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v2 | 3 | schema | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA_20261002_v2fp8_a3v2p1_r1_h1` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v2 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA_20261002_v2fp8_a3v2p1_r1_h2` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v2 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA_20261002_v2fp8_a3v2p1_r2_h1` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v2 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA_20261002_v2fp8_a3v2p1_r2_h2` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v2 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA_20261002_v2fp8_a3v2p1_r3_h1` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v2 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `s1decA_20261002_v2fp8_a3v2p1_r3_h2` | 25 | 25 | 1 | llama-3.1-8b-instruct | deepinfra | fp8 | v2 | 3 | plain | False | 0 | blank | (not stamped) | 2026-10-01 |
| `sweep_v2_20260902_arm2-3_qwen7b_50q` | 100 | 50 | 1 | qwen-2.5-7b-instruct | blank | (not stamped) | v2 | 2, 3 | (not stamped) | (not stamped) | (not stamped) | (not stamped) | (not stamped) | 2026-09-02 |
| `var_20260923_arm1_70b_p2_resumed` | 50 | 50 | 1 | llama-3.1-70b-instruct | deepinfra | fp8 | v5 | 1 | (not stamped) | False | 0 | blank | (not stamped) | 2026-09-22 to 2026-09-29 |
| `var_20260923_arm1_70b_p3_resumed` | 50 | 50 | 1 | llama-3.1-70b-instruct | deepinfra | fp8 | v5 | 1 | (not stamped) | False | 0 | blank | (not stamped) | 2026-09-22 to 2026-09-29 |
| `var_20260923_arm1_70b_p4` | 50 | 50 | 1 | llama-3.1-70b-instruct | deepinfra | fp8 | v5 | 1 | (not stamped) | False | 0 | blank | (not stamped) | 2026-09-22 |

259 run folders.
