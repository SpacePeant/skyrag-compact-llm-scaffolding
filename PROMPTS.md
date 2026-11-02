# Prompts, schemas and scoring rules

Where every prompt, schema and rule the study used lives in the code. Names are given rather than line numbers, so this index stays right when the files change; search the named file for the name.

The pipeline has four stages: **(1)** query rewrite, **(2)** step-back principles, **(3)** keyword selection, then retrieval, and **(4)** learning-path generation. Which prompt a stage uses depends on three run settings, all stamped on every result row: `sweep_version`, `arm` (Arm 3 switches the schema on) and `expansion_mode` (the rewrite, set independently of the arm).

## Which prompt family each run used

| `sweep_version` | Payload of the keyword selection | Runs in `outputs/` |
| :--- | :--- | :--- |
| `v5` | Flat map, keyword → score | The factorial (`fact_20260923_v5fp8_*`), the decomposition (`s1dec*`, `merged_s1dec_20261001/`), precision (`qprec5090_*`, `merged_qprec_20261006/`), the floor ranker (`floorhosted_*`, `merged_floor_20261007/`), the hosted 70B (`c70h_v5fp8_*`) and the fixed-candidate test (`a7fix_v5fp8_*`), and the Arm 1 70B runs |
| `v2` | Array of objects | The array runs: `sweep_v2_20260902_arm2-3_qwen7b_50q/`, `arm3_A_spec_only_20260831/`, `s1decA*` and `merged_s1decA_20261002/` |
| none stamped | Pre-sweep run; Arms 1 and 2 only, so no schema | `eval_20260827_arm1-2_llama70b-8b_100q_topup/` |

`v5` runs use the "V4 family" prompts for stages 1, 2 and 4 and the `V5` prompt and schema for stage 3. `v2` and earlier runs use the `V1` prompts and schemas.

## Stage 1: query rewrite (`src/skyrag/prompts.py`, chosen in `src/skyrag/pipelines/skyrag.py`)

| `expansion_mode` | Prompt | Schema (`src/skyrag/schemas.py`) | In the paper |
| :--- | :--- | :--- | :--- |
| `plain` (Arm 2's default) | `EXPAND_QUERY` | none | The default rewrite |
| `schema` (Arm 3's default) | `EXPAND_QUERY_SCHEMA_V4` (`v5`) or `EXPAND_QUERY_SCHEMA_V1` (`v2`) | `QUERY_EXPANSION_SCHEMA_V4` or `QUERY_EXPANSION_SCHEMA` | The short JSON rewrite |
| `schema_wording` (`v5` only) | `EXPAND_QUERY_SCHEMA_WORDING` | none | The schema's wording as plain text, no schema |

The mode is resolved by `resolve_expansion_mode` in `pipelines/skyrag.py`.

## Stage 2: step-back principles

`STEP_BACK` without the schema; with it, `STEP_BACK_SCHEMA_V4` (`v5`, schema `STEP_BACK_SCHEMA_V4`) or `STEP_BACK_SCHEMA_V1` (`v2`, schema `STEP_BACK_SCHEMA`). All in `prompts.py` and `schemas.py`.

## Stage 3: keyword selection (`src/skyrag/selection.py`, `select_keywords`)

| Configuration | Prompt | Schema |
| :--- | :--- | :--- |
| No schema (Arms 1 and 2) | `PROMPT` | none; the reply is parsed by `try_parse_keyword_response` |
| Schema, `v5` (Arm 3) | `PROMPT_SCHEMA_V5` | `keyword_selection_schema_v5` (flat map, at most `max_keywords` entries) |
| Schema, `v2` (Arm 3) | `PROMPT_SCHEMA_V1` | `keyword_selection_schema` (array of objects) |

* **Format levels of the factorial.** Level 0 is no schema (Arm 2); Level 1 adds the schema (Arm 3; `schema_prompt_variant` = `spec_only`, or `spec_plus_candidate_injection` under injection); Level 2 adds a worked example (`--schema-exemplar`; `spec_plus_exemplar`). The example text is rendered by `render` in `schemas.py`.
* **The vocabulary shown to the model.** Without injection, the 1,000 most frequent keywords per platform (`max_vocabulary`). With candidate injection (`--candidate-injection --candidate-limit 100`), the 100 keywords ranked most relevant by `rank_candidate_keywords`, in place of the frequency window.
* **The keyword cap.** The prompt asks for at most 40 keywords per platform (`max_keywords`, `SKYRAG_MAX_KEYWORDS` in `src/skyrag/config.py`). The reply's token budget is sized for a compliant 40-keyword answer. A reply that runs past it is cut off and recorded as an overrun (`OUTCOME_OVERRUN`), and that platform contributes no keywords; a reply cut off below the budget is recorded as a harness truncation (`OUTCOME_TRUNCATED`) instead. Under the schema, a complete reply with more than 40 entries fails validation (`maxProperties`), with no partial salvage.
* **Schema checking.** The schema is written into the prompt and the reply is validated after generation by `parse_and_validate` and `validate` in `schemas.py`. It is not enforced during decoding.

## The lexical ranker (`selection.py`)

`rank_candidate_keywords` with `RANKER_LEXICAL`: the catalogue's eligible keywords scored by token overlap with, and containment in, the ranking text. Under injection the ranking text is the stage 1 rewrite (`--candidate-source rewrite`, every published cell) or the learner's original query (`--candidate-source query`, the fixed-candidate test). `RANKER_FLOOR` is the frequency-floor variant of the floor-ranker run (minimum frequency `RANKER_FLOOR_MIN_FREQUENCY`).

## Invented keywords and near-misses (`selection.py`)

* `classify` sorts each selected keyword four ways: usable (in the shown window), out of window (in the catalogue but not shown), removed by cleaning, or not in the catalogue at all.
* `split_near_misses` then moves a not-in-catalogue keyword to the near-miss bucket when it matches a catalogue keyword after `normalise_keyword`: lower case, punctuation and separators to spaces, whitespace collapsed, and each token singularised by `_singularise` (`-ies` to `-y`; `-ses`, `-xes`, `-zes`, `-ches`, `-shes` lose `-es`; a final `-s` dropped except after `s`; tokens under four characters untouched).
* A near-miss is counted, never repaired: the keyword stays out of retrieval.
* What remains is the invented set. The invalid keyword rate and its two forms (over every run, and over runs with usable output) are computed from the exported rows by `scripts/aggregate_arms.py`.

## Stage 4: learning-path generation

`build_generation_prompt` (`GENERATE_PATH`) without the schema; with it, `GENERATE_PATH_SCHEMA_V4` (`v5`, schema `LEARNING_PATH_SCHEMA_V4`) or `GENERATE_PATH_SCHEMA_V1` (`v2`, schema `LEARNING_PATH_SCHEMA`).

## Not used by the paper

`PROMPT_SCHEMA_V4` and `keyword_selection_schema_v4` (the `v4` keyword payload), and the `v1`, `v3` and `v4` sweep versions, can be selected with `--sweep-version` but no shipped run used them.
