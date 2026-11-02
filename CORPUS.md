# Corpus

The study ran over **10,092 courses: 6,641 from Coursera and 3,451 from edX**. This file documents how that corpus was built from its two public sources, so that it can be rebuilt without being redistributed.

The catalogues themselves are **not included in this repository**. The edX source declares no licence, so no redistribution right is granted, and it is a scrape of edx.org page data. The keyword vocabulary drawn from the catalogues is withheld for the same reason. Course titles, keywords and URLs that appear inside the released per-query results remain the property of their sources and are not covered by this repository's CC BY 4.0 licence.

## Sources

| Platform | Hugging Face dataset | Pinned revision (commit SHA) | Licence | Records upstream |
| :--- | :--- | :--- | :--- | ---: |
| Coursera | [`azrai99/coursera-course-dataset`](https://huggingface.co/datasets/azrai99/coursera-course-dataset), file `coursera_course_2024.csv` | `ccd36c1660fee9610d5262e7517d8397c343168b` | Apache 2.0 | 6,645 rows |
| edX | [`Knowledge-Innovation-Centre/edX-raw`](https://huggingface.co/datasets/Knowledge-Innovation-Centre/edX-raw), one page file per record | `0d8ac6d3aa95a2484789c6ba82f597331d7deb62` | none declared | 5,203 pages |

Both repositories have been unmodified since 2024, so the pinned revisions are also the current ones.

## Building it

### Step 1. Fetch and clean the catalogues

```bash
python scripts/prepare_corpus.py --fetch --clean --as-published
```

This downloads both pinned revisions, checks that all 5,203 edX pages arrived, and writes `documents/CourseraDataset_clean.json` and `documents/EdxDataset-Clean.json`.

The edX half is 5,203 separate files. Anonymous downloads hit the Hugging Face rate limit partway through; the download waits and resumes by itself, but it is much faster with a free Hugging Face token set as `HF_TOKEN`.

**Coursera: 6,645 → 6,641.** Rows are deduplicated on the title plus the first 150 characters of the description. The four rows removed are syndicated re-listings of the same course.

**edX: 5,203 → 3,451.**

| Step | Records |
| :--- | ---: |
| Raw pages | 5,203 |
| Less pages that are not courses (552 programs, 63 degrees, 3 master's, 1 bootcamp) | −619 |
| Less non-English pages | −1,034 |
| Kept | 3,550 |
| Less duplicates (title plus first 150 characters of the description) | −73 |
| **Complete rebuild** | **3,477** |
| Less the 26 published omissions (`--as-published`) | −26 |
| **Corpus the study measured** | **3,451** |

**Why `--as-published` exists.** The edX catalogue used in the study was cleaned from a download that was interrupted partway, which left out 26 consecutive pages. They hold ordinary courses with full descriptions; no rule excluded them. The download was later completed, but the catalogue was not rebuilt, and every experiment ran against the 3,451-course version. The 26 are listed by URL and title in [`benchmarks/edx_published_omissions.json`](benchmarks/edx_published_omissions.json). With `--as-published` they are removed after cleaning, and the result equals the measured catalogue record for record. Without it a rebuild gives 3,477. That is the complete count, but it is not the corpus the results describe.

The 26 courses are 0.26% of the corpus and are not concentrated in any query domain.

**Checked on a fresh clone (2026-10-20).** The rebuilt edX catalogue matched the measured one in all 3,451 records. The Coursera catalogue matched in 6,639 of 6,641. The other two hold a skill list with quotation marks inside it, which the original cleaning stripped and this script keeps, so those two courses differ in a few tags.

### Step 2. Keyword tags

Each course carries keyword tags, which form the catalogue's keyword index that the pipeline selects from.

* **Platform tags.** Coursera tags come from the dataset's own skill list. edX tags come from each course's own subject labels.
* **Filled tags.** Where a course arrived with no tags, a language model extracted five from the title, the learning outcomes and the description (`skyrag.data.keywords`). This filled **1,952 Coursera records (29.4%) and 25 edX records (0.72%)**, 1,977 in total.

```bash
python scripts/build_corpus_index.py --only keywords
```

**Which model filled them.** The original run did not record its model. It was identified afterwards by regenerating 60 of the filled records with five candidate models at the original settings (temperature 0, five keywords) and comparing them with the measured tags. LLaMA 3.1-8B through OpenRouter, with no provider pin, matched far better than the others: mean overlap (Jaccard) 0.659, against 0.248 to 0.394 for LLaMA 3.1-70B, LLaMA 3.3-70B, Gemma 2 27B and Qwen 2.5-7B. It is the default here. The script that ran the comparison is `scripts/fingerprint_keyword_fill_model.py`.

**A rebuild approximates the filled tags.** Hosted models vary between identical requests even at temperature 0. On the fresh-clone rebuild, 566 of the 1,952 filled Coursera records and 7 of the 25 filled edX records came back identical (mean overlap 0.632 and 0.600). The platform tags are reproduced exactly. As a result, the rebuilt keyword vocabulary shares 91.2% of the measured Coursera vocabulary and 99.4% of the edX vocabulary. Because the invalid rate is measured against this vocabulary, a rerun on a rebuilt corpus will not give exactly the reported rates.

### Step 3. Course embeddings

Each course is embedded once with the `all-mpnet-base-v2` sentence encoder (768 dimensions), over the text the course schema renders for embedding (`skyrag.data.schema`, used by `skyrag.embedding.Embedder.embed_courses`). The stores are written to `embeddings/coursera.{npy,jsonl}` and `embeddings/edx.{npy,jsonl}`. The evaluation harness refuses to run without them, rather than falling back to a placeholder corpus.

```bash
python scripts/build_corpus_index.py --only embed
```

### Step 4. Check the result

```bash
python scripts/prepare_corpus.py --verify
```

This prints, for each platform, the count on disk against both reference counts (complete rebuild and published) and a short SHA-256. It exits non-zero when a catalogue matches neither, because a corpus that matches neither is not one this study describes.

The files the study ran against (all dated 2026-08-04, before the first experiment) had these SHA-256 digests:

| File | Records | SHA-256 |
| :--- | ---: | :--- |
| `documents/CourseraDataset_clean.json` | 6,641 | `02e33e114dd8a17572923a807bfd62eb7c5f745834e545a960484f599066264c` |
| `documents/EdxDataset-Clean.json` | 3,451 | `a8d0bb56a4a2d901b17747537ff387c8d25e358f0fb28fc25d58ab2a19f30cb8` |
| `documents/CourseraDataset_EN_with_keywords.json` | 6,641 | `b05bbea2267cf092f056a50e227374cae36d7fd6e8edd38a7727d74e429e133d` |
| `documents/EdxDataset_with_keywords.json` | 3,451 | `c0c47dcb66e68bab2f461b3b18d41013f467819e5780888dc45bdd1173204591` |
| `embeddings/coursera.npy` | 6,641 | `ec07d566fd98478ae1b3432a90a65d0eccbe213ff8a87483a4f6240db57b10cc` |
| `embeddings/edx.npy` | 3,451 | `8cc4a82bcd34535e3f7b07ff241e9f97f02dc7e9127cfbd7751fe5769ed51e26` |

A rebuilt file can hold the same records in a different byte layout, so a different digest alone does not mean a different corpus. Compare records, which is what `--verify` does for the catalogues.

## What a rebuild will and will not match

| Part | Match |
| :--- | :--- |
| Catalogue records, with `--as-published` | Exact |
| Platform keyword tags | Exact |
| The 1,977 filled keyword tags | Approximate (hosted language-model output; about 30% identical, mean overlap 0.63) |
| Keyword vocabulary | 91.2% shared (Coursera), 99.4% (edX) |
| Course embeddings | Close; floating-point results can differ slightly across hardware and library versions |

**Checked end to end (2026-10-20).** On a fresh clone, the corpus was rebuilt with these steps, and the 8B baseline cell (static window, 50 queries, one pass, the study's settings) was rerun on it. The rerun gave an invalid rate of 0.6579 and 80.8 courses retrieved per query, against 0.6666 and 78.9 reported. That is inside the spread between the study's own passes (0.6430 to 0.6822).

The reported statistics are computed from the released per-query results, so checking them does not require the corpus. A rebuild is needed only to rerun the pipeline.
