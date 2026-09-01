# ICWSM analysis pipeline

## Stage 0: profile the raw schemas

Before relying on the standard field mapping, regenerate the raw schema EDA:

```bash
python3 Code/00_raw_schema_eda.py
```

This produces a readable summary and a complete machine-readable field profile
under `Data/eda/`. The report contains aggregate schema information only and
does not reproduce post text or author names.

- `raw_schema_profile.md` is the human-readable methodological audit.
- `raw_schema_profile.json` contains the complete per-field counts and types so
  the standardization choices can be reproduced or revised later.

## Stage 1: standardize the Reddit exports

`01_standardize_data.py` streams the raw JSONL and compressed JSONL files into
a consistent schema. It does not modify the source files.

The untouched source exports are stored under `Data/Raw Data/`.

Run a small validation pass:

```bash
python3 Code/01_standardize_data.py \
  --limit-per-file 100 \
  --output-dir /tmp/icwsm-standardized-smoke
```

Run the full standardization:

```bash
python3 Code/01_standardize_data.py
```

Outputs are compressed JSONL files in `Data/standardized/`, grouped by
community and record type. `audit_summary.json` records input counts, date
coverage, and missing-field counts.

### Standard schema

- `record_type`: `post` or `comment`
- `community`: `gradadmissions`, `MSCS`, or `MBA`
- `admissions_cycle`: August-to-July cycle, such as `2022-23`
- `id`, `post_id`, `parent_id`: normalized Reddit identifiers
- `author`: Reddit author name when available
- `created_utc`, `created_at`: Unix and ISO-8601 UTC timestamps
- `title`, `body`, `text`: source text and combined model-ready text
- `score`, `num_comments`, `permalink`: available metadata
- `link_flair_text`, `upvote_ratio`, `is_self_post`: post context and engagement
- `removed_by_category`, `is_edited`, `edited_utc`: moderation/edit metadata
- `retrieved_utc`, `retrieved_at`: collection provenance when available
- `is_stickied`, `is_locked`, `is_nsfw`: analysis and filtering flags
- `subreddit_subscribers`: community size at collection when available
- `author_available`, `text_available`: analysis eligibility flags
- `source_file`: provenance for every record

## Stage 2: flag cleaned and analysis-eligible records

```bash
python3 Code/02_clean_data.py
```

This stage reads only `Data/standardized/` and writes `Data/cleaned/`. It does
not silently drop records. Each record receives flags for anchor eligibility,
thread-exposure eligibility, and longitudinal language measurement, along with
explicit exclusion reasons. The generated Markdown, JSON, and CSV flow reports
provide paper-ready counts by community and admissions cycle.

## Stage 3: identify anchor posts

Build the high-recall keyword candidate set:

```bash
python3 Code/03_identify_anchor_posts.py --stage candidates
```

Then apply the configured local zero-shot NLI model:

```bash
python3 Code/03_identify_anchor_posts.py --stage classify
```

The labels, lexicon, model, primary threshold, and sensitivity thresholds are
versioned in `Code/anchor_config.json`. Candidate and classification outputs
retain all scores and matched terms for auditing.

The binary construct, subtype boundaries, dominant-purpose rule, review key,
and refinement procedure are defined in `Code/anchor_label_codebook.md`.

Audit the full classification and create a reproducible review sample:

```bash
python3 Code/04_audit_anchor_classifications.py
```
