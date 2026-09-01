# Code organization

The active pipeline is split into small folders by purpose.

## `data_preparation/`

- `00_raw_schema_eda.py` profiles the raw Reddit export schemas.
- `01_standardize_data.py` writes a consistent compressed JSONL schema.

## `cleaning/`

- `bot_filter/` contains Agam Goyal's modular bot, tombstone, short-text, and
  style-detector filtration pipeline.
- `ai_detector/` contains the portable Moltbook-versus-Reddit style detector.
- `verify_filtered_data.py` independently checks the delivered filtered files
  and their record counts without changing them.

Cleaning scripts read the standardized-data path from
`REDDIT_STANDARDIZED_DIR` and write reports to `REDDIT_CLEANING_OUTPUT_DIR`.
The style detector model is supplied separately through `AI_DETECTOR_MODEL`
and optional metadata through `AI_DETECTOR_METADATA`.

Example:

```bash
export REDDIT_STANDARDIZED_DIR=/path/to/standardized_filtered
export REDDIT_CLEANING_OUTPUT_DIR=/path/to/cleaning_reports
export AI_DETECTOR_MODEL=/path/to/tfidf_logreg.joblib
python -m Code.cleaning.bot_filter.apply_pipeline
```

## `anchor_classification_supervised/`

Contains the supervised DeBERTa feasibility experiments and label-preparation
utilities. Research labels and predictions remain outside Git.

## `anchor_classification_llm/`

Reserved for the server-side structured-reasoning classifier and evaluation
scripts.
