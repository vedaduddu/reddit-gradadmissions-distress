"""Configuration for the frozen v6 whole-corpus anchor run."""
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "Data"
POSTS_DIR = DATA_DIR / "standardized_filtered"
GROUND_TRUTH_DIR = DATA_DIR / "anchor_ground_truth"
RESULTS_DIR = (
    DATA_DIR
    / "anchor_classification_llm"
    / "qwen_full_corpus_v6"
)
PREPARED_DIR = RESULTS_DIR / "prepared"

GROUND_TRUTH_FILE = GROUND_TRUTH_DIR / "combined_400_ground_truth.csv"
COMBINED_KEY_FILE = (
    GROUND_TRUTH_DIR
    / "internal_keys_do_not_share"
    / "combined_400_reddit_key.csv"
)
EXPANSION_KEY_FILE = (
    GROUND_TRUTH_DIR
    / "internal_keys_do_not_share"
    / "expansion_300_key.csv"
)
INITIAL_KEY_FILE = (
    PROJECT_DIR
    / "Archive"
    / "anchor-validation-filter-based-approach-2026-08-27"
    / "Data"
    / "annotations"
    / "keyword_validation_v2"
    / "internal_keys_do_not_share"
    / "keyword_validation_key.csv"
)

POST_FILES = [
    "MBA_posts_filtered.jsonl.gz",
    "MSCS_posts_filtered.jsonl.gz",
    "gradadmissions_posts_filtered.jsonl.gz",
]

ADMISSIONS_CYCLES = {
    "2022-23",
    "2023-24",
    "2024-25",
}
ELIGIBLE_MONTHS = {9, 10, 11}

N_SHARDS = 4
RANDOM_SEED = 20260828
DEMOS_PER_CLASS = 5
VALIDATION_PER_CLASS = 50
