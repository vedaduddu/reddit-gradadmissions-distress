"""Paths and fixed rules for user exposure extraction."""
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
COMMENT_DIR = PROJECT_DIR / "Data" / "standardized_filtered"
POST_LABEL_FILE = (
    PROJECT_DIR
    / "Data"
    / "anchor_classification_llm"
    / "qwen_full_corpus_v6"
    / "all_post_labels.jsonl.gz"
)
OUTPUT_DIR = PROJECT_DIR / "Data" / "exposure"

COMMENT_FILES = [
    "MBA_comments_filtered.jsonl.gz",
    "MSCS_comments_filtered.jsonl.gz",
    "gradadmissions_comments_filtered.jsonl.gz",
]

EXPOSURE_MONTHS = {9, 10, 11}
COMMUNITIES = ["MBA", "MSCS", "gradadmissions"]
