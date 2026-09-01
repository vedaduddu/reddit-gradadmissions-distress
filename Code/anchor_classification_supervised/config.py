"""Configuration for the supervised encoder feasibility experiment."""
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
GROUND_TRUTH_FILE = PROJECT_DIR / "Data" / "anchor_ground_truth" / "combined_400_ground_truth.csv"
OUTPUT_DIR = PROJECT_DIR / "Data" / "anchor_classification_supervised" / "combined_400"

MODEL_NAME = "microsoft/deberta-v3-base"
RANDOM_SEED = 20260828
N_SPLITS = 5
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
