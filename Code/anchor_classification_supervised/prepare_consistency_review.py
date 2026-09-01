"""Prepare a deterministic blinded audit sample for the Qwen v3 evaluation."""
from __future__ import annotations

import csv
import json
import random
import argparse
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
GROUND_TRUTH_FILE = PROJECT_DIR / "Data" / "anchor_ground_truth" / "combined_400_ground_truth.csv"
KEY_FILE = (
    PROJECT_DIR
    / "Data"
    / "anchor_ground_truth"
    / "internal_keys_do_not_share"
    / "qwen_v3_consistency_review_key.csv"
)
REVIEW_JSON = Path("/private/tmp/qwen_v3_consistency_review.json")
RANDOM_SEED = 20260830
SAMPLE_PER_GROUP = 25


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-file", type=Path, required=True)
    parser.add_argument("--ground-truth-file", type=Path, default=GROUND_TRUTH_FILE)
    parser.add_argument("--key-file", type=Path, default=KEY_FILE)
    parser.add_argument("--review-json", type=Path, default=REVIEW_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = read_csv(args.predictions_file)
    ground_truth = {row["item_id"]: row for row in read_csv(args.ground_truth_file)}
    if len(predictions) != 400 or len({row["item_id"] for row in predictions}) != 400:
        raise ValueError("Expected 400 unique Qwen v3 predictions")

    groups: dict[str, list[dict[str, str]]] = {
        "false_positive": [],
        "false_negative": [],
        "true_positive": [],
        "true_negative": [],
    }
    for prediction in predictions:
        true_label = int(prediction["true_label"])
        predicted_label = int(prediction["predicted_label"])
        if true_label == 0 and predicted_label == 1:
            group = "false_positive"
        elif true_label == 1 and predicted_label == 0:
            group = "false_negative"
        elif true_label == 1:
            group = "true_positive"
        else:
            group = "true_negative"
        source = ground_truth[prediction["item_id"]]
        groups[group].append({**prediction, "text": source["text"]})

    rng = random.Random(RANDOM_SEED)
    selected = []
    for group_name, rows in groups.items():
        if len(rows) < SAMPLE_PER_GROUP:
            raise ValueError(f"Not enough rows in {group_name}")
        selected.extend((group_name, row) for row in rng.sample(rows, SAMPLE_PER_GROUP))
    rng.shuffle(selected)

    review_rows = []
    key_rows = []
    for index, (group_name, row) in enumerate(selected, 1):
        review_id = f"Q3R-{index:03d}"
        review_rows.append(
            {
                "review_id": review_id,
                "community": row["community"],
                "admissions_cycle": row["admissions_cycle"],
                "text": row["text"],
                "reviewer_label": "",
                "confidence": "",
                "reviewer_notes": "",
            }
        )
        key_rows.append(
            {
                "review_id": review_id,
                "item_id": row["item_id"],
                "audit_group": group_name,
                "original_human_label": row["true_label"],
                "qwen_v3_prediction": row["predicted_label"],
                "qwen_v3_anchor_probability": row["threat_distress_probability"],
                "fold": row["fold"],
            }
        )

    args.key_file.parent.mkdir(parents=True, exist_ok=True)
    with args.key_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(key_rows[0]))
        writer.writeheader()
        writer.writerows(key_rows)
    args.review_json.write_text(json.dumps(review_rows, ensure_ascii=False), encoding="utf-8")
    print(f"Review rows: {len(review_rows)}")
    print("Groups:", {name: sum(row["audit_group"] == name for row in key_rows) for name in groups})
    print(f"Key: {args.key_file}")
    print(f"Review JSON: {args.review_json}")


if __name__ == "__main__":
    main()
