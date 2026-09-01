"""Add ranking and post-hoc threshold diagnostics to cross-validation results."""
from __future__ import annotations

import csv
import json

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from config import OUTPUT_DIR


PREDICTIONS_FILE = OUTPUT_DIR / "cross_validation_predictions.csv"
SUMMARY_FILE = OUTPUT_DIR / "cross_validation_summary.json"


def main():
    with PREDICTIONS_FILE.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    y_true = [int(row["true_label"]) for row in rows]
    scores = [float(row["anchor_probability"]) for row in rows]

    candidates = []
    for threshold in sorted(set(scores)):
        predictions = [int(score >= threshold) for score in scores]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, predictions, average="binary", zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        candidates.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(y_true, predictions),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            }
        )
    best = max(candidates, key=lambda row: (row["accuracy"], row["f1"]))

    summary = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    summary["aggregate_out_of_fold_ranking"] = {
        "roc_auc": roc_auc_score(y_true, scores),
        "average_precision": average_precision_score(y_true, scores),
        "minimum_anchor_probability": min(scores),
        "maximum_anchor_probability": max(scores),
    }
    summary["posthoc_best_accuracy_threshold"] = {
        **best,
        "warning": "Selected on the same out-of-fold predictions; descriptive only, not an unbiased estimate.",
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "aggregate_out_of_fold": summary["aggregate_out_of_fold"],
        "aggregate_out_of_fold_ranking": summary["aggregate_out_of_fold_ranking"],
        "posthoc_best_accuracy_threshold": summary["posthoc_best_accuracy_threshold"],
    }, indent=2))


if __name__ == "__main__":
    main()
