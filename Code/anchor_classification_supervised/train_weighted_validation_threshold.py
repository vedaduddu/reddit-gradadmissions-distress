"""Evaluate class-weighted DeBERTa with fold-local validation thresholds."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from config import (
    BATCH_SIZE,
    EPOCHS,
    GROUND_TRUTH_FILE,
    LEARNING_RATE,
    MAX_LENGTH,
    MODEL_NAME,
    N_SPLITS,
    PROJECT_DIR,
    RANDOM_SEED,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from train_cross_validation import TextDataset, set_seed


OUTPUT_DIR = PROJECT_DIR / "Data" / "anchor_classification_supervised" / "combined_400_weighted_validation_threshold"
PREDICTIONS_FILE = OUTPUT_DIR / "cross_validation_predictions.csv"
SUMMARY_FILE = OUTPUT_DIR / "cross_validation_summary.json"
VALIDATION_FRACTION = 0.20


def read_rows():
    with GROUND_TRUTH_FILE.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    labels = [int(row["label"]) for row in rows]
    if not rows or len({row["item_id"] for row in rows}) != len(rows) or set(labels) != {0, 1}:
        raise ValueError("Ground truth must contain unique records from both classes")
    return rows, labels


def predict_probabilities(model, rows, tokenizer, device):
    loader = DataLoader(TextDataset(rows, tokenizer), batch_size=BATCH_SIZE, shuffle=False)
    probabilities = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels")
            inputs = {key: value.to(device) for key, value in batch.items()}
            probabilities.extend(torch.softmax(model(**inputs).logits, dim=1)[:, 1].cpu().tolist())
    return probabilities


def choose_threshold(labels, scores):
    candidates = sorted(set(scores))
    results = []
    for threshold in candidates:
        predictions = [int(score >= threshold) for score in scores]
        _, _, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary", zero_division=0
        )
        results.append((balanced_accuracy_score(labels, predictions), f1, threshold))
    balanced_accuracy, f1, threshold = max(results)
    return threshold, balanced_accuracy, f1


def calculate_metrics(labels, predictions, scores):
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    specificity = tn / (tn + fp)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc_score(labels, scores),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train_fold(fold, development_rows, test_rows, tokenizer, device):
    development_labels = [int(row["label"]) for row in development_rows]
    train_rows, validation_rows = train_test_split(
        development_rows,
        test_size=VALIDATION_FRACTION,
        stratify=development_labels,
        random_state=RANDOM_SEED + fold,
    )
    train_labels = [int(row["label"]) for row in train_rows]
    counts = np.bincount(train_labels, minlength=2)
    class_weights = torch.tensor(
        [len(train_labels) / (2 * count) for count in counts], dtype=torch.float32, device=device
    )

    set_seed(RANDOM_SEED + fold)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, local_files_only=True
    ).to(device)
    loader = DataLoader(
        TextDataset(train_rows, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED + fold),
    )
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

    losses = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            loss = loss_function(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(float(loss.detach().cpu()))
        mean_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(mean_loss)
        print(f"Fold {fold}/{N_SPLITS}, epoch {epoch}/{EPOCHS}, loss={mean_loss:.4f}", flush=True)

    validation_scores = predict_probabilities(model, validation_rows, tokenizer, device)
    validation_labels = [int(row["label"]) for row in validation_rows]
    threshold, validation_balanced_accuracy, validation_f1 = choose_threshold(
        validation_labels, validation_scores
    )

    test_scores = predict_probabilities(model, test_rows, tokenizer, device)
    test_labels = [int(row["label"]) for row in test_rows]
    test_predictions = [int(score >= threshold) for score in test_scores]
    rows = []
    for row, label, prediction, score in zip(test_rows, test_labels, test_predictions, test_scores):
        rows.append({
            "item_id": row["item_id"],
            "community": row["community"],
            "admissions_cycle": row["admissions_cycle"],
            "true_label": label,
            "predicted_label": prediction,
            "anchor_probability": score,
            "fold": fold,
            "fold_threshold": threshold,
        })
    fold_metrics = calculate_metrics(test_labels, test_predictions, test_scores)
    details = {
        "fold": fold,
        "train_records": len(train_rows),
        "validation_records": len(validation_rows),
        "test_records": len(test_rows),
        "class_weights": {"non_anchor": float(class_weights[0]), "anchor": float(class_weights[1])},
        "selected_threshold": threshold,
        "validation_balanced_accuracy": validation_balanced_accuracy,
        "validation_f1": validation_f1,
        "epoch_losses": losses,
        **fold_metrics,
    }
    print(
        f"Fold {fold} threshold={threshold:.3f}, test balanced accuracy={fold_metrics['balanced_accuracy']:.3f}, F1={fold_metrics['f1']:.3f}",
        flush=True,
    )
    del model
    return rows, details


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, labels = read_rows()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True, use_fast=False)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    indices = np.arange(len(rows))
    predictions = []
    folds = []
    print(f"Device: {device}; records: {len(rows)}; model: {MODEL_NAME}", flush=True)
    for fold, (development_indices, test_indices) in enumerate(splitter.split(indices, labels), 1):
        development_rows = [rows[index] for index in development_indices]
        test_rows = [rows[index] for index in test_indices]
        fold_predictions, fold_details = train_fold(
            fold, development_rows, test_rows, tokenizer, device
        )
        predictions.extend(fold_predictions)
        folds.append(fold_details)

    predictions.sort(key=lambda row: row["item_id"])
    with PREDICTIONS_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)

    true_labels = [int(row["true_label"]) for row in predictions]
    predicted_labels = [int(row["predicted_label"]) for row in predictions]
    scores = [float(row["anchor_probability"]) for row in predictions]
    aggregate = calculate_metrics(true_labels, predicted_labels, scores)
    summary = {
        "experiment": "class-weighted DeBERTa with fold-local validation threshold selection",
        "model": MODEL_NAME,
        "ground_truth_file": str(GROUND_TRUTH_FILE),
        "records": len(rows),
        "class_counts": {"non_anchor": labels.count(0), "anchor": labels.count(1)},
        "outer_folds": N_SPLITS,
        "validation_fraction_within_outer_development_set": VALIDATION_FRACTION,
        "threshold_objective": "maximum validation balanced accuracy; F1 tie-break",
        "folds": folds,
        "aggregate_out_of_fold": aggregate,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "interpretation_status": "feasibility; test posts excluded from training and threshold selection",
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2), flush=True)
    print(f"Summary: {SUMMARY_FILE}", flush=True)


if __name__ == "__main__":
    main()
