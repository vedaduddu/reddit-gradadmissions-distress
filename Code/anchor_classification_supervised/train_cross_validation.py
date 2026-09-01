"""Fine-tune DeBERTa-v3-base with stratified five-fold cross-validation."""
from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from config import (
    BATCH_SIZE,
    EPOCHS,
    GROUND_TRUTH_FILE,
    LEARNING_RATE,
    MAX_LENGTH,
    MODEL_NAME,
    N_SPLITS,
    OUTPUT_DIR,
    RANDOM_SEED,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)


PREDICTIONS_FILE = OUTPUT_DIR / "cross_validation_predictions.csv"
SUMMARY_FILE = OUTPUT_DIR / "cross_validation_summary.json"
TRAINING_LOG_FILE = OUTPUT_DIR / "training_log.jsonl"


class TextDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_ground_truth():
    with GROUND_TRUTH_FILE.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows or len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError("Ground truth must contain unique records")
    labels = [int(row["label"]) for row in rows]
    if set(labels) != {0, 1}:
        raise ValueError("Ground truth must contain both binary classes")
    return rows, labels


def metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train_fold(fold, train_rows, test_rows, tokenizer, device):
    fold_seed = RANDOM_SEED + fold
    set_seed(fold_seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, local_files_only=True
    ).to(device)
    train_loader = DataLoader(
        TextDataset(train_rows, tokenizer), batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(fold_seed),
    )
    test_loader = DataLoader(TextDataset(test_rows, tokenizer), batch_size=BATCH_SIZE, shuffle=False)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    epoch_logs = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(output.loss.detach().cpu()))
        epoch_logs.append({"epoch": epoch, "mean_training_loss": sum(losses) / len(losses)})
        print(f"Fold {fold}/{N_SPLITS}, epoch {epoch}/{EPOCHS}, loss={epoch_logs[-1]['mean_training_loss']:.4f}", flush=True)

    model.eval()
    predictions = []
    cursor = 0
    with torch.inference_mode():
        for batch in test_loader:
            labels = batch.pop("labels")
            inputs = {key: value.to(device) for key, value in batch.items()}
            probabilities = torch.softmax(model(**inputs).logits, dim=1).cpu()
            for offset, (label, probs) in enumerate(zip(labels.tolist(), probabilities.tolist())):
                row = test_rows[cursor + offset]
                predictions.append(
                    {
                        "item_id": row["item_id"],
                        "community": row["community"],
                        "admissions_cycle": row["admissions_cycle"],
                        "true_label": label,
                        "predicted_label": int(probs[1] >= 0.5),
                        "non_anchor_probability": probs[0],
                        "anchor_probability": probs[1],
                        "fold": fold,
                    }
                )
            cursor += len(labels)
    del model
    return predictions, epoch_logs


def write_csv(rows):
    with PREDICTIONS_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, labels = read_ground_truth()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; model: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, local_files_only=True, use_fast=False
    )
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    all_predictions = []
    fold_summaries = []
    TRAINING_LOG_FILE.write_text("", encoding="utf-8")
    indices = np.arange(len(rows))
    for fold, (train_indices, test_indices) in enumerate(splitter.split(indices, labels), 1):
        train_rows = [rows[index] for index in train_indices]
        test_rows = [rows[index] for index in test_indices]
        predictions, epoch_logs = train_fold(fold, train_rows, test_rows, tokenizer, device)
        all_predictions.extend(predictions)
        fold_metrics = metrics(
            [row["true_label"] for row in predictions],
            [row["predicted_label"] for row in predictions],
        )
        fold_summaries.append({
            "fold": fold,
            "train_records": len(train_rows),
            "test_records": len(test_rows),
            **fold_metrics,
        })
        with TRAINING_LOG_FILE.open("a", encoding="utf-8") as file:
            file.write(json.dumps({"fold": fold, "epochs": epoch_logs, "metrics": fold_metrics}) + "\n")
        print(f"Fold {fold} accuracy={fold_metrics['accuracy']:.3f}, F1={fold_metrics['f1']:.3f}", flush=True)

    all_predictions.sort(key=lambda row: row["item_id"])
    write_csv(all_predictions)
    aggregate = metrics(
        [row["true_label"] for row in all_predictions],
        [row["predicted_label"] for row in all_predictions],
    )
    summary = {
        "experiment": "supervised DeBERTa-v3-base feasibility with stratified five-fold cross-validation",
        "model": MODEL_NAME,
        "ground_truth_file": str(GROUND_TRUTH_FILE),
        "records": len(rows),
        "class_counts": {
            "non_anchor": labels.count(0),
            "anchor": labels.count(1),
        },
        "random_seed": RANDOM_SEED,
        "n_splits": N_SPLITS,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "decision_threshold": 0.5,
        "folds": fold_summaries,
        "aggregate_out_of_fold": aggregate,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "interpretation_status": "feasibility only; not an independent final test",
        "saved_model_checkpoints": False,
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2), flush=True)
    print(f"Summary: {SUMMARY_FILE}", flush=True)


if __name__ == "__main__":
    main()
