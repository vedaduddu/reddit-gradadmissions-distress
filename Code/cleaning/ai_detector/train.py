"""Retrain the portable TF-IDF logistic detector from the saved RQ5 splits.

The original C_xgboost joblib was never kept. This retrains the near-equivalent
portable model (feature set B, logistic regression, C=3.0) which scored 0.948
macro-F1 on the original 16k-sample run, and evaluates it on the current test
split. The saved pipeline takes raw strings (we preprocess inside).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from .preprocess import preprocess_text

REPO = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "models"
TRAIN = REPO / "rq5_output" / "train_split.csv"
TEST = REPO / "rq5_output" / "test_split.csv"

# Match the original portable (feature-set B) hyperparameters from run_metadata.
TFIDF = dict(ngram_range=(1, 2), min_df=10, max_features=8000, sublinear_tf=True)
LOGREG_C = 3.0
SEED = 42
# Operating point from the original C_xgboost threshold sweep (max F1).
THRESHOLD = 0.406


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["text", "text_clean", "label", "platform"])
    df["text_clean"] = df["text_clean"].fillna("").astype(str)
    empty = df["text_clean"].str.strip() == ""
    df.loc[empty, "text_clean"] = df.loc[empty, "text"].map(preprocess_text)
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = _load(TRAIN)
    test = _load(TEST)
    print(f"[train] {len(train):,} rows  [test] {len(test):,} rows")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF)),
        ("clf", LogisticRegression(
            penalty="l2", C=LOGREG_C, class_weight="balanced",
            solver="saga", max_iter=3000, random_state=SEED, n_jobs=-1,
        )),
    ])
    pipe.fit(train["text_clean"], train["label"].values)

    y_true = test["label"].values
    y_proba = pipe.predict_proba(test["text_clean"])[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "test_auc": float(roc_auc_score(y_true, y_proba)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "positive_class": "Moltbook / AI agent (label=1)",
        "threshold_default": THRESHOLD,
        "note": (
            "Trained on Moltbook vs Reddit in 5 communities. This is a style "
            "detector, not a general LLM-text detector."
        ),
    }
    print(classification_report(y_true, y_pred, target_names=["human", "ai_agent"]))
    print(json.dumps({k: metrics[k] for k in ("test_accuracy", "test_macro_f1", "test_auc")}, indent=2))

    blob = {
        "pipeline": pipe,
        "threshold": THRESHOLD,
        "meta": metrics,
    }
    model_path = OUT / "tfidf_logreg.joblib"
    joblib.dump(blob, model_path)
    (OUT / "metadata.json").write_text(json.dumps({
        "model": "tfidf_logreg",
        "feature_set": "B (TF-IDF unigram+bigram)",
        "original_paper_model": "C_xgboost was best (0.952) but joblib was not saved; "
                                "this portable B_logreg was 0.948 on the 16k run.",
        **metrics,
        "tfidf": {**TFIDF, "ngram_range": list(TFIDF["ngram_range"])},
        "logreg_C": LOGREG_C,
    }, indent=2))
    print(f"[train] wrote {model_path}")


if __name__ == "__main__":
    main()
