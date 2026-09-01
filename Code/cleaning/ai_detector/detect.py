"""Load a trained AI-vs-human detector and score new text.

This is a *style* detector trained on Moltbook (AI agents) vs Reddit (humans)
in five matched communities. It is not a general ChatGPT/LLM-text detector.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

from .preprocess import preprocess_text

PKG_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = Path(os.environ.get("AI_DETECTOR_MODEL", PKG_DIR / "models" / "tfidf_logreg.joblib"))
DEFAULT_META = Path(os.environ.get("AI_DETECTOR_METADATA", PKG_DIR / "models" / "metadata.json"))

LABELS = {0: "human", 1: "ai_agent"}


class AIContentDetector:
    def __init__(self, model_path: Path | str | None = None):
        path = Path(model_path) if model_path else DEFAULT_MODEL
        if not path.exists():
            raise FileNotFoundError(
                f"No saved model at {path}. Train first: python -m ai_detector.train"
            )
        blob = joblib.load(path)
        self.pipeline = blob["pipeline"]
        self.threshold = float(blob.get("threshold", 0.5))
        self.meta = blob.get("meta", {})
        meta_path = Path(model_path).with_name("metadata.json") if model_path else DEFAULT_META
        if meta_path.exists():
            self.meta = {**json.loads(meta_path.read_text()), **self.meta}

    def score(self, texts: Iterable[str]) -> pd.DataFrame:
        raw = [str(t) if t is not None else "" for t in texts]
        cleaned = [preprocess_text(t) for t in raw]
        proba = self.pipeline.predict_proba(cleaned)[:, 1]
        pred = (proba >= self.threshold).astype(int)
        return pd.DataFrame({
            "text": raw,
            "p_ai_agent": proba,
            "label_id": pred,
            "label": [LABELS[int(x)] for x in pred],
        })

    def predict_one(self, text: str) -> dict:
        row = self.score([text]).iloc[0]
        return {
            "label": row["label"],
            "p_ai_agent": float(row["p_ai_agent"]),
            "threshold": self.threshold,
        }
