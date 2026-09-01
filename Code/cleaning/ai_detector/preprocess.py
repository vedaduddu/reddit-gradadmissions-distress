"""Text cleanup matching rq5_detection_analysis.preprocess_text."""
from __future__ import annotations

import re


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
