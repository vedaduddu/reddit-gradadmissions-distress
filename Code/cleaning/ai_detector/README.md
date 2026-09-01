# AI-agent vs human style detector

Portable scorer trained on **Moltbook (AI agents)** vs **Reddit (humans)** in five
matched communities (consciousness, philosophy, technology, trading, offmychest).

This is a **register/style detector**, not a general “was this written by ChatGPT”
classifier. Scores will not transfer cleanly to essays, emails, or other domains.

## Status of the original models

RQ5 trained six classifiers (LIWC / TF-IDF / combined × logreg / XGBoost). The
best in-distribution model was **XGBoost on combined features** (test acc 0.952).
The training script was supposed to write `rq5_output/best_model_pipeline.joblib`,
but that file was never kept. Metrics, splits, and feature tables are in
`rq5_output/`.

This package retrains the **portable** model: TF-IDF (uni+bigram) + L2 logistic
regression (`C=3.0`). On the original 16k-sample run that model scored **0.948**
test accuracy — nearly identical, and it only needs `sklearn` + `joblib`.

The combined XGBoost is not shipped here because it depends on the proprietary
LIWC dictionary, which we cannot redistribute.

## Train

```bash
python -m ai_detector.train
```

Writes `ai_detector/models/tfidf_logreg.joblib`.

## Use

```python
from ai_detector import AIContentDetector

det = AIContentDetector()
print(det.predict_one("unsigned self-modification of agent skills is reckless"))
# {'label': 'ai_agent' or 'human', 'p_ai_agent': 0.87, 'threshold': 0.406}

df = det.score(["text one", "text two"])
```

CLI:

```bash
python -m ai_detector "some post or comment"
python -m ai_detector --file texts.txt --json
```

`p_ai_agent` is P(Moltbook-like). Default threshold 0.406 is the original max-F1
operating point from the C_xgboost sweep; pass a different cutoff yourself if
you want higher precision.

## Copy elsewhere

Take the whole `ai_detector/` folder (including `models/tfidf_logreg.joblib`).
Dependencies: `scikit-learn`, `joblib`, `pandas`, `numpy`. Sklearn versions
should be close to the one used to train, or reload may fail.
