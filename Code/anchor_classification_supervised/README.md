# Supervised DeBERTa anchor-classification feasibility

This branch fine-tunes `microsoft/deberta-v3-base` on Veda's 100-post ground-truth set.
It uses stratified five-fold cross-validation, so every post receives one
out-of-fold prediction from a model that did not train on that post.

Run:

```bash
python3 Code/anchor_classification_supervised/train_cross_validation.py
```

Results are written to `Data/anchor_classification_supervised/`. The experiment uses
fixed hyperparameters and does not save fold checkpoints. Its aggregate metrics
are a feasibility estimate, not a final independent test-set result.

Add probability-ranking and descriptive threshold diagnostics with:

```bash
python3 Code/anchor_classification_supervised/summarize_results.py
```
