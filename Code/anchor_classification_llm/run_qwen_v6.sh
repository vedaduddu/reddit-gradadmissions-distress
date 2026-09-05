#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$HOME/reddit-gradadmissions-distress"
PYTHON_FILE="$PROJECT_DIR/Code/anchor_classification_llm/run_qwen_structured_cv.py"
LOG_DIR="$PROJECT_DIR/logs/qwen_structured_cv_6"
RESULTS_DIR="$PROJECT_DIR/Data/anchor_classification_llm/qwen_structured_cv_6"

cd "$PROJECT_DIR"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

export PYTHONUNBUFFERED=1

run_fold() {
    local fold="$1"
    local gpu="$2"
    local log_file="$LOG_DIR/fold${fold}.log"

    echo "Starting fold $fold on GPU $gpu"

    CUDA_VISIBLE_DEVICES="$gpu" \
        .venv/bin/python "$PYTHON_FILE" \
        --fold "$fold" \
        2>&1 | tee "$log_file"

    echo "Fold $fold process completed"
}

echo "Validating code and prompt"

test -s \
    "$PROJECT_DIR/Code/anchor_classification_llm/anchor_definition_v6.txt"

grep -q "qwen_structured_cv_6" "$PYTHON_FILE"
grep -q "anchor_definition_v6.txt" "$PYTHON_FILE"

.venv/bin/python -m tabnanny "$PYTHON_FILE"
.venv/bin/python -m py_compile "$PYTHON_FILE"

echo "Launching folds 1–4"

run_fold 1 0 &
pid1=$!

run_fold 2 1 &
pid2=$!

run_fold 3 2 &
pid3=$!

run_fold 4 3 &
pid4=$!

echo "Waiting for fold 1 to release GPU 0"

wait "$pid1"

echo "Fold 1 completed; starting fold 5 on GPU 0"

run_fold 5 0

echo "Fold 5 completed; waiting for folds 2–4"

wait "$pid2"
wait "$pid3"
wait "$pid4"

echo "All folds completed; combining results"

.venv/bin/python "$PYTHON_FILE" \
    --combine \
    2>&1 | tee "$LOG_DIR/combine.log"

echo "V6 evaluation completed successfully"
echo "Results: $RESULTS_DIR"
echo "Summary: $RESULTS_DIR/qwen_structured_cv_summary.json"