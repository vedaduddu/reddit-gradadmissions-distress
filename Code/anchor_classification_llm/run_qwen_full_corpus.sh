#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$HOME/reddit-gradadmissions-distress"
CODE_DIR="$PROJECT_DIR/Code/anchor_classification_llm"
LOG_DIR="$PROJECT_DIR/logs/qwen_full_corpus_v6"

cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1

echo "Validating code"

.venv/bin/python -m py_compile \
    "$CODE_DIR/prepare_full_corpus.py" \
    "$CODE_DIR/run_qwen_full_corpus.py" \
    "$CODE_DIR/combine_qwen_full_corpus.py"

echo "Preparing eligible posts and four deterministic shards"

.venv/bin/python "$CODE_DIR/prepare_full_corpus.py" \
    2>&1 | tee "$LOG_DIR/prepare.log"

run_shard() {
    local shard="$1"
    local gpu="$2"

    echo "Starting shard $shard on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" \
        .venv/bin/python "$CODE_DIR/run_qwen_full_corpus.py" \
        --shard "$shard" \
        2>&1 | tee "$LOG_DIR/shard${shard}.log"
}

echo "Launching four shards"

run_shard 0 0 &
pid0=$!

run_shard 1 1 &
pid1=$!

run_shard 2 2 &
pid2=$!

run_shard 3 3 &
pid3=$!

wait "$pid0"
wait "$pid1"
wait "$pid2"
wait "$pid3"

echo "All shards finished; combining results"

.venv/bin/python "$CODE_DIR/combine_qwen_full_corpus.py" \
    2>&1 | tee "$LOG_DIR/combine.log"

echo "Whole-corpus v6 classification completed"
