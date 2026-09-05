# Qwen anchor classification

`run_qwen_structured_cv.py` evaluates the frozen prompt with stratified
five-fold few-shot classification on the 400 labeled posts.

The whole-corpus pipeline is split into short steps:

1. `prepare_full_corpus.py` selects September-November original posts from the
   three final filtered community files, retains the 400 posts with human
   labels, and creates four deterministic machine-classification shards.
2. `run_qwen_full_corpus.py` classifies one shard with Qwen3-8B and the frozen
   v6 prompt. JSONL output is appended one record at a time and safely resumes
   from completed post IDs.
3. `combine_qwen_full_corpus.py` combines the four machine shards with the 400
   human-labeled posts. It writes the complete label file, the anchor-post
   file, a JSON summary, and a fresh 50-anchor/50-non-anchor manual-validation
   sample.
4. `run_qwen_full_corpus.sh` prepares, runs one shard on each of four GPUs, and
   combines the results. Run this script inside a detached `tmux` session.

Generated data, logs, and model files remain outside Git.
