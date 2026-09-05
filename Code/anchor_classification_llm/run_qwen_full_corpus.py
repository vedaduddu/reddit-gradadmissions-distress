"""Classify one prepared whole-corpus shard with the frozen v6 prompt."""
from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime, timezone

import pandas as pd

import run_qwen_structured_cv as qwen
from full_corpus_config import (
    DEMOS_PER_CLASS,
    GROUND_TRUTH_FILE,
    N_SHARDS,
    PREPARED_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard",
        required=True,
        type=int,
        choices=range(N_SHARDS),
        help="Prepared shard number, from 0 through 3.",
    )
    return parser.parse_args()


def choose_demonstrations() -> pd.DataFrame:
    ground_truth = pd.read_csv(GROUND_TRUTH_FILE)
    non_anchors = ground_truth[ground_truth["label"] == 0].sample(
        n=DEMOS_PER_CLASS,
        random_state=RANDOM_SEED,
    )
    anchors = ground_truth[ground_truth["label"] == 1].sample(
        n=DEMOS_PER_CLASS,
        random_state=RANDOM_SEED + 100,
    )
    return pd.concat([non_anchors, anchors], ignore_index=True).sample(
        frac=1,
        random_state=RANDOM_SEED + 200,
    )


def input_file(shard: int):
    return PREPARED_DIR / f"machine_posts_shard_{shard}.jsonl.gz"


def prediction_file(shard: int):
    return RESULTS_DIR / f"predictions_shard_{shard}.jsonl"


def error_file(shard: int):
    return RESULTS_DIR / f"errors_shard_{shard}.jsonl"


def append_json(path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def completed_ids(path) -> set[str]:
    if not path.exists():
        return set()
    identifiers = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                identifiers.add(str(json.loads(line)["id"]))
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(
                    f"Invalid existing prediction at {path}, line {line_number}"
                ) from error
    return identifiers


def prediction_record(post: dict, classification: dict) -> dict:
    result = classification["result"]
    predicted_label = int(result["final_label"] == "anchor")
    return {
        "id": str(post["id"]),
        "post_id": post.get("post_id"),
        "community": post.get("community"),
        "admissions_cycle": post.get("admissions_cycle"),
        "created_utc": post.get("created_utc"),
        "created_at": post.get("created_at"),
        "author": post.get("author"),
        "title": post.get("title"),
        "body": post.get("body"),
        "text": post.get("text"),
        "score": post.get("score"),
        "num_comments": post.get("num_comments"),
        "permalink": post.get("permalink"),
        "predicted_label": predicted_label,
        "predicted_label_text": result["final_label"],
        "label_source": "qwen_v6",
        "model_final_label": result.get("model_final_label"),
        "checklist_final_label": result.get("checklist_final_label"),
        "final_label_was_overridden": result.get(
            "final_label_was_overridden"
        ),
        "confidence": result.get("confidence"),
        "graduate_education_relevance": result.get(
            "graduate_education_relevance"
        ),
        "applicant_timing_scope": result.get("applicant_timing_scope"),
        "realized_or_salient_adverse_outcome": result.get(
            "realized_or_salient_adverse_outcome"
        ),
        "explicit_negative_affect": result.get("explicit_negative_affect"),
        "expected_failure_or_application_threat": result.get(
            "expected_failure_or_application_threat"
        ),
        "threatening_comparison_or_vicarious_threat": result.get(
            "threatening_comparison_or_vicarious_threat"
        ),
        "funding_affordability_or_roi_threat": result.get(
            "funding_affordability_or_roi_threat"
        ),
        "career_or_degree_outcome_threat": result.get(
            "career_or_degree_outcome_threat"
        ),
        "life_burden_withdrawal_or_abandonment": result.get(
            "life_burden_withdrawal_or_abandonment"
        ),
        "neutral_information_or_routine_uncertainty": result.get(
            "neutral_information_or_routine_uncertainty"
        ),
        "evidence_phrases": result.get("evidence_phrases", []),
        "counterevidence_phrases": result.get("counterevidence_phrases", []),
        "decision_basis": result.get("decision_basis", ""),
        "prompt_tokens": classification["prompt_tokens"],
        "generated_tokens": classification["generated_tokens"],
        "seconds": classification["seconds"],
        "model": qwen.MODEL_NAME,
    }


def main() -> None:
    args = parse_arguments()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    source = input_file(args.shard)
    destination = prediction_file(args.shard)
    completed = completed_ids(destination)
    demonstrations = choose_demonstrations()

    with gzip.open(source, "rt", encoding="utf-8") as file:
        total = sum(1 for line in file if line.strip())

    print(
        f"Shard {args.shard}: {total:,} posts; "
        f"resuming with {len(completed):,}",
        flush=True,
    )
    tokenizer, model = qwen.load_model()

    with gzip.open(source, "rt", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            post = json.loads(line)
            post_id = str(post["id"])
            if post_id in completed:
                continue

            try:
                classification = qwen.classify_post(
                    post.get("text") or post.get("body") or "",
                    demonstrations,
                    tokenizer,
                    model,
                )
                append_json(
                    destination,
                    prediction_record(post, classification),
                )
                completed.add(post_id)
                if len(completed) % 10 == 0 or len(completed) == total:
                    print(
                        f"Shard {args.shard}: completed "
                        f"{len(completed):,}/{total:,}",
                        flush=True,
                    )
            except Exception as error:
                append_json(
                    error_file(args.shard),
                    {
                        "id": post_id,
                        "shard": args.shard,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "recorded_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                print(
                    f"Shard {args.shard}: error for {post_id}: {error}",
                    flush=True,
                )

    print(f"Shard {args.shard} finished", flush=True)


if __name__ == "__main__":
    main()
