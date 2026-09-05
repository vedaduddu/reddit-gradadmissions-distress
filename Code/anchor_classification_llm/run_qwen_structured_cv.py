from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-8B"
RANDOM_SEED = 20260828
N_FOLDS = 5
DEMOS_PER_CLASS = 5

MAX_DEMO_CHARACTERS = 2500
MAX_TARGET_CHARACTERS = 7000
MAX_NEW_TOKENS = 6000

QUALIFYING_FIELDS = [
    "realized_or_salient_adverse_outcome",
    "explicit_negative_affect",
    "expected_failure_or_application_threat",
    "threatening_comparison_or_vicarious_threat",
    "funding_affordability_or_roi_threat",
    "career_or_degree_outcome_threat",
    "life_burden_withdrawal_or_abandonment",
]

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = (PROJECT_DIR /"Data"/"anchor_ground_truth"/"combined_400_ground_truth.csv")
RESULTS_DIR = (PROJECT_DIR /"Data"/"anchor_classification_llm"/"qwen_structured_cv_6")

PROMPT_FILE = Path(__file__).with_name("anchor_definition_v6.txt")

if not PROMPT_FILE.exists():
    raise FileNotFoundError(f"Prompt file not found: {PROMPT_FILE}")

DEFINITION = PROMPT_FILE.read_text(encoding="utf-8").strip()

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int,
        choices=range(1, N_FOLDS + 1),
        help="Run one fold, numbered 1 through 5.",
    )

    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine completed fold predictions and calculate final metrics.",
    )

    return parser.parse_args()

def shorten(text, maximum):
    text = str(text).strip()

    if len(text) <= maximum:
        return text

    return text[:maximum] + "\n[POST TRUNCATED]"


def load_data():
    df = pd.read_csv(DATA_FILE)

    required = {
        "item_id",
        "community",
        "admissions_cycle",
        "text",
        "label",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing)}"
        )

    if df["item_id"].astype(str).duplicated().any():
        raise ValueError("item_id values must be unique")

    df["item_id"] = df["item_id"].astype(str)
    df["label"] = df["label"].astype(int)

    labels = set(df["label"])

    if labels != {0, 1}:
        raise ValueError(
            f"Expected binary labels 0 and 1; found {labels}"
        )

    return assign_folds(df)


def assign_folds(df):
    df = df.copy()
    df["fold"] = 0

    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    for fold, (_, test_indices) in enumerate(
        splitter.split(df, df["label"]),
        start=1,
    ):
        df.loc[df.index[test_indices], "fold"] = fold

    return df


def choose_demonstrations(training_df, fold):
    non_anchors = training_df[
        training_df["label"] == 0
    ].sample(
        n=DEMOS_PER_CLASS,
        random_state=RANDOM_SEED + fold,
    )

    anchors = training_df[
        training_df["label"] == 1
    ].sample(
        n=DEMOS_PER_CLASS,
        random_state=RANDOM_SEED + 100 + fold,
    )

    demonstrations = pd.concat(
        [non_anchors, anchors],
        ignore_index=True,
    )

    return demonstrations.sample(
        frac=1,
        random_state=RANDOM_SEED + 200 + fold,
    )


def format_demonstrations(demonstrations):
    sections = []

    for number, row in enumerate(
        demonstrations.itertuples(),
        start=1,
    ):
        label = (
            "anchor"
            if int(row.label) == 1
            else "non_anchor"
        )

        text = shorten(
            row.text,
            MAX_DEMO_CHARACTERS,
        )

        sections.append(
            f"""EXAMPLE {number}

POST:
{text}

CORRECT FINAL LABEL:
{label}"""
        )

    return "\n\n".join(sections)


def build_messages(target_text, demonstrations):
    examples = format_demonstrations(demonstrations)

    target_text = shorten(
        target_text,
        MAX_TARGET_CHARACTERS,
    )

    system_message = f"""
You are a careful research classifier.

{DEFINITION}

Return one JSON object and no markdown.

Use this exact schema:

{{
  "graduate_education_relevance": true or false,
  "applicant_timing_scope": true or false,
  "realized_or_salient_adverse_outcome": true or false,
  "explicit_negative_affect": true or false,
  "expected_failure_or_application_threat": true or false,
  "threatening_comparison_or_vicarious_threat": true or false,
  "funding_affordability_or_roi_threat": true or false,
  "career_or_degree_outcome_threat": true or false,
  "life_burden_withdrawal_or_abandonment": true or false,
  "neutral_information_or_routine_uncertainty": true or false,
  "evidence_phrases": ["exact phrase from the post"],
  "counterevidence_phrases": ["exact phrase from the post"],
  "decision_basis": "A concise explanation of at most three sentences.",
  "final_label": "anchor" or "non_anchor",
  "confidence": number from 0.0 to 1.0
}}

Do not invent or paraphrase evidence.

Every phrase in evidence_phrases and counterevidence_phrases must appear
verbatim in the target post.

Confidence is the model's self-assessment and is not a calibrated
probability.
""".strip()

    user_message = f"""
Here are labeled examples drawn only from the training portion of this fold:

{examples}

Now classify this held-out post.

POST:
{target_text}
""".strip()

    return [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

def load_model():
    print(f"Loading {MODEL_NAME}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )

    model.eval()
    return tokenizer, model


def extract_thinking(response):
    match = re.search(r"<think>(.*?)</think>", response, flags=re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()

def extract_json(response):
    visible_response = re.sub(
        r"<think>.*?</think>",
        "",
        response,
        flags=re.DOTALL,
    ).strip()

    decoder = json.JSONDecoder()
    candidates = []

    for position, character in enumerate(visible_response):
        if character != "{":
            continue

        try:
            parsed, _ = decoder.raw_decode(
                visible_response[position:]
            )
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict) and "final_label" in parsed:
            candidates.append(parsed)

    if not candidates:
        raise ValueError("No valid classification JSON found")

    result = candidates[-1]

    required_boolean_fields = [
        "graduate_education_relevance",
        "applicant_timing_scope",
        *QUALIFYING_FIELDS,
        "neutral_information_or_routine_uncertainty",
    ]

    missing_fields = [
        field
        for field in required_boolean_fields
        if field not in result
    ]

    if missing_fields:
        raise ValueError(
            "Classification JSON is missing fields: "
            f"{missing_fields}"
        )

    non_boolean_fields = [
        field
        for field in required_boolean_fields
        if not isinstance(result[field], bool)
    ]

    if non_boolean_fields:
        raise ValueError(
            "Checklist fields must be JSON booleans: "
            f"{non_boolean_fields}"
        )

    model_final_label = str(
        result["final_label"]
    ).strip().lower()

    if model_final_label not in {"anchor", "non_anchor"}:
        raise ValueError(
            f"Invalid final_label: {model_final_label}"
        )

    in_scope = (
        result["graduate_education_relevance"]
        and result["applicant_timing_scope"]
    )

    has_qualifying_signal = any(
        result[field]
        for field in QUALIFYING_FIELDS
    )

    checklist_final_label = (
        "anchor"
        if in_scope and has_qualifying_signal
        else "non_anchor"
    )

    result["model_final_label"] = model_final_label
    result["checklist_final_label"] = checklist_final_label
    result["final_label_was_overridden"] = (
        model_final_label != checklist_final_label
    )

    # The final prediction follows the structured decision rule.
    result["final_label"] = checklist_final_label

    return result


def classify_post(text, demonstrations, tokenizer, model):
    messages = build_messages(text, demonstrations)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)

    started_at = time.time()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0, inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    result = extract_json(response)

    return {
        "result": result,
        "reasoning_trace": extract_thinking(response),
        "raw_response": response,
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "generated_tokens": int(generated_tokens.shape[0]),
        "seconds": time.time() - started_at,
    }


def prediction_file(fold):
    return RESULTS_DIR / f"predictions_fold_{fold}.csv"


def error_file(fold):
    return RESULTS_DIR / f"errors_fold_{fold}.jsonl"


def load_existing_predictions(fold):
    path = prediction_file(fold)

    if not path.exists():
        return []

    return pd.read_csv(path).to_dict("records")


def save_predictions(rows, fold):
    pd.DataFrame(rows).to_csv(
        prediction_file(fold),
        index=False,
    )


def append_error(error, fold):
    with error_file(fold).open("a", encoding="utf-8") as file:
        file.write(json.dumps(error, ensure_ascii=False) + "\n")


def run_fold(fold):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    training_df = df[df["fold"] != fold].copy()
    testing_df = df[df["fold"] == fold].copy()

    demonstrations = choose_demonstrations(training_df, fold)
    demonstration_ids = set(demonstrations["item_id"])

    predictions = load_existing_predictions(fold)
    completed_ids = {
        str(row["item_id"])
        for row in predictions
    }

    print(
        f"Fold {fold}: {len(training_df)} training records, "
        f"{len(testing_df)} held-out records",
        flush=True,
    )
    print(
        f"Resuming with {len(completed_ids)} completed predictions",
        flush=True,
    )

    tokenizer, model = load_model()

    for row in testing_df.itertuples():
        item_id = str(row.item_id)

        if item_id in completed_ids:
            continue

        if item_id in demonstration_ids:
            raise RuntimeError("Held-out item appeared in demonstrations")

        try:
            classification = classify_post(
                row.text,
                demonstrations,
                tokenizer,
                model,
            )

            result = classification["result"]
            predicted_label = int(result["final_label"] == "anchor")

            predictions.append(
                {
                    "item_id": item_id,
                    "community": row.community,
                    "admissions_cycle": row.admissions_cycle,
                    "fold": fold,
                    "true_label": int(row.label),
                                        "predicted_label": predicted_label,
                    "predicted_label_text": result["final_label"],
                    "model_final_label":
                        result.get("model_final_label"),
                    "checklist_final_label":
                        result.get("checklist_final_label"),
                    "final_label_was_overridden":
                        result.get("final_label_was_overridden"),
                    "confidence": result.get("confidence"),
                    "graduate_education_relevance":
                        result.get("graduate_education_relevance"),
                    "applicant_timing_scope":
                        result.get("applicant_timing_scope"),
                    "realized_or_salient_adverse_outcome":
                        result.get(
                            "realized_or_salient_adverse_outcome"
                        ),
                    "explicit_negative_affect":
                        result.get("explicit_negative_affect"),
                    "expected_failure_or_application_threat":
                        result.get(
                            "expected_failure_or_application_threat"
                        ),
                    "threatening_comparison_or_vicarious_threat":
                        result.get(
                            "threatening_comparison_or_vicarious_threat"
                        ),
                    "funding_affordability_or_roi_threat":
                        result.get(
                            "funding_affordability_or_roi_threat"
                        ),
                    "career_or_degree_outcome_threat":
                        result.get(
                            "career_or_degree_outcome_threat"
                        ),
                    "life_burden_withdrawal_or_abandonment":
                        result.get(
                            "life_burden_withdrawal_or_abandonment"
                        ),
                    "neutral_information_or_routine_uncertainty":
                        result.get(
                            "neutral_information_or_routine_uncertainty"
                        ),
                    "evidence_phrases": json.dumps(
                        result.get("evidence_phrases", []),
                        ensure_ascii=False,
                    ),
                    "counterevidence_phrases": json.dumps(
                        result.get("counterevidence_phrases", []),
                        ensure_ascii=False,
                    ),
                    "decision_basis": result.get("decision_basis", ""),
                    "reasoning_trace":
                        classification["reasoning_trace"],
                    "raw_response":
                        classification["raw_response"],
                    "prompt_tokens":
                        classification["prompt_tokens"],
                    "generated_tokens":
                        classification["generated_tokens"],
                    "seconds": classification["seconds"],
                    "model": MODEL_NAME,
                }
            )

            completed_ids.add(item_id)
            save_predictions(predictions, fold)

            print(
                f"Fold {fold}: completed "
                f"{len(completed_ids)}/{len(testing_df)}",
                flush=True,
            )

        except Exception as error:
            append_error(
                {
                    "item_id": item_id,
                    "fold": fold,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                },
                fold,
            )

            print(
                f"Fold {fold}: error for {item_id}: {error}",
                flush=True,
            )

    print(f"Fold {fold} finished", flush=True)


def calculate_metrics(predictions):
    y_true = predictions["true_label"].astype(int)
    y_pred = predictions["predicted_label"].astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    if predictions["confidence"].notna().all():
        anchor_score = predictions.apply(
            lambda row: (
                float(row["confidence"])
                if int(row["predicted_label"]) == 1
                else 1.0 - float(row["confidence"])
            ),
            axis=1,
        )

        metrics["roc_auc_from_self_reported_confidence"] = roc_auc_score(
            y_true,
            anchor_score,
        )

    return metrics


def combine_folds():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frames = []

    for fold in range(1, N_FOLDS + 1):
        path = prediction_file(fold)

        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")

        frame = pd.read_csv(path)
        frames.append(frame)

    predictions = pd.concat(frames, ignore_index=True)

    if predictions["item_id"].astype(str).duplicated().any():
        raise ValueError("Combined predictions contain duplicate item IDs")

    ground_truth = load_data()
    expected_records = len(ground_truth)

    expected_ids = set(
        ground_truth["item_id"].astype(str)
    )

    observed_ids = set(
        predictions["item_id"].astype(str)
    )

    missing_item_ids = sorted(
        expected_ids.difference(observed_ids)
    )

    if len(predictions) > expected_records:
        raise ValueError(
            f"Expected at most {expected_records} predictions; "
            f"found {len(predictions)}"
        )

    if missing_item_ids:
        print(
            f"Warning: combining with {len(missing_item_ids)} "
            f"missing predictions: {missing_item_ids}",
            flush=True,
        )

    predictions = predictions.sort_values("item_id")
    combined_file = RESULTS_DIR / "qwen_structured_cv_predictions.csv"
    predictions.to_csv(combined_file, index=False)

    errors = predictions[
        predictions["true_label"].astype(int)
        != predictions["predicted_label"].astype(int)
    ].copy()

    errors_file = RESULTS_DIR / "qwen_structured_cv_errors.csv"
    errors.to_csv(errors_file, index=False)

    metrics = calculate_metrics(predictions)
    summary = {
    "model": MODEL_NAME,
    "evaluation":
        "stratified five-fold few-shot structured-reasoning classification",
    "expected_records": expected_records,
    "records_evaluated": len(predictions),
    "missing_records": len(missing_item_ids),
    "missing_item_ids": missing_item_ids,
    "demonstrations_per_class": DEMOS_PER_CLASS,
    "random_seed": RANDOM_SEED,
    "thinking_enabled": True,
    "metrics": metrics,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "notes": [
        "Every post was predicted only in its held-out fold.",
        "Demonstrations were selected only from the corresponding training fold.",
        "Confidence is model-reported and is not a calibrated probability.",
        "Metrics exclude records for which the model did not return valid structured JSON.",
        "Final predictions were derived deterministically from graduate-education relevance, applicant-timing scope, and the qualifying-category checklist.",
        "The model's originally returned final label and any checklist override were retained in the prediction files.",
        ],
    }

    summary_file = RESULTS_DIR / "qwen_structured_cv_summary.json"
    summary_file.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2), flush=True)
    print(f"Predictions: {combined_file}", flush=True)
    print(f"Errors: {errors_file}", flush=True)
    print(f"Summary: {summary_file}", flush=True)


def main():
    args = parse_arguments()

    if args.combine:
        combine_folds()
    elif args.fold:
        run_fold(args.fold)
    else:
        raise SystemExit("Specify --fold 1 through 5, or --combine")


if __name__ == "__main__":
    main()