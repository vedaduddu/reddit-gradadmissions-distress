from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_PARENT = PROJECT_DIR / "Data" / "anchor_classification_llm"
V4_FILE = RESULTS_PARENT / "qwen_structured_cv_4" / "qwen_structured_cv_predictions.csv"
V5_DIR = RESULTS_PARENT / "qwen_structured_cv_5"
V5_FILE = V5_DIR / "qwen_structured_cv_predictions.csv"
GROUND_TRUTH_FILE = (
    PROJECT_DIR / "Data" / "anchor_ground_truth" / "combined_400_ground_truth.csv"
)
OUTPUT_FILE = V5_DIR / "v4_v5_item_comparison.csv"

CHECKLIST_FIELDS = [
    "graduate_education_relevance",
    "realized_or_salient_adverse_outcome",
    "explicit_negative_affect",
    "expected_failure_or_application_threat",
    "threatening_comparison_or_vicarious_threat",
    "funding_affordability_or_roi_threat",
    "career_or_degree_outcome_threat",
    "life_burden_withdrawal_or_abandonment",
    "neutral_information_or_routine_uncertainty",
]


def extract_result(raw_response):
    text = re.sub(
        r"<think>.*?</think>",
        "",
        str(raw_response),
        flags=re.DOTALL,
    ).strip()
    decoder = json.JSONDecoder()
    results = []

    for position, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "final_label" in value:
            results.append(value)

    return results[-1] if results else {}


def normalized_bool(value):
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return None
    if str(value).strip().lower() in {"true", "1"}:
        return True
    if str(value).strip().lower() in {"false", "0"}:
        return False
    return None


def result_value(row, parsed, field):
    if field in row.index:
        saved = normalized_bool(row[field])
        if saved is not None:
            return saved
    return normalized_bool(parsed.get(field))


def classify_change(true_label, v4_prediction, v5_prediction):
    if pd.isna(v5_prediction):
        return "missing_in_v5"

    v4_correct = int(v4_prediction) == int(true_label)
    v5_correct = int(v5_prediction) == int(true_label)

    if not v4_correct and v5_correct:
        return "corrected_by_v5"
    if v4_correct and not v5_correct:
        return "new_error_in_v5"
    if v4_correct and v5_correct:
        return "correct_in_both"
    if int(true_label) == 0:
        return "false_positive_in_both"
    return "false_negative_in_both"


def main():
    v4 = pd.read_csv(V4_FILE)
    v5 = pd.read_csv(V5_FILE)
    truth = pd.read_csv(GROUND_TRUTH_FILE)

    truth["item_id"] = truth["item_id"].astype(str)
    v4["item_id"] = v4["item_id"].astype(str)
    v5["item_id"] = v5["item_id"].astype(str)

    base = truth[
        ["item_id", "community", "admissions_cycle", "text", "label"]
    ].rename(columns={"label": "true_label"})

    v4_by_id = v4.set_index("item_id")
    v5_by_id = v5.set_index("item_id")
    output_rows = []

    for truth_row in base.itertuples(index=False):
        item_id = truth_row.item_id
        row4 = v4_by_id.loc[item_id]
        row5 = v5_by_id.loc[item_id] if item_id in v5_by_id.index else None
        parsed4 = extract_result(row4.get("raw_response", ""))
        parsed5 = (
            extract_result(row5.get("raw_response", ""))
            if row5 is not None
            else {}
        )

        pred4 = int(row4["predicted_label"])
        pred5 = (
            int(row5["predicted_label"])
            if row5 is not None
            else pd.NA
        )

        record = {
            "item_id": item_id,
            "community": truth_row.community,
            "admissions_cycle": truth_row.admissions_cycle,
            "true_label": int(truth_row.true_label),
            "true_label_text": (
                "anchor" if int(truth_row.true_label) == 1 else "non_anchor"
            ),
            "v4_predicted_label": pred4,
            "v4_predicted_label_text": row4.get("predicted_label_text", ""),
            "v5_predicted_label": pred5,
            "v5_predicted_label_text": (
                row5.get("predicted_label_text", "")
                if row5 is not None
                else ""
            ),
            "comparison_status": classify_change(
                truth_row.true_label,
                pred4,
                pred5,
            ),
            "v4_confidence": row4.get("confidence"),
            "v5_confidence": row5.get("confidence") if row5 is not None else pd.NA,
            "text": truth_row.text,
            "v4_evidence_phrases": row4.get("evidence_phrases", ""),
            "v5_evidence_phrases": (
                row5.get("evidence_phrases", "") if row5 is not None else ""
            ),
            "v4_counterevidence_phrases": row4.get(
                "counterevidence_phrases", ""
            ),
            "v5_counterevidence_phrases": (
                row5.get("counterevidence_phrases", "")
                if row5 is not None
                else ""
            ),
            "v4_decision_basis": row4.get("decision_basis", ""),
            "v5_decision_basis": (
                row5.get("decision_basis", "") if row5 is not None else ""
            ),
        }

        for field in CHECKLIST_FIELDS:
            record[f"v4_{field}"] = result_value(row4, parsed4, field)
            record[f"v5_{field}"] = (
                result_value(row5, parsed5, field)
                if row5 is not None
                else None
            )

        output_rows.append(record)

    output = pd.DataFrame(output_rows)
    status_order = {
        "corrected_by_v5": 0,
        "new_error_in_v5": 1,
        "false_positive_in_both": 2,
        "false_negative_in_both": 3,
        "missing_in_v5": 4,
        "correct_in_both": 5,
    }
    output["_status_order"] = output["comparison_status"].map(status_order)
    output = output.sort_values(["_status_order", "item_id"]).drop(
        columns="_status_order"
    )
    output.to_csv(OUTPUT_FILE, index=False)

    print(output["comparison_status"].value_counts().to_string())
    print(f"Rows: {len(output)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
