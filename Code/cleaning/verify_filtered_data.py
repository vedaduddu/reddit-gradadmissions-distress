"""Verify the delivered standardized filtered Reddit files.

This script does not change the data. It checks that each gzip file contains
valid JSON records and that its observed record count matches the filtration
pipeline report.
"""
from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "Data" / "standardized_filtered"
METRICS_FILE = DATA_DIR / "bot_filter" / "pipeline_metrics.json"
REPORT_DIR = PROJECT_DIR / "Data" / "verification"
REPORT_FILE = REPORT_DIR / "filtered_data_verification.json"

FILTERED_FILES = [
    "MBA_comments_filtered.jsonl.gz",
    "MBA_posts_filtered.jsonl.gz",
    "MSCS_comments_filtered.jsonl.gz",
    "MSCS_posts_filtered.jsonl.gz",
    "gradadmissions_comments_filtered.jsonl.gz",
    "gradadmissions_posts_filtered.jsonl.gz",
]


def expected_values(filename: str) -> tuple[str, str]:
    community, record_group = filename.split("_", maxsplit=1)
    record_type = "comment" if record_group.startswith("comments") else "post"
    return community, record_type


def source_name(filtered_name: str) -> str:
    return filtered_name.replace("_filtered.jsonl.gz", ".jsonl.gz")


def inspect_file(path: Path) -> dict:
    expected_community, expected_type = expected_values(path.name)
    count = 0
    duplicate_ids = 0
    wrong_community = 0
    wrong_type = 0
    missing_ids = 0
    seen_ids: set[str] = set()

    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path.name}, line {line_number}") from error

            count += 1
            record_id = record.get("id")
            if not record_id:
                missing_ids += 1
            elif record_id in seen_ids:
                duplicate_ids += 1
            else:
                seen_ids.add(record_id)

            wrong_community += record.get("community") != expected_community
            wrong_type += record.get("record_type") != expected_type

    return {
        "records": count,
        "duplicate_ids": duplicate_ids,
        "missing_ids": missing_ids,
        "wrong_community": wrong_community,
        "wrong_record_type": wrong_type,
    }


def main() -> None:
    metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    all_passed = True
    file_results = {}

    for filename in FILTERED_FILES:
        path = DATA_DIR / filename
        observed = inspect_file(path)
        expected_count = metrics["files"][source_name(filename)]["kept"]

        problems = {
            "count_difference": observed["records"] - expected_count,
            "duplicate_ids": observed["duplicate_ids"],
            "missing_ids": observed["missing_ids"],
            "wrong_community": observed["wrong_community"],
            "wrong_record_type": observed["wrong_record_type"],
        }
        passed = all(value == 0 for value in problems.values())
        all_passed &= passed
        file_results[filename] = {
            "passed": passed,
            "expected_records": expected_count,
            **observed,
            **problems,
        }

        status = "PASS" if passed else "FAIL"
        print(f"{status}  {filename}: {observed['records']:,} records")
        if not passed:
            print(f"      {problems}")

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "passed": all_passed,
        "metrics_file": str(METRICS_FILE.relative_to(PROJECT_DIR)),
        "files": file_results,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\nVerification passed." if all_passed else "\nVerification failed.")
    print(f"Report: {REPORT_FILE}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
