"""Prepare 150 keyword-pass and 150 keyword-fail posts for blinded labeling."""
from __future__ import annotations

import csv
import gzip
import json
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
FILTERED_DIR = PROJECT_DIR / "Data" / "standardized_filtered"
GROUND_TRUTH_DIR = PROJECT_DIR / "Data" / "anchor_ground_truth"
ARCHIVE_DIR = PROJECT_DIR / "Archive" / "anchor-validation-filter-based-approach-2026-08-27"
KEYWORD_DIR = ARCHIVE_DIR / "Data" / "keyword_candidates" / "v2"
OLD_KEY_FILE = (
    ARCHIVE_DIR
    / "Data"
    / "annotations"
    / "keyword_validation_v2"
    / "internal_keys_do_not_share"
    / "keyword_validation_key.csv"
)

POST_FILES = [
    "MBA_posts_filtered.jsonl.gz",
    "MSCS_posts_filtered.jsonl.gz",
    "gradadmissions_posts_filtered.jsonl.gz",
]
MONTHS = {9, 10, 11}
RANDOM_SEED = 20260828
SAMPLE_PER_ARM = 150

BLINDED_FILE = GROUND_TRUTH_DIR / "expansion_300_blinded.csv"
KEY_DIR = GROUND_TRUTH_DIR / "internal_keys_do_not_share"
KEY_FILE = KEY_DIR / "expansion_300_key.csv"
MANIFEST_FILE = GROUND_TRUTH_DIR / "expansion_300_manifest.json"


def read_gzip_json(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


def eligible_posts():
    for filename in POST_FILES:
        for record in read_gzip_json(FILTERED_DIR / filename):
            month = datetime.fromtimestamp(record["created_utc"], UTC).month
            if month in MONTHS:
                yield record


def allocate_proportionally(groups: dict, total: int) -> dict:
    allocation = {key: 1 for key in groups}
    remaining = total - len(groups)
    population = sum(len(records) - 1 for records in groups.values())
    exact = {key: remaining * (len(records) - 1) / population for key, records in groups.items()}
    for key, value in exact.items():
        allocation[key] += int(value)
    unassigned = total - sum(allocation.values())
    order = sorted(exact, key=lambda key: exact[key] - int(exact[key]), reverse=True)
    for key in order[:unassigned]:
        allocation[key] += 1
    return allocation


def stratified_sample(records, total, randomizer):
    groups = defaultdict(list)
    for record in records:
        groups[(record["community"], record["admissions_cycle"])].append(record)
    allocation = allocate_proportionally(groups, total)
    selected = []
    for key in sorted(groups):
        randomizer.shuffle(groups[key])
        selected.extend(groups[key][: allocation[key]])
    return selected


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    randomizer = random.Random(RANDOM_SEED)
    with OLD_KEY_FILE.open(encoding="utf-8", newline="") as file:
        existing_ids = {row["reddit_id"] for row in csv.DictReader(file)}

    positives = [
        record
        for record in read_gzip_json(KEYWORD_DIR / "keyword_candidates.jsonl.gz")
        if record["id"] not in existing_ids
    ]
    positive_ids = {record["id"] for record in read_gzip_json(KEYWORD_DIR / "keyword_candidates.jsonl.gz")}
    negatives = [
        record for record in eligible_posts()
        if record["id"] not in positive_ids and record["id"] not in existing_ids
    ]

    selected = [(record, "keyword_pass") for record in stratified_sample(positives, SAMPLE_PER_ARM, randomizer)]
    selected.extend(
        (record, "keyword_fail")
        for record in stratified_sample(negatives, SAMPLE_PER_ARM, randomizer)
    )
    randomizer.shuffle(selected)

    blinded_rows = []
    key_rows = []
    for number, (record, selection_arm) in enumerate(selected, 1):
        item_id = f"KVE-{number:03d}"
        blinded_rows.append(
            {
                "item_id": item_id,
                "community": record["community"],
                "admissions_cycle": record["admissions_cycle"],
                "created_at": record["created_at"],
                "text": record["text"],
                "anchor_label": "",
                "review_notes": "",
            }
        )
        key_rows.append(
            {
                "item_id": item_id,
                "reddit_id": record["id"],
                "selection_arm": selection_arm,
                "community": record["community"],
                "admissions_cycle": record["admissions_cycle"],
            }
        )

    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(BLINDED_FILE, list(blinded_rows[0]), blinded_rows)
    write_csv(KEY_FILE, list(key_rows[0]), key_rows)

    arm_counts = Counter(row["selection_arm"] for row in key_rows)
    group_counts = Counter(
        (row["selection_arm"], row["community"], row["admissions_cycle"])
        for row in key_rows
    )
    manifest = {
        "purpose": "supervised encoder training-label expansion",
        "random_seed": RANDOM_SEED,
        "records": len(blinded_rows),
        "composition": dict(arm_counts),
        "excluded_existing_ground_truth_records": len(existing_ids),
        "sampling": "minimum one per community-cycle, then proportional within each selection arm",
        "label_values": ["anchor", "non_anchor"],
        "selection_arm_is_blinded": True,
        "by_arm_community_cycle": [
            {
                "selection_arm": arm,
                "community": community,
                "admissions_cycle": cycle,
                "count": count,
            }
            for (arm, community, cycle), count in sorted(group_counts.items())
        ],
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Wrote {BLINDED_FILE}")


if __name__ == "__main__":
    main()
