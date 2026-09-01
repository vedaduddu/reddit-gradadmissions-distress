"""Combine the original ground truth with Veda's completed expansion labels."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from config import GROUND_TRUTH_FILE, PROJECT_DIR


ORIGINAL_FILE = PROJECT_DIR / "Data" / "anchor_ground_truth" / "veda_anchor_ground_truth.csv"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python prepare_interim_ground_truth.py LABELS_TSV")

    expansion_file = Path(sys.argv[1])
    with ORIGINAL_FILE.open(encoding="utf-8", newline="") as file:
        original = list(csv.DictReader(file))
    with expansion_file.open(encoding="utf-8-sig", newline="") as file:
        sample = file.read(4096)
        file.seek(0)
        delimiter = csv.Sniffer().sniff(sample, delimiters=",\t").delimiter
        expansion = list(csv.DictReader(file, delimiter=delimiter))

    label_map = {"general_information": "0", "threat_distress": "1"}
    rows = list(original)
    for row in expansion:
        label_name = row["anchor_label"].strip()
        if label_name not in label_map:
            continue
        rows.append({
            "item_id": row["item_id"],
            "community": row["community"],
            "admissions_cycle": row["admissions_cycle"],
            "created_at": row["created_at"],
            "text": row["text"],
            "original_anchor_label": label_name,
            "anchor_label": "anchor" if label_map[label_name] == "1" else "non_anchor",
            "label": label_map[label_name],
        })

    item_ids = [row["item_id"] for row in rows]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Duplicate item IDs found across ground-truth sources")

    GROUND_TRUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    with GROUND_TRUTH_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    counts = {label: sum(row["label"] == label for row in rows) for label in ("0", "1")}
    print(f"Wrote {len(rows)} records to {GROUND_TRUTH_FILE}")
    print(f"non_anchor={counts['0']}; anchor={counts['1']}")


if __name__ == "__main__":
    main()
