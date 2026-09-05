"""Combine v6 shard predictions with the 400 retained human labels."""
from __future__ import annotations

import csv
import gzip
import json
import random
from collections import Counter
from datetime import UTC, datetime

from full_corpus_config import (
    N_SHARDS,
    PREPARED_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    VALIDATION_PER_CLASS,
)


def read_jsonl(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}"
                ) from error


def reservoir_sample(records, size, seed):
    randomizer = random.Random(seed)
    sample = []
    for number, record in enumerate(records, start=1):
        if len(sample) < size:
            sample.append(record)
            continue
        replacement = randomizer.randrange(number)
        if replacement < size:
            sample[replacement] = record
    return sample


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    human_path = PREPARED_DIR / "human_labeled_posts.jsonl.gz"
    machine_paths = [
        RESULTS_DIR / f"predictions_shard_{shard}.jsonl"
        for shard in range(N_SHARDS)
    ]
    for path in [human_path, *machine_paths]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")

    all_path = RESULTS_DIR / "all_post_labels.jsonl.gz"
    anchor_path = RESULTS_DIR / "anchor_posts.jsonl.gz"
    counts = Counter()
    by_group = Counter()
    seen_ids = set()
    machine_anchors = []
    machine_non_anchors = []

    with gzip.open(all_path, "wt", encoding="utf-8") as all_file, gzip.open(
        anchor_path, "wt", encoding="utf-8"
    ) as anchor_file:
        sources = [read_jsonl(human_path)] + [
            read_jsonl(path) for path in machine_paths
        ]
        for records in sources:
            for record in records:
                record_id = str(record["id"])
                if record_id in seen_ids:
                    raise ValueError(f"Duplicate combined post ID: {record_id}")
                seen_ids.add(record_id)
                label = record["predicted_label_text"]
                source = record["label_source"]
                counts["all_posts"] += 1
                counts[label] += 1
                counts[f"source_{source}"] += 1
                if record.get("final_label_was_overridden") is True:
                    counts["checklist_overrides"] += 1
                by_group[(record["community"], record["admissions_cycle"], label)] += 1

                line = json.dumps(record, ensure_ascii=False) + "\n"
                all_file.write(line)
                if label == "anchor":
                    anchor_file.write(line)

                if source == "qwen_v6":
                    if label == "anchor":
                        machine_anchors.append(record)
                    else:
                        machine_non_anchors.append(record)

    anchor_sample = reservoir_sample(
        machine_anchors,
        min(VALIDATION_PER_CLASS, len(machine_anchors)),
        RANDOM_SEED,
    )
    non_anchor_sample = reservoir_sample(
        machine_non_anchors,
        min(VALIDATION_PER_CLASS, len(machine_non_anchors)),
        RANDOM_SEED + 1,
    )
    validation = anchor_sample + non_anchor_sample
    random.Random(RANDOM_SEED + 2).shuffle(validation)

    validation_path = RESULTS_DIR / "manual_validation_sample.csv"
    fields = [
        "review_id",
        "id",
        "community",
        "admissions_cycle",
        "created_at",
        "text",
        "predicted_label_text",
        "confidence",
        "evidence_phrases",
        "decision_basis",
        "prediction_correct",
        "review_notes",
    ]
    with validation_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for number, record in enumerate(validation, start=1):
            writer.writerow(
                {
                    "review_id": f"FV6-{number:03d}",
                    "id": record["id"],
                    "community": record["community"],
                    "admissions_cycle": record["admissions_cycle"],
                    "created_at": record["created_at"],
                    "text": record["text"],
                    "predicted_label_text": record["predicted_label_text"],
                    "confidence": record.get("confidence"),
                    "evidence_phrases": json.dumps(
                        record.get("evidence_phrases", []),
                        ensure_ascii=False,
                    ),
                    "decision_basis": record.get("decision_basis", ""),
                    "prediction_correct": "",
                    "review_notes": "",
                }
            )

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": "Qwen/Qwen3-8B",
        "prompt": "anchor_definition_v6.txt",
        "counts": {key: int(value) for key, value in sorted(counts.items())},
        "by_community_cycle_label": [
            {
                "community": community,
                "admissions_cycle": cycle,
                "label": label,
                "posts": int(value),
            }
            for (community, cycle, label), value in sorted(by_group.items())
        ],
        "manual_validation": {
            "machine_anchor_posts_sampled": len(anchor_sample),
            "machine_non_anchor_posts_sampled": len(non_anchor_sample),
            "random_seed": RANDOM_SEED,
            "file": validation_path.name,
        },
        "outputs": {
            "all_post_labels": all_path.name,
            "anchor_posts": anchor_path.name,
            "manual_validation_sample": validation_path.name,
        },
    }
    summary_path = RESULTS_DIR / "full_corpus_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["counts"], indent=2))
    print(f"Anchors: {anchor_path}")
    print(f"Validation sample: {validation_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
