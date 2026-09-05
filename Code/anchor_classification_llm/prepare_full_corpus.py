"""Prepare September-November posts for the frozen v6 whole-corpus run."""
from __future__ import annotations

import gzip
import json
import zlib
from collections import Counter
from datetime import UTC, datetime

import pandas as pd

from full_corpus_config import (
    ADMISSIONS_CYCLES,
    COMBINED_KEY_FILE,
    ELIGIBLE_MONTHS,
    EXPANSION_KEY_FILE,
    GROUND_TRUTH_FILE,
    INITIAL_KEY_FILE,
    N_SHARDS,
    POST_FILES,
    POSTS_DIR,
    PREPARED_DIR,
)


def build_combined_key() -> pd.DataFrame:
    """Join the 400 study IDs to stable Reddit post IDs and human labels."""
    if COMBINED_KEY_FILE.exists():
        return pd.read_csv(COMBINED_KEY_FILE, dtype={"reddit_id": str})

    if not INITIAL_KEY_FILE.exists():
        raise FileNotFoundError(
            f"Missing {COMBINED_KEY_FILE}. Build it locally, then copy it "
            "to the server before preparing the corpus."
        )

    truth = pd.read_csv(GROUND_TRUTH_FILE)
    keys = pd.concat(
        [
            pd.read_csv(INITIAL_KEY_FILE)[["item_id", "reddit_id"]],
            pd.read_csv(EXPANSION_KEY_FILE)[["item_id", "reddit_id"]],
        ],
        ignore_index=True,
    )
    combined = truth[
        ["item_id", "community", "admissions_cycle", "anchor_label", "label"]
    ].merge(keys, on="item_id", validate="one_to_one")

    if len(combined) != 400 or combined["reddit_id"].duplicated().any():
        raise ValueError("Expected 400 unique labeled Reddit post IDs")

    COMBINED_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(COMBINED_KEY_FILE, index=False)
    print(f"Wrote human-label key: {COMBINED_KEY_FILE}")
    return combined


def eligible(record: dict) -> bool:
    created_at = str(record.get("created_at") or "")
    if len(created_at) < 7:
        return False
    try:
        month = int(created_at[5:7])
    except ValueError:
        return False
    return (
        record.get("record_type") == "post"
        and record.get("admissions_cycle") in ADMISSIONS_CYCLES
        and month in ELIGIBLE_MONTHS
    )


def shard_for(record_id: str) -> int:
    return zlib.crc32(record_id.encode("utf-8")) % N_SHARDS


def write_record(file, record: dict) -> None:
    file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    human_key = build_combined_key()
    human_by_reddit_id = {
        str(row.reddit_id): row
        for row in human_key.itertuples(index=False)
    }

    human_path = PREPARED_DIR / "human_labeled_posts.jsonl.gz"
    shard_paths = [
        PREPARED_DIR / f"machine_posts_shard_{shard}.jsonl.gz"
        for shard in range(N_SHARDS)
    ]

    counts = Counter()
    by_community_cycle = Counter()
    seen_ids: set[str] = set()
    found_human_ids: set[str] = set()

    with gzip.open(human_path, "wt", encoding="utf-8") as human_file:
        shard_files = [
            gzip.open(path, "wt", encoding="utf-8")
            for path in shard_paths
        ]
        try:
            for filename in POST_FILES:
                source = POSTS_DIR / filename
                with gzip.open(source, "rt", encoding="utf-8") as input_file:
                    for line_number, line in enumerate(input_file, start=1):
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as error:
                            raise ValueError(
                                f"Invalid JSON in {filename}, line {line_number}"
                            ) from error

                        counts["source_posts"] += 1
                        if not eligible(record):
                            continue

                        record_id = str(record.get("id") or "")
                        if not record_id:
                            raise ValueError(f"Missing post ID in {filename}")
                        if record_id in seen_ids:
                            raise ValueError(f"Duplicate eligible post ID: {record_id}")
                        seen_ids.add(record_id)

                        counts["eligible_posts"] += 1
                        by_community_cycle[
                            (record["community"], record["admissions_cycle"])
                        ] += 1

                        if record_id in human_by_reddit_id:
                            label = human_by_reddit_id[record_id]
                            output = {
                                **record,
                                "ground_truth_item_id": label.item_id,
                                "predicted_label": int(label.label),
                                "predicted_label_text": (
                                    "anchor" if int(label.label) == 1 else "non_anchor"
                                ),
                                "label_source": "human",
                            }
                            write_record(human_file, output)
                            found_human_ids.add(record_id)
                            counts["human_labeled_posts"] += 1
                        else:
                            shard = shard_for(record_id)
                            write_record(shard_files[shard], record)
                            counts[f"machine_shard_{shard}"] += 1
                            counts["machine_posts"] += 1
        finally:
            for file in shard_files:
                file.close()

    missing_human_ids = sorted(
        set(human_by_reddit_id).difference(found_human_ids)
    )
    if missing_human_ids:
        raise ValueError(
            f"Could not locate {len(missing_human_ids)} labeled posts: "
            f"{missing_human_ids[:10]}"
        )

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_directory": str(POSTS_DIR),
        "input_files": POST_FILES,
        "admissions_cycles": sorted(ADMISSIONS_CYCLES),
        "eligible_months": sorted(ELIGIBLE_MONTHS),
        "shard_rule": "crc32(reddit_post_id) modulo 4",
        "counts": {key: int(value) for key, value in sorted(counts.items())},
        "by_community_cycle": [
            {
                "community": community,
                "admissions_cycle": cycle,
                "posts": int(value),
            }
            for (community, cycle), value in sorted(by_community_cycle.items())
        ],
    }
    manifest_path = PREPARED_DIR / "preparation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
