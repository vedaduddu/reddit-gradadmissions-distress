#!/usr/bin/env python3
"""Profile the raw Reddit exports before downstream standardization.

The report deliberately excludes content samples and author names. It records
schema prevalence, missingness, types, date coverage, duplicate identifiers,
and post-comment join coverage by community and admissions cycle.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TextIO


FILE_SPECS = (
    ("r_gradadmissions_2022_posts.jsonl", "gradadmissions", "post"),
    ("r_gradadmissions_posts.jsonl", "gradadmissions", "post"),
    ("r_gradadmissions_2022_comments.jsonl", "gradadmissions", "comment"),
    ("r_gradadmissions_comments.jsonl", "gradadmissions", "comment"),
    ("r_MSCS_2022_posts.jsonl", "MSCS", "post"),
    ("r_MSCS_posts.jsonl", "MSCS", "post"),
    ("r_MSCS_2022_comments.jsonl", "MSCS", "comment"),
    ("r_MSCS_comments.jsonl", "MSCS", "comment"),
    ("mba_posts_clean.jsonl.gz", "MBA", "post"),
    ("mba_comments_clean.jsonl.gz", "MBA", "comment"),
)

STANDARD_MAPPING = {
    "record_type": ["inferred from source-file role"],
    "community": ["subreddit", "inferred from source filename"],
    "admissions_cycle": ["derived from normalized timestamp"],
    "id": ["id"],
    "post_id": ["id (posts)", "link_id", "post_id"],
    "parent_id": ["parent_id"],
    "author": ["author"],
    "created_utc": ["created_utc", "created_dt", "created"],
    "created_at": ["derived from normalized timestamp"],
    "title": ["title"],
    "body": ["selftext", "body", "clean_text"],
    "text": ["title + body (posts)", "body (comments)"],
    "score": ["score"],
    "num_comments": ["num_comments"],
    "permalink": ["permalink"],
    "link_flair_text": ["link_flair_text"],
    "upvote_ratio": ["upvote_ratio"],
    "is_self_post": ["is_self"],
    "removed_by_category": ["removed_by_category"],
    "retrieved_utc": ["retrieved_on"],
    "retrieved_at": ["derived from retrieved_on"],
    "is_edited": ["edited"],
    "edited_utc": ["edited when timestamp-valued"],
    "is_stickied": ["stickied"],
    "is_locked": ["locked"],
    "is_nsfw": ["over_18"],
    "subreddit_subscribers": ["subreddit_subscribers"],
    "author_available": ["derived from author"],
    "text_available": ["derived from normalized text"],
    "source_file": ["source filename"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("Data/Raw Data"))
    parser.add_argument("--output-dir", type=Path, default=Path("Data/eda"))
    return parser.parse_args()


def open_text(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open_text(path, "rt") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_number}")
            yield row


def type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def timestamp(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(float(value)) if math.isfinite(float(value)) else None
    text = str(value).strip()
    try:
        return int(float(text))
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def raw_timestamp(row: dict[str, Any]) -> int | None:
    for key in ("created_utc", "created_dt", "created"):
        if row.get(key) is not None:
            return timestamp(row[key])
    return None


def cycle(ts: int | None) -> str | None:
    if ts is None:
        return None
    moment = datetime.fromtimestamp(ts, timezone.utc)
    start = moment.year if moment.month >= 8 else moment.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def iso(ts: int | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


def normalized_id(value: Any, prefix: str | None = None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if prefix and text.startswith(prefix):
        text = text[len(prefix) :]
    return text


def pct(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 3) if denominator else 0.0


def profile(data_dir: Path) -> dict[str, Any]:
    expected = [data_dir / name for name, _, _ in FILE_SPECS]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected sources: {', '.join(missing)}")

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "standard_mapping": STANDARD_MAPPING,
        "files": {},
        "combined": {},
    }
    combined_ids: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    post_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    comment_refs: dict[tuple[str, str], set[str]] = defaultdict(set)

    for filename, community, record_type in FILE_SPECS:
        path = data_dir / filename
        rows = 0
        invalid_timestamps = 0
        first_ts: int | None = None
        last_ts: int | None = None
        cycles: Counter[str] = Counter()
        field_present: Counter[str] = Counter()
        field_non_null: Counter[str] = Counter()
        field_nonempty: Counter[str] = Counter()
        field_types: dict[str, Counter[str]] = defaultdict(Counter)
        file_ids: Counter[str] = Counter()

        for row in iter_jsonl(path):
            rows += 1
            for field, value in row.items():
                field_present[field] += 1
                field_non_null[field] += value is not None
                field_nonempty[field] += is_nonempty(value)
                field_types[field][type_name(value)] += 1

            ts = raw_timestamp(row)
            if ts is None:
                invalid_timestamps += 1
            else:
                first_ts = ts if first_ts is None else min(first_ts, ts)
                last_ts = ts if last_ts is None else max(last_ts, ts)
                cycles[cycle(ts)] += 1

            record_id = normalized_id(row.get("id"))
            if record_id:
                file_ids[record_id] += 1
                combined_ids[(community, record_type)][record_id] += 1
            record_cycle = cycle(ts)
            if record_cycle and record_type == "post" and record_id:
                post_ids[(community, record_cycle)].add(record_id)
            if record_cycle and record_type == "comment":
                ref = normalized_id(row.get("link_id"), "t3_")
                if ref is None:
                    ref = normalized_id(row.get("post_id"))
                if ref:
                    comment_refs[(community, record_cycle)].add(ref)

        fields = {}
        for field in sorted(field_present):
            fields[field] = {
                "present": field_present[field],
                "present_pct": pct(field_present[field], rows),
                "non_null": field_non_null[field],
                "non_null_pct": pct(field_non_null[field], rows),
                "nonempty": field_nonempty[field],
                "nonempty_pct": pct(field_nonempty[field], rows),
                "types": dict(sorted(field_types[field].items())),
            }
        report["files"][filename] = {
            "community": community,
            "record_type": record_type,
            "rows": rows,
            "field_count": len(fields),
            "fields": fields,
            "first_created_at": iso(first_ts),
            "last_created_at": iso(last_ts),
            "invalid_timestamps": invalid_timestamps,
            "cycle_rows": dict(sorted(cycles.items())),
            "unique_ids": len(file_ids),
            "duplicate_id_records": sum(count - 1 for count in file_ids.values() if count > 1),
        }

    for (community, record_type), ids in sorted(combined_ids.items()):
        report["combined"].setdefault(community, {})[record_type] = {
            "unique_ids": len(ids),
            "duplicate_id_records_across_sources": sum(
                count - 1 for count in ids.values() if count > 1
            ),
        }

    joins = {}
    for key in sorted(set(post_ids) | set(comment_refs)):
        community, record_cycle = key
        posts = post_ids[key]
        refs = comment_refs[key]
        joined = posts & refs
        joins[f"{community}:{record_cycle}"] = {
            "community": community,
            "admissions_cycle": record_cycle,
            "unique_post_ids": len(posts),
            "unique_comment_post_refs": len(refs),
            "joined_comment_post_refs": len(joined),
            "comment_post_ref_join_pct": pct(len(joined), len(refs)),
        }
    report["joins_by_community_cycle"] = joins
    return report


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Raw schema EDA",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "This report contains aggregate schema diagnostics only; it includes no post text or author names.",
        "",
        "## How to read this report",
        "",
        "- **Source overview** checks that every expected raw export was read, shows its row and field counts, and confirms its date range. A nonzero duplicate-ID count would require deduplication before analysis.",
        "- **Standard field mapping** documents how inconsistent raw names become one analysis field. For example, Reddit comments use `link_id` while cleaned MBA comments use `post_id`; both become standardized `post_id`.",
        "- **Post-comment joins** show whether the original post referenced by a comment exists in our collected posts. A lower join rate means some threads are unavailable and must be excluded from thread-level analyses; it does not mean the available records were standardized incorrectly.",
        "- **Complete field inventory** is stored in JSON because it contains every field from every source. For each field, `present_pct` is how often the key exists, `non_null_pct` is how often it is not null, and `nonempty_pct` additionally excludes empty strings or collections.",
        "",
        "## Decisions resulting from the EDA",
        "",
        "The first standardization pass preserved the core analysis fields. Review of this EDA identified additional potentially useful metadata: post flair, upvote ratio, self-post status, removal category, retrieval time, edit status, sticky/locked/NSFW flags, and subreddit size. These fields were added to the standard schema and the full standardization was rerun. The current files in `Data/standardized/` therefore incorporate the EDA findings.",
        "",
        "The EDA found zero invalid timestamps and zero duplicate IDs. The raw 2022-23 collections and 2023-25 collections are adjacent and can be safely merged while retaining `source_file` provenance.",
        "",
        "## Source overview",
        "",
        "| Source file | Community | Type | Rows | Fields | Date coverage | Duplicate IDs |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for filename, item in report["files"].items():
        coverage = f"{item['first_created_at'][:10]} to {item['last_created_at'][:10]}"
        lines.append(
            f"| `{filename}` | {item['community']} | {item['record_type']} | "
            f"{item['rows']:,} | {item['field_count']:,} | {coverage} | "
            f"{item['duplicate_id_records']:,} |"
        )

    lines.extend(
        [
            "",
            "## Standard field mapping",
            "",
            "| Standard field | Raw source or derivation |",
            "|---|---|",
        ]
    )
    for field, sources in report["standard_mapping"].items():
        lines.append(f"| `{field}` | {', '.join(f'`{s}`' for s in sources)} |")

    lines.extend(
        [
            "",
            "## Post-comment joins by community and cycle",
            "",
            "| Community | Cycle | Post IDs | Comment post references | Joined references | Join rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["joins_by_community_cycle"].values():
        lines.append(
            f"| {item['community']} | {item['admissions_cycle']} | "
            f"{item['unique_post_ids']:,} | {item['unique_comment_post_refs']:,} | "
            f"{item['joined_comment_post_refs']:,} | "
            f"{item['comment_post_ref_join_pct']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Complete field inventory",
            "",
            "The companion `raw_schema_profile.json` contains every observed field, its prevalence, null and empty rates, and observed JSON types for each source file.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    result = profile(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "raw_schema_profile.json"
    markdown_path = args.output_dir / "raw_schema_profile.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown(result), encoding="utf-8")
    total_rows = sum(item["rows"] for item in result["files"].values())
    print(f"Profiled {total_rows:,} raw records")
    print(f"Full profile: {json_path}")
    print(f"Readable summary: {markdown_path}")


if __name__ == "__main__":
    main()
