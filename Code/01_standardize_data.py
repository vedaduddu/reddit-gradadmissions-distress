#!/usr/bin/env python3
"""Standardize the project Reddit exports into analysis-ready JSONL files.

The script streams every input so it can process the full corpus with bounded
memory. Raw files are read-only; standardized files and an audit summary are
written beneath Data/standardized by default.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter
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

REMOVED_TEXT = {"", "[deleted]", "[removed]"}
REMOVED_AUTHORS = {"", "[deleted]", "[removed]"}


def parse_args() -> argparse.Namespace:
    #creating a parser tool to find our folder
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("Data/Raw Data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("Data/standardized")
    )
    parser.add_argument(
        "--limit-per-file",
        type=int,
        default=None,
        help="Testing aid: process at most N valid JSON records per source file.",
    )
    return parser.parse_args()


def open_text(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def iter_jsonl(path: Path, limit: int | None) -> Iterator[dict[str, Any]]:
    seen = 0
    with open_text(path, "rt") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object in {path}:{line_number}")
            yield record
            seen += 1
            if limit is not None and seen >= limit:
                break


def first_present(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


def normalize_id(value: Any, prefix: str | None = None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if prefix and text.startswith(prefix):
        return text[len(prefix) :]
    return text


def timestamp(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return None
        return int(float(value))
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


def iso_utc(created_utc: int | None) -> str | None:
    if created_utc is None:
        return None
    return datetime.fromtimestamp(created_utc, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def admissions_cycle(created_utc: int | None) -> str | None:
    if created_utc is None:
        return None
    moment = datetime.fromtimestamp(created_utc, timezone.utc)
    start_year = moment.year if moment.month >= 8 else moment.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def clean_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def standardize(
    record: dict[str, Any], community: str, record_type: str, source_file: str
) -> dict[str, Any]:
    created_utc = timestamp(first_present(record, "created_utc", "created_dt", "created"))
    author = clean_string(record.get("author"))
    record_id = normalize_id(record.get("id"))

    if record_type == "post":
        post_id = record_id
        parent_id = None
        title = clean_string(record.get("title"))
        body = clean_string(first_present(record, "selftext", "clean_text"))
        text_parts = [part for part in (title, body) if part and part not in REMOVED_TEXT]
        num_comments = record.get("num_comments")
    else:
        post_id = normalize_id(first_present(record, "link_id", "post_id"), "t3_")
        parent_id = clean_string(record.get("parent_id"))
        title = None
        body = clean_string(first_present(record, "body", "clean_text"))
        text_parts = [body] if body and body not in REMOVED_TEXT else []
        num_comments = None

    text = "\n\n".join(text_parts) if text_parts else None
    retrieved_utc = timestamp(record.get("retrieved_on"))
    edited_value = record.get("edited")
    edited_utc = timestamp(edited_value) if not isinstance(edited_value, bool) else None
    return {
        "record_type": record_type,
        "community": community,
        "admissions_cycle": admissions_cycle(created_utc),
        "id": record_id,
        "post_id": post_id,
        "parent_id": parent_id,
        "author": author,
        "created_utc": created_utc,
        "created_at": iso_utc(created_utc),
        "title": title,
        "body": body,
        "text": text,
        "score": record.get("score"),
        "num_comments": num_comments,
        "permalink": clean_string(record.get("permalink")),
        "link_flair_text": clean_string(record.get("link_flair_text")),
        "upvote_ratio": record.get("upvote_ratio"),
        "is_self_post": record.get("is_self") if record_type == "post" else None,
        "removed_by_category": clean_string(record.get("removed_by_category")),
        "retrieved_utc": retrieved_utc,
        "retrieved_at": iso_utc(retrieved_utc),
        "is_edited": bool(edited_value),
        "edited_utc": edited_utc,
        "is_stickied": record.get("stickied"),
        "is_locked": record.get("locked"),
        "is_nsfw": record.get("over_18") if record_type == "post" else None,
        "subreddit_subscribers": record.get("subreddit_subscribers"),
        "author_available": author not in REMOVED_AUTHORS and author is not None,
        "text_available": text is not None,
        "source_file": source_file,
    }


def output_path(output_dir: Path, community: str, record_type: str) -> Path:
    return output_dir / f"{community}_{record_type}s.jsonl.gz"


def main() -> None:
    args = parse_args()
    missing = [name for name, _, _ in FILE_SPECS if not (args.data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected input files: {', '.join(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[tuple[str, str], TextIO] = {}
    stats: dict[str, Counter[str]] = {}
    date_bounds: dict[str, list[int]] = {}
    try:
        for _, community, record_type in FILE_SPECS:
            key = (community, record_type)
            if key not in writers:
                writers[key] = gzip.open(
                    output_path(args.output_dir, community, record_type),
                    "wt",
                    encoding="utf-8",
                )

        for filename, community, record_type in FILE_SPECS:
            source = args.data_dir / filename
            counter: Counter[str] = Counter()
            bounds: list[int] = []
            writer = writers[(community, record_type)]
            for raw in iter_jsonl(source, args.limit_per_file):
                row = standardize(raw, community, record_type, filename)
                writer.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                writer.write("\n")
                counter["records"] += 1
                counter["missing_id"] += row["id"] is None
                counter["missing_post_id"] += row["post_id"] is None
                counter["missing_author"] += not row["author_available"]
                counter["missing_text"] += not row["text_available"]
                counter["missing_timestamp"] += row["created_utc"] is None
                if row["created_utc"] is not None:
                    bounds.append(row["created_utc"])
            stats[filename] = counter
            date_bounds[filename] = bounds
    finally:
        for writer in writers.values():
            writer.close()

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "limited_run": args.limit_per_file is not None,
        "limit_per_file": args.limit_per_file,
        "files": {},
    }
    for filename, counter in stats.items():
        bounds = date_bounds[filename]
        summary["files"][filename] = {
            **dict(counter),
            "first_created_at": iso_utc(min(bounds)) if bounds else None,
            "last_created_at": iso_utc(max(bounds)) if bounds else None,
        }

    summary_path = args.output_dir / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    total = sum(item["records"] for item in summary["files"].values())
    print(f"Standardized {total:,} records into {args.output_dir}")
    print(f"Audit summary: {summary_path}")


if __name__ == "__main__":
    main()
