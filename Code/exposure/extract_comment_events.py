"""Link filtered September--November comments to classified posts."""
from __future__ import annotations

import gzip
import json
from collections import Counter
from datetime import datetime, timezone

from config import (
    COMMENT_DIR,
    COMMENT_FILES,
    EXPOSURE_MONTHS,
    OUTPUT_DIR,
    POST_LABEL_FILE,
)


def read_gzip_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def normalized_id(value):
    if value is None:
        return None
    return str(value).split("_", maxsplit=1)[-1]


def load_post_labels():
    posts = {}
    anchor_authors_by_cycle = {}
    for post in read_gzip_json(POST_LABEL_FILE):
        post_id = normalized_id(post.get("id") or post.get("post_id"))
        if not post_id:
            continue
        posts[post_id] = post
        if post["predicted_label_text"] == "anchor":
            cycle = post["admissions_cycle"]
            anchor_authors_by_cycle.setdefault(cycle, set()).add(
                str(post["author"]).casefold()
            )
    return posts, anchor_authors_by_cycle


def event_record(comment, post, commenter_is_anchor_author):
    parent_id = normalized_id(comment.get("parent_id"))
    post_id = normalized_id(post.get("id") or post.get("post_id"))
    delay_hours = (comment["created_utc"] - post["created_utc"]) / 3_600
    return {
        "user": comment["author"],
        "community": comment["community"],
        "admissions_cycle": comment["admissions_cycle"],
        "post_id": post_id,
        "post_label": post["predicted_label_text"],
        "post_label_source": post.get("label_source"),
        "post_author": post["author"],
        "post_created_utc": post["created_utc"],
        "post_created_at": post["created_at"],
        "comment_id": comment["id"],
        "comment_created_utc": comment["created_utc"],
        "comment_created_at": comment["created_at"],
        "comment_parent_id": parent_id,
        "is_direct_reply": parent_id == post_id,
        "is_nested_reply": parent_id != post_id,
        "commenter_is_post_author": (
            str(comment["author"]).casefold()
            == str(post["author"]).casefold()
        ),
        "commenter_is_anchor_author_in_cycle": commenter_is_anchor_author,
        "delay_hours": round(delay_hours, 3),
        "comment_text": comment.get("text"),
        "comment_score": comment.get("score"),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    posts, anchor_authors_by_cycle = load_post_labels()
    flow = Counter({"classified_posts": len(posts)})
    seen_comment_ids = set()

    anchor_path = OUTPUT_DIR / "anchor_comment_events.jsonl.gz"
    non_anchor_path = OUTPUT_DIR / "non_anchor_comment_events.jsonl.gz"
    with gzip.open(anchor_path, "wt", encoding="utf-8") as anchor_file, gzip.open(
        non_anchor_path, "wt", encoding="utf-8"
    ) as non_anchor_file:
        for filename in COMMENT_FILES:
            for comment in read_gzip_json(COMMENT_DIR / filename):
                flow["comments_read"] += 1
                post = posts.get(normalized_id(comment.get("post_id")))
                if post is None:
                    continue
                flow["comments_linked_to_classified_posts"] += 1

                if comment.get("community") != post.get("community"):
                    flow["excluded_community_mismatch"] += 1
                    continue
                if comment.get("admissions_cycle") != post.get("admissions_cycle"):
                    flow["excluded_cycle_mismatch"] += 1
                    continue
                if comment.get("created_utc") is None:
                    flow["excluded_missing_timestamp"] += 1
                    continue
                if comment["created_utc"] < post["created_utc"]:
                    flow["excluded_before_post"] += 1
                    continue
                month = datetime.fromtimestamp(
                    comment["created_utc"], timezone.utc
                ).month
                if month not in EXPOSURE_MONTHS:
                    flow["excluded_outside_september_november"] += 1
                    continue
                if comment["id"] in seen_comment_ids:
                    flow["excluded_duplicate_comment"] += 1
                    continue

                seen_comment_ids.add(comment["id"])
                cycle = comment["admissions_cycle"]
                is_anchor_author = (
                    str(comment["author"]).casefold()
                    in anchor_authors_by_cycle.get(cycle, set())
                )
                event = event_record(comment, post, is_anchor_author)
                label = post["predicted_label_text"]
                if label == "anchor":
                    anchor_file.write(json.dumps(event, ensure_ascii=False) + "\n")
                    flow["anchor_comment_events"] += 1
                else:
                    non_anchor_file.write(
                        json.dumps(event, ensure_ascii=False) + "\n"
                    )
                    flow["non_anchor_comment_events"] += 1

    report = {
        "rules": {
            "months": sorted(EXPOSURE_MONTHS),
            "comments_must_follow_post": True,
            "same_community_and_cycle_required": True,
            "anchor_author_exclusion_is_cycle_specific": True,
        },
        "extraction": dict(flow),
    }
    (OUTPUT_DIR / "cohort_flow.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
