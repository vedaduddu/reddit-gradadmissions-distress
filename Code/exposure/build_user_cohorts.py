"""Create unique treated user-cycles and unexposed control candidates."""
from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict

from config import COMMUNITIES, OUTPUT_DIR


def read_events(filename):
    with gzip.open(OUTPUT_DIR / filename, "rt", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_treated(anchor_events):
    all_exposed_cycles = {
        (event["user"].casefold(), event["admissions_cycle"])
        for event in anchor_events
        if not event["commenter_is_post_author"]
    }
    anchor_author_cycles = {
        (event["user"].casefold(), event["admissions_cycle"])
        for event in anchor_events
        if event["commenter_is_anchor_author_in_cycle"]
    }

    eligible = [
        event
        for event in anchor_events
        if not event["commenter_is_post_author"]
        and not event["commenter_is_anchor_author_in_cycle"]
    ]
    grouped = defaultdict(list)
    for event in eligible:
        grouped[(event["user"].casefold(), event["admissions_cycle"])].append(
            event
        )

    rows = []
    for events in grouped.values():
        ordered = sorted(
            events,
            key=lambda item: (item["comment_created_utc"], item["comment_id"]),
        )
        first = ordered[0]
        post_ids = {event["post_id"] for event in ordered}
        communities = {event["community"] for event in ordered}
        row = {
            "user": first["user"],
            "admissions_cycle": first["admissions_cycle"],
            "treatment": 1,
            "index_community": first["community"],
            "index_comment_id": first["comment_id"],
            "index_post_id": first["post_id"],
            "index_utc": first["comment_created_utc"],
            "index_at": first["comment_created_at"],
            "anchor_comment_count": len(ordered),
            "distinct_anchor_posts": len(post_ids),
            "distinct_exposure_communities": len(communities),
            "cross_community_exposure": len(communities) > 1,
        }
        for community in COMMUNITIES:
            safe = community.lower()
            row[f"{safe}_anchor_comments"] = sum(
                event["community"] == community for event in ordered
            )
            row[f"{safe}_anchor_posts"] = len(
                {
                    event["post_id"]
                    for event in ordered
                    if event["community"] == community
                }
            )
        rows.append(row)

    rows.sort(key=lambda row: (row["admissions_cycle"], row["index_utc"], row["user"].casefold()))
    return rows, all_exposed_cycles, anchor_author_cycles


def build_control_candidates(non_anchor_events, all_exposed_cycles, anchor_author_cycles):
    grouped = defaultdict(list)
    for event in non_anchor_events:
        user_cycle = (event["user"].casefold(), event["admissions_cycle"])
        if event["commenter_is_post_author"]:
            continue
        if user_cycle in all_exposed_cycles:
            continue
        if user_cycle in anchor_author_cycles or event["commenter_is_anchor_author_in_cycle"]:
            continue
        grouped[
            (
                event["user"].casefold(),
                event["community"],
                event["admissions_cycle"],
            )
        ].append(event)

    rows = []
    for events in grouped.values():
        ordered = sorted(
            events,
            key=lambda item: (item["comment_created_utc"], item["comment_id"]),
        )
        first, last = ordered[0], ordered[-1]
        rows.append(
            {
                "user": first["user"],
                "community": first["community"],
                "admissions_cycle": first["admissions_cycle"],
                "treatment": 0,
                "first_non_anchor_comment_id": first["comment_id"],
                "first_non_anchor_comment_utc": first["comment_created_utc"],
                "first_non_anchor_comment_at": first["comment_created_at"],
                "last_non_anchor_comment_utc": last["comment_created_utc"],
                "last_non_anchor_comment_at": last["comment_created_at"],
                "non_anchor_comment_count": len(ordered),
                "distinct_non_anchor_posts": len(
                    {event["post_id"] for event in ordered}
                ),
            }
        )
    rows.sort(key=lambda row: (row["admissions_cycle"], row["community"], row["user"].casefold()))
    return rows


def excluded_anchor_author_rows(anchor_events):
    grouped = defaultdict(list)
    for event in anchor_events:
        if event["commenter_is_anchor_author_in_cycle"]:
            grouped[(event["user"].casefold(), event["admissions_cycle"])].append(
                event
            )
    rows = []
    for events in grouped.values():
        first = min(events, key=lambda item: item["comment_created_utc"])
        rows.append(
            {
                "user": first["user"],
                "admissions_cycle": first["admissions_cycle"],
                "anchor_comment_count": len(events),
                "distinct_anchor_posts_commented": len(
                    {event["post_id"] for event in events}
                ),
                "also_commented_on_own_anchor": any(
                    event["commenter_is_post_author"] for event in events
                ),
            }
        )
    return sorted(rows, key=lambda row: (row["admissions_cycle"], row["user"].casefold()))


def main():
    anchor_events = list(read_events("anchor_comment_events.jsonl.gz"))
    non_anchor_events = list(read_events("non_anchor_comment_events.jsonl.gz"))
    treated, exposed_cycles, anchor_author_cycles = build_treated(anchor_events)
    controls = build_control_candidates(
        non_anchor_events, exposed_cycles, anchor_author_cycles
    )
    excluded_authors = excluded_anchor_author_rows(anchor_events)

    treated_fields = list(treated[0]) if treated else []
    control_fields = list(controls[0]) if controls else []
    excluded_fields = list(excluded_authors[0]) if excluded_authors else []
    if not treated_fields or not control_fields:
        raise ValueError("No treated users or control candidates were produced")

    write_csv(OUTPUT_DIR / "treated_user_cycles.csv", treated, treated_fields)
    write_csv(
        OUTPUT_DIR / "candidate_control_user_community_cycles.csv",
        controls,
        control_fields,
    )
    if excluded_fields:
        write_csv(
            OUTPUT_DIR / "excluded_anchor_author_user_cycles.csv",
            excluded_authors,
            excluded_fields,
        )

    flow_path = OUTPUT_DIR / "cohort_flow.json"
    flow = json.loads(flow_path.read_text(encoding="utf-8"))
    treated_by_stratum = Counter(
        (row["index_community"], row["admissions_cycle"]) for row in treated
    )
    controls_by_stratum = Counter(
        (row["community"], row["admissions_cycle"]) for row in controls
    )
    flow["cohorts"] = {
        "treated_user_cycles": len(treated),
        "unique_treated_users": len({row["user"].casefold() for row in treated}),
        "candidate_control_user_community_cycles": len(controls),
        "unique_candidate_control_users": len(
            {row["user"].casefold() for row in controls}
        ),
        "excluded_anchor_author_user_cycles": len(excluded_authors),
        "cross_community_treated_user_cycles": sum(
            row["cross_community_exposure"] for row in treated
        ),
        "treated_by_index_community_cycle": [
            {"community": community, "admissions_cycle": cycle, "users": count}
            for (community, cycle), count in sorted(treated_by_stratum.items())
        ],
        "candidate_controls_by_community_cycle": [
            {"community": community, "admissions_cycle": cycle, "users": count}
            for (community, cycle), count in sorted(controls_by_stratum.items())
        ],
    }
    flow_path.write_text(json.dumps(flow, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(flow["cohorts"], indent=2))


if __name__ == "__main__":
    main()
