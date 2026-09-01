"""After tombstones + lists, drop short comments and resweep detector thresholds.

Short filter applies to comments only (posts kept regardless of length).
Length is whitespace word count on the `text` field.
"""
from __future__ import annotations

import gzip
import json
import os
import random
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from ai_detector import AIContentDetector
from ai_detector.preprocess import preprocess_text
from bot_filter.lists import author_is_known_bot, text_has_bot_phrase
from bot_filter.threshold_curve import row_tombstone

DIR = Path(os.environ.get("REDDIT_STANDARDIZED_DIR", "Data/standardized_filtered"))
OUT = Path(os.environ.get("REDDIT_CLEANING_OUTPUT_DIR", DIR)) / "short_comment_curve.json"
BATCH = 8192
THRESHOLDS = [0.40, 0.406, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
MIN_WORDS = [0, 3, 5, 8, 10, 15, 20]
SAMPLE_K = 6
SEED = 42


def word_count(text: str) -> int:
    return len(text.split()) if text else 0


def main() -> None:
    rng = random.Random(SEED)
    det = AIContentDetector()
    files = sorted(p.name for p in DIR.glob("*.jsonl.gz"))

    n_orig = n_tomb = n_hp = 0
    n_comments = n_posts = 0
    comment_wc = Counter()  # word count -> n, cap display at 50+
    comment_wc_exact = []  # we'll just use Counter; also percentiles via cumulative

    # After tomb+lists, before short filter
    n_base = 0
    n_base_comments = n_base_posts = 0

    # surviving[min_w] after also dropping comments with wc < min_w
    surviving = Counter()
    surviving_comments = Counter()
    flags = defaultdict(Counter)  # min_w -> threshold -> n
    flags_comments = defaultdict(Counter)

    # samples of comments dropped at 5 and 10 words, plus high-p short comments
    dropped_at = {5: [], 10: []}
    dropped_n = Counter()
    flagged_short = []  # comments with wc < 10 and p >= 0.5
    flagged_short_n = 0

    batch_clean: list[str] = []
    batch_meta: list[dict] = []

    def flush() -> None:
        nonlocal flagged_short_n
        if not batch_clean:
            return
        proba = det.pipeline.predict_proba(batch_clean)[:, 1]
        for meta, p in zip(batch_meta, proba):
            wc = meta["n_words"]
            is_comment = meta["record_type"] == "comment"
            for min_w in MIN_WORDS:
                if is_comment and wc < min_w:
                    continue
                surviving[min_w] += 1
                if is_comment:
                    surviving_comments[min_w] += 1
                for t in THRESHOLDS:
                    if p >= t:
                        flags[min_w][t] += 1
                        if is_comment:
                            flags_comments[min_w][t] += 1
            if is_comment:
                for cut in (5, 10):
                    if wc < cut:
                        dropped_n[cut] += 1
                        bucket = dropped_at[cut]
                        row = {
                            "wc": wc,
                            "p": round(float(p), 3),
                            "community": meta["community"],
                            "preview": meta["preview"],
                        }
                        if len(bucket) < SAMPLE_K:
                            bucket.append(row)
                        else:
                            j = rng.randrange(dropped_n[cut])
                            if j < SAMPLE_K:
                                bucket[j] = row
                if wc < 10 and p >= 0.5:
                    flagged_short_n += 1
                    row = {
                        "wc": wc,
                        "p": round(float(p), 3),
                        "community": meta["community"],
                        "preview": meta["preview"],
                    }
                    if len(flagged_short) < SAMPLE_K:
                        flagged_short.append(row)
                    else:
                        j = rng.randrange(flagged_short_n)
                        if j < SAMPLE_K:
                            flagged_short[j] = row
        batch_clean.clear()
        batch_meta.clear()

    for fname in files:
        print(f"[short] {fname}", flush=True)
        with gzip.open(DIR / fname, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line)
                n_orig += 1
                dead, _ = row_tombstone(obj)
                if dead:
                    n_tomb += 1
                    continue
                author = obj.get("author") or ""
                text = obj.get("text") if obj.get("text") is not None else obj.get("body")
                text = text if isinstance(text, str) else ""
                if author_is_known_bot(author) or text_has_bot_phrase(text)[0]:
                    n_hp += 1
                    continue
                n_base += 1
                rtype = obj.get("record_type")
                wc = word_count(text)
                if rtype == "comment":
                    n_base_comments += 1
                    comment_wc[min(wc, 51)] += 1  # 51 = 50+
                    n_comments += 1
                else:
                    n_base_posts += 1
                    n_posts += 1
                words = text.split()
                batch_meta.append({
                    "record_type": rtype,
                    "community": obj.get("community"),
                    "n_words": wc,
                    "preview": " ".join(words[:18]),
                })
                batch_clean.append(preprocess_text(text))
                if len(batch_clean) >= BATCH:
                    flush()
        flush()

    # percentiles from comment_wc (51 = 50+)
    # rebuild approximate cdf for 0..50
    cdf = []
    acc = 0
    for w in range(0, 51):
        acc += comment_wc.get(w, 0)
        cdf.append(acc)

    def pctile(p: float) -> int:
        target = p * n_base_comments
        for w, c in enumerate(cdf):
            if c >= target:
                return w
        return 50

    curve = []
    for min_w in MIN_WORDS:
        dropped_c = n_base_comments - surviving_comments[min_w]
        rec = {
            "min_comment_words": min_w,
            "n_remaining": surviving[min_w],
            "comments_remaining": surviving_comments[min_w],
            "posts_remaining": surviving[min_w] - surviving_comments[min_w],
            "comments_dropped": dropped_c,
            "comments_dropped_pct": round(100 * dropped_c / n_base_comments, 3) if n_base_comments else 0,
            "by_threshold": [],
        }
        for t in THRESHOLDS:
            k = flags[min_w][t]
            rec["by_threshold"].append({
                "threshold": t,
                "n": int(k),
                "pct_of_remaining": round(100 * k / surviving[min_w], 4) if surviving[min_w] else 0,
                "comment_n": int(flags_comments[min_w][t]),
            })
        curve.append(rec)

    results = {
        "n_original": n_orig,
        "n_tombstones": n_tomb,
        "n_lists": n_hp,
        "n_after_tomb_and_lists": n_base,
        "n_comments": n_base_comments,
        "n_posts": n_base_posts,
        "comment_word_count": {
            "n": n_base_comments,
            "p10": pctile(0.10),
            "p25": pctile(0.25),
            "p50": pctile(0.50),
            "p75": pctile(0.75),
            "histogram_0_to_50plus": [int(comment_wc.get(w, 0)) for w in range(0, 52)],
        },
        "curve": curve,
        "samples_dropped_under_5_words": dropped_at[5],
        "samples_dropped_under_10_words": dropped_at[10],
        "samples_flagged_under_10_words_p_ge_0.5": flagged_short,
        "n_comments_under_10_words_p_ge_0.5": flagged_short_n,
    }
    OUT.write_text(json.dumps(results, indent=2))
    summary = {
        "base": n_base,
        "comments": n_base_comments,
        "posts": n_base_posts,
        "comment_p10_p25_p50": [
            results["comment_word_count"]["p10"],
            results["comment_word_count"]["p25"],
            results["comment_word_count"]["p50"],
        ],
        "grid": [
            {
                "min_w": min_w,
                "remain": surviving[min_w],
                "c_drop_pct": round(100 * (n_base_comments - surviving_comments[min_w]) / n_base_comments, 2),
                "det": {str(t): round(100 * flags[min_w][t] / surviving[min_w], 2) if surviving[min_w] else 0
                        for t in (0.406, 0.5, 0.6, 0.8)},
            }
            for min_w in MIN_WORDS
        ],
        "wrote": str(OUT),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    os.chdir(DIR)
    main()
