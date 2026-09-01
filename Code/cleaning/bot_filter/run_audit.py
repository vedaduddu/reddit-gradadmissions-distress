"""Stream jsonl.gz records and measure three-signal bot-content overlap.

Does not write decompressed copies. Detector is the portable TF-IDF logistic
model from the Moltbook paper (RQ2 / Appendix A.3; feature set B).
"""
from __future__ import annotations

import gzip
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_detector import AIContentDetector
from ai_detector.preprocess import preprocess_text
from bot_filter.lists import (
    KNOWN_BOT_AUTHORS,
    author_is_known_bot,
    text_has_bot_phrase,
)

DIR = Path(os.environ.get("REDDIT_STANDARDIZED_DIR", "Data/standardized_filtered"))
OUT = Path(os.environ.get("REDDIT_CLEANING_OUTPUT_DIR", DIR)) / "audit_results.json"
BATCH = 8192
THRESH = 0.406  # paper Table A4 max-F1 operating point
THRESH_HI_PREC = 0.445
THRESH_HALF = 0.50
MIN_WORDS = 20
SAMPLE_K = 8
SEED = 42


def venn_key(a: bool, b: bool, c: bool) -> str:
    return f"{int(a)}{int(b)}{int(c)}"


VENN_LABELS = {
    "100": "username only",
    "010": "phrase only",
    "001": "detector only",
    "110": "username ∩ phrase (not detector)",
    "101": "username ∩ detector (not phrase)",
    "011": "phrase ∩ detector (not username)",
    "111": "all three",
}


def main() -> None:
    rng = random.Random(SEED)
    det = AIContentDetector()
    files = sorted(p.name for p in DIR.glob("*.jsonl.gz"))

    n = 0
    n_scored = 0
    n_empty = 0
    n_user = n_phrase = n_det = 0
    n_det_hi = n_det_half = 0
    n_det_long = 0
    n_long = 0
    venn = Counter()
    venn_by = defaultdict(Counter)  # (community, record_type) -> venn
    phrase_hits = Counter()
    user_authors = Counter()
    phrase_authors = Counter()
    det_authors = Counter()
    score_hist = [0] * 20  # 0.05 bins
    samples = defaultdict(list)
    samples_n = Counter()

    batch_meta: list[dict] = []
    batch_clean: list[str] = []

    def flush() -> None:
        nonlocal n_scored, n_det, n_det_hi, n_det_half, n_det_long, n_long
        if not batch_clean:
            return
        proba = det.pipeline.predict_proba(batch_clean)[:, 1]
        for meta, p in zip(batch_meta, proba):
            n_scored += 1
            b = int(min(19, max(0, p * 20)))
            score_hist[b] += 1
            flag_d = bool(p >= THRESH)
            flag_d_hi = bool(p >= THRESH_HI_PREC)
            flag_d_half = bool(p >= THRESH_HALF)
            if flag_d:
                n_det += 1
                det_authors[meta["author"]] += 1
            if flag_d_hi:
                n_det_hi += 1
            if flag_d_half:
                n_det_half += 1
            if meta["n_words"] >= MIN_WORDS:
                n_long += 1
                if flag_d:
                    n_det_long += 1
            k = venn_key(meta["user"], meta["phrase"], flag_d)
            venn[k] += 1
            venn_by[(meta["community"], meta["record_type"])][k] += 1
            if k != "000":
                bucket = samples[k]
                samples_n[k] += 1
                if len(bucket) < SAMPLE_K:
                    bucket.append({**meta, "p_ai_agent": float(p), "region": VENN_LABELS[k]})
                else:
                    j = rng.randrange(samples_n[k])
                    if j < SAMPLE_K:
                        bucket[j] = {**meta, "p_ai_agent": float(p), "region": VENN_LABELS[k]}
        batch_meta.clear()
        batch_clean.clear()

    for fname in files:
        path = DIR / fname
        print(f"[audit] {fname}", flush=True)
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line)
                n += 1
                author = obj.get("author") or ""
                text = obj.get("text") if obj.get("text") is not None else obj.get("body")
                text = text if isinstance(text, str) else ""
                if not text.strip():
                    n_empty += 1
                    text = ""
                flag_u = author_is_known_bot(author)
                flag_p, phrase_lab = text_has_bot_phrase(text)
                if flag_u:
                    n_user += 1
                    user_authors[author] += 1
                if flag_p:
                    n_phrase += 1
                    phrase_hits[phrase_lab] += 1
                    phrase_authors[author] += 1
                words = text.split()
                batch_meta.append({
                    "file": fname,
                    "community": obj.get("community"),
                    "record_type": obj.get("record_type"),
                    "id": obj.get("id"),
                    "author": author,
                    "user": flag_u,
                    "phrase": flag_p,
                    "phrase_label": phrase_lab,
                    "n_words": len(words),
                    "text_preview": " ".join(words[:60]),
                })
                batch_clean.append(preprocess_text(text))
                if len(batch_clean) >= BATCH:
                    flush()
        flush()

    union = n - venn["000"]
    any2 = venn["110"] + venn["101"] + venn["011"] + venn["111"]
    all3 = venn["111"]

    per_slice = []
    for (comm, rtype), ctr in sorted(venn_by.items()):
        total = sum(ctr.values())
        u = ctr["100"] + ctr["110"] + ctr["101"] + ctr["111"]
        p = ctr["010"] + ctr["110"] + ctr["011"] + ctr["111"]
        d = ctr["001"] + ctr["101"] + ctr["011"] + ctr["111"]
        un = total - ctr["000"]
        per_slice.append({
            "community": comm,
            "record_type": rtype,
            "n": total,
            "username": u,
            "phrase": p,
            "detector": d,
            "union": un,
            "union_pct": round(100.0 * un / total, 3) if total else 0,
            "regions": {VENN_LABELS[k]: int(ctr[k]) for k in VENN_LABELS},
        })

    def top(counter: Counter, k=15):
        return [{"author": a, "n": c} for a, c in counter.most_common(k)]

    # strip previews of samples for JSON
    sample_out = {}
    for k, rows in samples.items():
        sample_out[VENN_LABELS[k]] = [
            {kk: vv for kk, vv in row.items() if kk != "user"}
            for row in rows
        ]

    results = {
        "n_records": n,
        "n_empty_text": n_empty,
        "n_scored": n_scored,
        "known_bot_list_size": len(KNOWN_BOT_AUTHORS),
        "threshold": {
            "primary_max_f1": THRESH,
            "high_precision_paper": THRESH_HI_PREC,
            "half": THRESH_HALF,
        },
        "signals": {
            "username": n_user,
            "phrase": n_phrase,
            "detector_0.406": n_det,
            "detector_0.445": n_det_hi,
            "detector_0.50": n_det_half,
            "detector_0.406_among_ge20_words": n_det_long,
            "n_ge20_words": n_long,
        },
        "combinations": {
            "union_any": union,
            "union_pct": round(100.0 * union / n, 4) if n else 0,
            "any_two_or_three": any2,
            "all_three": all3,
            "username_or_phrase_high_precision": None,
        },
        "venn": {VENN_LABELS[k]: int(venn[k]) for k in VENN_LABELS},
        "venn_none": int(venn["000"]),
        "phrase_pattern_hits": dict(phrase_hits),
        "unique_authors_flagged": {
            "username": len(user_authors),
            "phrase": len(phrase_authors),
            "detector": len(det_authors),
        },
        "top_authors": {
            "username": top(user_authors),
            "phrase": top(phrase_authors),
            "detector": top(det_authors),
        },
        "score_histogram": {
            "bin_width": 0.05,
            "counts": score_hist,
        },
        "by_community_type": per_slice,
        "samples": sample_out,
        "notes": {
            "detector": (
                "Portable TF-IDF unigram+bigram logistic regression from the "
                "Moltbook paper RQ2 detection experiment (Appendix A.3 feature "
                "set B; test acc 0.948). Trained on Moltbook agents vs matched "
                "Reddit humans. Style/register detector, not a general ChatGPT detector. "
                "Default threshold 0.406 is the original max-F1 operating point."
            ),
            "domain_shift": (
                "Admissions Reddit is more formal/expository than the training "
                "Reddit communities, so detector-only flags are expected to include "
                "false positives. Username and phrase are high-precision platform-bot "
                "filters; the detector is a soft LLM-agent-style flag."
            ),
        },
    }

    # username OR phrase (high precision union): inclusion-exclusion
    user_or_phrase = (
        venn["100"] + venn["010"] + venn["110"]
        + venn["101"] + venn["011"] + venn["111"]
    )
    # wait that's username OR phrase OR (username∩det) etc. which includes detector overlaps
    # correct: any record with user or phrase, regardless of detector
    user_or_phrase = n_user + n_phrase - (
        venn["110"] + venn["111"]  # counted in both user and phrase, detector or not
        # n_user includes 100,110,101,111
        # n_phrase includes 010,110,011,111
        # intersection of user and phrase = 110 + 111
    )
    results["combinations"]["username_or_phrase_high_precision"] = user_or_phrase
    results["combinations"]["username_or_phrase_pct"] = round(100.0 * user_or_phrase / n, 4) if n else 0

    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps({
        "n": n,
        "username": n_user,
        "phrase": n_phrase,
        "detector": n_det,
        "union": union,
        "all3": all3,
        "venn": results["venn"],
        "wrote": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    os.chdir(DIR)
    main()
