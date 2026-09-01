"""Drop Reddit tombstones, keep username/phrase lists, sweep detector thresholds."""
from __future__ import annotations

import gzip
import json
import os
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from ai_detector import AIContentDetector
from ai_detector.preprocess import preprocess_text
from bot_filter.lists import author_is_known_bot, text_has_bot_phrase

DIR = Path(os.environ.get("REDDIT_STANDARDIZED_DIR", "Data/standardized_filtered"))
OUT = Path(os.environ.get("REDDIT_CLEANING_OUTPUT_DIR", DIR)) / "threshold_curve.json"
BATCH = 8192
THRESHOLDS = [0.40, 0.406, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
TOMBSTONE = {"[deleted]", "[removed]"}
MIN_WORDS = 20


def is_tombstone_token(value) -> bool:
    if value is None:
        return False
    if not isinstance(value, str):
        return False
    return value.strip().lower() in TOMBSTONE


def row_tombstone(obj: dict) -> tuple[bool, str | None]:
    author = obj.get("author")
    if is_tombstone_token(author):
        return True, "author_[deleted]" if (author or "").strip().lower() == "[deleted]" else "author_[removed]"
    body = obj.get("body")
    text = obj.get("text")
    title = obj.get("title")
    # Native content field: comments use body; posts use body (selftext).
    if is_tombstone_token(body):
        token = body.strip().lower()
        return True, f"body_{token}"
    if is_tombstone_token(text):
        token = text.strip().lower()
        return True, f"text_{token}"
    if is_tombstone_token(title):
        token = title.strip().lower()
        return True, f"title_{token}"
    return False, None


def main() -> None:
    det = AIContentDetector()
    files = sorted(p.name for p in DIR.glob("*.jsonl.gz"))

    n = 0
    tomb = Counter()
    tomb_by = defaultdict(Counter)
    n_after_tomb = 0
    n_user = n_phrase = n_hp = 0
    n_keep = 0  # after tomb + username/phrase
    n_keep_long = 0

    flags = Counter()       # threshold -> count on keep set
    flags_long = Counter()  # keep set AND >=20 words
    flags_after_tomb_only = Counter()  # after tomb, before username/phrase
    hist_keep = [0] * 20

    by_slice_keep = defaultdict(int)
    by_slice_flags = defaultdict(Counter)

    batch_clean: list[str] = []
    batch_meta: list[dict] = []

    def flush() -> None:
        if not batch_clean:
            return
        proba = det.pipeline.predict_proba(batch_clean)[:, 1]
        for meta, p in zip(batch_meta, proba):
            b = int(min(19, max(0, p * 20)))
            if meta["keep"]:
                hist_keep[b] += 1
            for t in THRESHOLDS:
                hit = p >= t
                if hit:
                    flags_after_tomb_only[t] += 1
                    if meta["keep"]:
                        flags[t] += 1
                        by_slice_flags[(meta["community"], meta["record_type"])][t] += 1
                        if meta["n_words"] >= MIN_WORDS:
                            flags_long[t] += 1
        batch_clean.clear()
        batch_meta.clear()

    for fname in files:
        print(f"[curve] {fname}", flush=True)
        with gzip.open(DIR / fname, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line)
                n += 1
                dead, reason = row_tombstone(obj)
                if dead:
                    tomb[reason] += 1
                    tomb_by[(obj.get("community"), obj.get("record_type"))][reason] += 1
                    continue
                n_after_tomb += 1
                author = obj.get("author") or ""
                text = obj.get("text") if obj.get("text") is not None else obj.get("body")
                text = text if isinstance(text, str) else ""
                flag_u = author_is_known_bot(author)
                flag_p, _ = text_has_bot_phrase(text)
                if flag_u:
                    n_user += 1
                if flag_p:
                    n_phrase += 1
                hp = flag_u or flag_p
                if hp:
                    n_hp += 1
                keep = not hp
                if keep:
                    n_keep += 1
                    by_slice_keep[(obj.get("community"), obj.get("record_type"))] += 1
                    if len(text.split()) >= MIN_WORDS:
                        n_keep_long += 1
                batch_meta.append({
                    "keep": keep,
                    "community": obj.get("community"),
                    "record_type": obj.get("record_type"),
                    "n_words": len(text.split()),
                })
                batch_clean.append(preprocess_text(text))
                if len(batch_clean) >= BATCH:
                    flush()
        flush()

    def row_for(t: float) -> dict:
        k = flags[t]
        kl = flags_long[t]
        tomb_only = flags_after_tomb_only[t]
        return {
            "threshold": t,
            "flagged_after_tombstones": tomb_only,
            "flagged_after_tombstones_pct_of_after_tomb": round(100 * tomb_only / n_after_tomb, 4) if n_after_tomb else 0,
            "flagged_after_tomb_and_lists": k,
            "pct_of_keep": round(100 * k / n_keep, 4) if n_keep else 0,
            "pct_of_original": round(100 * k / n, 4) if n else 0,
            "flagged_keep_ge20_words": kl,
            "pct_of_keep_ge20": round(100 * kl / n_keep_long, 4) if n_keep_long else 0,
        }

    slice_out = []
    for (comm, rtype), nk in sorted(by_slice_keep.items()):
        rec = {"community": comm, "record_type": rtype, "n_keep": nk, "by_threshold": {}}
        for t in THRESHOLDS:
            rec["by_threshold"][str(t)] = {
                "n": int(by_slice_flags[(comm, rtype)][t]),
                "pct": round(100 * by_slice_flags[(comm, rtype)][t] / nk, 3) if nk else 0,
            }
        slice_out.append(rec)

    results = {
        "n_original": n,
        "tombstones": {
            "total": sum(tomb.values()),
            "pct_of_original": round(100 * sum(tomb.values()) / n, 4) if n else 0,
            "by_reason": dict(tomb),
            "by_slice": {
                f"{c}/{r}": dict(ctr) for (c, r), ctr in sorted(tomb_by.items())
            },
        },
        "after_tombstones": n_after_tomb,
        "username_or_phrase_among_remaining": n_hp,
        "username": n_user,
        "phrase": n_phrase,
        "n_keep_for_detector_curve": n_keep,
        "n_keep_ge20_words": n_keep_long,
        "curve": [row_for(t) for t in THRESHOLDS],
        "score_histogram_keep": {"bin_width": 0.05, "counts": hist_keep},
        "by_community_type": slice_out,
        "threshold_origin": {
            "value": 0.406,
            "source": "Moltbook paper Appendix A.3 Table A4, best model (combined features + XGBoost)",
            "meaning": (
                "Max-F1 operating point on the held-out in-domain test set "
                "(precision 0.943, recall 0.964, F1 0.953). High-precision "
                "(≥95%) was 0.445; high-recall (≥95%) was 0.456. The portable "
                "TF-IDF logistic model shipped here uses the same 0.406 cutoff "
                "so the operating point matches the paper, even though this "
                "classifier is feature-set B rather than C_xgboost."
            ),
        },
    }
    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps({
        "n_original": n,
        "tombstones": sum(tomb.values()),
        "tomb_reasons": dict(tomb),
        "after_tomb": n_after_tomb,
        "hp_lists": n_hp,
        "keep": n_keep,
        "curve": [
            {"t": t, "n": flags[t], "pct_keep": round(100 * flags[t] / n_keep, 3) if n_keep else 0}
            for t in THRESHOLDS
        ],
        "wrote": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    os.chdir(DIR)
    main()
