"""Apply the agreed filtration pipeline and write *_filtered.jsonl.gz files.

Order (first match drops the row):
  1. Tombstone: author [deleted]/[removed], or body/text/title [removed]/[deleted]
  2. Known bot username or bot self-ID phrase
  3. Fewer than MIN_WORDS whitespace tokens in `text` (fallback: body)
  4. Moltbook-style detector p(ai_agent) >= DETECTOR_THRESHOLD
Kept rows are written unchanged.
"""
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
from bot_filter.threshold_curve import row_tombstone

DIR = Path(os.environ.get("REDDIT_STANDARDIZED_DIR", "Data/standardized_filtered"))
METRICS_OUT = Path(os.environ.get("REDDIT_CLEANING_OUTPUT_DIR", DIR)) / "pipeline_metrics.json"
BATCH = 8192
MIN_WORDS = 5
DETECTOR_THRESHOLD = 0.75

INPUTS = [
    "MBA_comments.jsonl.gz",
    "MBA_posts.jsonl.gz",
    "MSCS_comments.jsonl.gz",
    "MSCS_posts.jsonl.gz",
    "gradadmissions_comments.jsonl.gz",
    "gradadmissions_posts.jsonl.gz",
]


def analysis_text(obj: dict) -> str:
    text = obj.get("text") if obj.get("text") is not None else obj.get("body")
    return text if isinstance(text, str) else ""


def filtered_name(fname: str) -> str:
    return fname.replace(".jsonl.gz", "_filtered.jsonl.gz")


def main() -> None:
    det = AIContentDetector()
    overall = Counter()
    per_file = {}

    for fname in INPUTS:
        src = DIR / fname
        dst = DIR / filtered_name(fname)
        print(f"[filter] {fname} -> {dst.name}", flush=True)
        counts = Counter()
        by_comm = Counter()
        by_type = Counter()
        batch_objs: list[dict] = []
        batch_lines: list[str] = []
        batch_clean: list[str] = []

        def flush(out_fh) -> None:
            if not batch_clean:
                return
            proba = det.pipeline.predict_proba(batch_clean)[:, 1]
            for obj, raw, p in zip(batch_objs, batch_lines, proba):
                if float(p) >= DETECTOR_THRESHOLD:
                    counts["detector"] += 1
                    continue
                counts["kept"] += 1
                by_comm[obj.get("community")] += 1
                by_type[obj.get("record_type")] += 1
                out_fh.write(raw if raw.endswith("\n") else raw + "\n")
            batch_objs.clear()
            batch_lines.clear()
            batch_clean.clear()

        with gzip.open(src, "rt", encoding="utf-8") as inf, gzip.open(
            dst, "wt", encoding="utf-8", compresslevel=6
        ) as outf:
            for line in inf:
                if not line.strip():
                    continue
                obj = json.loads(line)
                counts["n"] += 1
                if row_tombstone(obj)[0]:
                    counts["tombstone"] += 1
                    continue
                text = analysis_text(obj)
                if author_is_known_bot(obj.get("author") or "") or text_has_bot_phrase(text)[0]:
                    counts["bot_list"] += 1
                    continue
                if len(text.split()) < MIN_WORDS:
                    counts["short"] += 1
                    continue
                batch_objs.append(obj)
                batch_lines.append(line)
                batch_clean.append(preprocess_text(text))
                if len(batch_clean) >= BATCH:
                    flush(outf)
            flush(outf)

        dst_size = dst.stat().st_size
        per_file[fname] = {
            "output": dst.name,
            "output_bytes": dst_size,
            **{k: int(counts[k]) for k in ("n", "tombstone", "bot_list", "short", "detector", "kept")},
            "kept_by_community": dict(by_comm),
            "kept_by_type": dict(by_type),
        }
        for k in ("n", "tombstone", "bot_list", "short", "detector", "kept"):
            overall[k] += counts[k]
        dropped = counts["n"] - counts["kept"]
        print(
            f"  n={counts['n']:,} kept={counts['kept']:,} "
            f"({100 * counts['kept'] / counts['n']:.2f}%)  "
            f"drop tomb={counts['tombstone']:,} list={counts['bot_list']:,} "
            f"short={counts['short']:,} det={counts['detector']:,}  "
            f"out={dst_size / 1e6:.1f} MB",
            flush=True,
        )

    metrics = {
        "pipeline": {
            "order": [
                "tombstone (author [deleted]/[removed] or body/text/title [removed]/[deleted])",
                "known bot username or bot self-ID phrase",
                f"whitespace word count of text < {MIN_WORDS} (posts and comments)",
                f"Moltbook-style TF-IDF logistic detector p(ai_agent) >= {DETECTOR_THRESHOLD}",
            ],
            "min_words": MIN_WORDS,
            "detector_threshold": DETECTOR_THRESHOLD,
            "detector_note": (
                "0.75 is a high-precision out-of-domain operating point, not the "
                "paper max-F1 cutoff of 0.406 (Appendix A.3 Table A4)."
            ),
        },
        "overall": {k: int(overall[k]) for k in ("n", "tombstone", "bot_list", "short", "detector", "kept")},
        "overall_pct_of_original": {
            k: round(100 * overall[k] / overall["n"], 4)
            for k in ("tombstone", "bot_list", "short", "detector", "kept")
        },
        "files": per_file,
    }
    METRICS_OUT.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["overall"], indent=2))
    print("wrote", METRICS_OUT)


if __name__ == "__main__":
    main()
