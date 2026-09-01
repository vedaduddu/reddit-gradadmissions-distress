"""CLI: python -m ai_detector "some text"  |  python -m ai_detector --file texts.txt"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .detect import AIContentDetector


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score text as AI-agent-like (Moltbook) vs human-like (Reddit)."
    )
    ap.add_argument("text", nargs="?", help="Text to score")
    ap.add_argument("--file", type=Path, help="File with one text per line")
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    det = AIContentDetector(args.model)
    if args.file:
        lines = [ln.rstrip("\n") for ln in args.file.read_text().splitlines() if ln.strip()]
        out = det.score(lines)
        if args.json:
            print(out.to_json(orient="records"))
        else:
            print(out[["p_ai_agent", "label"]].to_string(index=False))
        return
    if not args.text:
        args.text = sys.stdin.read()
    result = det.predict_one(args.text)
    if args.json:
        print(json.dumps(result))
    else:
        print(f"{result['label']}  p(ai_agent)={result['p_ai_agent']:.3f}  "
              f"(threshold={result['threshold']:.3f})")


if __name__ == "__main__":
    main()
