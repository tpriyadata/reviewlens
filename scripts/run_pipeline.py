"""Run ReviewLens over a CSV with columns: review_id, text.

    python scripts/run_pipeline.py --input data/sample_reviews.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from azure.core.exceptions import ClientAuthenticationError

from reviewlens.client import ConfigError, build_client
from reviewlens.pipeline import ReviewPipeline
from reviewlens.report import summarize

REQUIRED_COLUMNS = {"review_id", "text"}


def load_reviews(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return [(row["review_id"], row["text"]) for row in reader]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/sample_reviews.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("output"))
    parser.add_argument("--languages", default="en", help="comma-separated ISO 639-1 codes")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    # azure-core logs full HTTP traffic at INFO; keep it quiet unless debugging.
    logging.getLogger("azure").setLevel(logging.WARNING)

    try:
        reviews = load_reviews(args.input)
        client = build_client()
        pipeline = ReviewPipeline(client, allowed_languages=args.languages.split(","))
        results = pipeline.run(reviews)
    except (ConfigError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ClientAuthenticationError:
        print("error: authentication failed - check LANGUAGE_KEY matches LANGUAGE_ENDPOINT",
              file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")

    summary = summarize(results)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {results_path} and {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
