#!/usr/bin/env python
"""Quick test: run answer_judge on first N records (default 100)."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pipeline.ops  # noqa: F401 — register all processors

from pipeline.core.io import load_yaml, read_canonical_records, write_canonical_records
from pipeline.core.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Test answer_judge on a small subset")
    parser.add_argument("--config", required=True, help="Pipeline YAML config path")
    parser.add_argument("--limit", type=int, default=100, help="Max records to process")
    parser.add_argument("--output", default="", help="Override output path (default: adds _testN suffix)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_yaml(args.config)
    input_path = config["input_path"]

    log = logging.getLogger("test_answer_judge")
    log.info("Reading records from %s ...", input_path)
    all_records = read_canonical_records(input_path)
    log.info("Total records in file: %d", len(all_records))

    records = all_records[: args.limit]
    log.info("Using first %d records for testing", len(records))

    flagged = sum(
        1 for r in records
        if dict(r.verification).get("majority_matches_gt") is False
    )
    log.info("Of which %d are flagged (majority_matches_gt=False) and will actually call LLM", flagged)

    if flagged == 0:
        log.warning(
            "No flagged records in the first %d rows. "
            "Try increasing --limit to include flagged records.",
            args.limit,
        )

    if args.output:
        output_path = args.output
    else:
        p = Path(config["output_path"])
        output_path = str(p.with_name(f"{p.stem}_test{args.limit}.parquet"))

    result = run_pipeline(records, config, checkpoint_dir=None, workers=1)
    write_canonical_records(result.records, output_path)

    log.info("Done. Output: %s (%d records)", output_path, len(result.records))


if __name__ == "__main__":
    main()
