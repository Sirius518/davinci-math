#!/usr/bin/env python3
"""Compute per-dataset_name record counts at each checkpoint stage.

Usage:
    python scripts/checkpoint_stats.py <checkpoint_dir> [--csv output.csv]

Reads every *.parquet under checkpoint_dir (sorted by mtime),
groups records by dataset_name, and prints a table showing how
each dataset's count changes across stages.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def _load_counts(parquet_path: Path) -> Counter[str]:
    table = pq.read_table(parquet_path, columns=["dataset_name"])
    return Counter(table.column("dataset_name").to_pylist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-dataset checkpoint statistics")
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--csv", type=Path, default=None, help="Save CSV to this path")
    args = parser.parse_args()

    ckpt_dir: Path = args.checkpoint_dir
    if not ckpt_dir.is_dir():
        print(f"Error: {ckpt_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    parquet_files = sorted(ckpt_dir.glob("*.parquet"), key=lambda p: p.stat().st_mtime)
    if not parquet_files:
        print(f"No parquet files found in {ckpt_dir}", file=sys.stderr)
        sys.exit(1)

    stage_names: list[str] = []
    stage_counts: list[Counter[str]] = []
    all_datasets: set[str] = set()

    for pf in parquet_files:
        stage = pf.stem
        counts = _load_counts(pf)
        stage_names.append(stage)
        stage_counts.append(counts)
        all_datasets.update(counts.keys())

    datasets = sorted(all_datasets)

    header = ["dataset_name"] + stage_names + ["final_kept", "total_dropped"]
    rows: list[list[str | int]] = []
    for ds in datasets:
        row: list[str | int] = [ds]
        first = stage_counts[0].get(ds, 0) if stage_counts else 0
        last = stage_counts[-1].get(ds, 0) if stage_counts else 0
        for sc in stage_counts:
            row.append(sc.get(ds, 0))
        row.append(last)
        row.append(first - last)
        rows.append(row)

    total_row: list[str | int] = ["__TOTAL__"]
    for sc in stage_counts:
        total_row.append(sum(sc.values()))
    total_row.append(sum(stage_counts[-1].values()) if stage_counts else 0)
    first_total = sum(stage_counts[0].values()) if stage_counts else 0
    last_total = sum(stage_counts[-1].values()) if stage_counts else 0
    total_row.append(first_total - last_total)
    rows.append(total_row)

    col_widths = [max(len(str(header[i])), *(len(str(r[i])) for r in rows)) for i in range(len(header))]

    def _fmt_row(values: list) -> str:
        parts = []
        for i, v in enumerate(values):
            if i == 0:
                parts.append(str(v).ljust(col_widths[i]))
            else:
                parts.append(str(v).rjust(col_widths[i]))
        return "  ".join(parts)

    print(_fmt_row(header))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(_fmt_row(row))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"\nCSV saved to {args.csv}")


if __name__ == "__main__":
    main()
