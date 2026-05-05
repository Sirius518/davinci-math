"""Split pipeline output into drop / midtrain / posttrain Parquet files.

Usage:
    python scripts/split_by_training_phase.py \
        --input /path/to/classified.parquet \
        --output-dir /path/to/splits
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


_PHASE_MAP = {
    "drop": "drop",
    "midtrain": "midtrain",
}


def split(input_path: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(input_path)
    phases = table.column("training_phase").to_pylist()

    buckets: dict[str, list[int]] = {"drop": [], "midtrain": [], "posttrain": []}
    for idx, phase in enumerate(phases):
        bucket = _PHASE_MAP.get(phase, "posttrain")
        buckets[bucket].append(idx)

    for label, indices in buckets.items():
        subset = table.take(indices)
        dest = out / f"{label}.parquet"
        pq.write_table(subset, dest)
        print(f"{label}: {len(indices)} records -> {dest}")

    print(f"\ntotal: {table.num_rows}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split classified data by training_phase")
    parser.add_argument("--input", required=True, help="Path to pipeline output Parquet")
    parser.add_argument("--output-dir", required=True, help="Directory for split Parquet files")
    args = parser.parse_args()
    split(args.input, args.output_dir)


if __name__ == "__main__":
    main()
