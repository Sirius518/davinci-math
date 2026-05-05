"""Extract records that need retry (samples < num_samples) from result parquet files.

Streams in batches to avoid loading multi-GB files into RAM.
Produces a single merged parquet with all retry-needed records across slices.

Usage:
    # Extract from a single file
    python scripts/extract_retry_records.py \
        --input data/rollout_slices_decontam/slice_00_midtrain_result.parquet \
        --output data/rollout_slices_decontam/slice_00_midtrain_needretry.parquet

    # Extract from all 10 slices into one merged file
    python scripts/extract_retry_records.py --all-midtrain
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "rollout_slices_decontam"


def _filter_batch(
    batch: pa.RecordBatch,
    num_samples: int,
    target_phase: str | None,
    max_question_chars: int,
) -> pa.RecordBatch:
    """Return only rows that need retry."""
    n = batch.num_rows
    keep_mask = []
    verification_col = batch.column("verification")
    question_col = batch.column("question")
    phase_col = batch.column("training_phase") if "training_phase" in batch.schema.names else None

    for i in range(n):
        if target_phase and phase_col is not None:
            if phase_col[i].as_py() != target_phase:
                keep_mask.append(False)
                continue

        v_str = verification_col[i].as_py()
        try:
            v = json.loads(v_str) if v_str else {}
        except (json.JSONDecodeError, TypeError):
            v = {}
        if len(v.get("samples") or []) >= num_samples:
            keep_mask.append(False)
            continue

        q = question_col[i].as_py() or ""
        if len(q) > max_question_chars:
            keep_mask.append(False)
            continue

        keep_mask.append(True)

    return batch.filter(pa.array(keep_mask, type=pa.bool_()))


def extract_retry(
    input_path: Path,
    output_path: Path,
    num_samples: int = 8,
    target_phase: str | None = "midtrain",
    max_question_chars: int = 12000,
    batch_size: int = 1000,
) -> int:
    t0 = time.time()
    parquet_file = pq.ParquetFile(input_path)
    schema = parquet_file.schema_arrow
    total_rows = parquet_file.metadata.num_rows
    print(f"Input : {input_path}  ({total_rows:,} rows)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = pa_ds.dataset(str(input_path), format="parquet")
    written = 0
    scanned = 0
    writer = None

    try:
        for batch in dataset.to_batches(batch_size=batch_size):
            scanned += batch.num_rows
            filtered = _filter_batch(batch, num_samples, target_phase, max_question_chars)
            if filtered.num_rows > 0:
                if writer is None:
                    writer = pq.ParquetWriter(output_path, schema, compression="snappy")
                writer.write_table(pa.Table.from_batches([filtered], schema=schema))
                written += filtered.num_rows
            if scanned % 50000 == 0 or scanned == total_rows:
                print(f"  scanned {scanned:,}/{total_rows:,}  need_retry={written:,}")
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        pq.write_table(
            pa.table({name: pa.array([], type=schema.field(name).type) for name in schema.names}),
            output_path, compression="snappy",
        )

    elapsed = time.time() - t0
    print(f"Output: {output_path}  ({written:,} rows, {elapsed:.1f}s)")
    return written


def _extract_one_slice(args: tuple[str, Path, int, int]) -> tuple[str, Path, int, float]:
    """Extract retry records from one slice to a temp file. Used by ProcessPoolExecutor."""
    sl, input_path, num_samples, max_question_chars = args
    tmp_output = input_path.with_name(f"_tmp_needretry_{sl}.parquet")

    t0 = time.time()
    parquet_file = pq.ParquetFile(input_path)
    schema = parquet_file.schema_arrow
    total_rows = parquet_file.metadata.num_rows
    print(f"[slice_{sl}] start: {input_path.name}  ({total_rows:,} rows)")

    dataset = pa_ds.dataset(str(input_path), format="parquet")
    scanned = 0
    written = 0
    writer = None

    try:
        for batch in dataset.to_batches(batch_size=1000):
            scanned += batch.num_rows
            filtered = _filter_batch(batch, num_samples, "midtrain", max_question_chars)
            if filtered.num_rows > 0:
                if writer is None:
                    writer = pq.ParquetWriter(tmp_output, schema, compression="snappy")
                writer.write_table(pa.Table.from_batches([filtered], schema=schema))
                written += filtered.num_rows
    finally:
        if writer is not None:
            writer.close()

    elapsed = time.time() - t0
    print(f"[slice_{sl}] done: {written:,} records ({elapsed:.1f}s)")
    return sl, tmp_output, written, elapsed


def run_all_midtrain(num_samples: int, max_question_chars: int, workers: int = 4) -> None:
    from concurrent.futures import ProcessPoolExecutor

    slices = {
        "00": "slice_00_midtrain_result.parquet",
        "01": "slice_01_midtrain_result.parquet",
        "02": "slice_02_midtrain_result.parquet",
        "03": "slice_03_midtrain_result.parquet",
        "04": "slice_04_midtrain_result.parquet",
        "05": "slice_05_midtrain_result.parquet",
        "06": "slice_06_midtrain_result.parquet",
        "07": "slice_07_midtrain_result.parquet",
        "08": "slice_08_midtrain_result.parquet",
        "09": "slice_09_midtrain_result_resumed.parquet",
    }

    merged_output = DATA_DIR / "midtrain_needretry_all.parquet"
    merged_output.parent.mkdir(parents=True, exist_ok=True)

    # Build task list
    tasks = []
    for sl, filename in slices.items():
        input_path = DATA_DIR / filename
        if not input_path.exists():
            print(f"SKIP slice_{sl}: {input_path} not found")
            continue
        tasks.append((sl, input_path, num_samples, max_question_chars))

    # Parallel extraction: each process writes a temp file
    t0_all = time.time()
    print(f"Extracting {len(tasks)} slices with {workers} workers...")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_extract_one_slice, tasks))

    # Merge temp files into one output (sequential, fast — only small files)
    total = 0
    writer = None
    try:
        for sl, tmp_path, written, _ in results:
            if written == 0:
                continue
            table = pq.read_table(tmp_path)
            if writer is None:
                writer = pq.ParquetWriter(merged_output, table.schema, compression="snappy")
            writer.write_table(table)
            total += written
            tmp_path.unlink()
    finally:
        if writer is not None:
            writer.close()

    # Clean up empty temp files
    for _, tmp_path, written, _ in results:
        if tmp_path.exists():
            tmp_path.unlink()

    elapsed_all = time.time() - t0_all
    print(f"\n{'='*60}")
    print(f"Merged output: {merged_output}")
    print(f"Total records needing retry: {total:,}  ({elapsed_all:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract records needing retry")
    parser.add_argument("--input", help="Single input parquet file")
    parser.add_argument("--output", help="Output parquet file")
    parser.add_argument("--all-midtrain", action="store_true",
                        help="Extract from all 10 midtrain result slices into one merged file")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-question-chars", type=int, default=12000)
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers for --all-midtrain")
    args = parser.parse_args()

    if args.all_midtrain:
        run_all_midtrain(args.num_samples, args.max_question_chars, workers=args.workers)
    elif args.input and args.output:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = ROOT / input_path
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        extract_retry(
            input_path, output_path,
            num_samples=args.num_samples,
            max_question_chars=args.max_question_chars,
        )
    else:
        parser.error("Provide --input/--output or --all-midtrain")


if __name__ == "__main__":
    main()
