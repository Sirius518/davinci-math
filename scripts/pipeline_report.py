"""Analyze pipeline output: per-dataset breakdown at each processor stage.

Reads the final output parquet (which contains trace events per record)
and the input shards (to count records removed by DatasetProcessors).

Usage:
    PYTHONPATH=src python scripts/pipeline_report.py \
        --input  data/intermediate/data_same_format_shards \
        --output data/processed/math_clean_classified.parquet
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def _load_dataset_counts(path: str) -> dict[str, int]:
    """Count records per dataset_name from parquet path (file or directory)."""
    counts: dict[str, int] = defaultdict(int)
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.parquet"))
    else:
        files = [p]
    for f in files:
        table = pq.read_table(f, columns=["dataset_name"])
        for name in table.column("dataset_name").to_pylist():
            counts[str(name or "")] += 1
    return dict(counts)


def _parse_trace(raw: str | list) -> list[dict]:
    if isinstance(raw, list):
        return raw
    if not raw or raw == "[]":
        return []
    try:
        return json.loads(raw)
    except Exception:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline per-dataset report")
    parser.add_argument("--input", required=True, help="Path to input shards (pre-pipeline)")
    parser.add_argument("--output", required=True, help="Path to pipeline output parquet")
    args = parser.parse_args()

    print("=" * 80)
    print("Loading input data counts ...")
    input_counts = _load_dataset_counts(args.input)
    total_input = sum(input_counts.values())
    print(f"Input: {total_input:,} records across {len(input_counts)} datasets\n")

    print("Loading output data ...")
    output_table = pq.read_table(args.output, columns=["dataset_name", "training_phase", "filter_tag", "trace"])
    rows = output_table.to_pylist()
    total_output = len(rows)
    print(f"Output: {total_output:,} records")
    print(f"Removed by DatasetProcessors (exact_dedup + fuzzy_dedup): {total_input - total_output:,}")
    print("=" * 80)

    # --- Per-dataset: input vs output vs phase ---
    output_counts: dict[str, int] = defaultdict(int)
    phase_by_dataset: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tag_by_dataset: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    processor_actions: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    for row in rows:
        ds = str(row.get("dataset_name", ""))
        phase = str(row.get("training_phase", ""))
        tag = str(row.get("filter_tag", ""))
        label = phase if phase else "posttrain"

        output_counts[ds] += 1
        phase_by_dataset[ds][label] += 1
        if tag:
            tag_by_dataset[ds][tag] += 1

        trace = _parse_trace(row.get("trace", "[]"))
        for event in trace:
            proc = str(event.get("processor", ""))
            status = str(event.get("status", ""))
            if proc:
                processor_actions[ds][proc][status] += 1

    all_datasets = sorted(set(list(input_counts.keys()) + list(output_counts.keys())))

    # --- Summary table ---
    print(f"\n{'Dataset':<45} {'Input':>10} {'Output':>10} {'Dedup-ed':>10} {'drop':>8} {'midtrain':>8} {'posttrain':>9}")
    print("-" * 105)
    totals = {"input": 0, "output": 0, "deduped": 0, "drop": 0, "midtrain": 0, "posttrain": 0}
    for ds in all_datasets:
        inp = input_counts.get(ds, 0)
        out = output_counts.get(ds, 0)
        deduped = inp - out
        drop = phase_by_dataset[ds].get("drop", 0)
        mid = phase_by_dataset[ds].get("midtrain", 0)
        post = phase_by_dataset[ds].get("posttrain", 0)
        name = ds if len(ds) <= 44 else ds[:41] + "..."
        print(f"{name:<45} {inp:>10,} {out:>10,} {deduped:>10,} {drop:>8,} {mid:>8,} {post:>9,}")
        totals["input"] += inp
        totals["output"] += out
        totals["deduped"] += deduped
        totals["drop"] += drop
        totals["midtrain"] += mid
        totals["posttrain"] += post
    print("-" * 105)
    print(f"{'TOTAL':<45} {totals['input']:>10,} {totals['output']:>10,} {totals['deduped']:>10,} {totals['drop']:>8,} {totals['midtrain']:>8,} {totals['posttrain']:>9,}")

    # --- Per-dataset filter_tag breakdown ---
    print(f"\n{'='*80}")
    print("Filter tag breakdown (why records were classified):")
    print(f"{'='*80}")
    for ds in all_datasets:
        tags = tag_by_dataset.get(ds, {})
        if not tags:
            continue
        print(f"\n  {ds}:")
        for tag, count in sorted(tags.items(), key=lambda x: -x[1]):
            print(f"    {tag:<35} {count:>8,}")

    # --- Per-dataset processor action breakdown ---
    print(f"\n{'='*80}")
    print("Processor action breakdown per dataset:")
    print(f"{'='*80}")
    for ds in all_datasets:
        procs = processor_actions.get(ds, {})
        if not procs:
            continue
        print(f"\n  {ds}:")
        for proc in sorted(procs.keys()):
            actions = procs[proc]
            parts = [f"{status}={count:,}" for status, count in sorted(actions.items())]
            print(f"    {proc:<40} {', '.join(parts)}")

    print(f"\n{'='*80}")
    print("Done.")


if __name__ == "__main__":
    main()
