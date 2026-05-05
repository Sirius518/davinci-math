#!/usr/bin/env python3
"""Run low-threshold fuzzy dedup on a stratified sample and inspect Jaccard buckets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.core.io import load_yaml
from pipeline.core.schema import CanonicalRecord
from pipeline.ops.dedup._minhash_engine import VerifyResult
from pipeline.ops.dedup.fuzzy import _union_find_drop_ids, minhash_candidate_pairs
from pipeline.utils.priority import build_record_priority_map, parse_tier_config


def truncate(text: str, max_len: int = 400) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_len:
        return normalized
    return normalized[:max_len] + "..."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input parquet path.")
    parser.add_argument(
        "--config",
        default="configs/pipelines/math_clean_pipeline_v2.yaml",
        help="Pipeline config used to load dataset_priority and round2 defaults.",
    )
    parser.add_argument("--sample-size", type=int, default=100_000, help="Active-record sample size.")
    parser.add_argument("--threshold", type=float, default=0.3, help="Low threshold for exploration run.")
    parser.add_argument("--num-perm", type=int, default=128, help="MinHash num_perm.")
    parser.add_argument("--bands", type=int, default=32, help="LSH bands.")
    parser.add_argument("--ngram-size", type=int, default=3, help="N-gram size.")
    parser.add_argument("--tokenizer", default="char", help="Tokenizer name.")
    parser.add_argument("--workers", type=int, default=32, help="Fuzzy dedup worker count.")
    parser.add_argument("--query-workers", type=int, default=8, help="Query worker count.")
    parser.add_argument("--verify-workers", type=int, default=32, help="Verify worker count.")
    parser.add_argument("--query-backend", default="array", help="LSH query backend.")
    parser.add_argument("--query-chunk-size", type=int, default=2048, help="Query chunk size.")
    parser.add_argument("--emit-chunk-size", type=int, default=32768, help="Emit chunk size.")
    parser.add_argument("--candidate-queue-maxsize", type=int, default=1024, help="Candidate queue size.")
    parser.add_argument("--max-candidates-per-record", type=int, default=100, help="Cap LSH hits per record.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--examples-per-bucket", type=int, default=15, help="Examples to show per bucket.")
    parser.add_argument(
        "--bucket-edges",
        default="0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.01",
        help="Comma-separated Jaccard bucket edges.",
    )
    parser.add_argument(
        "--threshold-grid",
        default="0.30,0.40,0.50,0.60,0.70,0.80",
        help="Thresholds for union-find sensitivity simulation.",
    )
    parser.add_argument("--max-question-chars", type=int, default=400, help="Truncate question text in report.")
    parser.add_argument("--output", required=True, help="Markdown output path.")
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one float value")
    return values


def bucket_label(value: float, edges: list[float]) -> str:
    for left, right in zip(edges[:-1], edges[1:]):
        if left <= value < right:
            return f"{left:.2f}-{right:.2f}"
    return f"{edges[-2]:.2f}-{edges[-1]:.2f}"


def find_round2_processor_config(config_path: str) -> dict:
    config = load_yaml(config_path)
    for processor in config.get("processors", []):
        if processor.get("step_name") == "fuzzy_dedup_round2":
            return dict(processor)
    raise ValueError(f"Could not find fuzzy_dedup_round2 in {config_path}")


def stratified_sample_indices(
    table: pa.Table,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[list[int], dict[str, int], int]:
    dataset_names = table.column("dataset_name").to_pylist()
    training_phases = table.column("training_phase").to_pylist()
    groups: dict[str, list[int]] = defaultdict(list)
    active_count = 0
    for index, (dataset_name, training_phase) in enumerate(zip(dataset_names, training_phases)):
        if str(training_phase or "") == "drop":
            continue
        groups[str(dataset_name or "")].append(index)
        active_count += 1

    if active_count == 0:
        return [], {}, 0
    if sample_size >= active_count:
        all_indices = [idx for indices in groups.values() for idx in indices]
        return all_indices, {dataset_name: len(indices) for dataset_name, indices in groups.items()}, active_count

    desired: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for dataset_name, indices in groups.items():
        raw = sample_size * (len(indices) / active_count)
        base = min(len(indices), int(raw))
        desired[dataset_name] = base
        remainders.append((raw - base, dataset_name))

    assigned = sum(desired.values())
    for _, dataset_name in sorted(remainders, reverse=True):
        if assigned >= sample_size:
            break
        if desired[dataset_name] >= len(groups[dataset_name]):
            continue
        desired[dataset_name] += 1
        assigned += 1

    sampled: list[int] = []
    for dataset_name, indices in groups.items():
        want = desired[dataset_name]
        if want <= 0:
            continue
        if want >= len(indices):
            sampled.extend(indices)
            continue
        chosen = rng.choice(np.asarray(indices, dtype=np.int64), size=want, replace=False)
        sampled.extend(int(value) for value in chosen.tolist())
    return sampled, desired, active_count


def materialize_records(table: pa.Table, row_indices: list[int]) -> list[CanonicalRecord]:
    if not row_indices:
        return []
    sorted_indices = sorted(row_indices)
    record_ids = table.column("record_id")
    questions = table.column("question")
    dataset_names = table.column("dataset_name")
    training_phases = table.column("training_phase")
    filter_tags = table.column("filter_tag") if "filter_tag" in table.column_names else None

    records: list[CanonicalRecord] = []
    for row_index in sorted_indices:
        records.append(
            CanonicalRecord(
                record_id=str(record_ids[row_index].as_py() or ""),
                question=str(questions[row_index].as_py() or ""),
                raw_dataset_answer="",
                confirmed_answer="",
                dataset_name=str(dataset_names[row_index].as_py() or ""),
                training_phase=str(training_phases[row_index].as_py() or ""),
                filter_tag=str(filter_tags[row_index].as_py() or "") if filter_tags is not None else "",
            )
        )
    return records


def normalize_pair_result(
    pair_result: VerifyResult | list,
    records: list[CanonicalRecord],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(pair_result, VerifyResult):
        return pair_result.accepted_left, pair_result.accepted_right, pair_result.accepted_similarity

    id_to_index = {record.record_id: index for index, record in enumerate(records)}
    left: list[int] = []
    right: list[int] = []
    similarity: list[float] = []
    for item in pair_result:
        left.append(id_to_index[item.left_id])
        right.append(id_to_index[item.right_id])
        similarity.append(float(item.similarity))
    return (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(similarity, dtype=np.float64),
    )


def sensitivity_rows(
    records: list[CanonicalRecord],
    accepted_left: np.ndarray,
    accepted_right: np.ndarray,
    accepted_similarity: np.ndarray,
    thresholds: list[float],
    tier_config: dict[int, list[str]] | None,
) -> list[tuple[float, int, float]]:
    priority_map = build_record_priority_map(records, tier_config) if tier_config is not None else None
    record_ids = [record.record_id for record in records]
    rows: list[tuple[float, int, float]] = []
    for threshold in thresholds:
        mask = accepted_similarity >= threshold
        indices = np.flatnonzero(mask)
        edges = [(record_ids[int(accepted_left[i])], record_ids[int(accepted_right[i])]) for i in indices]
        to_drop = _union_find_drop_ids(edges, priority_map=priority_map)
        rows.append((threshold, len(to_drop), len(to_drop) / len(records) if records else 0.0))
    return rows


def render_bar(count: int, total: int, width: int = 32) -> str:
    if total <= 0 or count <= 0:
        return ""
    filled = max(1, int(round(width * (count / total))))
    return "#" * filled


def write_markdown(
    output_path: Path,
    records: list[CanonicalRecord],
    sample_counts: dict[str, int],
    active_count: int,
    accepted_left: np.ndarray,
    accepted_right: np.ndarray,
    accepted_similarity: np.ndarray,
    bucket_edges: list[float],
    examples_per_bucket: int,
    threshold_rows: list[tuple[float, int, float]],
    max_question_chars: int,
    seed: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    bucket_to_indices: dict[str, list[int]] = defaultdict(list)
    for pair_index, similarity in enumerate(accepted_similarity.tolist()):
        bucket_to_indices[bucket_label(float(similarity), bucket_edges)].append(pair_index)

    lines: list[str] = []
    lines.append("# Dedup Jaccard Exploration")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- 输入 active 记录数: {active_count:,}")
    lines.append(f"- 分层采样记录数: {len(records):,}")
    lines.append(f"- 接受的候选对数: {accepted_similarity.shape[0]:,}")
    lines.append("")
    lines.append("## Sample Composition")
    lines.append("")
    lines.append("| Dataset | Sampled Rows |")
    lines.append("| --- | ---: |")
    for dataset_name, count in sorted(sample_counts.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"| `{dataset_name}` | {count:,} |")

    lines.append("")
    lines.append("## Jaccard Distribution")
    lines.append("")
    total_pairs = int(accepted_similarity.shape[0])
    for left, right in zip(bucket_edges[:-1], bucket_edges[1:]):
        label = f"{left:.2f}-{right:.2f}"
        count = len(bucket_to_indices.get(label, []))
        lines.append(f"- `J={label}`: {count:,} {render_bar(count, total_pairs)}")

    lines.append("")
    lines.append("## Threshold Sensitivity")
    lines.append("")
    lines.append("| Threshold | Dropped Records | Drop Rate |")
    lines.append("| ---: | ---: | ---: |")
    for threshold, dropped, rate in threshold_rows:
        lines.append(f"| {threshold:.2f} | {dropped:,} | {rate:.2%} |")

    for left, right in zip(bucket_edges[:-1], bucket_edges[1:]):
        label = f"{left:.2f}-{right:.2f}"
        pair_indices = bucket_to_indices.get(label, [])
        if not pair_indices:
            continue
        choose_count = min(examples_per_bucket, len(pair_indices))
        chosen = rng.choice(np.asarray(pair_indices, dtype=np.int64), size=choose_count, replace=False)
        lines.append("")
        lines.append(f"## Bucket J={label}")
        lines.append("")
        lines.append(f"- 候选对数: {len(pair_indices):,}")
        lines.append(f"- 展示样本数: {choose_count}")
        lines.append("")
        for example_index, pair_index in enumerate(chosen.tolist(), start=1):
            left_record = records[int(accepted_left[int(pair_index)])]
            right_record = records[int(accepted_right[int(pair_index)])]
            similarity = float(accepted_similarity[int(pair_index)])
            lines.append(f"### Pair {example_index}")
            lines.append("")
            lines.append(f"- Jaccard: {similarity:.4f}")
            lines.append(f"- A: `{left_record.dataset_name}` / `{left_record.record_id}` / len={len(left_record.question)}")
            lines.append(f"- B: `{right_record.dataset_name}` / `{right_record.record_id}` / len={len(right_record.question)}")
            lines.append("")
            lines.append("```text")
            lines.append("[A]")
            lines.append(truncate(left_record.question, max_len=max_question_chars))
            lines.append("")
            lines.append("[B]")
            lines.append(truncate(right_record.question, max_len=max_question_chars))
            lines.append("```")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    bucket_edges = parse_float_list(args.bucket_edges)
    threshold_grid = parse_float_list(args.threshold_grid)
    if args.workers < 2:
        raise ValueError("workers must be >= 2 so fuzzy dedup returns VerifyResult with pair indices")

    started_at = time.perf_counter()
    rng = np.random.default_rng(args.seed)

    print("Loading round2 processor config...")
    round2_config = find_round2_processor_config(args.config)
    tier_config = parse_tier_config(round2_config.get("dataset_priority"))

    print("Reading input table...")
    table = pq.read_table(
        args.input,
        columns=["record_id", "question", "dataset_name", "training_phase", "filter_tag"],
    )
    sampled_indices, sample_counts, active_count = stratified_sample_indices(table, args.sample_size, rng)
    print(f"  active rows: {active_count:,}")
    print(f"  sampled rows: {len(sampled_indices):,}")

    print("Materializing sampled CanonicalRecord rows...")
    records = materialize_records(table, sampled_indices)

    print("Running low-threshold fuzzy dedup...")
    pair_result = minhash_candidate_pairs(
        records,
        threshold=args.threshold,
        num_perm=args.num_perm,
        bands=args.bands,
        ngram_size=args.ngram_size,
        tokenizer=args.tokenizer,
        preprocess_text=True,
        workers=args.workers,
        max_candidates_per_record=args.max_candidates_per_record,
        query_workers=min(args.query_workers, args.workers),
        verify_workers=min(args.verify_workers, args.workers),
        query_backend=args.query_backend,
        query_chunk_size=args.query_chunk_size,
        emit_chunk_size=args.emit_chunk_size,
        candidate_queue_maxsize=args.candidate_queue_maxsize,
    )
    accepted_left, accepted_right, accepted_similarity = normalize_pair_result(pair_result, records)
    print(f"  accepted pairs: {accepted_similarity.shape[0]:,}")

    print("Simulating threshold sensitivity...")
    threshold_rows = sensitivity_rows(
        records,
        accepted_left,
        accepted_right,
        accepted_similarity,
        threshold_grid,
        tier_config=tier_config,
    )

    output_path = Path(args.output)
    write_markdown(
        output_path,
        records,
        sample_counts,
        active_count,
        accepted_left,
        accepted_right,
        accepted_similarity,
        bucket_edges,
        args.examples_per_bucket,
        threshold_rows,
        args.max_question_chars,
        args.seed,
    )
    print(f"Report written to {output_path}")
    print(f"Elapsed: {time.perf_counter() - started_at:.1f}s")


if __name__ == "__main__":
    main()
