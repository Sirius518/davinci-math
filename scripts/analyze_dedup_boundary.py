#!/usr/bin/env python3
"""Inspect fuzzy dedup boundary cases between round 1 and round 2."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq


@dataclass(frozen=True)
class RecordRow:
    record_id: str
    question: str
    dataset_name: str
    training_phase: str


@dataclass(frozen=True)
class MatchRow:
    removed: RecordRow
    kept: RecordRow | None
    similarity: float


def char_ngrams(text: str, n: int = 3) -> set[str]:
    text = text.strip().lower()
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def word_shingles(text: str, n: int = 2) -> set[str]:
    words = text.strip().lower().split()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(set_a: set[str], set_b: set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def truncate(text: str, max_len: int = 400) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_len:
        return normalized
    return normalized[:max_len] + "..."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round1", required=True, help="Round 1 parquet path.")
    parser.add_argument("--round2", required=True, help="Round 2 parquet path.")
    parser.add_argument("--num-examples", type=int, default=200, help="Reservoir sample size from removed records.")
    parser.add_argument("--examples-per-bucket", type=int, default=10, help="Max examples to print per Jaccard bucket.")
    parser.add_argument("--top-k", type=int, default=50, help="Top inverted-index candidates to re-rank.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--bucket-edges",
        default="0.00,0.30,0.40,0.50,0.60,0.70,0.80,1.01",
        help="Comma-separated bucket boundaries.",
    )
    parser.add_argument("--max-question-chars", type=int, default=400, help="Truncate question text in report.")
    parser.add_argument("--output", required=True, help="Markdown output path.")
    return parser.parse_args()


def parse_bucket_edges(raw: str) -> list[float]:
    edges = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(edges) < 2:
        raise ValueError("bucket edges must contain at least two values")
    if any(edges[index] >= edges[index + 1] for index in range(len(edges) - 1)):
        raise ValueError("bucket edges must be strictly increasing")
    return edges


def bucket_label(value: float, edges: list[float]) -> str:
    for left, right in zip(edges[:-1], edges[1:]):
        if left <= value < right:
            return f"{left:.2f}-{right:.2f}"
    return f"{edges[-2]:.2f}-{edges[-1]:.2f}"


def reservoir_sample_removed(
    round1_path: str,
    kept_ids: set[str],
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[int, list[RecordRow]]:
    table = pq.read_table(round1_path, columns=["record_id", "question", "dataset_name", "training_phase"])
    record_ids = table.column("record_id").to_pylist()
    questions = table.column("question").to_pylist()
    dataset_names = table.column("dataset_name").to_pylist()
    training_phases = table.column("training_phase").to_pylist()

    removed_count = 0
    sample: list[RecordRow] = []
    for index, record_id in enumerate(record_ids):
        if record_id in kept_ids:
            continue
        removed_count += 1
        row = RecordRow(
            record_id=str(record_id or ""),
            question=str(questions[index] or ""),
            dataset_name=str(dataset_names[index] or ""),
            training_phase=str(training_phases[index] or ""),
        )
        if len(sample) < sample_size:
            sample.append(row)
            continue
        replace_at = int(rng.integers(0, removed_count))
        if replace_at < sample_size:
            sample[replace_at] = row
    return removed_count, sample


def build_query_shingle_set(rows: list[RecordRow]) -> set[str]:
    shingle_set: set[str] = set()
    for row in rows:
        shingle_set.update(word_shingles(row.question))
    return shingle_set


def load_kept_records(round2_path: str) -> tuple[list[RecordRow], dict[str, int]]:
    table = pq.read_table(round2_path, columns=["record_id", "question", "dataset_name", "training_phase"])
    record_ids = table.column("record_id").to_pylist()
    questions = table.column("question").to_pylist()
    dataset_names = table.column("dataset_name").to_pylist()
    training_phases = table.column("training_phase").to_pylist()
    rows = [
        RecordRow(
            record_id=str(record_ids[index] or ""),
            question=str(questions[index] or ""),
            dataset_name=str(dataset_names[index] or ""),
            training_phase=str(training_phases[index] or ""),
        )
        for index in range(len(record_ids))
    ]
    id_to_index = {row.record_id: index for index, row in enumerate(rows)}
    return rows, id_to_index


def build_partial_inverted_index(records: list[RecordRow], query_shingles: set[str]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = defaultdict(list)
    for row_index, row in enumerate(records):
        for shingle in word_shingles(row.question):
            if shingle in query_shingles:
                index[shingle].append(row_index)
    return index


def find_best_match(
    query_text: str,
    kept_records: list[RecordRow],
    inv_index: dict[str, list[int]],
    top_k: int,
) -> tuple[int, float]:
    q_shingles = word_shingles(query_text)
    candidate_counts: dict[int, int] = defaultdict(int)
    for shingle in q_shingles:
        for idx in inv_index.get(shingle, []):
            candidate_counts[idx] += 1
    if not candidate_counts:
        return -1, 0.0

    top_candidates = sorted(candidate_counts, key=candidate_counts.get, reverse=True)[:top_k]
    query_ngrams = char_ngrams(query_text)
    best_idx = -1
    best_sim = 0.0
    for idx in top_candidates:
        similarity = jaccard(query_ngrams, char_ngrams(kept_records[idx].question))
        if similarity > best_sim:
            best_idx = idx
            best_sim = similarity
    return best_idx, best_sim


def render_bar(count: int, total: int, width: int = 32) -> str:
    if total <= 0 or count <= 0:
        return ""
    filled = max(1, int(round(width * (count / total))))
    return "#" * filled


def write_markdown(
    path: Path,
    matches: list[MatchRow],
    total_removed: int,
    total_kept: int,
    bucket_edges: list[float],
    examples_per_bucket: int,
    max_question_chars: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matched = [match for match in matches if match.kept is not None]
    bucket_to_matches: dict[str, list[MatchRow]] = defaultdict(list)
    for match in matched:
        bucket_to_matches[bucket_label(match.similarity, bucket_edges)].append(match)

    dataset_counts = Counter(match.kept.dataset_name for match in matched if match.kept is not None)
    avg_sim_by_dataset: dict[str, float] = {}
    for dataset_name in dataset_counts:
        sims = [match.similarity for match in matched if match.kept and match.kept.dataset_name == dataset_name]
        avg_sim_by_dataset[dataset_name] = sum(sims) / len(sims)

    lines: list[str] = []
    lines.append("# Fuzzy Dedup Boundary Analysis")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Round 2 存活记录数: {total_kept:,}")
    lines.append(f"- Round 1 -> Round 2 被移除记录数: {total_removed:,}")
    lines.append(f"- 人工审查样本数: {len(matches):,}")
    lines.append(f"- 找到匹配的样本数: {len(matched):,}")
    lines.append("")
    lines.append("## Jaccard Bucket Distribution")
    lines.append("")
    for label in [f"{left:.2f}-{right:.2f}" for left, right in zip(bucket_edges[:-1], bucket_edges[1:])]:
        bucket_matches = bucket_to_matches.get(label, [])
        bar = render_bar(len(bucket_matches), len(matched))
        lines.append(f"- `J={label}`: {len(bucket_matches):,} {bar}")
    lines.append("")
    lines.append("## Kept Match Dataset Breakdown")
    lines.append("")
    lines.append("| Dataset | Match Count | Avg Jaccard |")
    lines.append("| --- | ---: | ---: |")
    for dataset_name, count in dataset_counts.most_common():
        lines.append(f"| `{dataset_name}` | {count:,} | {avg_sim_by_dataset[dataset_name]:.3f} |")

    for label in [f"{left:.2f}-{right:.2f}" for left, right in zip(bucket_edges[:-1], bucket_edges[1:])]:
        bucket_matches = bucket_to_matches.get(label, [])
        if not bucket_matches:
            continue
        avg_removed_len = sum(len(match.removed.question) for match in bucket_matches) / len(bucket_matches)
        lines.append("")
        lines.append(f"## Bucket J={label}")
        lines.append("")
        lines.append(f"- 样本数: {len(bucket_matches):,}")
        lines.append(f"- 平均被移除 question 长度: {avg_removed_len:.1f}")
        lines.append("")
        for index, match in enumerate(bucket_matches[:examples_per_bucket], start=1):
            kept = match.kept
            if kept is None:
                continue
            lines.append(f"### Example {index}")
            lines.append("")
            lines.append(f"- Jaccard: {match.similarity:.4f}")
            lines.append(f"- Removed: `{match.removed.dataset_name}` / `{match.removed.record_id}` / `{match.removed.training_phase}`")
            lines.append(f"- Kept: `{kept.dataset_name}` / `{kept.record_id}` / `{kept.training_phase}`")
            lines.append("")
            lines.append("```text")
            lines.append("[REMOVED]")
            lines.append(truncate(match.removed.question, max_len=max_question_chars))
            lines.append("")
            lines.append("[KEPT]")
            lines.append(truncate(kept.question, max_len=max_question_chars))
            lines.append("```")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    bucket_edges = parse_bucket_edges(args.bucket_edges)
    started_at = time.perf_counter()
    rng = np.random.default_rng(args.seed)

    print("Loading Round 2 kept records...")
    kept_records, kept_id_to_index = load_kept_records(args.round2)
    kept_ids = set(kept_id_to_index)
    print(f"  kept records: {len(kept_records):,}")

    print("Sampling removed records from Round 1...")
    removed_count, removed_sample = reservoir_sample_removed(args.round1, kept_ids, args.num_examples, rng)
    print(f"  removed records: {removed_count:,}")
    print(f"  sampled removed records: {len(removed_sample):,}")

    print("Building partial inverted index for sampled queries...")
    query_shingles = build_query_shingle_set(removed_sample)
    inverted_index = build_partial_inverted_index(kept_records, query_shingles)
    print(f"  query shingles: {len(query_shingles):,}")
    print(f"  indexed shingles: {len(inverted_index):,}")

    print("Finding best matches...")
    matches: list[MatchRow] = []
    for removed in removed_sample:
        kept_index, similarity = find_best_match(removed.question, kept_records, inverted_index, args.top_k)
        kept = kept_records[kept_index] if kept_index >= 0 else None
        matches.append(MatchRow(removed=removed, kept=kept, similarity=similarity))

    output_path = Path(args.output)
    write_markdown(
        output_path,
        matches,
        total_removed=removed_count,
        total_kept=len(kept_records),
        bucket_edges=bucket_edges,
        examples_per_bucket=args.examples_per_bucket,
        max_question_chars=args.max_question_chars,
    )
    print(f"Report written to {output_path}")
    print(f"Elapsed: {time.perf_counter() - started_at:.1f}s")


if __name__ == "__main__":
    main()
