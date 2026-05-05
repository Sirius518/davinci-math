#!/usr/bin/env python3
"""Create a small test Parquet file with 10 math questions for rollout testing."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.core.schema import CANONICAL_RECORD_SCHEMA, CanonicalRecord


QUESTIONS = [
    {
        "record_id": "test-001",
        "question": "Find all real solutions to $x^2 - 5x + 6 = 0$.",
        "raw_dataset_answer": "2, 3",
    },
    {
        "record_id": "test-002",
        "question": "What is the value of $\\sqrt{144}$?",
        "raw_dataset_answer": "12",
    },
    {
        "record_id": "test-003",
        "question": "Compute $\\frac{3}{4} + \\frac{1}{6}$.",
        "raw_dataset_answer": "\\frac{11}{12}",
    },
    {
        "record_id": "test-004",
        "question": "What is the derivative of $f(x) = x^3 - 2x + 1$?",
        "raw_dataset_answer": "3x^2 - 2",
    },
    {
        "record_id": "test-005",
        "question": "Compute $\\binom{10}{3}$.",
        "raw_dataset_answer": "120",
    },
    {
        "record_id": "test-006",
        "question": "What is the area of a circle with radius 5?",
        "raw_dataset_answer": "25\\pi",
    },
    {
        "record_id": "test-007",
        "question": "Simplify $\\log_2 32$.",
        "raw_dataset_answer": "5",
    },
    {
        "record_id": "test-008",
        "question": "What is the sum of the first 100 positive integers?",
        "raw_dataset_answer": "5050",
    },
    {
        "record_id": "test-009",
        "question": "Solve for $x$: $2^x = 64$.",
        "raw_dataset_answer": "6",
    },
    {
        "record_id": "test-010",
        "question": "What is the GCD of 48 and 36?",
        "raw_dataset_answer": "12",
    },
]


def main() -> None:
    records = []
    for q in QUESTIONS:
        rec = CanonicalRecord(
            record_id=q["record_id"],
            question=q["question"],
            raw_dataset_answer=q["raw_dataset_answer"],
            confirmed_answer=q["raw_dataset_answer"],
            dataset_name="rollout_test",
            training_phase="posttrain",
        )
        records.append(rec.to_storage_dict())

    table = pa.Table.from_pylist(records, schema=CANONICAL_RECORD_SCHEMA)
    out_path = ROOT / "data" / "test" / "rollout_test_10.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(out_path))
    print(f"Wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
