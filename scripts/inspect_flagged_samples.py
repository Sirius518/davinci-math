#!/usr/bin/env python
"""Inspect flagged records: no pipeline imports, just pyarrow + json + re."""
from __future__ import annotations

import json
import re
import sys

import pyarrow.parquet as pq

INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "data/rollout_slices/slice_00_result.parquet"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 100

ANSWER_PATTERN = re.compile(r"(?im)^(?:final answer|answer)\s*[:：]\s*(.+)$")
BOXED_PATTERN = re.compile(r"\\boxed\s*\{")


def parse_answer(text: str) -> str:
    matches = list(ANSWER_PATTERN.finditer(text))
    if matches:
        return matches[-1].group(1).strip()
    return text.splitlines()[-1].strip() if text.strip() else ""


def extract_boxed(text: str) -> str:
    last_start = -1
    for m in BOXED_PATTERN.finditer(text):
        last_start = m.end()
    if last_start < 0:
        return ""
    depth, pos = 1, last_start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    return text[last_start:pos - 1].strip() if depth == 0 else ""


def is_junk(text: str) -> bool:
    return not text or bool(re.match(r"^[\s\\\[\]$(){}*#`]+$", text))


def main() -> None:
    print(f"Reading {INPUT_PATH} (first {LIMIT} rows) ...")
    table = pq.read_table(INPUT_PATH)
    total_rows = table.num_rows
    table = table.slice(0, min(LIMIT, total_rows))
    rows = table.to_pylist()
    print(f"File has {total_rows} rows, inspecting first {len(rows)}\n")

    flagged = []
    for row in rows:
        v = row.get("verification")
        if isinstance(v, str):
            v = json.loads(v)
        if not isinstance(v, dict):
            continue
        if v.get("majority_matches_gt") is False:
            flagged.append((row, v))

    print(f"Flagged (majority_matches_gt=False): {len(flagged)} / {len(rows)}\n")

    stats = {"total_flagged": len(flagged), "total_samples": 0,
             "junk_parse_answer": 0, "has_boxed": 0, "boxed_is_useful": 0,
             "parse_answer_useful": 0}

    for idx, (row, v) in enumerate(flagged):
        samples = v.get("samples", [])
        if isinstance(samples, str):
            samples = json.loads(samples)
        gt_answer = v.get("raw_dataset_answer", row.get("raw_dataset_answer", ""))
        majority_answer = v.get("majority_answer", "")
        majority_count = v.get("majority_count", 0)
        num_samples = v.get("num_samples", len(samples))
        question = row.get("question", "")

        print("=" * 80)
        print(f"[{idx}] record_id: {str(row.get('record_id', ''))[:24]}...")
        print(f"    question (120c): {question[:120]}...")
        print(f"    gt_answer:       {gt_answer}")
        print(f"    majority_answer: {majority_answer}")
        print(f"    majority_count:  {majority_count}/{num_samples}")

        for i, sample in enumerate(samples):
            solution = sample.get("solution", "") or ""
            reasoning = sample.get("reasoning", "") or ""
            pre_extracted = sample.get("answer", "")
            stop_reason = sample.get("stop_reason", "")

            re_extracted = parse_answer(solution)
            boxed = extract_boxed(solution)
            if not boxed and reasoning and reasoning != solution:
                boxed = extract_boxed(reasoning)

            stats["total_samples"] += 1
            if is_junk(re_extracted):
                stats["junk_parse_answer"] += 1
            else:
                stats["parse_answer_useful"] += 1
            if boxed:
                stats["has_boxed"] += 1
                if not is_junk(boxed):
                    stats["boxed_is_useful"] += 1

            tail_lines = solution.strip().splitlines()[-3:] if solution.strip() else []
            tail_str = " | ".join(l.strip() for l in tail_lines)[:200]

            print(f"    [{i}] stop={stop_reason:8s}  "
                  f"pre_extracted={pre_extracted!r:30s}  "
                  f"re_extracted={re_extracted!r:30s}  "
                  f"boxed={boxed!r:30s}")
            if is_junk(re_extracted) and not boxed:
                print(f"         tail: {tail_str}")

        print()

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print(f"  Flagged records:         {stats['total_flagged']}")
    print(f"  Total samples:           {stats['total_samples']}")
    print(f"  parse_answer useful:     {stats['parse_answer_useful']} "
          f"({100*stats['parse_answer_useful']/max(stats['total_samples'],1):.1f}%)")
    print(f"  parse_answer junk:       {stats['junk_parse_answer']} "
          f"({100*stats['junk_parse_answer']/max(stats['total_samples'],1):.1f}%)")
    print(f"  has \\boxed:              {stats['has_boxed']} "
          f"({100*stats['has_boxed']/max(stats['total_samples'],1):.1f}%)")
    print(f"  \\boxed useful:           {stats['boxed_is_useful']} "
          f"({100*stats['boxed_is_useful']/max(stats['total_samples'],1):.1f}%)")
    could_rescue = stats["boxed_is_useful"]
    total_junk = stats["junk_parse_answer"]
    print(f"  junk that boxed rescues: {could_rescue} / {total_junk}")


if __name__ == "__main__":
    main()
