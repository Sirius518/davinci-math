"""Decontamination sanity check: sample benchmark questions and show FAISS top-k matches.

Usage:
    PYTHONPATH=src python scripts/decontam_topk_preview.py [--seed 42] [--samples 3]
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

CHECKPOINT_DIR = Path("./artifacts/checkpoints/decontamination")
INPUT_PATH = Path("./data/processed/math_clean_llm_with_cascade_v2.parquet")
EMBEDDING_MODEL = "Qwen3-Embedding-4B"
TARGET_PHASES = {"posttrain", "midtrain"}
TOP_K_LEVELS = [5, 10, 20, 50]
MAX_DISPLAY_CHARS = 300


def wrap(text: str, width: int = MAX_DISPLAY_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) > width:
        return text[:width] + " ..."
    return text


def load_benchmarks() -> list[dict]:
    path = CHECKPOINT_DIR / "benchmark_questions.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_embeddings(prefix: str, num_texts: int) -> np.ndarray:
    safe = EMBEDDING_MODEL.replace("/", "_")
    npy_path = CHECKPOINT_DIR / f"{prefix}_{safe}.npy"
    emb = np.load(npy_path, allow_pickle=False).astype(np.float32)
    assert emb.shape[0] == num_texts, f"{prefix}: expected {num_texts}, got {emb.shape[0]}"
    return emb


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return arr / norms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    print("=" * 80)
    print("Loading benchmark questions ...")
    benchmarks = load_benchmarks()
    print(f"  benchmark questions: {len(benchmarks)}")

    print("Loading training records (question + record_id + training_phase + dataset_name) ...")
    table = pq.read_table(INPUT_PATH, columns=["record_id", "question", "training_phase", "dataset_name"])
    all_questions = table.column("question").to_pylist()
    all_record_ids = table.column("record_id").to_pylist()
    all_phases = table.column("training_phase").to_pylist()
    all_datasets = table.column("dataset_name").to_pylist()

    target_indices = [i for i, p in enumerate(all_phases) if p in TARGET_PHASES]
    train_questions = [all_questions[i] for i in target_indices]
    train_record_ids = [all_record_ids[i] for i in target_indices]
    train_datasets = [all_datasets[i] for i in target_indices]
    print(f"  total records: {len(all_questions)}")
    print(f"  target records (posttrain+midtrain): {len(target_indices)}")

    print("Loading embeddings ...")
    train_emb = load_embeddings("train", len(target_indices))
    bench_emb = load_embeddings("benchmark", len(benchmarks))
    train_emb = l2_normalize(train_emb)
    bench_emb = l2_normalize(bench_emb)
    print(f"  train embedding shape: {train_emb.shape}")
    print(f"  benchmark embedding shape: {bench_emb.shape}")

    print("Building FAISS index ...")
    import faiss
    dim = train_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(train_emb)
    print(f"  FAISS index size: {index.ntotal}")

    rng = np.random.default_rng(args.seed)
    sample_ids = rng.choice(len(benchmarks), size=min(args.samples, len(benchmarks)), replace=False)

    max_k = max(TOP_K_LEVELS)
    sample_embs = bench_emb[sample_ids]
    distances, neighbors = index.search(sample_embs, max_k)

    print("=" * 80)
    for rank, bench_idx in enumerate(sample_ids):
        bq = benchmarks[bench_idx]
        print(f"\n{'#' * 80}")
        print(f"BENCHMARK [{rank+1}/{len(sample_ids)}]  source={bq['dataset_name']}  uid={bq['uid']}")
        print(f"{'#' * 80}")
        print(f"Q: {wrap(bq['question'], 500)}")
        print(f"A: {wrap(bq.get('ground_truth', ''), 200)}")

        for k in TOP_K_LEVELS:
            print(f"\n  {'─' * 70}")
            print(f"  Top-{k} matches:")
            print(f"  {'─' * 70}")
            for j in range(k):
                train_pos = int(neighbors[rank, j])
                if train_pos < 0:
                    continue
                sim = float(distances[rank, j])
                q_text = wrap(train_questions[train_pos])
                rid = train_record_ids[train_pos]
                ds_name = train_datasets[train_pos]
                marker = ""
                if j < 5:
                    marker = " ★"
                print(f"    [{j+1:3d}] sim={sim:.4f}  dataset={ds_name:<30s}  rid={rid[:20]}{marker}")
                print(f"          {q_text}")
        print()

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
