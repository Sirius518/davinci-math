"""Fuzzy deduplication processor and public entry point.

Implementation details are split across internal modules:
  - _verify_kernel: Numba JIT Jaccard kernel and packed token storage
  - _minhash_engine: signature building, LSH indexing, query workers, streaming
"""
from __future__ import annotations

import gc as _gc
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Any, cast

from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, DatasetProcessor, DatasetProcessorResult, DedupPair
from pipeline.utils.procinfo import _log_snapshot, _log_stage2

from datasketch import MinHash  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]

from ._minhash_engine import (
    CandidateChunkMsg,
    QueryWorkerDoneMsg,
    VerifyResult,
    _build_minhash,
    _init_question_worker,
    _iter_candidate_pairs_via_lsh,
    _resolve_tokenizer,
    _stream_query_verify,
    _tokenize_minhash_worker,
    _tokenize_question,
)
from ._verify_kernel import (
    _pack_token_sequences,
    _prime_verify_kernel,
    _verify_candidate_arrays,
    _verify_candidate_chunk,
    _verify_candidate_chunk_numba,
)
import pipeline.ops.dedup._verify_kernel as _vk
import pipeline.ops.dedup._minhash_engine as _me


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def minhash_candidate_pairs(
    records: list[CanonicalRecord],
    *,
    threshold: float = 0.8,
    num_perm: int = 64,
    bands: int = 8,
    ngram_size: int = 3,
    tokenizer: str = "char",
    preprocess_text: bool = True,
    workers: int = 1,
    bucket_block_size: int = 512,
    max_candidates_per_record: int = 200,
    query_workers: int | None = None,
    verify_workers: int | None = None,
    query_backend: str = "datasketch",
    query_chunk_size: int = 2048,
    emit_chunk_size: int = 8192,
    candidate_queue_maxsize: int = 256,
) -> VerifyResult | list[DedupPair]:
    del bucket_block_size
    workers = max(1, min(int(workers), 256))
    query_workers = max(1, min(int(query_workers if query_workers is not None else workers), 256))
    verify_workers = max(1, min(int(verify_workers if verify_workers is not None else min(32, workers)), 256))
    query_chunk_size = max(int(query_chunk_size), 256)
    emit_chunk_size = max(int(emit_chunk_size), 256)
    candidate_queue_maxsize = max(int(candidate_queue_maxsize), 4)
    started_at = time.perf_counter()
    resolved_tokenizer = _resolve_tokenizer(tokenizer)
    questions = [record.question for record in records]
    _log_stage2(
        f"minhash_start records={len(records)} workers={workers} "
        f"query_workers={query_workers} verify_workers={verify_workers} "
        f"ngram={ngram_size} num_perm={num_perm} bands={bands} "
        f"tokenizer={resolved_tokenizer} preprocess={preprocess_text} "
        f"signature_backend=datasketch query_backend={query_backend}"
    )
    _log_snapshot("minhash_start", records=len(records), workers=workers)
    build_started_at = time.perf_counter()
    _log_stage2("pre_alloc_start")
    tokenized_questions: list[tuple[int, ...] | None] = [None] * len(records)
    minhashes: list[MinHash | None] = [None] * len(records)
    _log_stage2(f"pre_alloc_done elapsed_sec={time.perf_counter() - build_started_at:.1f}")
    if workers > 1:
        _log_stage2(f"signature_pool_start workers={workers}")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("fork"),
            initializer=_init_question_worker,
            initargs=(questions, resolved_tokenizer, preprocess_text),
        ) as executor:
            tasks = ((index, ngram_size, num_perm) for index in range(len(records)))
            completed_sigs = 0
            for index, tokens, minhash in executor.map(_tokenize_minhash_worker, tasks, chunksize=128):
                tokenized_questions[index] = tokens
                minhashes[index] = minhash
                completed_sigs += 1
                if completed_sigs % 500_000 == 0 or completed_sigs == len(records):
                    _log_stage2(
                        f"signature_progress completed={completed_sigs}/{len(records)} "
                        f"elapsed_sec={time.perf_counter() - build_started_at:.1f}"
                    )
    else:
        for index, question in enumerate(questions):
            tokens = _tokenize_question(
                question,
                ngram_size=ngram_size,
                tokenizer=resolved_tokenizer,
                preprocess_text=preprocess_text,
            )
            tokenized_questions[index] = tokens
            minhashes[index] = _build_minhash(tokens, num_perm=num_perm)

    unresolved_minhashes = sum(1 for item in minhashes if item is None)
    unresolved_tokens = sum(1 for item in tokenized_questions if item is None)
    if unresolved_minhashes or unresolved_tokens:
        raise RuntimeError(
            "signature stage produced unresolved entries: "
            f"minhashes_none={unresolved_minhashes} tokenized_none={unresolved_tokens}"
        )
    resolved_minhashes = cast(list[MinHash], minhashes)
    resolved_tokens = cast(list[tuple[int, ...]], tokenized_questions)

    _log_stage2(f"signature_done elapsed_sec={time.perf_counter() - build_started_at:.1f}")
    _log_snapshot("signature_done", tokenized=len(resolved_tokens), minhashes=len(resolved_minhashes))
    questions = []

    record_ids = [r.record_id for r in records]
    token_sizes = [len(t) for t in resolved_tokens]
    _log_snapshot("pre_query_prepared")

    # ---- Streaming path (workers > 1): query workers + verify overlap ----
    if workers > 1:
        verify_result = _stream_query_verify(
            resolved_minhashes,
            token_sizes,
            resolved_tokens,
            record_ids,
            threshold=threshold,
            num_perm=num_perm,
            bands=bands,
            query_workers=query_workers,
            verify_workers=verify_workers,
            query_backend=query_backend,
            query_chunk_size=query_chunk_size,
            max_candidates_per_record=max_candidates_per_record,
            emit_chunk_size=emit_chunk_size,
            candidate_queue_maxsize=candidate_queue_maxsize,
        )
        _log_stage2(f"minhash_done total_elapsed_sec={time.perf_counter() - started_at:.1f}")
        return verify_result

    # ---- Legacy non-streaming fallback (workers == 1) ----
    candidate_started_at = time.perf_counter()
    candidate_count = 0
    candidate_chunks = _iter_candidate_pairs_via_lsh(
        resolved_minhashes,
        token_sizes,
        threshold=threshold,
        num_perm=num_perm,
        bands=bands,
        workers=1,
        query_backend=query_backend,
        query_chunk_size=query_chunk_size,
        max_candidates_per_record=max_candidates_per_record,
    )

    _log_stage2(f"post_query_release releasing minhashes ({len(resolved_minhashes)}) and LSH globals")
    del resolved_minhashes
    minhashes = []  # type: ignore[assignment]
    _me._release_query_globals(query_backend)
    del token_sizes
    _gc.collect()
    _log_snapshot("post_query_released")

    token_buffer, token_offsets, token_lengths = _pack_token_sequences(resolved_tokens)
    total_token_count = int(token_lengths.sum())
    resolved_tokens = []
    _gc.collect()
    _log_snapshot(
        "verify_tokens_packed",
        records=len(record_ids),
        token_buffer_count=total_token_count,
        token_buffer_gb=f"{token_buffer.nbytes / (1024 ** 3):.2f}",
    )

    nonempty_candidate_chunks = sum(1 for chunk in candidate_chunks if chunk)
    candidate_pairs_total = sum(len(chunk) for chunk in candidate_chunks)
    max_candidate_pairs_in_chunk = max((len(chunk) for chunk in candidate_chunks), default=0)
    avg_candidate_pairs_per_chunk = (
        candidate_pairs_total / len(candidate_chunks) if candidate_chunks else 0.0
    )
    _log_stage2(
        "verify_baseline "
        f"records_count={len(records)} record_ids_count={len(record_ids)} "
        f"resolved_tokens_count={len(token_lengths)} "
        f"candidate_chunks_count={len(candidate_chunks)} nonempty_candidate_chunks={nonempty_candidate_chunks} "
        f"candidate_pairs_total={candidate_pairs_total} "
        f"avg_candidate_pairs_per_chunk={avg_candidate_pairs_per_chunk:.1f} "
        f"max_candidate_pairs_in_chunk={max_candidate_pairs_in_chunk}"
    )

    pairs: list[DedupPair] = []
    _vk._WORKER_TOKEN_BUFFER = token_buffer
    _vk._WORKER_TOKEN_OFFSETS = token_offsets
    _vk._WORKER_TOKEN_LENGTHS = token_lengths
    _prime_verify_kernel()
    verify_started_at = time.perf_counter()
    verify_chunk_counter = 0
    for chunk in candidate_chunks:
        verify_chunk_counter += 1
        candidate_count += len(chunk)
        for left_index, right_index, similarity in _verify_candidate_chunk((chunk, threshold)):
            pairs.append(
                DedupPair(
                    left_id=record_ids[left_index],
                    right_id=record_ids[right_index],
                    left_question="",
                    right_question="",
                    method="minhash",
                    similarity=similarity,
                )
            )
        if verify_chunk_counter % 100 == 0:
            accepted_pairs = len(pairs)
            elapsed_sec = time.perf_counter() - verify_started_at
            _log_stage2(
                f"candidate_verify_progress chunks={verify_chunk_counter} "
                f"candidates={candidate_count} accepted_pairs={accepted_pairs} "
                f"elapsed_sec={elapsed_sec:.1f}"
            )
            _log_snapshot(
                "candidate_verify_progress",
                workers=1,
                chunks=verify_chunk_counter,
                candidates=candidate_count,
                accepted_pairs=accepted_pairs,
                elapsed_sec=f"{elapsed_sec:.1f}",
            )
    _log_stage2(
        f"candidate_build_done method=lsh candidates={candidate_count} "
        f"accepted_pairs={len(pairs)} "
        f"elapsed_sec={time.perf_counter() - candidate_started_at:.1f}"
    )
    _log_snapshot(
        "candidate_verify_done",
        workers=1,
        chunks=verify_chunk_counter,
        candidates=candidate_count,
        accepted_pairs=len(pairs),
        elapsed_sec=f"{time.perf_counter() - verify_started_at:.1f}",
    )
    _vk._WORKER_TOKEN_BUFFER = None
    _vk._WORKER_TOKEN_OFFSETS = None
    _vk._WORKER_TOKEN_LENGTHS = None
    _log_stage2(f"minhash_done total_elapsed_sec={time.perf_counter() - started_at:.1f}")
    return pairs


# ---------------------------------------------------------------------------
# Union-Find for dedup clustering
# ---------------------------------------------------------------------------

def _union_find_drop_ids(
    edges: list[tuple[str, str]],
    *,
    priority_map: dict[str, int] | None = None,
) -> set[str]:
    _prio = priority_map or {}
    id_to_index: dict[str, int] = {}
    index_to_id: list[str] = []
    for left, right in edges:
        if left not in id_to_index:
            id_to_index[left] = len(index_to_id)
            index_to_id.append(left)
        if right not in id_to_index:
            id_to_index[right] = len(index_to_id)
            index_to_id.append(right)
    parent = list(range(len(index_to_id)))
    rank = [0] * len(index_to_id)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if rank[root_left] < rank[root_right]:
            parent[root_left] = root_right
        elif rank[root_left] > rank[root_right]:
            parent[root_right] = root_left
        else:
            parent[root_right] = root_left
            rank[root_left] += 1

    for left, right in edges:
        union(id_to_index[left], id_to_index[right])

    root_best_id: dict[int, str] = {}
    for index, record_id in enumerate(index_to_id):
        root = find(index)
        current = root_best_id.get(root)
        if current is None or (_prio.get(record_id, 0), record_id) < (_prio.get(current, 0), current):
            root_best_id[root] = record_id
    to_drop: set[str] = set()
    for index, record_id in enumerate(index_to_id):
        root = find(index)
        if record_id != root_best_id[root]:
            to_drop.add(record_id)
    return to_drop


# ---------------------------------------------------------------------------
# Registered processor
# ---------------------------------------------------------------------------

@register_processor("fuzzy_dedup")
class FuzzyDedupProcessor(DatasetProcessor):
    name = "fuzzy_dedup"

    def process(self, records: list[CanonicalRecord], *, pipeline_artifacts: dict[str, Any] | None = None) -> DatasetProcessorResult:
        from pipeline.utils.priority import build_record_priority_map, parse_tier_config

        skip_phases = {
            phase.strip()
            for phase in str(self.config.get("skip_training_phases", "drop")).split(",")
            if phase.strip()
        }
        active = [r for r in records if r.training_phase not in skip_phases]
        passthrough = [r for r in records if r.training_phase in skip_phases]
        if not active:
            all_records = sorted(records, key=lambda item: item.record_id)
            return DatasetProcessorResult(kept_records=all_records, artifacts={"candidate_pairs": []})

        tier_config = parse_tier_config(self.config.get("dataset_priority"))
        priority_map: dict[str, int] | None = None
        if tier_config is not None:
            priority_map = build_record_priority_map(active, tier_config)
            _log_stage2(f"dataset_priority enabled tiers={sorted(tier_config.keys())}")

        threshold = float(self.config.get("threshold", 0.8))
        num_perm = int(self.config.get("num_perm", 64))
        bands = int(self.config.get("bands", 8))
        ngram_size = int(self.config.get("ngram_size", 3))
        tokenizer = str(self.config.get("tokenizer", "char"))
        preprocess_text = bool(self.config.get("preprocess_text", True))
        workers = max(int(self.config.get("workers", 1)), 1)
        max_candidates = max(int(self.config.get("max_candidates_per_record", 0)), 0)
        query_workers = max(int(self.config.get("query_workers", workers)), 1)
        verify_workers = max(int(self.config.get("verify_workers", min(32, workers))), 1)
        query_backend = str(self.config.get("query_backend", "datasketch"))
        query_chunk_size = max(int(self.config.get("query_chunk_size", 2048)), 256)
        emit_chunk_size = max(int(self.config.get("emit_chunk_size", 8192)), 256)
        candidate_queue_maxsize = max(int(self.config.get("candidate_queue_maxsize", 256)), 4)
        pair_result = minhash_candidate_pairs(
            active,
            threshold=threshold,
            num_perm=num_perm,
            bands=bands,
            ngram_size=ngram_size,
            tokenizer=tokenizer,
            preprocess_text=preprocess_text,
            workers=workers,
            max_candidates_per_record=max_candidates,
            query_workers=query_workers,
            verify_workers=verify_workers,
            query_backend=query_backend,
            query_chunk_size=query_chunk_size,
            emit_chunk_size=emit_chunk_size,
            candidate_queue_maxsize=candidate_queue_maxsize,
        )
        if isinstance(pair_result, VerifyResult):
            edges = pair_result.to_edges()
            artifact_pairs: VerifyResult | list[DedupPair] = pair_result
        else:
            edges = [(item.left_id, item.right_id) for item in pair_result]
            artifact_pairs = pair_result
        if not edges:
            all_records = sorted(active + passthrough, key=lambda item: item.record_id)
            return DatasetProcessorResult(kept_records=all_records, artifacts={"candidate_pairs": artifact_pairs})
        to_drop = _union_find_drop_ids(edges, priority_map=priority_map)
        deduped = [r for r in active if r.record_id not in to_drop]
        all_records = sorted(deduped + passthrough, key=lambda item: item.record_id)
        return DatasetProcessorResult(kept_records=all_records, artifacts={"candidate_pairs": artifact_pairs})
