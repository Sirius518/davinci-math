"""MinHash signature building, LSH indexing, query workers, and streaming
query-verify orchestration for fuzzy deduplication.

This is an internal module consumed by fuzzy.py — not part of the public API.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass
import hashlib
from multiprocessing import Process, get_context
import os
import time
import traceback
from typing import Any, cast

from pipeline.core.schema import DedupPair
from pipeline.utils.procinfo import _log_snapshot, _log_stage2, _rss_bytes
from pipeline.utils.text import ngrams, preprocess_for_minhash

import numpy as np  # type: ignore[reportMissingImports]
from datasketch import MinHash, MinHashLSH  # type: ignore[reportMissingImports]
import tiktoken  # type: ignore[reportMissingImports]

from ._band_index import (
    _clear_array_index_globals,
    _init_array_index_globals,
    _query_array_range,
    _query_stream_worker_array,
    build_array_band_index,
)
from ._verify_kernel import (
    _WORKER_TOKEN_BUFFER,
    _WORKER_TOKEN_LENGTHS,
    _WORKER_TOKEN_OFFSETS,
    _pack_token_sequences,
    _prime_verify_kernel,
    _verify_candidate_arrays_raw,
    _verify_candidate_arrays,
    _verify_candidate_chunk,
)
import pipeline.ops.dedup._verify_kernel as _vk

# ---------------------------------------------------------------------------
# Streaming message types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CandidateChunkMsg:
    left_indices: np.ndarray
    right_indices: np.ndarray
    pair_count: int

@dataclass(slots=True)
class QueryWorkerDoneMsg:
    worker_id: int
    error: str = ""
    capped: int = 0
    total_candidates: int = 0


@dataclass(slots=True)
class VerifyResult:
    accepted_left: np.ndarray
    accepted_right: np.ndarray
    accepted_similarity: np.ndarray
    record_ids: list[str]

    def to_edges(self) -> list[tuple[str, str]]:
        return [
            (
                self.record_ids[int(self.accepted_left[index])],
                self.record_ids[int(self.accepted_right[index])],
            )
            for index in range(self.accepted_left.shape[0])
        ]

    def to_dedup_pairs(self) -> list[DedupPair]:
        return [
            DedupPair(
                left_id=self.record_ids[int(self.accepted_left[index])],
                right_id=self.record_ids[int(self.accepted_right[index])],
                left_question="",
                right_question="",
                method="minhash",
                similarity=float(self.accepted_similarity[index]),
            )
            for index in range(self.accepted_left.shape[0])
        ]

# ---------------------------------------------------------------------------
# Module-level query/signature state
# ---------------------------------------------------------------------------

_WORKER_QUESTIONS: list[str] = []
_WORKER_MINHASHES: list[MinHash] = []
_WORKER_TOKEN_SIZES: np.ndarray = np.empty(0, dtype=np.int32)
_WORKER_LSH: MinHashLSH | None = None
_WORKER_TOKENIZER = "char"
_WORKER_PREPROCESS = True
_TOKENIZER_CACHE: dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def _hashed_ngrams(text: str, *, ngram_size: int) -> tuple[int, ...]:
    values: set[int] = set()
    for shingle in ngrams(text, ngram_size):
        digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
        values.add(int.from_bytes(digest, "big", signed=False))
    return tuple(sorted(values))


def _encoding_name(tokenizer: str) -> str:
    name = tokenizer.strip().lower()
    if name in {"cl100k", "cl100k_base"}:
        return "cl100k_base"
    if name in {"p50k", "p50k_base"}:
        return "p50k_base"
    raise ValueError(f"Unsupported tokenizer: {tokenizer}")


def _resolve_tokenizer(tokenizer: str) -> str:
    lowered = tokenizer.strip().lower()
    if lowered in {"char", "character"}:
        return "char"
    _encoding_name(lowered)
    return lowered


def _get_tiktoken_encoding(tokenizer: str) -> Any:
    encoding_name = _encoding_name(tokenizer)
    cached = _TOKENIZER_CACHE.get(encoding_name)
    if cached is not None:
        return cached
    encoding = tiktoken.get_encoding(encoding_name)
    _TOKENIZER_CACHE[encoding_name] = encoding
    return encoding


def _hashed_bpe_ngrams(text: str, *, ngram_size: int, tokenizer: str) -> tuple[int, ...]:
    encoding = _get_tiktoken_encoding(tokenizer)
    token_ids = encoding.encode(text)
    values: set[int] = set()
    if len(token_ids) < ngram_size:
        for token_id in token_ids:
            digest = hashlib.blake2b(token_id.to_bytes(4, "big", signed=False), digest_size=8).digest()
            values.add(int.from_bytes(digest, "big", signed=False))
        return tuple(sorted(values))
    for index in range(len(token_ids) - ngram_size + 1):
        hasher = hashlib.blake2b(digest_size=8)
        for token_id in token_ids[index : index + ngram_size]:
            hasher.update(int(token_id).to_bytes(4, "big", signed=False))
        values.add(int.from_bytes(hasher.digest(), "big", signed=False))
    return tuple(sorted(values))


def _tokenize_question(
    question: str,
    *,
    ngram_size: int,
    tokenizer: str,
    preprocess_text: bool,
) -> tuple[int, ...]:
    text = preprocess_for_minhash(question) if preprocess_text else question
    if tokenizer == "char":
        return _hashed_ngrams(text, ngram_size=ngram_size)
    return _hashed_bpe_ngrams(text, ngram_size=ngram_size, tokenizer=tokenizer)

# ---------------------------------------------------------------------------
# MinHash signature building
# ---------------------------------------------------------------------------

def _build_minhash(tokens: tuple[int, ...], *, num_perm: int) -> MinHash:
    minhash = MinHash(num_perm=num_perm)
    token_bytes = [value.to_bytes(8, "big", signed=False) for value in tokens]
    if hasattr(minhash, "update_batch"):
        minhash.update_batch(token_bytes)
    else:  # pragma: no cover
        for value in token_bytes:
            minhash.update(value)
    return minhash


def _init_question_worker(questions: list[str], tokenizer: str, preprocess_text: bool) -> None:
    global _WORKER_QUESTIONS, _WORKER_TOKENIZER, _WORKER_PREPROCESS
    _WORKER_QUESTIONS = questions
    _WORKER_TOKENIZER = tokenizer
    _WORKER_PREPROCESS = preprocess_text


def _tokenize_minhash_worker(args: tuple[int, int, int]) -> tuple[int, tuple[int, ...], MinHash]:
    index, ngram_size, minhash_num_perm = args
    tokens = _tokenize_question(
        _WORKER_QUESTIONS[index],
        ngram_size=ngram_size,
        tokenizer=_WORKER_TOKENIZER,
        preprocess_text=_WORKER_PREPROCESS,
    )
    minhash = _build_minhash(tokens, num_perm=minhash_num_perm)
    return index, tokens, minhash

# ---------------------------------------------------------------------------
# LSH query
# ---------------------------------------------------------------------------

def _token_ratio_within_threshold(left_count: int, right_count: int, threshold: float) -> bool:
    if threshold <= 0:
        return True
    if left_count == 0 or right_count == 0:
        return True
    ratio = (left_count / right_count) if left_count >= right_count else (right_count / left_count)
    return ratio <= (1.0 / threshold)


def _query_lsh_range(
    start: int, end: int, threshold: float, max_candidates_per_record: int,
) -> tuple[list[tuple[int, int]], int]:
    """Query LSH for a range of records. Returns (candidate_pairs, capped_count)."""
    if _WORKER_LSH is None:
        raise RuntimeError("LSH index is not initialized for query worker")
    rows: list[tuple[int, int]] = []
    capped = 0
    for left_index in range(start, end):
        minhash = _WORKER_MINHASHES[left_index]
        emitted = 0
        left_size = _WORKER_TOKEN_SIZES[left_index]
        for raw in _WORKER_LSH.query(minhash):
            right_index = int(raw)
            if right_index <= left_index:
                continue
            if not _token_ratio_within_threshold(left_size, _WORKER_TOKEN_SIZES[right_index], threshold):
                continue
            rows.append((left_index, right_index))
            emitted += 1
            if max_candidates_per_record > 0 and emitted >= max_candidates_per_record:
                capped += 1
                break
    return rows, capped


def _query_range_worker(args: tuple[int, int, float, int]) -> tuple[list[tuple[int, int]], int]:
    start, end, threshold, max_candidates_per_record = args
    return _query_lsh_range(start, end, threshold, max_candidates_per_record)


def _query_stream_worker(
    task_queue: Any,
    candidate_queue: Any,
    worker_id: int,
    threshold: float,
    max_candidates_per_record: int,
    emit_chunk_size: int,
) -> None:
    """Streaming query worker: pulls (start,end) ranges, pushes CandidateChunkMsg."""
    try:
        left_buf = np.empty(emit_chunk_size, dtype=np.int64)
        right_buf = np.empty(emit_chunk_size, dtype=np.int64)
        buf_pos = 0
        total_capped = 0
        total_candidates = 0

        def _flush() -> None:
            nonlocal buf_pos, total_candidates
            if buf_pos <= 0:
                return
            candidate_queue.put(
                CandidateChunkMsg(
                    left_indices=left_buf[:buf_pos].copy(),
                    right_indices=right_buf[:buf_pos].copy(),
                    pair_count=buf_pos,
                )
            )
            total_candidates += buf_pos
            buf_pos = 0

        while True:
            task = task_queue.get()
            if task is None:
                break
            start, end = task
            for left_index in range(start, end):
                if _WORKER_LSH is None:
                    raise RuntimeError("LSH index is not initialized for query worker")
                minhash = _WORKER_MINHASHES[left_index]
                emitted = 0
                left_size = int(_WORKER_TOKEN_SIZES[left_index])
                for raw in _WORKER_LSH.query(minhash):
                    right_index = int(raw)
                    if right_index <= left_index:
                        continue
                    if not _token_ratio_within_threshold(left_size, int(_WORKER_TOKEN_SIZES[right_index]), threshold):
                        continue
                    left_buf[buf_pos] = left_index
                    right_buf[buf_pos] = right_index
                    buf_pos += 1
                    if buf_pos >= emit_chunk_size:
                        _flush()
                    emitted += 1
                    if max_candidates_per_record > 0 and emitted >= max_candidates_per_record:
                        total_capped += 1
                        break
        _flush()
        candidate_queue.put(
            QueryWorkerDoneMsg(
                worker_id=worker_id,
                capped=total_capped,
                total_candidates=total_candidates,
            )
        )
    except Exception:
        candidate_queue.put(QueryWorkerDoneMsg(worker_id=worker_id, error=traceback.format_exc()))

# ---------------------------------------------------------------------------
# LSH index
# ---------------------------------------------------------------------------

def _build_lsh_index(
    minhashes: list[MinHash],
    *,
    num_perm: int,
    bands: int,
) -> MinHashLSH:
    if bands < 1:
        raise ValueError(f"bands must be >= 1, got {bands}")
    rows_per_band = max(1, num_perm // bands)
    if bands * rows_per_band > num_perm:
        raise ValueError(
            f"Invalid LSH params bands={bands} rows_per_band={rows_per_band} for num_perm={num_perm}"
        )
    lsh = MinHashLSH(num_perm=num_perm, params=(bands, rows_per_band))
    insert_started_at = time.perf_counter()
    _log_stage2(
        f"lsh_insert_start records={len(minhashes)} bands={bands} rows_per_band={rows_per_band}"
    )
    with lsh.insertion_session() as session:
        for index, minhash in enumerate(minhashes, start=1):
            session.insert(str(index - 1), minhash, check_duplication=False)
            if index % 1_000_000 == 0 or index == len(minhashes):
                percent = (index / len(minhashes)) * 100 if minhashes else 100.0
                _log_stage2(
                    f"lsh_insert_progress inserted={index}/{len(minhashes)} "
                    f"percent={percent:.2f} elapsed_sec={time.perf_counter() - insert_started_at:.1f}"
                )
    _log_stage2(f"lsh_insert_done elapsed_sec={time.perf_counter() - insert_started_at:.1f}")
    _log_snapshot("candidate_query_index_ready", records=len(minhashes), bands=bands, rows_per_band=rows_per_band)
    return lsh


def _build_query_index(
    minhashes: list[MinHash],
    *,
    query_backend: str,
    num_perm: int,
    bands: int,
) -> Any:
    backend = query_backend.strip().lower()
    if backend == "datasketch":
        return _build_lsh_index(minhashes, num_perm=num_perm, bands=bands)
    if backend == "array":
        build_started_at = time.perf_counter()
        _log_stage2(f"array_index_build_start records={len(minhashes)} bands={bands}")
        band_index = build_array_band_index(minhashes, num_perm=num_perm, bands=bands)
        _log_stage2(
            f"array_index_build_done elapsed_sec={time.perf_counter() - build_started_at:.1f} "
            f"bands={band_index.bands} rows_per_band={band_index.rows_per_band}"
        )
        _log_snapshot(
            "candidate_query_index_ready",
            records=len(minhashes),
            bands=band_index.bands,
            rows_per_band=band_index.rows_per_band,
            query_backend="array",
        )
        return band_index
    raise ValueError(f"Unsupported query_backend: {query_backend}")


def _release_query_globals(query_backend: str) -> None:
    global _WORKER_LSH, _WORKER_MINHASHES, _WORKER_TOKEN_SIZES
    _WORKER_LSH = None
    _WORKER_MINHASHES = []
    _WORKER_TOKEN_SIZES = np.empty(0, dtype=np.int32)
    backend = query_backend.strip().lower()
    if backend == "datasketch":
        return
    if backend == "array":
        _clear_array_index_globals()
        return
    raise ValueError(f"Unsupported query_backend: {query_backend}")

# ---------------------------------------------------------------------------
# Non-streaming query path (workers==1)
# ---------------------------------------------------------------------------

def _iter_candidate_pairs_via_lsh(
    minhashes: list[MinHash],
    token_sizes: list[int],
    *,
    threshold: float,
    num_perm: int,
    bands: int,
    workers: int,
    query_backend: str = "datasketch",
    query_chunk_size: int = 2048,
    max_candidates_per_record: int = 200,
) -> list[list[tuple[int, int]]]:
    backend = query_backend.strip().lower()
    token_sizes_array = np.asarray(token_sizes, dtype=np.int32)
    query_index = _build_query_index(minhashes, query_backend=backend, num_perm=num_perm, bands=bands)
    record_count = len(minhashes)

    ranges = [(start, min(start + query_chunk_size, record_count)) for start in range(0, record_count, query_chunk_size)]
    query_workers = max(1, min(workers, len(ranges)))
    _log_stage2(
        f"candidate_query_plan chunks={len(ranges)} chunk_size={query_chunk_size} "
        f"workers={query_workers} query_backend={backend}"
    )
    query_args = [(start, end, threshold, max_candidates_per_record) for start, end in ranges]

    global _WORKER_LSH, _WORKER_MINHASHES, _WORKER_TOKEN_SIZES
    if backend == "datasketch":
        _WORKER_LSH = cast(MinHashLSH, query_index)
        _WORKER_MINHASHES = minhashes
        _WORKER_TOKEN_SIZES = token_sizes_array
    else:
        _init_array_index_globals(query_index, token_sizes_array)
        minhashes.clear()

    query_started_at = time.perf_counter()
    all_chunks: list[list[tuple[int, int]]] = []
    cumulative_candidates = 0
    cumulative_capped = 0
    for index, args in enumerate(query_args, start=1):
        rows, capped = _query_range_worker(args) if backend == "datasketch" else _query_array_range(*args)
        cumulative_candidates += len(rows)
        cumulative_capped += capped
        if index % 100 == 0 or index == len(query_args):
            _log_stage2(
                f"candidate_query_progress chunks={index}/{len(query_args)} "
                f"cumulative_candidates={cumulative_candidates} capped_records={cumulative_capped} "
                f"query_backend={backend} "
                f"rss_gb={_rss_bytes(os.getpid()) / (1024 ** 3):.2f} "
                f"elapsed_sec={time.perf_counter() - query_started_at:.1f}"
            )
        if rows:
            all_chunks.append(rows)
    return all_chunks

# ---------------------------------------------------------------------------
# Streaming query + verify path (workers > 1)
# ---------------------------------------------------------------------------

def _stream_query_verify(
    minhashes: list[MinHash],
    token_sizes: list[int],
    resolved_tokens: list[tuple[int, ...]],
    record_ids: list[str],
    *,
    threshold: float,
    num_perm: int,
    bands: int,
    query_workers: int,
    verify_workers: int,
    query_backend: str = "datasketch",
    query_chunk_size: int = 2048,
    max_candidates_per_record: int = 200,
    emit_chunk_size: int = 8192,
    candidate_queue_maxsize: int = 256,
) -> VerifyResult:
    """Streaming query->verify: query workers push candidate subchunks onto a
    bounded queue, the main thread feeds them to the verify thread pool in
    real-time, avoiding full candidate materialization."""
    import gc as _gc

    backend = query_backend.strip().lower()
    token_sizes_array = np.asarray(token_sizes, dtype=np.int32)
    record_count = len(minhashes)
    query_index: Any = None
    if backend == "datasketch":
        query_index = _build_query_index(minhashes, query_backend=backend, num_perm=num_perm, bands=bands)
    elif backend == "array":
        query_index = _build_query_index(minhashes, query_backend=backend, num_perm=num_perm, bands=bands)
        _init_array_index_globals(query_index, token_sizes_array)
        minhashes.clear()
    else:
        raise ValueError(f"Unsupported query_backend: {query_backend}")

    ranges = [(start, min(start + query_chunk_size, record_count)) for start in range(0, record_count, query_chunk_size)]
    query_workers = max(1, min(query_workers, len(ranges)))
    verify_workers = max(1, verify_workers)

    _log_stage2(
        f"candidate_query_plan chunks={len(ranges)} chunk_size={query_chunk_size} "
        f"query_workers={query_workers} verify_workers={verify_workers} "
        f"streaming=true emit_chunk_size={emit_chunk_size} "
        f"candidate_queue_maxsize={candidate_queue_maxsize} query_backend={backend}"
    )

    global _WORKER_LSH, _WORKER_MINHASHES, _WORKER_TOKEN_SIZES
    if backend == "datasketch":
        _WORKER_LSH = cast(MinHashLSH, query_index)
        _WORKER_MINHASHES = minhashes
        _WORKER_TOKEN_SIZES = token_sizes_array

    mp_ctx = get_context("fork")
    task_queue = mp_ctx.Queue()  # type: ignore[attr-defined]
    candidate_queue = mp_ctx.Queue(maxsize=candidate_queue_maxsize)  # type: ignore[attr-defined]

    for start, end in ranges:
        task_queue.put((start, end))
    for _ in range(query_workers):
        task_queue.put(None)

    worker_procs: list[Process] = []
    for wid in range(query_workers):
        proc = mp_ctx.Process(
            target=_query_stream_worker if backend == "datasketch" else _query_stream_worker_array,
            args=(task_queue, candidate_queue, wid, threshold, max_candidates_per_record, emit_chunk_size),
            daemon=True,
        )
        proc.start()
        worker_procs.append(proc)

    token_buffer, token_offsets, token_lengths = _pack_token_sequences(resolved_tokens)
    total_token_count = int(token_lengths.sum())
    resolved_tokens.clear()
    _gc.collect()

    _vk._WORKER_TOKEN_BUFFER = token_buffer
    _vk._WORKER_TOKEN_OFFSETS = token_offsets
    _vk._WORKER_TOKEN_LENGTHS = token_lengths
    _prime_verify_kernel()

    _log_snapshot(
        "streaming_ready",
        records=len(record_ids),
        token_buffer_count=total_token_count,
        token_buffer_gb=f"{token_buffer.nbytes / (1024 ** 3):.2f}",
        query_workers=query_workers,
        verify_workers=verify_workers,
        query_backend=backend,
    )

    result_chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    verify_pending: set[Any] = set()
    verify_pending_limit = max(4, verify_workers * 2)
    candidate_count = 0
    verify_chunk_counter = 0
    workers_done = 0
    query_started_at = time.perf_counter()
    verify_started_at = query_started_at
    query_released = False

    stat_candidate_chunks = 0
    stat_nonempty_chunks = 0
    stat_max_pairs_in_chunk = 0
    stat_query_capped = 0
    stat_query_candidates = 0
    stat_accepted_pairs = 0

    def _collect_verify_results(done_futures: set[Any]) -> None:
        nonlocal stat_accepted_pairs
        for future in done_futures:
            left_indices, right_indices, similarities = future.result()
            result_chunks.append((left_indices, right_indices, similarities))
            stat_accepted_pairs += int(left_indices.shape[0])

    with ThreadPoolExecutor(max_workers=verify_workers) as verify_executor:
        while workers_done < query_workers:
            try:
                msg = candidate_queue.get(timeout=0.5)
            except Exception:
                for proc in worker_procs:
                    if not proc.is_alive() and proc.exitcode != 0:
                        workers_done = query_workers
                        break
                continue

            if isinstance(msg, QueryWorkerDoneMsg):
                workers_done += 1
                stat_query_capped += msg.capped
                stat_query_candidates += msg.total_candidates
                if msg.error:
                    _log_stage2(f"query_worker_error worker={msg.worker_id} error={msg.error}")
                else:
                    _log_stage2(
                        f"candidate_query_worker_done worker={msg.worker_id} "
                        f"workers_done={workers_done}/{query_workers} "
                        f"worker_candidates={msg.total_candidates} worker_capped={msg.capped} "
                        f"cumulative_candidates={stat_query_candidates} "
                        f"cumulative_capped={stat_query_capped}"
                    )
                continue

            if isinstance(msg, CandidateChunkMsg):
                stat_candidate_chunks += 1
                pair_count = msg.pair_count
                candidate_count += pair_count
                if pair_count > 0:
                    stat_nonempty_chunks += 1
                    if pair_count > stat_max_pairs_in_chunk:
                        stat_max_pairs_in_chunk = pair_count
                    verify_chunk_counter += 1
                    verify_pending.add(verify_executor.submit(
                        _verify_candidate_arrays_raw, msg.left_indices, msg.right_indices, threshold
                    ))
                    if len(verify_pending) >= verify_pending_limit:
                        done, verify_pending = wait(verify_pending, return_when=FIRST_COMPLETED)
                        _collect_verify_results(done)

                if verify_chunk_counter > 0 and verify_chunk_counter % 100 == 0:
                    accepted_pairs = stat_accepted_pairs
                    elapsed_sec = time.perf_counter() - query_started_at
                    _log_stage2(
                        f"candidate_verify_progress chunks={verify_chunk_counter} "
                        f"pending={len(verify_pending)} candidates={candidate_count} "
                        f"accepted_pairs={accepted_pairs} elapsed_sec={elapsed_sec:.1f}"
                    )
                    _log_snapshot(
                        "candidate_verify_progress",
                        workers=verify_workers,
                        chunks=verify_chunk_counter,
                        pending=len(verify_pending),
                        candidates=candidate_count,
                        accepted_pairs=accepted_pairs,
                        elapsed_sec=f"{elapsed_sec:.1f}",
                    )

        for proc in worker_procs:
            proc.join(timeout=5)
        if not query_released:
            _log_stage2(f"post_query_release releasing query backend globals backend={backend}")
            _release_query_globals(backend)
            _gc.collect()
            _log_snapshot("post_query_released")
            query_released = True
        verify_started_at = time.perf_counter()

        while verify_pending:
            done, verify_pending = wait(verify_pending, return_when=FIRST_COMPLETED)
            _collect_verify_results(done)

    avg_pairs_per_chunk = (candidate_count / stat_candidate_chunks) if stat_candidate_chunks else 0.0
    _log_stage2(
        "verify_baseline "
        f"records_count={len(record_ids)} "
        f"candidate_chunks_count={stat_candidate_chunks} nonempty_candidate_chunks={stat_nonempty_chunks} "
        f"candidate_pairs_total={candidate_count} "
        f"query_candidates_total={stat_query_candidates} "
        f"capped_records={stat_query_capped} "
        f"query_backend={backend} "
        f"avg_candidate_pairs_per_chunk={avg_pairs_per_chunk:.1f} "
        f"max_candidate_pairs_in_chunk={stat_max_pairs_in_chunk}"
    )
    _log_stage2(
        f"candidate_build_done method=lsh query_backend={backend} candidates={candidate_count} "
        f"accepted_pairs={stat_accepted_pairs} "
        f"elapsed_sec={time.perf_counter() - query_started_at:.1f}"
    )
    _log_snapshot(
        "candidate_verify_done",
        workers=verify_workers,
        chunks=verify_chunk_counter,
        candidates=candidate_count,
        accepted_pairs=stat_accepted_pairs,
        elapsed_sec=f"{time.perf_counter() - verify_started_at:.1f}",
    )

    _vk._WORKER_TOKEN_BUFFER = None
    _vk._WORKER_TOKEN_OFFSETS = None
    _vk._WORKER_TOKEN_LENGTHS = None
    if result_chunks:
        accepted_left = np.concatenate([chunk[0] for chunk in result_chunks]).astype(np.int64, copy=False)
        accepted_right = np.concatenate([chunk[1] for chunk in result_chunks]).astype(np.int64, copy=False)
        accepted_similarity = np.concatenate([chunk[2] for chunk in result_chunks]).astype(np.float64, copy=False)
    else:
        accepted_left = np.empty(0, dtype=np.int64)
        accepted_right = np.empty(0, dtype=np.int64)
        accepted_similarity = np.empty(0, dtype=np.float64)
    return VerifyResult(
        accepted_left=accepted_left,
        accepted_right=accepted_right,
        accepted_similarity=accepted_similarity,
        record_ids=record_ids,
    )
