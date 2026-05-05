"""Array-backed band index for MinHash LSH query.

This module replaces Python-object-heavy datasketch query with compact CSR-style
arrays plus Numba query kernels. Signature generation remains unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import traceback
from typing import Any

import numpy as np  # type: ignore[reportMissingImports]
from numba import njit  # type: ignore[reportMissingImports]

from datasketch import MinHash  # type: ignore[reportMissingImports]


@dataclass(slots=True)
class BandIndexArrays:
    band_keys: np.ndarray
    csr_keys: list[np.ndarray]
    csr_offsets: list[np.ndarray]
    csr_indices: list[np.ndarray]
    bands: int
    rows_per_band: int


_WORKER_BAND_KEYS: np.ndarray = np.empty((0, 0), dtype=np.uint64)
_WORKER_CSR_KEYS: list[np.ndarray] = []
_WORKER_CSR_OFFSETS: list[np.ndarray] = []
_WORKER_CSR_INDICES: list[np.ndarray] = []
_WORKER_TOKEN_SIZES: np.ndarray = np.empty(0, dtype=np.int32)


def _extract_signature_matrix(minhashes: list[MinHash]) -> np.ndarray:
    if not minhashes:
        return np.empty((0, 0), dtype=np.uint64)
    matrix = np.empty((len(minhashes), len(minhashes[0].hashvalues)), dtype=np.uint64)
    for index, minhash in enumerate(minhashes):
        matrix[index, :] = np.asarray(minhash.hashvalues, dtype=np.uint64)
    return matrix


@njit(cache=True, nogil=True)
def _compute_band_keys_numba(sig_matrix: np.ndarray, bands: int, rows_per_band: int) -> np.ndarray:
    record_count = sig_matrix.shape[0]
    band_keys = np.empty((bands, record_count), dtype=np.uint64)
    multiplier = np.uint64(1099511628211)
    seed = np.uint64(1469598103934665603)
    for band in range(bands):
        start_col = band * rows_per_band
        end_col = start_col + rows_per_band
        for row in range(record_count):
            value = seed ^ np.uint64(band + 1)
            for col in range(start_col, end_col):
                value ^= sig_matrix[row, col] + np.uint64(col - start_col + 1)
                value *= multiplier
            band_keys[band, row] = value
    return band_keys


@njit(cache=True, nogil=True)
def _lookup_csr_numba(sorted_keys: np.ndarray, offsets: np.ndarray, query_key: np.uint64) -> tuple[int, int]:
    left = 0
    right = sorted_keys.shape[0]
    while left < right:
        mid = (left + right) // 2
        if sorted_keys[mid] < query_key:
            left = mid + 1
        else:
            right = mid
    if left >= sorted_keys.shape[0] or sorted_keys[left] != query_key:
        return 0, 0
    return int(offsets[left]), int(offsets[left + 1])


@njit(cache=True, nogil=True)
def _token_ratio_within_threshold_numba(left_count: int, right_count: int, threshold: float) -> bool:
    if threshold <= 0.0:
        return True
    if left_count == 0 or right_count == 0:
        return True
    ratio = left_count / right_count if left_count >= right_count else right_count / left_count
    return ratio <= (1.0 / threshold)


@njit(cache=True, nogil=True)
def _scan_posting_numba(
    left_index: int,
    posting_indices: np.ndarray,
    start_pos: int,
    end: int,
    token_sizes: np.ndarray,
    threshold: float,
    seen_gen: np.ndarray,
    current_gen: int,
    max_candidates: int,
    out_left: np.ndarray,
    out_right: np.ndarray,
    out_pos: int,
) -> tuple[int, int, int, int]:
    emitted = 0
    left_size = int(token_sizes[left_index])
    pos = start_pos
    while pos < end:
        if out_pos >= out_left.shape[0]:
            break
        right_index = int(posting_indices[pos])
        if right_index <= left_index:
            pos += 1
            continue
        if seen_gen[right_index] == current_gen:
            pos += 1
            continue
        if not _token_ratio_within_threshold_numba(left_size, int(token_sizes[right_index]), threshold):
            pos += 1
            continue
        seen_gen[right_index] = current_gen
        out_left[out_pos] = left_index
        out_right[out_pos] = right_index
        out_pos += 1
        emitted += 1
        if max_candidates > 0 and emitted >= max_candidates:
            pos += 1
            break
        pos += 1
    return pos, out_pos, emitted, 1 if pos >= end else 0


def _build_csr_per_band(band_keys_for_band: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if band_keys_for_band.size == 0:
        return (
            np.empty(0, dtype=np.uint64),
            np.zeros(1, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )
    order = np.argsort(band_keys_for_band, kind="mergesort")
    sorted_band_keys = band_keys_for_band[order]
    unique_mask = np.empty(sorted_band_keys.shape[0], dtype=bool)
    unique_mask[0] = True
    unique_mask[1:] = sorted_band_keys[1:] != sorted_band_keys[:-1]
    unique_keys = sorted_band_keys[unique_mask]
    starts = np.flatnonzero(unique_mask).astype(np.int32)
    offsets = np.empty(unique_keys.shape[0] + 1, dtype=np.int32)
    offsets[:-1] = starts
    offsets[-1] = np.int32(sorted_band_keys.shape[0])
    return unique_keys, offsets, order.astype(np.int32)


def build_array_band_index(minhashes: list[MinHash], num_perm: int, bands: int) -> BandIndexArrays:
    if bands < 1:
        raise ValueError(f"bands must be >= 1, got {bands}")
    rows_per_band = max(1, num_perm // bands)
    if bands * rows_per_band > num_perm:
        raise ValueError(
            f"Invalid LSH params bands={bands} rows_per_band={rows_per_band} for num_perm={num_perm}"
        )
    sig_matrix = _extract_signature_matrix(minhashes)
    if sig_matrix.shape[1] < bands * rows_per_band:
        raise ValueError(
            f"Signature matrix columns={sig_matrix.shape[1]} insufficient for "
            f"bands={bands}, rows_per_band={rows_per_band}"
        )
    band_keys = _compute_band_keys_numba(sig_matrix[:, : bands * rows_per_band], bands, rows_per_band)
    csr_keys: list[np.ndarray] = []
    csr_offsets: list[np.ndarray] = []
    csr_indices: list[np.ndarray] = []
    for band in range(bands):
        keys, offsets, indices = _build_csr_per_band(band_keys[band])
        csr_keys.append(keys)
        csr_offsets.append(offsets)
        csr_indices.append(indices)
    return BandIndexArrays(
        band_keys=band_keys,
        csr_keys=csr_keys,
        csr_offsets=csr_offsets,
        csr_indices=csr_indices,
        bands=bands,
        rows_per_band=rows_per_band,
    )


def _init_array_index_globals(band_index: BandIndexArrays, token_sizes: np.ndarray) -> None:
    global _WORKER_BAND_KEYS, _WORKER_CSR_KEYS, _WORKER_CSR_OFFSETS, _WORKER_CSR_INDICES, _WORKER_TOKEN_SIZES
    _WORKER_BAND_KEYS = band_index.band_keys
    _WORKER_CSR_KEYS = band_index.csr_keys
    _WORKER_CSR_OFFSETS = band_index.csr_offsets
    _WORKER_CSR_INDICES = band_index.csr_indices
    _WORKER_TOKEN_SIZES = token_sizes


def _clear_array_index_globals() -> None:
    global _WORKER_BAND_KEYS, _WORKER_CSR_KEYS, _WORKER_CSR_OFFSETS, _WORKER_CSR_INDICES, _WORKER_TOKEN_SIZES
    _WORKER_BAND_KEYS = np.empty((0, 0), dtype=np.uint64)
    _WORKER_CSR_KEYS = []
    _WORKER_CSR_OFFSETS = []
    _WORKER_CSR_INDICES = []
    _WORKER_TOKEN_SIZES = np.empty(0, dtype=np.int32)


def _query_array_range(
    start: int,
    end: int,
    threshold: float,
    max_candidates_per_record: int,
) -> tuple[list[tuple[int, int]], int]:
    record_count = int(_WORKER_TOKEN_SIZES.shape[0])
    seen_gen = np.zeros(record_count, dtype=np.int32)
    current_gen = 0
    capped = 0
    rows: list[tuple[int, int]] = []
    per_record_limit = max_candidates_per_record if max_candidates_per_record > 0 else max(1024, min(record_count, 65536))
    out_left = np.empty(max(256, per_record_limit), dtype=np.int64)
    out_right = np.empty_like(out_left)
    for left_index in range(start, end):
        current_gen += 1
        emitted = 0
        out_pos = 0
        for band in range(_WORKER_BAND_KEYS.shape[0]):
            key = np.uint64(_WORKER_BAND_KEYS[band, left_index])
            posting_start, posting_end = _lookup_csr_numba(_WORKER_CSR_KEYS[band], _WORKER_CSR_OFFSETS[band], key)
            if posting_start == posting_end:
                continue
            scan_pos = posting_start
            while True:
                remaining = 0 if max_candidates_per_record <= 0 else max_candidates_per_record - emitted
                scan_pos, out_pos, newly_emitted, finished = _scan_posting_numba(
                    left_index,
                    _WORKER_CSR_INDICES[band],
                    scan_pos,
                    posting_end,
                    _WORKER_TOKEN_SIZES,
                    threshold,
                    seen_gen,
                    current_gen,
                    remaining,
                    out_left,
                    out_right,
                    out_pos,
                )
                emitted += newly_emitted
                if finished:
                    break
                for idx in range(out_pos):
                    rows.append((int(out_left[idx]), int(out_right[idx])))
                out_pos = 0
                if max_candidates_per_record > 0 and emitted >= max_candidates_per_record:
                    break
            if max_candidates_per_record > 0 and emitted >= max_candidates_per_record:
                capped += 1
                break
        for idx in range(out_pos):
            rows.append((int(out_left[idx]), int(out_right[idx])))
    return rows, capped


def _query_stream_worker_array(
    task_queue: Any,
    candidate_queue: Any,
    worker_id: int,
    threshold: float,
    max_candidates_per_record: int,
    emit_chunk_size: int,
) -> None:
    from ._minhash_engine import CandidateChunkMsg, QueryWorkerDoneMsg

    try:
        record_count = int(_WORKER_TOKEN_SIZES.shape[0])
        seen_gen = np.zeros(record_count, dtype=np.int32)
        current_gen = 0
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
                current_gen += 1
                emitted = 0
                for band in range(_WORKER_BAND_KEYS.shape[0]):
                    key = np.uint64(_WORKER_BAND_KEYS[band, left_index])
                    posting_start, posting_end = _lookup_csr_numba(
                        _WORKER_CSR_KEYS[band], _WORKER_CSR_OFFSETS[band], key
                    )
                    if posting_start == posting_end:
                        continue
                    scan_pos = posting_start
                    while True:
                        remaining = 0 if max_candidates_per_record <= 0 else max_candidates_per_record - emitted
                        scan_pos, buf_pos, newly_emitted, finished = _scan_posting_numba(
                            left_index,
                            _WORKER_CSR_INDICES[band],
                            scan_pos,
                            posting_end,
                            _WORKER_TOKEN_SIZES,
                            threshold,
                            seen_gen,
                            current_gen,
                            remaining,
                            left_buf,
                            right_buf,
                            buf_pos,
                        )
                        emitted += newly_emitted
                        if buf_pos >= emit_chunk_size:
                            _flush()
                        if finished:
                            break
                        if max_candidates_per_record > 0 and emitted >= max_candidates_per_record:
                            break
                    if max_candidates_per_record > 0 and emitted >= max_candidates_per_record:
                        total_capped += 1
                        break
            _flush()
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
