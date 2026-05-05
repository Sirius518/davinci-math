"""Packed-token verify kernel for fuzzy deduplication.

Provides the Numba-JIT two-pointer Jaccard kernel, packed token storage,
and the verify entry points used by both the streaming and legacy paths.
"""
from __future__ import annotations

import numpy as np  # type: ignore[reportMissingImports]
from numba import njit  # type: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Module-level verify state (set by minhash_candidate_pairs / _stream_query_verify)
# ---------------------------------------------------------------------------

_WORKER_TOKEN_BUFFER: np.ndarray | None = None
_WORKER_TOKEN_OFFSETS: np.ndarray | None = None
_WORKER_TOKEN_LENGTHS: np.ndarray | None = None


def _pack_token_sequences(token_sequences: list[tuple[int, ...]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.fromiter((len(tokens) for tokens in token_sequences), dtype=np.int64, count=len(token_sequences))
    offsets = np.empty(len(token_sequences), dtype=np.int64)
    total_token_count = int(lengths.sum())
    token_buffer = np.empty(total_token_count, dtype=np.uint64)
    cursor = 0
    for index, tokens in enumerate(token_sequences):
        offsets[index] = cursor
        next_cursor = cursor + len(tokens)
        if tokens:
            token_buffer[cursor:next_cursor] = np.asarray(tokens, dtype=np.uint64)
        cursor = next_cursor
    return token_buffer, offsets, lengths


@njit(cache=True, nogil=True)
def _verify_candidate_chunk_numba(
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    threshold: float,
    token_buffer: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    accepted_left = np.empty(left_indices.shape[0], dtype=np.int64)
    accepted_right = np.empty(right_indices.shape[0], dtype=np.int64)
    accepted_similarity = np.empty(left_indices.shape[0], dtype=np.float64)
    accepted_count = 0

    for candidate_index in range(left_indices.shape[0]):
        left_record_index = left_indices[candidate_index]
        right_record_index = right_indices[candidate_index]
        left_start = offsets[left_record_index]
        right_start = offsets[right_record_index]
        left_length = lengths[left_record_index]
        right_length = lengths[right_record_index]

        if left_length == 0 and right_length == 0:
            similarity = 1.0
        elif left_length == 0 or right_length == 0:
            similarity = 0.0
        else:
            left_cursor = left_start
            right_cursor = right_start
            left_end = left_start + left_length
            right_end = right_start + right_length
            intersection = 0
            while left_cursor < left_end and right_cursor < right_end:
                left_value = token_buffer[left_cursor]
                right_value = token_buffer[right_cursor]
                if left_value == right_value:
                    intersection += 1
                    left_cursor += 1
                    right_cursor += 1
                elif left_value < right_value:
                    left_cursor += 1
                else:
                    right_cursor += 1
            union = left_length + right_length - intersection
            similarity = 1.0 if union == 0 else intersection / union

        if similarity >= threshold:
            accepted_left[accepted_count] = left_record_index
            accepted_right[accepted_count] = right_record_index
            accepted_similarity[accepted_count] = similarity
            accepted_count += 1

    return (
        accepted_left[:accepted_count],
        accepted_right[:accepted_count],
        accepted_similarity[:accepted_count],
    )


def _prime_verify_kernel() -> None:
    token_buffer = _WORKER_TOKEN_BUFFER
    offsets = _WORKER_TOKEN_OFFSETS
    lengths = _WORKER_TOKEN_LENGTHS
    if token_buffer is None or offsets is None or lengths is None:
        return
    empty = np.empty(0, dtype=np.int64)
    _verify_candidate_chunk_numba(empty, empty, 1.0, token_buffer, offsets, lengths)


def _candidate_chunk_arrays(chunk: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    left_indices = np.empty(len(chunk), dtype=np.int64)
    right_indices = np.empty(len(chunk), dtype=np.int64)
    for index, (left_index, right_index) in enumerate(chunk):
        left_indices[index] = left_index
        right_indices[index] = right_index
    return left_indices, right_indices


def _verify_candidate_arrays_raw(
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if left_indices.shape[0] == 0:
        empty_indices = np.empty(0, dtype=np.int64)
        empty_similarity = np.empty(0, dtype=np.float64)
        return empty_indices, empty_indices, empty_similarity
    return _verify_candidate_chunk_numba(
        left_indices,
        right_indices,
        threshold,
        _WORKER_TOKEN_BUFFER,
        _WORKER_TOKEN_OFFSETS,
        _WORKER_TOKEN_LENGTHS,
    )


def _verify_candidate_arrays(left_indices: np.ndarray, right_indices: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    accepted_left, accepted_right, accepted_similarity = _verify_candidate_arrays_raw(
        left_indices,
        right_indices,
        threshold,
    )
    return [
        (int(accepted_left[i]), int(accepted_right[i]), float(accepted_similarity[i]))
        for i in range(accepted_left.shape[0])
    ]


def _verify_candidate_chunk(args: tuple[list[tuple[int, int]], float]) -> list[tuple[int, int, float]]:
    chunk, threshold = args
    if not chunk:
        return []
    left_indices, right_indices = _candidate_chunk_arrays(chunk)
    return _verify_candidate_arrays(left_indices, right_indices, threshold)
