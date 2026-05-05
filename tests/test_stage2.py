from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.core.schema import CanonicalRecord  # type: ignore[reportMissingImports]
from pipeline.core.io import (  # type: ignore[reportMissingImports]
    filter_canonical_records_to_output,
    read_canonical_records,
    write_canonical_records,
)
from pipeline.ops.dedup.exact import exact_deduplicate  # type: ignore[reportMissingImports]
from pipeline.ops.dedup import fuzzy as fuzzy_dedup  # type: ignore[reportMissingImports]
from pipeline.ops.dedup._verify_kernel import (  # type: ignore[reportMissingImports]
    _pack_token_sequences,
    _prime_verify_kernel,
    _verify_candidate_chunk,
    _verify_candidate_arrays,
    _verify_candidate_arrays_raw,
)
import pipeline.ops.dedup._verify_kernel as _vk  # type: ignore[reportMissingImports]
import pipeline.ops.dedup._minhash_engine as _me  # type: ignore[reportMissingImports]
import pipeline.ops.dedup._band_index as _bi  # type: ignore[reportMissingImports]
from pipeline.ops.dedup.fuzzy import minhash_candidate_pairs  # type: ignore[reportMissingImports]


def _edge_set(result: object) -> set[tuple[str, str]]:
    if isinstance(result, _me.VerifyResult):
        return set(result.to_edges())
    return {(pair.left_id, pair.right_id) for pair in result}  # type: ignore[attr-defined]


class Stage2Tests(unittest.TestCase):
    def test_pack_token_sequences_round_trip(self) -> None:
        token_sequences = [(1, 3, 5), (), (8, 13)]
        token_buffer, offsets, lengths = _pack_token_sequences(token_sequences)
        self.assertEqual(token_buffer.tolist(), [1, 3, 5, 8, 13])
        self.assertEqual(offsets.tolist(), [0, 3, 3])
        self.assertEqual(lengths.tolist(), [3, 0, 2])

    def test_verify_candidate_chunk_uses_packed_tokens(self) -> None:
        token_sequences = [(1, 2, 3, 4), (1, 2, 4, 5), (9, 10)]
        token_buffer, offsets, lengths = _pack_token_sequences(token_sequences)
        _vk._WORKER_TOKEN_BUFFER = token_buffer
        _vk._WORKER_TOKEN_OFFSETS = offsets
        _vk._WORKER_TOKEN_LENGTHS = lengths
        try:
            _prime_verify_kernel()
            results = _verify_candidate_chunk(([(0, 1), (0, 2)], 0.5))
        finally:
            _vk._WORKER_TOKEN_BUFFER = None
            _vk._WORKER_TOKEN_OFFSETS = None
            _vk._WORKER_TOKEN_LENGTHS = None
        self.assertEqual(results, [(0, 1, 0.6)])

    def test_verify_chunk_and_arrays_agree(self) -> None:
        """_verify_candidate_chunk and _verify_candidate_arrays produce the same results."""
        import numpy as _np
        token_sequences = [(2, 4, 6, 8), (2, 4, 6, 9), (10, 12)]
        token_buffer, offsets, lengths = _pack_token_sequences(token_sequences)
        _vk._WORKER_TOKEN_BUFFER = token_buffer
        _vk._WORKER_TOKEN_OFFSETS = offsets
        _vk._WORKER_TOKEN_LENGTHS = lengths
        try:
            _prime_verify_kernel()
            via_chunk = _verify_candidate_chunk(([(0, 1), (0, 2)], 0.5))
            left = _np.array([0, 0], dtype=_np.int64)
            right = _np.array([1, 2], dtype=_np.int64)
            via_arrays = _verify_candidate_arrays(left, right, 0.5)
        finally:
            _vk._WORKER_TOKEN_BUFFER = None
            _vk._WORKER_TOKEN_OFFSETS = None
            _vk._WORKER_TOKEN_LENGTHS = None
        self.assertEqual(via_chunk, via_arrays)

    def test_verify_candidate_arrays_raw(self) -> None:
        import numpy as _np
        token_sequences = [(2, 4, 6, 8), (2, 4, 6, 9), (10, 12)]
        token_buffer, offsets, lengths = _pack_token_sequences(token_sequences)
        _vk._WORKER_TOKEN_BUFFER = token_buffer
        _vk._WORKER_TOKEN_OFFSETS = offsets
        _vk._WORKER_TOKEN_LENGTHS = lengths
        try:
            _prime_verify_kernel()
            left = _np.array([0, 0], dtype=_np.int64)
            right = _np.array([1, 2], dtype=_np.int64)
            via_raw = _verify_candidate_arrays_raw(left, right, 0.5)
            via_arrays = _verify_candidate_arrays(left, right, 0.5)
        finally:
            _vk._WORKER_TOKEN_BUFFER = None
            _vk._WORKER_TOKEN_OFFSETS = None
            _vk._WORKER_TOKEN_LENGTHS = None
        self.assertEqual(via_raw[0].tolist(), [0])
        self.assertEqual(via_raw[1].tolist(), [1])
        self.assertEqual(via_raw[2].tolist(), [0.6])
        self.assertEqual(via_arrays, [(0, 1, 0.6)])

    def test_exact_dedup_uses_question_hash(self) -> None:
        records = [
            CanonicalRecord(record_id="a", question="Solve x+1=2", raw_dataset_answer="1"),
            CanonicalRecord(record_id="b", question="Solve x+1=2", raw_dataset_answer="1"),
        ]
        kept, duplicates = exact_deduplicate(records)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(duplicates), 1)

    def test_minhash_finds_near_duplicates(self) -> None:
        records = [
            CanonicalRecord(record_id="a", question="Compute the derivative of x squared.", raw_dataset_answer="2x"),
            CanonicalRecord(record_id="b", question="Compute derivative of x squared.", raw_dataset_answer="2x"),
        ]
        pairs = minhash_candidate_pairs(records, threshold=0.5)
        self.assertEqual(len(pairs), 1)

    def test_minhash_parallel_arg_keeps_behavior(self) -> None:
        records = [
            CanonicalRecord(record_id="a", question="Find derivative of x^2", raw_dataset_answer="2x"),
            CanonicalRecord(record_id="b", question="Find the derivative of x^2.", raw_dataset_answer="2x"),
            CanonicalRecord(record_id="c", question="Integrate x^2", raw_dataset_answer="x^3/3"),
        ]
        single = minhash_candidate_pairs(records, threshold=0.4, workers=1)
        multi = minhash_candidate_pairs(records, threshold=0.4, workers=4)
        self.assertEqual(_edge_set(single), _edge_set(multi))

    def test_minhash_parallel_workers_produce_pairs(self) -> None:
        records = [
            CanonicalRecord(record_id="a", question="Compute the derivative of x squared.", raw_dataset_answer="2x"),
            CanonicalRecord(record_id="b", question="Compute derivative of x squared.", raw_dataset_answer="2x"),
            CanonicalRecord(record_id="c", question="Integrate x squared.", raw_dataset_answer="x^3/3"),
        ]
        pairs = minhash_candidate_pairs(records, threshold=0.4, workers=2)
        self.assertGreaterEqual(len(_edge_set(pairs)), 1)

    def test_max_candidates_per_record_limits_output(self) -> None:
        records = [
            CanonicalRecord(record_id=f"id-{index}", question=f"Compute derivative of x^2 variant {index % 3}", raw_dataset_answer="2x")
            for index in range(12)
        ]
        uncapped = minhash_candidate_pairs(records, threshold=0.3, workers=1, num_perm=16, bands=4)
        capped = minhash_candidate_pairs(
            records,
            threshold=0.3,
            workers=1,
            num_perm=16,
            bands=4,
            max_candidates_per_record=1,
        )
        self.assertLessEqual(len(capped), len(uncapped))

    def test_streaming_vs_single_worker_equivalence(self) -> None:
        """The streaming path (workers>1) must produce the same (left_id, right_id)
        set as the legacy single-worker path, regardless of arrival order."""
        records = [
            CanonicalRecord(record_id="a", question="Compute the derivative of x squared.", raw_dataset_answer="2x"),
            CanonicalRecord(record_id="b", question="Compute derivative of x squared.", raw_dataset_answer="2x"),
            CanonicalRecord(record_id="c", question="Integrate x squared over zero to one.", raw_dataset_answer="1/3"),
            CanonicalRecord(record_id="d", question="Compute the derivative of x squared exactly.", raw_dataset_answer="2x"),
        ]
        single = minhash_candidate_pairs(records, threshold=0.4, workers=1, num_perm=64, bands=8)
        streaming = minhash_candidate_pairs(records, threshold=0.4, workers=2, num_perm=64, bands=8)
        single_set = _edge_set(single)
        streaming_set = _edge_set(streaming)
        self.assertEqual(single_set, streaming_set)

    def test_max_candidates_per_record_enforced_in_streaming(self) -> None:
        """cap semantics must still apply per source record in streaming mode."""
        records = [
            CanonicalRecord(record_id=f"id-{i}", question=f"Compute derivative of x^2 variant {i % 3}", raw_dataset_answer="2x")
            for i in range(12)
        ]
        uncapped_streaming = minhash_candidate_pairs(records, threshold=0.3, workers=2, num_perm=16, bands=4)
        capped_streaming = minhash_candidate_pairs(
            records,
            threshold=0.3,
            workers=2,
            num_perm=16,
            bands=4,
            max_candidates_per_record=1,
        )
        self.assertLessEqual(len(_edge_set(capped_streaming)), len(_edge_set(uncapped_streaming)))

    def test_streaming_new_knobs_accepted(self) -> None:
        records = [
            CanonicalRecord(record_id="a", question="Compute the derivative of x squared.", raw_dataset_answer="2x"),
            CanonicalRecord(record_id="b", question="Compute derivative of x squared.", raw_dataset_answer="2x"),
            CanonicalRecord(record_id="c", question="Integrate x squared over zero to one.", raw_dataset_answer="1/3"),
            CanonicalRecord(record_id="d", question="Compute the derivative of x squared exactly.", raw_dataset_answer="2x"),
        ]
        baseline = minhash_candidate_pairs(records, threshold=0.4, workers=2, num_perm=64, bands=8)
        tuned = minhash_candidate_pairs(
            records,
            threshold=0.4,
            workers=2,
            query_workers=2,
            verify_workers=1,
            query_chunk_size=256,
            emit_chunk_size=256,
            candidate_queue_maxsize=8,
            num_perm=64,
            bands=8,
        )
        self.assertEqual(_edge_set(baseline), _edge_set(tuned))

    def test_incremental_emit_equivalence(self) -> None:
        records = [
            CanonicalRecord(record_id=f"id-{i}", question=f"Compute derivative of x squared variant {i % 4}", raw_dataset_answer="2x")
            for i in range(16)
        ]
        coarse = minhash_candidate_pairs(
            records,
            threshold=0.3,
            workers=2,
            query_workers=2,
            verify_workers=2,
            query_chunk_size=2048,
            emit_chunk_size=8192,
            num_perm=16,
            bands=4,
        )
        fine = minhash_candidate_pairs(
            records,
            threshold=0.3,
            workers=2,
            query_workers=2,
            verify_workers=2,
            query_chunk_size=256,
            emit_chunk_size=1,
            candidate_queue_maxsize=8,
            num_perm=16,
            bands=4,
        )
        self.assertEqual(_edge_set(coarse), _edge_set(fine))

    def test_token_sizes_numpy(self) -> None:
        records = [
            CanonicalRecord(record_id="a", question="Find derivative of x^2", raw_dataset_answer="2x"),
            CanonicalRecord(record_id="b", question="Find the derivative of x^2.", raw_dataset_answer="2x"),
            CanonicalRecord(record_id="c", question="Integrate x^2", raw_dataset_answer="x^3/3"),
        ]
        try:
            minhash_candidate_pairs(
                records,
                threshold=0.4,
                workers=1,
                num_perm=16,
                bands=4,
                query_chunk_size=256,
            )
            self.assertEqual(str(_me._WORKER_TOKEN_SIZES.dtype), "int32")
        finally:
            _me._WORKER_LSH = None
            _me._WORKER_MINHASHES = []
            _me._WORKER_TOKEN_SIZES = _me.np.empty(0, dtype=_me.np.int32)

    def test_array_backend_matches_datasketch(self) -> None:
        records = [
            CanonicalRecord(record_id="a", question="Compute the derivative of x squared.", raw_dataset_answer="2x"),
            CanonicalRecord(record_id="b", question="Compute derivative of x squared.", raw_dataset_answer="2x"),
            CanonicalRecord(record_id="c", question="Integrate x squared over zero to one.", raw_dataset_answer="1/3"),
            CanonicalRecord(record_id="d", question="Compute the derivative of x squared exactly.", raw_dataset_answer="2x"),
            CanonicalRecord(record_id="e", question="Evaluate the integral of x squared from 0 to 1.", raw_dataset_answer="1/3"),
        ]
        datasketch_pairs = minhash_candidate_pairs(
            records,
            threshold=0.4,
            workers=1,
            query_backend="datasketch",
            num_perm=64,
            bands=8,
        )
        array_pairs = minhash_candidate_pairs(
            records,
            threshold=0.4,
            workers=1,
            query_backend="array",
            num_perm=64,
            bands=8,
        )
        self.assertEqual(
            {(pair.left_id, pair.right_id) for pair in datasketch_pairs},
            {(pair.left_id, pair.right_id) for pair in array_pairs},
        )

    def test_array_backend_streaming(self) -> None:
        records = [
            CanonicalRecord(record_id=f"id-{i}", question=f"Compute derivative of x squared variant {i % 4}", raw_dataset_answer="2x")
            for i in range(16)
        ]
        single = minhash_candidate_pairs(
            records,
            threshold=0.3,
            workers=1,
            query_backend="array",
            num_perm=16,
            bands=4,
        )
        streaming = minhash_candidate_pairs(
            records,
            threshold=0.3,
            workers=2,
            query_backend="array",
            query_workers=2,
            verify_workers=2,
            query_chunk_size=256,
            emit_chunk_size=4,
            candidate_queue_maxsize=8,
            num_perm=16,
            bands=4,
        )
        self.assertEqual(
            _edge_set(single),
            _edge_set(streaming),
        )

    def test_stream_verify_returns_verify_result(self) -> None:
        records = [
            CanonicalRecord(record_id="a", question="Compute the derivative of x squared.", raw_dataset_answer="2x"),
            CanonicalRecord(record_id="b", question="Compute derivative of x squared.", raw_dataset_answer="2x"),
            CanonicalRecord(record_id="c", question="Integrate x squared over zero to one.", raw_dataset_answer="1/3"),
            CanonicalRecord(record_id="d", question="Compute the derivative of x squared exactly.", raw_dataset_answer="2x"),
        ]
        single = minhash_candidate_pairs(records, threshold=0.4, workers=1, num_perm=64, bands=8)
        streaming = minhash_candidate_pairs(records, threshold=0.4, workers=2, num_perm=64, bands=8)
        self.assertIsInstance(streaming, _me.VerifyResult)
        self.assertEqual(_edge_set(single), _edge_set(streaming))
        self.assertEqual(
            _edge_set(streaming),
            {(pair.left_id, pair.right_id) for pair in streaming.to_dedup_pairs()},
        )

    def test_band_index_csr_correctness(self) -> None:
        band_keys = _me.np.array([30, 10, 30, 20, 20], dtype=_me.np.uint64)
        sorted_keys, offsets, indices = _bi._build_csr_per_band(band_keys)
        self.assertEqual(sorted_keys.tolist(), [10, 20, 30])
        self.assertEqual(offsets.tolist(), [0, 1, 3, 5])
        start, end = _bi._lookup_csr_numba(sorted_keys, offsets, _me.np.uint64(20))
        self.assertEqual(indices[start:end].tolist(), [3, 4])
        missing_start, missing_end = _bi._lookup_csr_numba(sorted_keys, offsets, _me.np.uint64(999))
        self.assertEqual((missing_start, missing_end), (0, 0))

    def test_processor_order_independence(self) -> None:
        """FuzzyDedupProcessor.process() keeps the same records regardless of
        internal pair ordering."""
        from pipeline.ops.dedup.fuzzy import FuzzyDedupProcessor
        records = [
            CanonicalRecord(record_id="a", question="Compute the derivative of x squared.", raw_dataset_answer="2x"),
            CanonicalRecord(record_id="b", question="Compute derivative of x squared.", raw_dataset_answer="2x"),
            CanonicalRecord(record_id="c", question="Integrate x squared over zero to one.", raw_dataset_answer="1/3"),
        ]
        proc = FuzzyDedupProcessor(config={"threshold": 0.4, "workers": 1, "num_perm": 64, "bands": 8})
        result_single = proc.process(records)
        proc_multi = FuzzyDedupProcessor(config={"threshold": 0.4, "workers": 2, "num_perm": 64, "bands": 8})
        result_multi = proc_multi.process(records)
        kept_single = sorted(r.record_id for r in result_single.kept_records)
        kept_multi = sorted(r.record_id for r in result_multi.kept_records)
        self.assertEqual(kept_single, kept_multi)

    def test_materialize_output_enforces_unique_record_id(self) -> None:
        records = [
            CanonicalRecord(record_id="a", question="Q-a-first", raw_dataset_answer="1"),
            CanonicalRecord(record_id="a", question="Q-a-second", raw_dataset_answer="2"),
            CanonicalRecord(record_id="b", question="Q-b", raw_dataset_answer="3"),
        ]
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.parquet"
            output_path = Path(temp_dir) / "output.parquet"
            write_canonical_records(records, input_path)
            written = filter_canonical_records_to_output(
                input_path,
                output_path,
                keep_ids={"a", "b"},
                workers=1,
                unique_by_record_id=True,
            )
            output_records = read_canonical_records(output_path)
        self.assertEqual(written, 2)
        self.assertEqual([item.record_id for item in output_records], ["a", "b"])
        self.assertEqual(output_records[0].question, "Q-a-first")

    def test_materialize_output_is_consistent_across_workers(self) -> None:
        shard_a = [
            CanonicalRecord(record_id="x", question="x-1", raw_dataset_answer="1"),
            CanonicalRecord(record_id="y", question="y-1", raw_dataset_answer="1"),
        ]
        shard_b = [
            CanonicalRecord(record_id="x", question="x-2", raw_dataset_answer="2"),
            CanonicalRecord(record_id="z", question="z-1", raw_dataset_answer="1"),
        ]
        with TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_one = Path(temp_dir) / "out.one.parquet"
            output_many = Path(temp_dir) / "out.many.parquet"
            write_canonical_records(shard_a, input_dir / "part-000.parquet")
            write_canonical_records(shard_b, input_dir / "part-001.parquet")
            filter_canonical_records_to_output(
                input_dir,
                output_one,
                keep_ids={"x", "y", "z"},
                workers=1,
                unique_by_record_id=True,
            )
            filter_canonical_records_to_output(
                input_dir,
                output_many,
                keep_ids={"x", "y", "z"},
                workers=4,
                unique_by_record_id=True,
            )
            rows_one = read_canonical_records(output_one)
            rows_many = read_canonical_records(output_many)
        self.assertEqual(
            [(item.record_id, item.question) for item in rows_one],
            [(item.record_id, item.question) for item in rows_many],
        )

    # ------------------------------------------------------------------
    # Dataset priority tests
    # ------------------------------------------------------------------

    def test_priority_build_exact_match(self) -> None:
        from pipeline.utils.priority import build_record_priority_map
        records = [
            CanonicalRecord(record_id="r1", question="Q1", raw_dataset_answer="A", dataset_name="alpha"),
            CanonicalRecord(record_id="r2", question="Q2", raw_dataset_answer="A", dataset_name="beta"),
        ]
        tier_config = {1: ["alpha"], 2: ["beta"]}
        pm = build_record_priority_map(records, tier_config)
        self.assertEqual(pm["r1"], 1)
        self.assertEqual(pm["r2"], 2)

    def test_priority_build_prefix_match(self) -> None:
        from pipeline.utils.priority import build_record_priority_map
        records = [
            CanonicalRecord(record_id="r1", question="Q1", raw_dataset_answer="A", dataset_name="OpenThoughts-114k"),
            CanonicalRecord(record_id="r2", question="Q2", raw_dataset_answer="A", dataset_name="OpenThoughts-3-1.2M"),
        ]
        tier_config = {2: ["OpenThoughts"]}
        pm = build_record_priority_map(records, tier_config)
        self.assertEqual(pm["r1"], 2)
        self.assertEqual(pm["r2"], 2)

    def test_priority_build_unknown_raises(self) -> None:
        from pipeline.utils.priority import build_record_priority_map
        records = [
            CanonicalRecord(record_id="r1", question="Q1", raw_dataset_answer="A", dataset_name="unknown_ds"),
        ]
        with self.assertRaises(ValueError):
            build_record_priority_map(records, {1: ["alpha"]})

    def test_exact_dedup_respects_priority(self) -> None:
        """With priority, exact dedup should keep tier-1 record over tier-3."""
        records = [
            CanonicalRecord(record_id="zzz", question="Solve x+1=2", raw_dataset_answer="1", dataset_name="alpha"),
            CanonicalRecord(record_id="aaa", question="Solve x+1=2", raw_dataset_answer="1", dataset_name="beta"),
        ]
        kept_no_prio, _ = exact_deduplicate(records)
        self.assertEqual(kept_no_prio[0].record_id, "aaa")

        priority_map = {"zzz": 1, "aaa": 3}
        kept_prio, _ = exact_deduplicate(records, priority_map=priority_map)
        self.assertEqual(kept_prio[0].record_id, "zzz")

    def test_fuzzy_union_find_respects_priority(self) -> None:
        """Union-find should keep the higher-priority record."""
        from pipeline.ops.dedup.fuzzy import _union_find_drop_ids
        edges = [("zzz", "aaa")]
        dropped_no_prio = _union_find_drop_ids(edges)
        self.assertIn("zzz", dropped_no_prio)

        priority_map = {"zzz": 1, "aaa": 3}
        dropped_prio = _union_find_drop_ids(edges, priority_map=priority_map)
        self.assertIn("aaa", dropped_prio)
        self.assertNotIn("zzz", dropped_prio)

    def test_exact_union_find_respects_priority(self) -> None:
        from pipeline.ops.dedup.exact import _union_find_drop_ids
        edges = [("zzz", "aaa")]
        dropped_no_prio = _union_find_drop_ids(edges)
        self.assertIn("zzz", dropped_no_prio)

        priority_map = {"zzz": 1, "aaa": 3}
        dropped_prio = _union_find_drop_ids(edges, priority_map=priority_map)
        self.assertIn("aaa", dropped_prio)
        self.assertNotIn("zzz", dropped_prio)

    def test_fuzzy_processor_with_priority(self) -> None:
        """FuzzyDedupProcessor should keep tier-1 record when duplicates span tiers."""
        from pipeline.ops.dedup.fuzzy import FuzzyDedupProcessor
        records = [
            CanonicalRecord(record_id="a", question="Compute the derivative of x squared.", raw_dataset_answer="2x", dataset_name="alpha"),
            CanonicalRecord(record_id="b", question="Compute derivative of x squared.", raw_dataset_answer="2x", dataset_name="beta"),
            CanonicalRecord(record_id="c", question="Integrate x squared over zero to one.", raw_dataset_answer="1/3", dataset_name="alpha"),
        ]
        proc = FuzzyDedupProcessor(config={
            "threshold": 0.4, "workers": 1, "num_perm": 64, "bands": 8,
            "dataset_priority": {1: ["beta"], 2: ["alpha"]},
        })
        result = proc.process(records)
        kept_ids = {r.record_id for r in result.kept_records}
        self.assertIn("b", kept_ids)

    def test_parse_tier_config_none(self) -> None:
        from pipeline.utils.priority import parse_tier_config
        self.assertIsNone(parse_tier_config(None))
        self.assertIsNone(parse_tier_config({}))

    def test_parse_tier_config_string_keys(self) -> None:
        from pipeline.utils.priority import parse_tier_config
        result = parse_tier_config({"1": ["alpha"], "2": ["beta"]})
        self.assertEqual(result, {1: ["alpha"], 2: ["beta"]})


if __name__ == "__main__":
    unittest.main()
