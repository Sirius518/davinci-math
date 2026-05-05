from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.core.io import read_canonical_records
from pipeline.sources.base import SourceConfig
from pipeline.sources.same_format import SameFormatAdapter


class SameFormatAdapterTests(unittest.TestCase):
    def test_maps_same_format_to_canonical_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "same_format.parquet"
            output_dir = tmp_path / "output"

            table = pa.Table.from_pylist(
                [
                    {
                        "question": "Solve x+1=2",
                        "uid": "a",
                        "ground_truth": "1",
                        "dataset_name": "demo",
                        "meta_data": {"cot_solution": [{"model": "m1", "solution": "x=1"}]},
                    },
                    {
                        "question": "Solve x+2=4",
                        "uid": "b",
                        "ground_truth": "2",
                        "dataset_name": "demo",
                        "meta_data": {"source": "unit-test"},
                    },
                ]
            )
            pq.write_table(table, input_path)

            config = SourceConfig(
                name="same_format",
                dataset_name="",
                input_path=str(input_path),
                output_path=str(output_dir),
                options={"target_shard_size_mb": 0, "shard_prefix": "demo"},
            )
            summary = SameFormatAdapter(config).write()

            self.assertEqual(summary.record_count, 2)
            self.assertEqual(len(summary.output_paths), 2)

            records = read_canonical_records(output_dir)
            self.assertEqual(len(records), 2)
            self.assertFalse(records[0].has_cot)
            self.assertNotIn("source_meta", records[0].meta)
            self.assertFalse(records[0].meta["_ingest"]["meta_data_loaded"])


if __name__ == "__main__":
    unittest.main()
