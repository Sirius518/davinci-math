from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.core.schema import CanonicalRecord


class RecordSchemaTests(unittest.TestCase):
    def test_generated_responses_round_trip(self) -> None:
        record = CanonicalRecord(
            record_id="r1",
            question="What is 1+1?",
            ground_truth="2",
            dataset_name="demo",
            source_name="demo",
        )
        record.add_generated_response(
            "correctness::sglang::demo-model",
            {"sample_index": 0, "response_text": "Final Answer: 2"},
        )
        restored = CanonicalRecord.from_storage_dict(record.to_storage_dict())
        self.assertIn("correctness::sglang::demo-model", restored.generated_responses)
        self.assertEqual(restored.generated_responses["correctness::sglang::demo-model"][0]["response_text"], "Final Answer: 2")


if __name__ == "__main__":
    unittest.main()
