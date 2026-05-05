from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.core.schema import CanonicalRecord
from pipeline.ops.evaluate.answer_judge import AnswerJudgeProcessor


PROMPT_PATH = ROOT / "configs" / "prompts" / "answer_judge.yaml"


def _record(
    *,
    record_id: str,
    majority_count: int,
    num_samples: int = 8,
    majority_matches_gt: bool = False,
    include_samples: bool = True,
    extra_verification: dict | None = None,
) -> CanonicalRecord:
    verification = {
        "raw_dataset_answer": "1/2",
        "majority_answer": "0.5",
        "majority_count": majority_count,
        "majority_matches_gt": majority_matches_gt,
        "num_samples": num_samples,
        "samples": [{"index": i, "answer": "0.5", "solution": "x", "reasoning": "r", "success": True, "stop_reason": "natural"} for i in range(num_samples)] if include_samples else [],
    }
    if extra_verification:
        verification.update(extra_verification)
    return CanonicalRecord(
        record_id=record_id,
        question="What is one half?",
        raw_dataset_answer="1/2",
        training_phase="posttrain",
        verification=verification,
    )


class AnswerJudgeTests(unittest.TestCase):
    def _processor(self) -> AnswerJudgeProcessor:
        return AnswerJudgeProcessor(
            {
                "prompt_path": str(PROMPT_PATH),
                "api_base": "http://localhost:8000/v1",
                "model": "demo-model",
                "target_phase": "posttrain",
            }
        )

    def test_clear_and_equivalent_keeps_posttrain(self) -> None:
        record = _record(record_id="r1", majority_count=5)
        processor = self._processor()
        with patch(
            "pipeline.ops.evaluate.answer_judge._judge_batch",
            return_value={"r1": ('{"gt_quality":"clear","judgement":"equivalent"}', "", 1)},
        ):
            result = processor.process([record])
        updated = result.kept_records[0]
        self.assertEqual(updated.training_phase, "posttrain")
        self.assertEqual(updated.filter_tag, "")
        self.assertTrue(updated.verification["majority_matches_gt"])
        self.assertEqual(updated.verification["llm_judge_gt_quality"], "clear")

    def test_clear_and_not_equivalent_routes_to_midtrain(self) -> None:
        record = _record(record_id="r2", majority_count=6)
        processor = self._processor()
        with patch(
            "pipeline.ops.evaluate.answer_judge._judge_batch",
            return_value={"r2": ('{"gt_quality":"clear","judgement":"not_equivalent"}', "", 2)},
        ):
            result = processor.process([record])
        updated = result.kept_records[0]
        self.assertEqual(updated.training_phase, "midtrain")
        self.assertEqual(updated.filter_tag, "answer_mismatch")
        self.assertFalse(updated.verification["majority_matches_gt"])

    def test_unclear_gt_with_confident_majority_routes_to_midtrain_with_mark(self) -> None:
        record = _record(record_id="r3", majority_count=5)
        processor = self._processor()
        with patch(
            "pipeline.ops.evaluate.answer_judge._judge_batch",
            return_value={"r3": ('{"gt_quality":"unclear","judgement":"not_applicable"}', "", 1)},
        ):
            result = processor.process([record])
        updated = result.kept_records[0]
        self.assertEqual(updated.training_phase, "midtrain")
        self.assertEqual(updated.filter_tag, "gt_unclear")

    def test_unclear_gt_with_unstable_majority_routes_to_midtrain(self) -> None:
        record = _record(record_id="r4", majority_count=4)
        processor = self._processor()
        with patch(
            "pipeline.ops.evaluate.answer_judge._judge_batch",
            return_value={"r4": ('{"gt_quality":"unclear","judgement":"not_applicable"}', "", 1)},
        ):
            result = processor.process([record])
        updated = result.kept_records[0]
        self.assertEqual(updated.training_phase, "midtrain")
        self.assertEqual(updated.filter_tag, "answer_uncertain")

    def test_missing_rollout_samples_routes_to_midtrain_without_api(self) -> None:
        record = _record(record_id="r5", majority_count=0, include_samples=False)
        processor = self._processor()
        with patch("pipeline.ops.evaluate.answer_judge._judge_batch") as mocked:
            result = processor.process([record])
        mocked.assert_not_called()
        updated = result.kept_records[0]
        self.assertEqual(updated.training_phase, "midtrain")
        self.assertEqual(updated.filter_tag, "answer_uncertain")


if __name__ == "__main__":
    unittest.main()
