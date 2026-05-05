from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.core.pipeline import run_pipeline
from pipeline.core.schema import CanonicalRecord
from pipeline.ops.clean.cot_cleaning import CotCleaningProcessor
from pipeline.ops.clean.language_filter import LanguageFilterProcessor
from pipeline.ops.clean.length_filter import LengthFilterProcessor
from pipeline.ops.clean.mcq_converter import McqConverterProcessor
from pipeline.ops.clean.rule_correctness import RuleCorrectnessProcessor
from pipeline.ops.clean.text_cleaning import TextCleaningProcessor

DEFAULT_PROCESSORS = (
    TextCleaningProcessor(),
    LanguageFilterProcessor(),
    LengthFilterProcessor(),
    McqConverterProcessor(),
    CotCleaningProcessor(),
    RuleCorrectnessProcessor(),
)


def run_stage1(
    records: list[CanonicalRecord],
    *,
    processors: tuple[object, ...] | list[object] | None = None,
):
    active = processors if processors is not None else DEFAULT_PROCESSORS
    config = {
        "processors": [{"name": processor.name} for processor in active if getattr(processor, "name", "")]
    }
    return run_pipeline(records, config)


class Stage1Tests(unittest.TestCase):
    def test_filters_non_english_question(self) -> None:
        record = CanonicalRecord(
            record_id="1",
            question="求解 x + 1 = 2",
            ground_truth="1",
            dataset_name="demo",
            source_name="demo",
        )
        result = run_stage1([record])
        self.assertEqual(result.records, [])

    def test_converts_multiple_choice_answer(self) -> None:
        record = CanonicalRecord(
            record_id="2",
            question="What is 1+1?\nA. 1\nB. 2\nC. 3\nD. 4",
            ground_truth="B",
            dataset_name="demo",
            source_name="demo",
        )
        result = run_stage1([record])
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0].ground_truth, "2")

    def test_first_five_processors_skip_rule_based_correctness(self) -> None:
        record = CanonicalRecord(
            record_id="3",
            question="Please compute the value of 1+1 and provide the final numeric answer.",
            ground_truth="2",
            final_answer="3",
            dataset_name="demo",
            source_name="demo",
        )
        full_result = run_stage1([record])
        first_five_result = run_stage1([record], processors=DEFAULT_PROCESSORS[:5])
        self.assertEqual(len(full_result.records), 0)
        self.assertEqual(len(first_five_result.records), 1)


if __name__ == "__main__":
    unittest.main()
