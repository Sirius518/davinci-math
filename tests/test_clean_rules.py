from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.core.schema import CanonicalRecord
from pipeline.ops.clean.instruction_wrapper_cleaner import InstructionWrapperCleanerProcessor
from pipeline.ops.clean.language_filter import LanguageFilterProcessor
from pipeline.ops.clean.math_filter import MathFilterProcessor
from pipeline.ops.clean.mcq_converter import McqConverterProcessor
from pipeline.ops.clean.meta_task_filter import MetaTaskFilterProcessor
from pipeline.ops.clean.missing_context_filter import MissingContextFilterProcessor
from pipeline.ops.clean.prompt_contamination_filter import PromptContaminationFilterProcessor


def make_record(question: str, answer: str = "2") -> CanonicalRecord:
    return CanonicalRecord(
        record_id="demo",
        question=question,
        raw_dataset_answer=answer,
        confirmed_answer=answer,
        dataset_name="demo",
    )


class CleanRuleTests(unittest.TestCase):
    def test_instruction_wrapper_cleaner_strips_transcript_noise(self) -> None:
        processor = InstructionWrapperCleanerProcessor()
        record = make_record("User: Solve x+1=2\n</think>")
        result = processor.process(record)
        self.assertEqual(result.record.question, "Solve x+1=2")

    def test_prompt_contamination_filter_routes_answer_leak(self) -> None:
        processor = PromptContaminationFilterProcessor()
        record = make_record("Solve 1+1. Final Answer: 2", "2")
        result = processor.process(record)
        self.assertEqual(result.record.training_phase, "midtrain")
        self.assertEqual(result.record.filter_tag, "answer_leakage")

    def test_missing_context_filter_drops_undefined_target(self) -> None:
        processor = MissingContextFilterProcessor()
        record = make_record("Find g(10).")
        result = processor.process(record)
        self.assertEqual(result.record.training_phase, "drop")
        self.assertEqual(result.record.filter_tag, "missing_context")

    def test_meta_task_filter_routes_code_context_prompt(self) -> None:
        processor = MetaTaskFilterProcessor()
        record = make_record("Retrieve and repeat the exact described function from the code context.")
        result = processor.process(record)
        self.assertEqual(result.record.training_phase, "midtrain")
        self.assertEqual(result.record.filter_tag, "meta_task")

    def test_math_filter_drops_non_math_generation_prompt(self) -> None:
        processor = MathFilterProcessor()
        record = make_record("For my YouTube video, give me 50 tags and 50 movie titles.")
        result = processor.process(record)
        self.assertEqual(result.record.training_phase, "drop")
        self.assertEqual(result.record.filter_tag, "non_math_signal")

    def test_mcq_converter_routes_compact_options(self) -> None:
        processor = McqConverterProcessor()
        record = make_record("Which value is correct? (A) 1 (B) 2 (C) 3", "B")
        result = processor.process(record)
        self.assertEqual(result.record.training_phase, "midtrain")
        self.assertEqual(result.record.filter_tag, "boolean_and_mcq")

    def test_language_filter_drops_cjk_questions(self) -> None:
        processor = LanguageFilterProcessor({"cjk_threshold": 0.02, "other_threshold": 0.02})
        record = make_record("求解 x + 1 = 2")
        result = processor.process(record)
        self.assertEqual(result.record.training_phase, "drop")
        self.assertEqual(result.record.filter_tag, "chinese")


if __name__ == "__main__":
    unittest.main()
