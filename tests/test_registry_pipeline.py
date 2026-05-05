from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.core.checkpoint import Checkpoint
from pipeline.core.pipeline import run_pipeline
from pipeline.core.registry import list_processors
from pipeline.core.schema import CanonicalRecord


class RegistryTests(unittest.TestCase):
    def test_registry_contains_all_expected_processors(self) -> None:
        import pipeline.ops  # noqa: F401  -- triggers registration

        names = set(list_processors())
        expected = {
            "answer_judge",
            "text_cleaning",
            "language_filter",
            "length_filter",
            "mcq_converter",
            "rule_correctness",
            "math_filter",
            "cot_cleaning",
            "exact_dedup",
            "fuzzy_dedup",
            "dedup_judge",
            "rollout_correctness",
            "difficulty_annotation",
            "pass_ratio_filter",
        }
        for processor in expected:
            self.assertIn(processor, names, f"Missing registered processor: {processor}")


class PipelineTests(unittest.TestCase):
    def test_pipeline_runs_processor_sequence(self) -> None:
        records = [
            CanonicalRecord(record_id="a", question="What is 1+1?\nA. 1\nB. 2", ground_truth="B", dataset_name="demo"),
            CanonicalRecord(record_id="b", question="What is 1+1?\nA. 1\nB. 2", ground_truth="B", dataset_name="demo"),
        ]
        config = {
            "processors": [
                {"name": "text_cleaning", "enabled": True},
                {"name": "mcq_converter", "enabled": True},
                {"name": "exact_dedup", "enabled": True},
            ]
        }
        result = run_pipeline(records, config)
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0].ground_truth, "2")

    def test_disabled_processor_is_skipped(self) -> None:
        records = [
            CanonicalRecord(record_id="a", question="求解 x+1=2", ground_truth="1", dataset_name="demo"),
        ]
        config = {
            "processors": [
                {"name": "language_filter", "enabled": False},
            ]
        }
        result = run_pipeline(records, config)
        self.assertEqual(len(result.records), 1)

    def test_stages_key_compat_maps_to_processors(self) -> None:
        records = [
            CanonicalRecord(record_id="a", question="What is 1+1?\nA. 1\nB. 2", ground_truth="B", dataset_name="demo"),
        ]
        config = {
            "stages": [
                {"name": "stage1_non_cot", "enabled": True},
            ]
        }
        result = run_pipeline(records, config)
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0].ground_truth, "2")

    def test_stages_key_maps_stage0(self) -> None:
        records = [
            CanonicalRecord(record_id="a", question="Write a react component", ground_truth="code", dataset_name="demo"),
        ]
        config = {
            "stages": [
                {"name": "stage0_text_filter", "enabled": True},
            ]
        }
        result = run_pipeline(records, config)
        self.assertEqual(len(result.records), 0)


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_per_processor_granularity(self) -> None:
        records = [
            CanonicalRecord(record_id="a", question="What is 1+1?\nA. 1\nB. 2", ground_truth="B", dataset_name="demo"),
        ]
        config = {
            "processors": [
                {"name": "text_cleaning", "enabled": True},
                {"name": "mcq_converter", "enabled": True},
            ]
        }
        with TemporaryDirectory() as tmp:
            result = run_pipeline(records, config, checkpoint_dir=tmp)
            self.assertEqual(len(result.records), 1)

            ckpt = Checkpoint(tmp)
            self.assertTrue(ckpt.is_completed("text_cleaning"))
            self.assertTrue(ckpt.is_completed("mcq_converter"))

    def test_checkpoint_resume_skips_completed(self) -> None:
        records = [
            CanonicalRecord(record_id="a", question="What is 1+1?", ground_truth="2", dataset_name="demo"),
        ]
        config = {
            "processors": [
                {"name": "text_cleaning", "enabled": True},
                {"name": "language_filter", "enabled": True},
            ]
        }
        with TemporaryDirectory() as tmp:
            run_pipeline(records, config, checkpoint_dir=tmp)
            result2 = run_pipeline(records, config, checkpoint_dir=tmp)
            self.assertEqual(len(result2.records), 1)


class RemovedLegacyModulesTests(unittest.TestCase):
    def test_stage1_module_removed(self) -> None:
        import importlib

        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("pipeline.stages.stage1_non_cot")


if __name__ == "__main__":
    unittest.main()
