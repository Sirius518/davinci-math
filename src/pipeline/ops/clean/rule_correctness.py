from __future__ import annotations

import importlib
from functools import lru_cache

from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, ProcessorResult, RecordProcessor


@lru_cache(maxsize=1)
def _load_math_verify_module():
    return importlib.import_module("math_verify")


def _math_verify_equivalent(left: str, right: str) -> bool:
    module = _load_math_verify_module()
    for attribute in ("verify", "is_equivalent", "math_equal"):
        func = getattr(module, attribute, None)
        if callable(func):
            try:
                return bool(func(left, right))
            except TypeError:
                continue
    raise RuntimeError("math_verify does not expose verify/is_equivalent/math_equal")


def rule_based_correctness(record: CanonicalRecord) -> ProcessorResult:
    candidate = record.confirmed_answer or record.raw_dataset_answer
    if not _math_verify_equivalent(candidate, record.raw_dataset_answer):
        record.add_trace(
            stage="clean",
            processor="rule_based_correctness",
            status="dropped",
            reason_code="rule_verify_failed",
        )
        return ProcessorResult(
            keep=False,
            record=record,
            stage="clean",
            processor="rule_based_correctness",
            reason_code="rule_verify_failed",
        )
    record.add_trace(stage="clean", processor="rule_based_correctness", status="kept")
    return ProcessorResult(keep=True, record=record, stage="clean", processor="rule_based_correctness")


@register_processor("rule_correctness")
class RuleCorrectnessProcessor(RecordProcessor):
    name = "rule_correctness"

    def process(self, record: CanonicalRecord) -> ProcessorResult:
        return rule_based_correctness(record)
