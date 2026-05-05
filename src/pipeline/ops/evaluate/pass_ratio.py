from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, DatasetProcessor, DatasetProcessorResult


@dataclass(slots=True)
class PassRatioFilterConfig:
    min_ratio: float = 0.0
    max_ratio: float = 1.0


@dataclass(slots=True)
class PassRatioOutput:
    kept_records: list[CanonicalRecord]
    dropped_records: list[CanonicalRecord]


def filter_by_pass_ratio(records: list[CanonicalRecord], config: PassRatioFilterConfig) -> PassRatioOutput:
    kept: list[CanonicalRecord] = []
    dropped: list[CanonicalRecord] = []
    for record in records:
        pass_ratio = float(dict(record.verification).get("pass_ratio", 0.0))
        if pass_ratio < config.min_ratio or pass_ratio > config.max_ratio:
            reason_code = "pass_ratio_too_low" if pass_ratio < config.min_ratio else "pass_ratio_too_high"
            filter_tag = "too_hard" if pass_ratio < config.min_ratio else "too_easy"
            updated = record.clone(training_phase="drop", filter_tag=filter_tag)
            updated.add_trace(
                stage="evaluate",
                processor="pass_ratio_filter",
                status="dropped",
                reason_code=reason_code,
                details={"pass_ratio": pass_ratio},
            )
            dropped.append(updated)
            continue
        all_correct = bool(dict(record.verification).get("all_correct", False))
        if all_correct:
            kept.append(record.clone(training_phase="midtrain", filter_tag="all_correct_easy"))
        else:
            kept.append(record.clone(training_phase=record.training_phase or "posttrain", filter_tag=record.filter_tag))
    return PassRatioOutput(kept_records=kept, dropped_records=dropped)


@register_processor("pass_ratio_filter")
class PassRatioFilterProcessor(DatasetProcessor):
    name = "pass_ratio_filter"

    def process(self, records: list[CanonicalRecord], *, pipeline_artifacts: dict[str, Any] | None = None) -> DatasetProcessorResult:
        config = PassRatioFilterConfig(
            min_ratio=float(self.config.get("min_ratio", 0.0)),
            max_ratio=float(self.config.get("max_ratio", 1.0)),
        )
        output = filter_by_pass_ratio(records, config)
        return DatasetProcessorResult(kept_records=output.kept_records, artifacts={"dropped_records": output.dropped_records})
