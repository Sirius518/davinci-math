from __future__ import annotations

from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, ProcessorResult, RecordProcessor

_BOOLEAN_ANSWERS = {"true", "false", "yes", "no", "t", "f"}


@register_processor("boolean_answer_filter")
class BooleanAnswerFilterProcessor(RecordProcessor):
    name = "boolean_answer_filter"

    def process(self, record: CanonicalRecord) -> ProcessorResult:
        if record.training_phase:
            record.add_trace(
                stage="clean",
                processor=self.name,
                status="skipped",
                details={"training_phase": record.training_phase},
            )
            return ProcessorResult(keep=True, record=record, stage="clean", processor=self.name)

        answer = record.raw_dataset_answer.strip().lower()
        if answer in _BOOLEAN_ANSWERS:
            updated = record.clone(training_phase="midtrain", filter_tag="boolean_and_mcq")
            updated.add_trace(
                stage="clean",
                processor=self.name,
                status="routed",
                reason_code="boolean_and_mcq",
                details={"answer": record.raw_dataset_answer},
            )
            return ProcessorResult(
                keep=True,
                record=updated,
                stage="clean",
                processor=self.name,
                reason_code="boolean_and_mcq",
                details={"answer": record.raw_dataset_answer},
            )
        record.add_trace(stage="clean", processor=self.name, status="kept")
        return ProcessorResult(keep=True, record=record, stage="clean", processor=self.name)
