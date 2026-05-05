from __future__ import annotations

from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, ProcessorResult, RecordProcessor
from pipeline.utils.text import should_filter_short_question


def filter_short_question_text(record: CanonicalRecord) -> ProcessorResult:
    if should_filter_short_question(record.question):
        updated = record.clone(training_phase="drop", filter_tag="short_question")
        updated.add_trace(
            stage="clean",
            processor="filter_short_question_text",
            status="dropped",
            reason_code="short_question",
        )
        return ProcessorResult(
            keep=True,
            record=updated,
            stage="clean",
            processor="filter_short_question_text",
            reason_code="short_question",
        )
    record.add_trace(stage="clean", processor="filter_short_question_text", status="kept")
    return ProcessorResult(keep=True, record=record, stage="clean", processor="filter_short_question_text")


@register_processor("length_filter")
class LengthFilterProcessor(RecordProcessor):
    name = "length_filter"

    def process(self, record: CanonicalRecord) -> ProcessorResult:
        return filter_short_question_text(record)
