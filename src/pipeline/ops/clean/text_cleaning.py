from __future__ import annotations

from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, ProcessorResult, RecordProcessor
from pipeline.utils.text import clean_urls_and_images, format_text


def _result(record: CanonicalRecord, *, processor: str) -> ProcessorResult:
    record.add_trace(stage="clean", processor=processor, status="kept")
    return ProcessorResult(keep=True, record=record, stage="clean", processor=processor)


def clean_urls_and_images_from_question(record: CanonicalRecord) -> ProcessorResult:
    updated = record.clone(question=clean_urls_and_images(record.question))
    return _result(updated, processor="clean_urls_and_images_from_question")


def format_question_text(record: CanonicalRecord) -> ProcessorResult:
    updated = record.clone(question=format_text(record.question))
    return _result(updated, processor="format_question_text")


@register_processor("text_cleaning")
class TextCleaningProcessor(RecordProcessor):
    name = "text_cleaning"

    def process(self, record: CanonicalRecord) -> ProcessorResult:
        cleaned = clean_urls_and_images_from_question(record).record
        return format_question_text(cleaned)
