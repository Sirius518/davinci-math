from __future__ import annotations

from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, ProcessorResult, RecordProcessor
from pipeline.utils.text import normalize_text


@register_processor("text_normalize")
class TextNormalizeProcessor(RecordProcessor):
    name = "text_normalize"

    def process(self, record: CanonicalRecord) -> ProcessorResult:
        normalize_punctuation = bool(self.config.get("normalize_punctuation", True))
        normalized_question = normalize_text(record.question, normalize_punctuation=normalize_punctuation)
        normalized_raw_answer = normalize_text(
            record.raw_dataset_answer, normalize_punctuation=normalize_punctuation
        )
        normalized_confirmed_answer = normalize_text(
            record.confirmed_answer, normalize_punctuation=normalize_punctuation
        )

        changed = (
            normalized_question != record.question
            or normalized_raw_answer != record.raw_dataset_answer
            or normalized_confirmed_answer != record.confirmed_answer
        )

        if changed:
            updated = record.clone(
                question=normalized_question,
                raw_dataset_answer=normalized_raw_answer,
                confirmed_answer=normalized_confirmed_answer,
            )
            updated.add_trace(
                stage="clean",
                processor=self.name,
                status="kept",
                details={"normalized": True, "normalize_punctuation": normalize_punctuation},
            )
            return ProcessorResult(keep=True, record=updated, stage="clean", processor=self.name)

        record.add_trace(
            stage="clean",
            processor=self.name,
            status="kept",
            details={"normalized": False, "normalize_punctuation": normalize_punctuation},
        )
        return ProcessorResult(keep=True, record=record, stage="clean", processor=self.name)
