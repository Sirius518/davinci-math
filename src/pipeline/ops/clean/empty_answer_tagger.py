"""Tag records with empty ground-truth answer as midtrain."""
from __future__ import annotations

from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, ProcessorResult, RecordProcessor


@register_processor("empty_answer_tagger")
class EmptyAnswerTaggerProcessor(RecordProcessor):
    name = "empty_answer_tagger"

    def process(self, record: CanonicalRecord) -> ProcessorResult:
        if record.training_phase:
            return ProcessorResult(keep=True, record=record, stage="clean", processor=self.name)

        if not record.raw_dataset_answer.strip():
            updated = record.clone(training_phase="midtrain", filter_tag="empty_ground_truth")
            updated.add_trace(
                stage="clean",
                processor=self.name,
                status="routed",
                reason_code="empty_ground_truth",
            )
            return ProcessorResult(
                keep=True,
                record=updated,
                stage="clean",
                processor=self.name,
                reason_code="empty_ground_truth",
            )

        record.add_trace(stage="clean", processor=self.name, status="kept")
        return ProcessorResult(keep=True, record=record, stage="clean", processor=self.name)
