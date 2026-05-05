from __future__ import annotations

from dataclasses import dataclass

from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, ProcessorResult, RecordProcessor
from pipeline.utils.text import clean_urls_and_images, contains_non_english_script, format_text


@dataclass(slots=True)
class CotProcessingOutput:
    records: list[CanonicalRecord]


def process_cot(records: list[CanonicalRecord]) -> CotProcessingOutput:
    processor = CotCleaningProcessor()
    output: list[CanonicalRecord] = []
    for record in records:
        output.append(processor.process(record).record)
    return CotProcessingOutput(records=output)


@register_processor("cot_cleaning")
class CotCleaningProcessor(RecordProcessor):
    name = "cot_cleaning"

    def process(self, record: CanonicalRecord) -> ProcessorResult:
        if not record.distillation:
            record.add_trace(stage="clean", processor=self.name, status="kept")
            return ProcessorResult(keep=True, record=record, stage="clean", processor=self.name)
        cleaned_distillation: dict[str, dict[str, object]] = {}
        for model_name, payload in record.distillation.items():
            item = dict(payload or {})
            solution = format_text(clean_urls_and_images(str(item.get("solution", ""))))
            reasoning = format_text(clean_urls_and_images(str(item.get("reasoning", ""))))
            quality_tags = [str(tag) for tag in item.get("quality_tags", []) if str(tag).strip()]
            combined_text = "\n".join([solution, reasoning]).strip()
            if combined_text and contains_non_english_script(combined_text):
                if "non_english" not in quality_tags:
                    quality_tags.append("non_english")
                record.add_trace(
                    stage="clean",
                    processor="remove_non_english_chars_from_cot",
                    status="flagged",
                    reason_code="cot_non_english",
                    details={"model": str(model_name)},
                )
            is_gpt_oss = "gpt-oss" in str(model_name).lower()
            truncated = bool((is_gpt_oss and not solution.strip()) or (not is_gpt_oss and reasoning.strip() and "</think>" not in reasoning))
            if truncated:
                if "truncated" not in quality_tags:
                    quality_tags.append("truncated")
                record.add_trace(
                    stage="clean",
                    processor="truncation_detection",
                    status="flagged",
                    reason_code="cot_truncated",
                    details={"model": str(model_name)},
                )
            cleaned_distillation[str(model_name)] = {
                "model": str(item.get("model", model_name)),
                "solution": solution,
                "reasoning": reasoning,
                "quality_status": "invalid" if quality_tags else "valid",
                "quality_tags": quality_tags,
            }
        updated = record.clone(distillation=cleaned_distillation)
        updated.add_trace(stage="clean", processor=self.name, status="kept")
        return ProcessorResult(keep=True, record=updated, stage="clean", processor=self.name)
