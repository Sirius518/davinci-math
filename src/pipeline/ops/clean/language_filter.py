from __future__ import annotations

from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, ProcessorResult, RecordProcessor
from pipeline.utils.text import analyze_script_profile

_DEFAULT_OTHER_THRESHOLD = 0.1
_DEFAULT_CJK_THRESHOLD = 0.1


def remove_non_english_chars_from_question(
    record: CanonicalRecord,
    *,
    other_threshold: float = _DEFAULT_OTHER_THRESHOLD,
    cjk_threshold: float = _DEFAULT_CJK_THRESHOLD,
) -> ProcessorResult:
    profile = analyze_script_profile(record.question)
    details = {
        "total_letters": profile.total_letters,
        "latin_ratio": round(profile.latin_ratio, 4),
        "cjk_ratio": round(profile.cjk_ratio, 4),
        "other_ratio": round(profile.other_ratio, 4),
    }

    if profile.other_ratio >= other_threshold:
        updated = record.clone(training_phase="drop", filter_tag="other_non_english")
        updated.add_trace(
            stage="clean",
            processor="remove_non_english_chars_from_question",
            status="dropped",
            reason_code="other_non_english_question",
            details=details,
        )
        return ProcessorResult(
            keep=True,
            record=updated,
            stage="clean",
            processor="remove_non_english_chars_from_question",
            reason_code="other_non_english_question",
        )

    if profile.cjk_ratio >= cjk_threshold:
        updated = record.clone(training_phase="midtrain", filter_tag="chinese")
        updated.add_trace(
            stage="clean",
            processor="remove_non_english_chars_from_question",
            status="midtrain",
            reason_code="chinese_question",
            details=details,
        )
        return ProcessorResult(
            keep=True,
            record=updated,
            stage="clean",
            processor="remove_non_english_chars_from_question",
            reason_code="chinese_question",
        )

    record.add_trace(
        stage="clean",
        processor="remove_non_english_chars_from_question",
        status="kept",
        details=details,
    )
    return ProcessorResult(keep=True, record=record, stage="clean", processor="remove_non_english_chars_from_question")


@register_processor("language_filter")
class LanguageFilterProcessor(RecordProcessor):
    name = "language_filter"

    def process(self, record: CanonicalRecord) -> ProcessorResult:
        if record.training_phase:
            record.add_trace(
                stage="clean",
                processor=self.name,
                status="skipped",
                details={"training_phase": record.training_phase},
            )
            return ProcessorResult(keep=True, record=record, stage="clean", processor=self.name)
        other_threshold = float(self.config.get("other_threshold", _DEFAULT_OTHER_THRESHOLD))
        cjk_threshold = float(self.config.get("cjk_threshold", _DEFAULT_CJK_THRESHOLD))
        return remove_non_english_chars_from_question(
            record, other_threshold=other_threshold, cjk_threshold=cjk_threshold,
        )
