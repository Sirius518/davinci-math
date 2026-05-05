from __future__ import annotations

from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, ProcessorResult, RecordProcessor
from pipeline.utils.text import (
    INLINE_OPTION_PATTERN,
    LATEX_OPTION_PATTERN,
    THE_OPTIONS_ARE_PATTERN,
    WHICH_OF_PATTERN,
    WHICH_OF_THESE_PATTERN,
    extract_answer_letter,
    extract_options,
    is_unconvertible_mcq,
    looks_like_compact_options,
    strip_option_lines,
)


def _route_midtrain(
    record: CanonicalRecord,
    *,
    reason: str,
    details: dict[str, object] | None = None,
) -> ProcessorResult:
    updated = record.clone(training_phase="midtrain", filter_tag="boolean_and_mcq")
    payload = {"reason": reason}
    if details:
        payload.update(details)
    updated.add_trace(
        stage="clean",
        processor="convert_to_open_ended",
        status="routed",
        reason_code="mcq_conversion_failed",
        details=payload,
    )
    return ProcessorResult(
        keep=True,
        record=updated,
        stage="clean",
        processor="convert_to_open_ended",
        reason_code="mcq_conversion_failed",
        details=payload,
    )


def convert_to_open_ended(record: CanonicalRecord) -> ProcessorResult:
    if record.training_phase:
        record.add_trace(
            stage="clean",
            processor="convert_to_open_ended",
            status="skipped",
            details={"training_phase": record.training_phase},
        )
        return ProcessorResult(keep=True, record=record, stage="clean", processor="convert_to_open_ended")

    options = extract_options(record.question)
    if not options:
        q_text = record.question
        if INLINE_OPTION_PATTERN.search(q_text):
            return _route_midtrain(
                record,
                reason="unstructured_mcq_detected",
                details={"mcq_detected": True},
            )
        if LATEX_OPTION_PATTERN.search(q_text):
            return _route_midtrain(
                record,
                reason="latex_mcq_detected",
                details={"mcq_detected": True},
            )
        if looks_like_compact_options(q_text):
            return _route_midtrain(
                record,
                reason="compact_mcq_detected",
                details={"mcq_detected": True},
            )
        if THE_OPTIONS_ARE_PATTERN.search(q_text):
            return _route_midtrain(
                record,
                reason="options_header_detected",
                details={"mcq_detected": True},
            )
        if (WHICH_OF_PATTERN.search(q_text) or WHICH_OF_THESE_PATTERN.search(q_text)):
            return _route_midtrain(
                record,
                reason="which_of_without_options",
                details={"mcq_detected": True},
            )
        record.add_trace(stage="clean", processor="convert_to_open_ended", status="kept", details={"mcq_detected": False})
        return ProcessorResult(
            keep=True,
            record=record,
            stage="clean",
            processor="convert_to_open_ended",
            details={"mcq_detected": False},
        )

    answer_letter = extract_answer_letter(record.raw_dataset_answer)
    if not answer_letter:
        return _route_midtrain(
            record,
            reason="answer_letter_missing",
            details={"mcq_detected": True},
        )

    option_value = options.get(answer_letter)
    if not option_value:
        return _route_midtrain(
            record,
            reason="answer_letter_not_found_in_options",
            details={"answer_letter": answer_letter},
        )

    is_unconvertible, reason = is_unconvertible_mcq(record.question, option_value)
    if is_unconvertible:
        return _route_midtrain(
            record,
            reason=reason,
            details={"answer_letter": answer_letter},
        )

    converted_question = strip_option_lines(record.question)
    converted_answer = option_value.strip()
    if len(converted_question) < 10:
        return _route_midtrain(
            record,
            reason="question_too_short_after_strip",
            details={"length": len(converted_question), "answer_letter": answer_letter},
        )
    if not converted_answer:
        return _route_midtrain(
            record,
            reason="empty_option_value",
            details={"answer_letter": answer_letter},
        )

    updated = record.clone(
        question=converted_question,
        raw_dataset_answer=converted_answer,
        confirmed_answer=converted_answer,
    )
    updated.add_trace(
        stage="clean",
        processor="convert_to_open_ended",
        status="kept",
        details={
            "answer_letter": answer_letter,
            "converted": True,
            "question_stripped": True,
            "converted_question_length": len(converted_question),
        },
    )
    return ProcessorResult(
        keep=True,
        record=updated,
        stage="clean",
        processor="convert_to_open_ended",
        details={
            "answer_letter": answer_letter,
            "converted": True,
            "question_stripped": True,
            "converted_question_length": len(converted_question),
        },
    )


@register_processor("mcq_converter")
class McqConverterProcessor(RecordProcessor):
    name = "mcq_converter"

    def process(self, record: CanonicalRecord) -> ProcessorResult:
        return convert_to_open_ended(record)
