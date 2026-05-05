from __future__ import annotations

import re

from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, ProcessorResult, RecordProcessor

_PROOF_KEYWORD_PATTERN = re.compile(r"\b(?:proof|prove|qed|shown|demonstrated|established)\b", re.IGNORECASE)
_PROOF_BLOCK_PATTERN = re.compile(r"\\begin\{proof\}|\\end\{proof\}", re.IGNORECASE)
_PROVE_PROMPT_PATTERN = re.compile(r"^\s*(?:prove|show|demonstrate)\s+that\b", re.IGNORECASE)
_MATH_SYMBOL_PATTERN = re.compile(r"\$|\\[A-Za-z]+|[0-9]|[=+\-*/^<>]|≤|≥|∑|∫|√")
_SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]")


def _is_non_verifiable_answer(question: str, answer: str) -> tuple[bool, str]:
    normalized_answer = answer.strip()
    if not normalized_answer:
        return False, ""
    sentence_count = len(_SENTENCE_SPLIT_PATTERN.findall(normalized_answer))
    has_math_symbol = bool(_MATH_SYMBOL_PATTERN.search(normalized_answer))

    if _PROOF_BLOCK_PATTERN.search(normalized_answer):
        return True, "proof_block"
    if _PROVE_PROMPT_PATTERN.search(question) and len(normalized_answer) > 200:
        return True, "proof_task_long_answer"
    if _PROOF_KEYWORD_PATTERN.search(normalized_answer) and (len(normalized_answer) > 120 or sentence_count > 1):
        return True, "proof_keyword_answer"
    if len(normalized_answer) > 500 and not has_math_symbol:
        return True, "long_plain_text_answer"
    if sentence_count > 3 and not has_math_symbol:
        return True, "multi_sentence_plain_text_answer"
    return False, ""


@register_processor("answer_quality_filter")
class AnswerQualityFilterProcessor(RecordProcessor):
    name = "answer_quality_filter"

    def process(self, record: CanonicalRecord) -> ProcessorResult:
        if record.training_phase:
            record.add_trace(
                stage="clean",
                processor=self.name,
                status="skipped",
                details={"training_phase": record.training_phase},
            )
            return ProcessorResult(keep=True, record=record, stage="clean", processor=self.name)

        flagged, reason = _is_non_verifiable_answer(record.question, record.raw_dataset_answer)
        if flagged:
            updated = record.clone(training_phase="midtrain", filter_tag="non_verifiable_answer")
            updated.add_trace(
                stage="clean",
                processor=self.name,
                status="routed",
                reason_code="non_verifiable_answer",
                details={"reason": reason},
            )
            return ProcessorResult(
                keep=True,
                record=updated,
                stage="clean",
                processor=self.name,
                reason_code="non_verifiable_answer",
                details={"reason": reason},
            )
        record.add_trace(stage="clean", processor=self.name, status="kept")
        return ProcessorResult(keep=True, record=record, stage="clean", processor=self.name)
