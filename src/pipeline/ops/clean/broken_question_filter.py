from __future__ import annotations

import re

from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, ProcessorResult, RecordProcessor

_BROKEN_PATTERNS = [
    re.compile(r"\b(?:given the same setup as|compare with|as in) (?:problem|example)\s*\d", re.I),
    re.compile(r"\bassert(?:ion)? in problem\s*\d", re.I),
    re.compile(r"\(p\.\s*\d+\)", re.I),
    re.compile(r"^\s*(?:subject|example)\s+\d+\.\s*$", re.I | re.M),
]

_LATEX_RE = re.compile(r"\$[^$]*\$|\\[a-zA-Z]+(?:\{[^}]*\})*")
_VERB_RE = re.compile(
    r"\b(?:solve|find|determine|evaluate|prove|show|compute|calculate|"
    r"simplify|express|factor|expand|verify|derive|graph|sketch|list|"
    r"describe|explain|give|state|write|how many|what is|let)\b",
    re.I,
)


def _is_bare_expression(question: str) -> bool:
    stripped = _LATEX_RE.sub("", question).strip()
    if len(stripped) >= 30:
        return False
    return not _VERB_RE.search(stripped)


@register_processor("broken_question_filter")
class BrokenQuestionFilterProcessor(RecordProcessor):
    name = "broken_question_filter"

    def process(self, record: CanonicalRecord) -> ProcessorResult:
        if record.training_phase:
            record.add_trace(
                stage="clean",
                processor=self.name,
                status="skipped",
                details={"training_phase": record.training_phase},
            )
            return ProcessorResult(keep=True, record=record, stage="clean", processor=self.name)

        text = record.question
        for pattern in _BROKEN_PATTERNS:
            if pattern.search(text):
                updated = record.clone(training_phase="drop", filter_tag="broken_reference")
                updated.add_trace(
                    stage="clean",
                    processor=self.name,
                    status="dropped",
                    reason_code="broken_reference",
                    details={"matched_pattern": pattern.pattern[:80]},
                )
                return ProcessorResult(
                    keep=True,
                    record=updated,
                    stage="clean",
                    processor=self.name,
                    reason_code="broken_reference",
                )

        if _is_bare_expression(text):
            updated = record.clone(training_phase="drop", filter_tag="bare_expression")
            updated.add_trace(
                stage="clean",
                processor=self.name,
                status="dropped",
                reason_code="bare_expression",
                details={"question_length": len(text)},
            )
            return ProcessorResult(
                keep=True,
                record=updated,
                stage="clean",
                processor=self.name,
                reason_code="bare_expression",
            )

        record.add_trace(stage="clean", processor=self.name, status="kept")
        return ProcessorResult(keep=True, record=record, stage="clean", processor=self.name)
