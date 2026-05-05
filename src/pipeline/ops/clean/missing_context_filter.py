from __future__ import annotations

import re

from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, ProcessorResult, RecordProcessor

_MISSING_CONTEXT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(?:as|shown|described|defined|given|proved|derived|stated|illustrated|indicated|depicted|demonstrated)\s+"
            r"(?:above|below|earlier|previously|before|after|in the (?:figure|table|chart|graph|diagram|image|picture|photo|passage|text|paragraph|section|chapter))\b",
            re.I,
        ),
        "relative_context_reference",
    ),
    (
        re.compile(
            r"\b(?:from|using|based on|according to|refer(?:ring)? to)\s+(?:the\s+)?"
            r"(?:conversation|chat|dialogue|code context|context above|previous answer|"
            r"passage|article|text|table|chart|graph|diagram|figure|image|picture|"
            r"data set|dataset|spreadsheet|document|source|excerpt|reading|map|"
            r"information (?:above|below|given|provided)|results? (?:above|below|shown))\b",
            re.I,
        ),
        "external_context_reference",
    ),
    (
        re.compile(
            r"\b(?:same|following)\s+(?:game|setup|construction|figure|diagram|problem|example|context|scenario|conditions?)\s+"
            r"(?:as|described|given|above|below)\b",
            re.I,
        ),
        "same_setup_reference",
    ),
    (
        re.compile(
            r"\b(?:problem|example|exercise|question|theorem|lemma|corollary|chapter|section|"
            r"proposition|definition|remark|figure|table|equation|page)\s+"
            r"\d+(?:\.\d+)*\b",
            re.I,
        ),
        "external_problem_reference",
    ),
    (
        re.compile(
            r"\b(?:see|refer to|using|look at|consult|check)\s+(?:the )?"
            r"(?:attachment|attached (?:image|figure|file|document|diagram|table|chart|spreadsheet|pdf)|"
            r"(?:image|figure|table|chart|graph|diagram|picture|photo|map) (?:below|above|shown|provided|attached)|"
            r"accompanying (?:figure|table|chart|image|diagram|document)|"
            r"(?:following|above|below) (?:figure|table|chart|image|diagram|graph))\b",
            re.I,
        ),
        "attachment_reference",
    ),
    # references to visual content that's not present
    (
        re.compile(
            r"\b(?:in|from|on) (?:the |this )?(?:figure|table|chart|graph|diagram|image|picture|photo|map|screenshot)\b",
            re.I,
        ),
        "visual_reference",
    ),
    # "the following data/table/information" without actual content
    (
        re.compile(
            r"\b(?:the |this )?following (?:data|table|information|passage|text|excerpt|code|"
            r"figure|image|diagram|chart|graph|results?|output|content|list|set of)\b"
            r"(?!.*[\n].*[\n].*[\n])",
            re.I,
        ),
        "dangling_following_reference",
    ),
    # "use the data in Table X" / "refer to Figure X"
    (
        re.compile(
            r"\b(?:use|refer to|see|look at|consider|examine|inspect|study|analyze)\s+(?:the )?"
            r"(?:Table|Figure|Fig\.|Chart|Graph|Diagram|Exhibit|Appendix)\s*(?:\d+|[A-Z](?:\.\d+)?)\b",
            re.I,
        ),
        "numbered_visual_reference",
    ),
]

_UNDERSPECIFIED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"^\s*(?:find|compute|evaluate|determine)\s+"
            r"[A-Za-z]\w*\s*\(\s*[-+]?\d+(?:\.\d+)?\s*\)\s*[.?!]?\s*$",
            re.I,
        ),
        "undefined_symbol_target",
    ),
    (
        re.compile(
            r"^\s*(?:find|determine|compute|evaluate)\s+the\s+"
            r"(?:value|answer|result)\s*[.?!]?\s*$",
            re.I,
        ),
        "missing_target",
    ),
    # extremely short questions (< 15 chars) that lack context
    (
        re.compile(r"^.{1,14}$", re.S),
        "too_short",
    ),
    # questions that are just a number, variable, or expression with no instruction
    (
        re.compile(r"^\s*[\d\s+\-*/^=().]+\s*$"),
        "bare_expression",
    ),
    # "What is the answer?" without any problem
    (
        re.compile(
            r"^\s*(?:what is|what's|find|give|tell me|state) (?:the )?(?:answer|result|solution|value|output)\s*\??\s*$",
            re.I,
        ),
        "missing_problem",
    ),
]

_BROKEN_FORMAT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"</think>|<think>|<assistant>|<user>|<system>", re.I), "transcript_tag"),
    (re.compile(r"^```|```$", re.M), "code_fence_residue"),
    (re.compile(r"^diff --git\b|^deleted file mode\b|^\+\+\+ |^--- |^@@ ", re.M), "diff_residue"),
    (re.compile(r"!\[\]\($|!\[[^\]]*\]\($", re.M), "truncated_image_markup"),
    # HTML artifacts
    (re.compile(r"<(?:div|span|table|tr|td|th|img|script|style|link|meta|head|body|html)\b", re.I), "html_artifact"),
    # base64 data
    (re.compile(r"data:image/[a-z]+;base64,", re.I), "base64_artifact"),
    # excessive Unicode garbage (>10% non-printable non-whitespace)
    (re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]{5,}"), "control_char_garbage"),
    # escaped Unicode sequences as literal text
    (re.compile(r"(?:\\u[0-9a-fA-F]{4}){3,}"), "escaped_unicode_garbage"),
    # LaTeX compile errors
    (re.compile(r"(?:Undefined control sequence|Missing \$ inserted|Extra \}|Runaway argument)", re.I), "latex_error_artifact"),
]


def detect_missing_context_reason(question: str) -> tuple[str, str]:
    text = question.strip()
    for pattern, reason in _MISSING_CONTEXT_PATTERNS:
        if pattern.search(text):
            return "missing_context", reason
    for pattern, reason in _UNDERSPECIFIED_PATTERNS:
        if pattern.search(text):
            return "missing_context", reason
    for pattern, reason in _BROKEN_FORMAT_PATTERNS:
        if pattern.search(text):
            return "broken_format", reason
    if text.count("(") != text.count(")") and len(text) < 400:
        return "broken_format", "unbalanced_parentheses"
    if text.count("[") != text.count("]") and len(text) < 400:
        return "broken_format", "unbalanced_brackets"
    if text.count("{") != text.count("}") and len(text) < 400:
        return "broken_format", "unbalanced_braces"
    return "", ""


@register_processor("missing_context_filter")
class MissingContextFilterProcessor(RecordProcessor):
    name = "missing_context_filter"

    def process(self, record: CanonicalRecord) -> ProcessorResult:
        if record.training_phase:
            record.add_trace(
                stage="clean",
                processor=self.name,
                status="skipped",
                details={"training_phase": record.training_phase},
            )
            return ProcessorResult(keep=True, record=record, stage="clean", processor=self.name)

        filter_tag, reason = detect_missing_context_reason(record.question)
        if filter_tag:
            updated = record.clone(training_phase="drop", filter_tag=filter_tag)
            updated.add_trace(
                stage="clean",
                processor=self.name,
                status="dropped",
                reason_code=reason,
                details={"filter_tag": filter_tag},
            )
            return ProcessorResult(
                keep=True,
                record=updated,
                stage="clean",
                processor=self.name,
                reason_code=reason,
                details={"filter_tag": filter_tag},
            )

        record.add_trace(stage="clean", processor=self.name, status="kept")
        return ProcessorResult(keep=True, record=record, stage="clean", processor=self.name)
