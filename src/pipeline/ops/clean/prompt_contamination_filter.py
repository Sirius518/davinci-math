from __future__ import annotations

import re

from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, ProcessorResult, RecordProcessor
from pipeline.utils.text import normalize_answer_text

_PROMPT_CONTAMINATION_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\bsolution\s*:", re.I), "prompt_contamination", "solution_marker"),
    (re.compile(r"\bfinal answer\s*:", re.I), "answer_leakage", "final_answer_marker"),
    (re.compile(r"\bthe answer is\s*\(?[A-F0-9\\]", re.I), "answer_leakage", "direct_answer_statement"),
    (re.compile(r"</think>|<think>", re.I), "prompt_contamination", "reasoning_tag"),
    (re.compile(r"\b(?:assistant|user|system)\s*:", re.I), "prompt_contamination", "chat_role_marker"),
    (re.compile(r"\\boxed\{[^}]{1,120}\}"), "answer_leakage", "boxed_answer"),
    (re.compile(r"\b(?:therefore|thus|hence)\b.{0,80}\b(?:answer|result|value)\b", re.I), "answer_leakage", "conclusion_phrase"),
    # step-by-step reasoning traces leaked into question
    (re.compile(r"\bstep\s+\d+\s*:", re.I), "prompt_contamination", "step_marker"),
    (re.compile(r"\b(?:first|second|third|fourth|fifth|next|finally|lastly),?\s+(?:we |I |let's |let us )?(?:find|compute|calculate|determine|note|observe|solve|simplify|use|apply)\b", re.I), "prompt_contamination", "reasoning_trace"),
    # answer embedded in question phrasing
    (re.compile(r"\b(?:the |our )?(?:answer|result|solution) (?:is|equals?|=)\s*(?:\$|\\|\d)", re.I), "answer_leakage", "answer_equals"),
    (re.compile(r"\b(?:so|thus|hence|therefore|accordingly)\s+(?:the )?(?:answer|result|value|probability|sum|area|volume)\s+(?:is|=|equals)\b", re.I), "answer_leakage", "derived_answer"),
    # chain-of-thought markers
    (re.compile(r"<\|(?:startoftext|endoftext|pad|sep|cls|mask|im_start|im_end)\|>", re.I), "prompt_contamination", "special_token"),
    (re.compile(r"\[/?(?:INST|SYS)\]", re.I), "prompt_contamination", "instruction_token"),
    (re.compile(r"<</?SYS>>", re.I), "prompt_contamination", "llama_sys_token"),
    # multi-turn chat formatting leaked
    (re.compile(r"(?:^|\n)(?:Human|AI|User|Assistant|System|Bot|Agent|ChatGPT|GPT)\s*:", re.I | re.M), "prompt_contamination", "chat_turn_marker"),
    # reward/preference annotation artifacts
    (re.compile(r"\b(?:chosen|rejected|preferred|reward|score)\s*(?::|=)\s*", re.I), "prompt_contamination", "reward_annotation"),
    # full solution walkthrough that accidentally ended up in question
    (re.compile(r"\b(?:solving|working through|let's work|working out)\s+(?:the |this )?(?:problem|equation|expression)\b", re.I), "prompt_contamination", "solution_walkthrough"),
]


def detect_prompt_contamination(question: str, raw_answer: str) -> tuple[str, str]:
    text = question.strip()
    for pattern, filter_tag, reason in _PROMPT_CONTAMINATION_PATTERNS:
        if pattern.search(text):
            return filter_tag, reason

    normalized_question = normalize_answer_text(text)
    normalized_answer = normalize_answer_text(raw_answer)
    if len(normalized_answer) >= 8 and normalized_answer in normalized_question:
        if any(cue in normalized_question for cue in ("answer", "solution", "final answer", "boxed")):
            return "answer_leakage", "answer_overlap"

    return "", ""


@register_processor("prompt_contamination_filter")
class PromptContaminationFilterProcessor(RecordProcessor):
    name = "prompt_contamination_filter"

    def process(self, record: CanonicalRecord) -> ProcessorResult:
        if record.training_phase:
            record.add_trace(
                stage="clean",
                processor=self.name,
                status="skipped",
                details={"training_phase": record.training_phase},
            )
            return ProcessorResult(keep=True, record=record, stage="clean", processor=self.name)

        filter_tag, reason = detect_prompt_contamination(record.question, record.raw_dataset_answer)
        if filter_tag:
            updated = record.clone(training_phase="midtrain", filter_tag=filter_tag)
            updated.add_trace(
                stage="clean",
                processor=self.name,
                status="routed",
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
