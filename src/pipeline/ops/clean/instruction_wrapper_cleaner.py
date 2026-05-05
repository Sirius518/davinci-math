from __future__ import annotations

import re

from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, ProcessorResult, RecordProcessor

_WRAPPER_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # --- prefix wrappers ---
    (re.compile(
        r"^Below is a (?:math|chemistry|physics) question\.?\s*"
        r"I want you to reason through the steps.*?"
        r"(?:final answer.*?\\boxed\{?\}?\.?\s*\n?)",
        re.I,
    ), ""),
    (re.compile(
        r"^Think step by step\.?\s*put the answer.*?\\boxed\{?\}?\.?\s*\n?",
        re.I,
    ), ""),
    (re.compile(
        r"^Solve the following (?:math|chemical|physics|biological) problem step by step\.?\s*"
        r"The last line.*?\\boxed\{\$Answer\}.*?\n?",
        re.I,
    ), ""),
    (re.compile(
        r"^Solve the following math problem\.?\s*"
        r"Make sure to put the answer.*?\\boxed\{?\}?\.?\s*\n?",
        re.I,
    ), ""),
    (re.compile(
        r"^Return your final response within \\boxed\{?\}?\.?\s*\n?",
        re.I,
    ), ""),
    # \{\}.\n prefix (Llama-Nemotron)
    (re.compile(r"^\s*\\?\{\\?\}\s*\.?\s*\n?", re.I), ""),
    (re.compile(r"^\s*Question:\s*", re.I), ""),
    (re.compile(r"^\s*Your (?:entire )?response should.*?\.\s*\n?", re.I), ""),
    (re.compile(r"^\s*(?:Final Answer Format|Provide|Note):.*?\n", re.I), ""),
    (re.compile(r"^\s*(?:Solution|Final Answer|Answer)\s*:\s*", re.I), ""),
    (re.compile(r"^\s*(?:Let us|Let's)\s+(?:solve|reason)\s+step by step\.?\s*", re.I), ""),
    (re.compile(r"^\s*<(?:think|assistant|user|system)>\s*", re.I), ""),
    (re.compile(r"^\s*(?:System|User|Assistant)\s*:\s*", re.I), ""),
    # "Please reason step by step, and put your final answer within \boxed{}."
    (re.compile(
        r"^Please reason step by step,?\s*(?:and )?put your final answer "
        r"within \\boxed\{?\}?\.?\s*\n?",
        re.I,
    ), ""),
    # role / persona preambles (aggressive strip)
    (re.compile(
        r"^\s*(?:You are|You're|I want you to (?:act|be|play|serve) as|"
        r"Act(?:ing)? as|Imagine (?:you are|you're|that you)|"
        r"Pretend (?:you are|you're|to be)|"
        r"Assume (?:you are|you're|the role)|"
        r"Your role is|My role is|"
        r"Play(?:ing)? the role of|"
        r"Respond (?:as|like) (?:a |an ))"
        r"[^.!?\n]{3,80}[.!?]?\s*\n?",
        re.I,
    ), ""),
    # "Here is the/a problem:" / "Here is a math question:" preambles
    (re.compile(r"^\s*(?:Here is|Here's|Below is|Consider)\s+(?:a|an|the)\s+(?:math |)(?:problem|question|exercise|challenge|task)\s*[:.]?\s*\n?", re.I), ""),
    # "I need you to..." / "I'd like you to..." / "Can you please..."
    (re.compile(r"^\s*(?:I (?:need|want|would like|'d like) you to|Can you (?:please )?|Could you (?:please )?|Would you (?:please )?|Please )\s*", re.I), ""),
    # "Answer the following question:" / "Solve the following:"
    (re.compile(r"^\s*(?:Answer|Solve|Evaluate|Compute|Calculate|Find|Determine)\s+the\s+following\s*(?:question|problem|expression|equation)?s?\s*[:.]?\s*\n?", re.I), ""),
    # --- suffix wrappers ---
    (re.compile(
        r"\s*(?:Can you help me solve this problem and )?put (?:the|your) final answer "
        r"(?:inside|within|in) \\boxed\{?\*?\}?\??\s*$",
        re.I,
    ), ""),
    (re.compile(
        r"\s*(?:Please )?(?:reason step by step,? and )?put your final answer "
        r"within \\boxed\{?\*?\}?\.?\s*$",
        re.I,
    ), ""),
    (re.compile(
        r"\s*Remember to put your final answer within \\boxed\{?\*?\}?\.?\s*$",
        re.I,
    ), ""),
    # "Can you help me solve this problem and put the final answer inside \boxed{}?"
    (re.compile(
        r"\s*Can you help me solve this (?:math )?problem"
        r"(?:\s+and put (?:the|your) final answer (?:inside|within|in) \\boxed\{?\}?)?\??\s*$",
        re.I,
    ), ""),
    (re.compile(r"\s*Express your answer as.*$", re.I), ""),
    # "(without quotes) where $Answer is the answer to the problem."
    (re.compile(
        r"\s*\(?without quotes\)?\s*where\s*\$Answer\s*is\s*the\s*answer\s*"
        r"to\s*the\s*problem\.?\s*$",
        re.I,
    ), ""),
    # truncated suffixes
    (re.compile(r"\s*Remember to\s*$", re.I), ""),
    (re.compile(r"\s*Please reason step by step,?\s*(?:and)?\s*$", re.I), ""),
    (re.compile(r"\s*Please put the final answer\s*$", re.I), ""),
    # "Present the answer in LaTex format: \boxed{Your answer}"
    (re.compile(
        r"\s*Present the answer in La[Tt]e[Xx] format:?\s*\\boxed\{.*?\}\s*$",
        re.I,
    ), ""),
    # "Answer with a json with the final response inside an 'answer' field..."
    (re.compile(
        r"\s*Answer with a json with the final response.*$",
        re.I,
    ), ""),
    # "Answer in the format: $\\boxed{answer}$"
    (re.compile(
        r"\s*Answer in the format:?\s*\$?\\boxed\{.*?\}\$?\.?\s*$",
        re.I,
    ), ""),
    (re.compile(r"\s*(?:Final Answer|Answer)\s*:\s*\\boxed\{.*?\}\s*$", re.I), ""),
    # "Give your final answer in a \\boxed{}."
    (re.compile(
        r"\s*Give (?:your|the) final answer in (?:a )?\\boxed\{?\}?\.?\s*$",
        re.I,
    ), ""),
    # "Make sure to put the answer (and only answer) inside \boxed{}."
    (re.compile(
        r"\s*Make sure to put the answer\s*(?:\(and only (?:the )?answer\)\s*)?"
        r"(?:inside|within|in) \\boxed\{?\}?\.?\s*$",
        re.I,
    ), ""),
    # "The answer is in the format \boxed{answer}."
    (re.compile(
        r"\s*The answer (?:is|should be) in the format\s*\\boxed\{.*?\}\.?\s*$",
        re.I,
    ), ""),
    # hex / json answer format instructions
    (re.compile(
        r"\s*(?:Provide|Give|Return) (?:the|your) answer (?:as|in) "
        r"(?:a )?(?:hex(?:adecimal)?|json|JSON|XML|csv).*$",
        re.I,
    ), ""),
    (re.compile(r"\s*</(?:think|assistant|user|system)>\s*$", re.I), ""),
    # "Show your work." / "Show all steps." / "Show your reasoning."
    (re.compile(r"\s*Show (?:your |all )?(?:work|steps?|reasoning|calculations?|process)\.?\s*$", re.I), ""),
    # "Be concise." / "Be precise." / "Be thorough."
    (re.compile(r"\s*Be (?:concise|precise|thorough|detailed|brief|accurate|careful|clear|specific)\.?\s*$", re.I), ""),
    # "Think carefully." / "Think step by step."
    (re.compile(r"\s*Think (?:carefully|step by step|about it|through|hard|logically)\.?\s*$", re.I), ""),
    # "Do not explain." / "Do not show work." / "Only give the answer."
    (re.compile(r"\s*(?:Do not|Don't|Please don't|Please do not)\s+(?:explain|show|include|add|provide|write|give|use).*$", re.I), ""),
    (re.compile(r"\s*Only (?:give|provide|show|output|return|state) (?:the |your )?(?:answer|result|final answer|number)\.?\s*$", re.I), ""),
    # "Note: ..." at the end
    (re.compile(r"\s*Note\s*:.*$", re.I), ""),
    # "Important: ..." at the end
    (re.compile(r"\s*Important\s*:.*$", re.I), ""),
    # "Hint: ..." at the end
    (re.compile(r"\s*Hint\s*:.*$", re.I), ""),
    # "[/INST]" / "[INST]" / "<<SYS>>" / "<|im_start|>" etc.
    (re.compile(r"\s*(?:\[/?INST\]|<<?/?SYS>>?|<\|(?:im_start|im_end|endoftext|pad)\|>)\s*$", re.I), ""),
    (re.compile(r"^\s*(?:\[/?INST\]|<<?/?SYS>>?|<\|(?:im_start|im_end|endoftext|pad)\|>)\s*", re.I), ""),
]

_MAX_CLEAN_ITERATIONS = 5
_NOISE_LINE_PATTERNS = [
    re.compile(r"^\s*```[\w-]*\s*$"),
    re.compile(r"^\s*diff --git\b"),
    re.compile(r"^\s*deleted file mode\b"),
    re.compile(r"^\s*index [0-9a-f]+\.\.[0-9a-f]+\s+\d+", re.I),
    re.compile(r"^\s*(?:---|\+\+\+) [ab/].*"),
    re.compile(r"^\s*@@ .* @@\s*$"),
    re.compile(r"^\s*<(?:think|assistant|user|system)>\s*$", re.I),
    re.compile(r"^\s*</(?:think|assistant|user|system)>\s*$", re.I),
    re.compile(r"^\s*(?:System|User|Assistant)\s*:\s*$", re.I),
]


def _strip_noise_lines(text: str) -> str:
    lines = text.splitlines()
    while lines and any(pattern.search(lines[0]) for pattern in _NOISE_LINE_PATTERNS):
        lines.pop(0)
    while lines and any(pattern.search(lines[-1]) for pattern in _NOISE_LINE_PATTERNS):
        lines.pop()
    return "\n".join(lines).strip()


@register_processor("instruction_wrapper_cleaner")
class InstructionWrapperCleanerProcessor(RecordProcessor):
    name = "instruction_wrapper_cleaner"

    def process(self, record: CanonicalRecord) -> ProcessorResult:
        text = record.question
        cleaned = text
        for _ in range(_MAX_CLEAN_ITERATIONS):
            prev = cleaned
            for pattern, replacement in _WRAPPER_PATTERNS:
                cleaned = pattern.sub(replacement, cleaned)
            cleaned = _strip_noise_lines(cleaned)
            cleaned = cleaned.strip()
            if cleaned == prev:
                break

        if cleaned != text.strip():
            updated = record.clone(question=cleaned)
            updated.add_trace(
                stage="clean",
                processor=self.name,
                status="kept",
                details={"wrapper_stripped": True, "original_length": len(text), "cleaned_length": len(cleaned)},
            )
            return ProcessorResult(keep=True, record=updated, stage="clean", processor=self.name)

        record.add_trace(stage="clean", processor=self.name, status="kept", details={"wrapper_stripped": False})
        return ProcessorResult(keep=True, record=record, stage="clean", processor=self.name)
