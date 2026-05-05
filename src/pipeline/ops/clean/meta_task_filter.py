from __future__ import annotations

import re

from pipeline.core.registry import register_processor
from pipeline.core.schema import CanonicalRecord, ProcessorResult, RecordProcessor

_META_TASK_PATTERNS = [
    # --- original solution-selection / meta-evaluation ---
    re.compile(r"you will be given a challenging math problem followed by \d+ solutions", re.I),
    re.compile(r"your task is to systematically analyze these solutions", re.I),
    re.compile(r"identify the most mathematically sound approach", re.I),
    re.compile(r"solutions:\s*detailed solutions indexed", re.I),
    re.compile(r"Input Format:\s*Problem:", re.I),
    re.compile(r"Solution [0-9]+:.*Solution [0-9]+:", re.DOTALL),
    re.compile(r"each concluding with an answer in \\boxed\{\}", re.I),
    re.compile(r"provide a good and informational response.+like a helpful human", re.I),
    re.compile(r"answer the following question.*\n\[Q\]:", re.I),
    re.compile(r"now provide the response and nothing else", re.I),
    re.compile(r"\b(?:follow|obey)\s+(?:the )?system rules\b", re.I),
    re.compile(r"\bretrieve and repeat the exact described function\b", re.I),
    re.compile(r"\bexact(?:ly)? \d+ bullet points\b", re.I),
    re.compile(r"\bgive me \d+ (?:tags|titles|keywords)\b", re.I),
    re.compile(r"\bfor my youtube video\b", re.I),
    re.compile(r"\bwrite a hypothetical\b", re.I),
    re.compile(r"\b(?:summarize|rewrite|proofread|paraphrase|analyze)\b.*\b(?:film|video|document|article|text)\b", re.I),
    re.compile(r"\b(?:repeat|extract|retrieve)\b.*\b(?:code context|function|snippet)\b", re.I),
    re.compile(r"\b(?:platform|product|app|feature)\b.*\b(?:design|requirements?|specification)\b", re.I),
    re.compile(r"\b(?:roleplay|persona|you are an ai named)\b", re.I),
    # --- grading / rating / evaluation meta-tasks ---
    re.compile(r"\b(?:grade|rate|score|rank|evaluate|assess|judge|critique|review)\s+"
               r"(?:the |this |these |my |each )?(?:solution|answer|response|essay|submission|work|paper|code|argument)\b", re.I),
    re.compile(r"\bon a scale (?:of|from) \d+", re.I),
    re.compile(r"\bprovide (?:a |detailed )?feedback\b", re.I),
    re.compile(r"\bcheck (?:the |this |my )?(?:solution|answer|work|homework|proof)\s+(?:for|and)", re.I),
    # --- instruction / prompt engineering ---
    re.compile(r"\byou must (?:always|never|only|strictly|exactly)\b", re.I),
    re.compile(r"\brule(?:s)?:\s*\d+\.", re.I),
    re.compile(r"\b(?:constraint|instruction|guideline|requirement)s?\s*:\s*(?:\n|1\.|-)", re.I),
    re.compile(r"\brespond (?:only |exclusively )?(?:in|with|using) (?:json|xml|yaml|csv|markdown|html|code)\b", re.I),
    re.compile(r"\bdo not (?:use|include|mention|add|generate|output|provide|write|say)\b", re.I),
    re.compile(r"\byou (?:should|must|shall|will) (?:not |never )?(?:include|output|generate|respond|provide|use)\b", re.I),
    # --- content generation / creative tasks ---
    re.compile(r"\b(?:generate|produce|come up with|brainstorm|suggest)\s+\d+\s+(?:\w+\s+)?"
               r"(?:ideas?|names?|titles?|slogans?|taglines?|captions?|headlines?|"
               r"questions?|examples?|scenarios?|prompts?|alternatives?|variations?|options?)\b", re.I),
    re.compile(r"\b(?:write|create|draft|compose|generate)\s+(?:a |an )?"
               r"(?:cover letter|resignation|recommendation|apology|thank you|invitation|"
               r"complaint|reference|condolence|congratulation)\s*(?:letter|note|message|email)?\b", re.I),
    re.compile(r"\b(?:write|create|generate)\s+(?:a |an )?"
               r"(?:resume|cv|curriculum vitae|bio(?:graphy)?|profile|portfolio)\b", re.I),
    # --- data / text processing meta-tasks ---
    re.compile(r"\b(?:classify|categorize|label|tag|annotate|segment|cluster)\s+"
               r"(?:the |this |these |each |every )?(?:text|sentence|document|review|comment|tweet|post|data|sample|record)\b", re.I),
    re.compile(r"\b(?:extract|identify|detect|recognize|find)\s+"
               r"(?:the |all |every )?(?:entities|keywords?|topics?|sentiments?|intents?|named entities|relations?)\b", re.I),
    re.compile(r"\bsentiment (?:analysis|classification|detection|scoring)\b", re.I),
    re.compile(r"\bnamed entity (?:recognition|extraction|tagging)\b", re.I),
    re.compile(r"\btopic (?:modeling|classification|extraction|detection)\b", re.I),
    re.compile(r"\btext (?:classification|clustering|summarization|generation|mining|preprocessing)\b", re.I),
    # --- Q&A about self / AI ---
    re.compile(r"\bwhat (?:kind|type) of (?:AI|model|assistant|bot|system) are you\b", re.I),
    re.compile(r"\bwhat (?:are|is) your (?:capabilities?|limitations?|strengths?|weaknesses?|features?)\b", re.I),
    re.compile(r"\b(?:who|what) (?:made|created|trained|built|designed|developed) you\b", re.I),
    # --- comparison / selection meta-tasks ---
    re.compile(r"\bwhich (?:solution|answer|approach|method|option|response) is (?:better|best|correct|more accurate)\b", re.I),
    re.compile(r"\bcompare (?:the |these |both )?(?:\w+ )?(?:solutions?|answers?|approaches?|methods?|responses?)\b", re.I),
    re.compile(r"\bselect the (?:best|correct|most accurate|optimal) (?:solution|answer|approach|response)\b", re.I),
    # --- teaching / tutoring meta-tasks ---
    re.compile(r"\b(?:explain|teach|tutor|guide|walk) (?:me|us|the student|a student)\s+"
               r"(?:through|about|how to|the concept|step by step)\b", re.I),
    re.compile(r"\bbreak (?:it|this|the problem) down (?:into|for|step)\b", re.I),
    re.compile(r"\b(?:hint|clue|scaffolding|guidance)\s*:\s*", re.I),
    # --- system / formatting / output instructions ---
    re.compile(r"\b(?:system prompt|system message|system instruction)\b", re.I),
    re.compile(r"\b(?:output format|response format|answer format)\s*:\s*", re.I),
    re.compile(r"\breturn (?:the |your )?(?:answer|response|output|result) (?:as|in) (?:a )?(?:json|xml|yaml|csv|table|list|markdown)\b", re.I),
    re.compile(r"\bformat (?:the |your )?(?:answer|response|output) (?:as|in|like|using)\b", re.I),
    # --- role-playing / persona (aggressive) ---
    re.compile(r"\byou are (?:a |an )?(?:helpful|friendly|creative|expert|knowledgeable|professional|experienced|senior)\s+"
               r"(?:assistant|AI|bot|agent|advisor|consultant|writer|editor|programmer|developer|designer)\b", re.I),
    re.compile(r"\b(?:your|my) role is\b", re.I),
    re.compile(r"\bplay(?:ing)? the role of\b", re.I),
    re.compile(r"\b(?:imagine|pretend|assume|suppose) (?:you are|you're|that you)\b", re.I),
    re.compile(r"\bact(?:ing)? as (?:a |an |the |if )?\w+", re.I),
    re.compile(r"\brespond (?:as if you (?:are|were)|like|as) (?:a |an )\w+", re.I),
    # --- multi-turn / conversation artifacts ---
    re.compile(r"\b(?:continue|resume) (?:the |our |this )?(?:conversation|discussion|chat|dialogue)\b", re.I),
    re.compile(r"\b(?:in |from )(?:the |our )?(?:previous|last|earlier) (?:message|response|conversation|turn)\b", re.I),
    re.compile(r"\b(?:as I |as we )(?:discussed|mentioned|said|noted|talked about)\b", re.I),
    # --- broad "list / enumerate" non-math tasks ---
    re.compile(r"\b(?:list|enumerate|name|mention|give me|provide)\s+(?:the |all |some |a few )?\d+\s+"
               r"(?:reasons?|things?|ways?|steps?|benefits?|advantages?|disadvantages?|"
               r"features?|characteristics?|types?|categories?|methods?|strategies?|"
               r"techniques?|approaches?|tools?|resources?|books?|articles?|websites?|"
               r"apps?|products?|services?|companies?|countries?|cities?|languages?|"
               r"tips?|facts?|myths?|mistakes?|challenges?|problems?|solutions?|"
               r"principles?|rules?|laws?|theories?|concepts?|ideas?|trends?)\b", re.I),
    # --- translation tasks ---
    re.compile(r"\btranslate (?:the (?:following )?|this |these |following )?"
               r"(?:text|sentence|paragraph|passage|phrase|word|document)\b", re.I),
    re.compile(r"\btranslat(?:e|ion)\s+(?:from|into|to)\s+"
               r"(?:english|french|spanish|german|chinese|japanese|korean|arabic|russian|"
               r"portuguese|italian|dutch|hindi|bengali|thai|turkish|vietnamese)\b", re.I),
]


@register_processor("meta_task_filter")
class MetaTaskFilterProcessor(RecordProcessor):
    name = "meta_task_filter"

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
        for pattern in _META_TASK_PATTERNS:
            if pattern.search(text):
                updated = record.clone(training_phase="midtrain", filter_tag="meta_task")
                updated.add_trace(
                    stage="clean",
                    processor=self.name,
                    status="routed",
                    reason_code="meta_task",
                    details={"matched_pattern": pattern.pattern[:80]},
                )
                return ProcessorResult(
                    keep=True,
                    record=updated,
                    stage="clean",
                    processor=self.name,
                    reason_code="meta_task",
                )

        record.add_trace(stage="clean", processor=self.name, status="kept")
        return ProcessorResult(keep=True, record=record, stage="clean", processor=self.name)
